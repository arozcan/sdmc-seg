import pytorch_lightning as pl
import torch
import inspect
import torch.nn.functional as F
from monai.networks.nets import FlexibleUNet, UNet, UNETR, SwinUNETR, BasicUNetPlusPlus, AttentionUnet, SegResNet
from monai.networks.nets.basic_unet import BasicUNet
from monai.losses import DiceLoss, TverskyLoss, DiceCELoss, DiceFocalLoss, FocalLoss, GeneralizedDiceLoss
from typing import Optional, Sequence, Literal
#from utils.dice import DiceLoss
from monai.metrics import DiceMetric, MeanIoU
from utils.image_processing_utils import modify_batch_image_channels_for_stroke_dataset, overlay_images_batch, colorize_multichannel_segment, overlay_images
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from torch import autocast
from tqdm import tqdm
from scripts.slice2seg import dice_score, iou_score
import os
from PIL import Image
from utils.tolerant_metrics import TolerantMeanIoU
from utils.metrics import _dice_iou_onehot, _nanmean_safe, compute_tolerant_iou, sensitivity_specificity_onehot, hd95_onehot
from omegaconf import ListConfig
from monai.data import list_data_collate
from monai.inferers import sliding_window_inference
import csv
import math


from ldm.models.transunet import TransUNet

class UNetLightning(pl.LightningModule):
    def __init__(
        self,
        model_type: str = "UNet",  # 'flexible' or 'unet'
        spatial_dims: int = 2,
        in_channels: int = 1,
        out_channels: int = 3,
        learning_rate: float = 1e-4,
        num_classes: int = 3,
        monitor: str = "val/val_iou",
        # Basic UNet only params
        features: tuple = (32, 32, 64, 128, 256, 32),
        # UNet and AttentionUnet only params
        channels: tuple = (16, 32, 64, 128, 256),
        # UNet only params
        strides: tuple = (2, 2, 2, 2),
        num_res_units: int = 2,
        # FlexibleUNet only params
        backbone: str = "resnet34",
        pretrained: bool = False,
        decoder_channels: tuple = (256, 128, 64, 32, 16),
        # SwinUNETR and TransUNet-style (ViT encoder + UNet decoder) params
        feature_size: int = 24,
        transunet_variant: str = "R50-ViT-B_16",
        # SegResNet only params
        init_filters: int = 16,
        # Optional per-class weights for loss [C] (including background)
        class_weights: Optional[Sequence[float]] = None,
        # Optional per-slice lesion-ratio based loss weighting
        use_lesion_ratio_weighting: bool = False,
        lesion_ratio_r_ref: float = 0.07,
        lesion_ratio_p: float = 0.5,
        lesion_ratio_a_min: float = 0.8,
        lesion_ratio_a_max: float = 3.0,
        lesion_ratio_eps: float = 1e-6,
        # Loss selection
        loss_name: Literal[
            "dice",
            "dice_alpha",
            "dice_ce",
            "focal",
            "dice_focal",
            "generalized_dice",
            "focal_tversky",
        ] = "dice",
        # Dice+CE params
        dice_ce_lambda_dice: float = 1.0,
        dice_ce_lambda_ce: float = 1.0,
        ce_weight: Optional[Sequence[float]] = None,
        # Focal / DiceFocal params
        focal_gamma: float = 2.0,
        focal_alpha: Optional[float] = None,
        dice_focal_lambda_dice: float = 1.0,
        dice_focal_lambda_focal: float = 1.0,
        # Focal Tversky params
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        focal_tversky_gamma: float = 1.33,
        # Scheduler (nnU-Net style)
        use_poly_lr: bool = True,
        poly_lr_power: float = 0.9,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_type = model_type
        self.num_classes = num_classes
        self.monitor = monitor
        self.learning_rate = learning_rate
        self.loss_name = loss_name
        self.use_poly_lr = use_poly_lr
        self.poly_lr_power = poly_lr_power
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.focal_tversky_gamma = focal_tversky_gamma

        # Dice+CE hyperparams
        self.dice_ce_lambda_dice = dice_ce_lambda_dice
        self.dice_ce_lambda_ce = dice_ce_lambda_ce

        # Optional CE class weights (for CrossEntropy / Focal components)
        self.ce_weight = None
        if ce_weight is not None:
            if isinstance(ce_weight, (list, tuple, np.ndarray)):
                ce_w = torch.tensor(ce_weight, dtype=torch.float32)
            elif isinstance(ce_weight, torch.Tensor):
                ce_w = ce_weight.float()
            else:
                raise ValueError(f"Unsupported type for ce_weight: {type(ce_weight)}")
            self.register_buffer("ce_weight", ce_w)

        # Focal / DiceFocal hyperparams
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.dice_focal_lambda_dice = dice_focal_lambda_dice
        self.dice_focal_lambda_focal = dice_focal_lambda_focal
        # If loss_name is dice_alpha, enable lesion-ratio weighting regardless of the flag
        self.use_lesion_ratio_weighting = use_lesion_ratio_weighting or (loss_name == "dice_alpha")
        self.lesion_ratio_r_ref = lesion_ratio_r_ref
        self.lesion_ratio_p = lesion_ratio_p
        self.lesion_ratio_a_min = lesion_ratio_a_min
        self.lesion_ratio_a_max = lesion_ratio_a_max
        self.lesion_ratio_eps = lesion_ratio_eps

        # Optional per-class weights (including background) for pixel-level balancing
        if class_weights is not None:
            if isinstance(class_weights, (list, tuple, np.ndarray)):
                cw_tensor = torch.tensor(class_weights, dtype=torch.float32)
            elif isinstance(class_weights, torch.Tensor):
                cw_tensor = class_weights.float()
            elif ListConfig is not None and isinstance(class_weights, ListConfig):
                cw_tensor = torch.tensor(list(class_weights), dtype=torch.float32)
            else:
                raise ValueError(f"Unsupported type for class_weights: {type(class_weights)}")
            # register_buffer ensures the weights move with the model to the correct device
            self.register_buffer("class_weights", cw_tensor)
        else:
            self.class_weights = None

        def _sig_has(obj, name: str) -> bool:
            try:
                return name in inspect.signature(obj).parameters
            except Exception:
                return False

        def _kw_softmax(loss_cls, kwargs: dict) -> dict:
            """Add the correct 'softmax' kwarg for the installed MONAI version (if supported)."""
            if _sig_has(loss_cls.__init__, "softmax"):
                kwargs["softmax"] = True
            elif _sig_has(loss_cls.__init__, "use_softmax"):
                kwargs["use_softmax"] = True
            return kwargs

        # Loss function
        # NOTE:
        # - "dice": standard DiceLoss (scalar)
        # - "dice_alpha": DiceLoss with per-slice lesion-ratio weighting (needs unreduced output)
        # - "dice_ce": Dice + CrossEntropy (common strong baseline; nnU-Net style)
        # - "focal": Focal loss (pixel-wise) for class imbalance
        # - "dice_focal": Dice + Focal (addresses reviewer "focal loss" request while keeping overlap term)
        # - "generalized_dice": Generalized Dice (more robust under severe imbalance)
        # - "focal_tversky": (existing) Tversky + gamma focusing

        if self.loss_name in ("dice", "dice_alpha"):
            dice_reduction = "none" if self.loss_name == "dice_alpha" else "mean"
            dice_kwargs = {
                "include_background": True,
                "to_onehot_y": True,
                "jaccard": False,
                "smooth_nr": 1e-20,
                "smooth_dr": 1e-20,
                "weight": self.class_weights,
                "reduction": dice_reduction,
            }
            dice_kwargs = _kw_softmax(DiceLoss, dice_kwargs)
            self.loss_fn = DiceLoss(**dice_kwargs)

        elif self.loss_name == "dice_ce":
            # Dice + CE combined loss (returns scalar)
            # NOTE: In this MONAI version DiceCELoss does NOT accept `ce_weight`.
            # It uses a single `weight` tensor for BOTH Dice and CrossEntropy components.
            # We therefore choose `ce_weight` if provided, otherwise fall back to `class_weights`.
            _w = self.ce_weight if self.ce_weight is not None else self.class_weights
            dice_ce_kwargs = {
                "include_background": True,
                "to_onehot_y": True,
                "smooth_nr": 1e-20,
                "smooth_dr": 1e-20,
                "lambda_dice": self.dice_ce_lambda_dice,
                "lambda_ce": self.dice_ce_lambda_ce,
                "weight": _w,
                "reduction": "mean",
            }
            dice_ce_kwargs = _kw_softmax(DiceCELoss, dice_ce_kwargs)
            self.loss_fn = DiceCELoss(**dice_ce_kwargs)

        elif self.loss_name == "focal":
            # Pixel-wise focal loss (scalar). Uses softmax over classes.
            # NOTE: alpha here is scalar balancing between classes; for per-class weighting use ce_weight.
            focal_kwargs = {
                "include_background": True,
                "to_onehot_y": True,
                "gamma": self.focal_gamma,
                "alpha": self.focal_alpha,
                "weight": self.ce_weight,
                "reduction": "mean",
            }
            focal_kwargs = _kw_softmax(FocalLoss, focal_kwargs)
            self.loss_fn = FocalLoss(**focal_kwargs)

        elif self.loss_name == "dice_focal":
            # Dice + Focal combined (scalar)
            dice_focal_kwargs = {
                "include_background": True,
                "to_onehot_y": True,
                "smooth_nr": 1e-20,
                "smooth_dr": 1e-20,
                "gamma": self.focal_gamma,
                "alpha": self.focal_alpha,
                "lambda_dice": self.dice_focal_lambda_dice,
                "lambda_focal": self.dice_focal_lambda_focal,
                "weight": self.class_weights,
                "reduction": "mean",
            }
            dice_focal_kwargs = _kw_softmax(DiceFocalLoss, dice_focal_kwargs)
            self.loss_fn = DiceFocalLoss(**dice_focal_kwargs)

        elif self.loss_name == "generalized_dice":
            gd_kwargs = {
                "include_background": True,
                "to_onehot_y": True,
                "smooth_nr": 1e-20,
                "smooth_dr": 1e-20,
                "reduction": "mean",
            }
            gd_kwargs = _kw_softmax(GeneralizedDiceLoss, gd_kwargs)
            self.loss_fn = GeneralizedDiceLoss(**gd_kwargs)

        elif self.loss_name == "focal_tversky":
            # TverskyLoss returns a scalar (with reduction="mean")
            tv_kwargs = {
                "include_background": True,
                "to_onehot_y": True,
                "alpha": self.tversky_alpha,
                "beta": self.tversky_beta,
                "smooth_nr": 1e-20,
                "smooth_dr": 1e-20,
                "reduction": "mean",
            }
            tv_kwargs = _kw_softmax(TverskyLoss, tv_kwargs)
            self.loss_fn = TverskyLoss(**tv_kwargs)

        else:
            raise ValueError(f"Unsupported loss_name: {self.loss_name}")
        self.val_dice = DiceMetric(include_background=True, reduction="mean", get_not_nans=False, ignore_empty=True)
        self.val_iou = MeanIoU(include_background=True, reduction="mean", get_not_nans=False, ignore_empty=True)
        self.test_dice = DiceMetric(include_background=True, reduction="mean", get_not_nans=False, ignore_empty=True)
        self.test_iou = MeanIoU(include_background=True, reduction="mean", get_not_nans=False, ignore_empty=True)
        self.test_tiou = TolerantMeanIoU(include_background=True, reduction="mean", get_not_nans=False, ignore_empty=True)

        
        if model_type == "BasicUNet":
            self.model = BasicUNet(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                features=features
            )
        elif model_type == "ResUNet":
            self.model = UNet(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                channels=channels,
                strides=strides,
                num_res_units=num_res_units,
            )
        elif model_type == "FlexibleUNet":
            self.model = FlexibleUNet(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                decoder_channels=decoder_channels,
                backbone=backbone,
                pretrained=pretrained,
            )
        elif model_type == "UNETR":
            self.model = UNETR(
                img_size=(256,256),
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                feature_size=feature_size,
            )
        elif model_type == "TransUNet":
            self.model = TransUNet(
                in_channels=in_channels,
                out_channels=out_channels,
                img_size=(256,256),
                variant=transunet_variant,
                vis=False,
            )
        elif model_type == "SwinUNETR":
            self.model = SwinUNETR(
                img_size=(256,256),
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                feature_size=feature_size,
                use_checkpoint=False
            )
        elif model_type == "AttentionUnet":
            self.model = AttentionUnet(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                channels=channels,
                strides=strides
            )
        elif model_type == "SegResNet":
            self.model = SegResNet(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                init_filters = init_filters
            )
        else:
            raise ValueError(
                f"Invalid model_type '{model_type}'. Choose 'BasicUNet','ResUNet','FlexibleUNet','UNETR','TransUNet','SwinUNETR','AttentionUnet' or 'SegResNet'."
            )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch["image"], batch["segmentation"]
        images = images.permute(0, 3, 1, 2)
        targets = targets.permute(0, 3, 1, 2)
        preds = self(images)

        if self.loss_name == "focal_tversky":
            # scalar loss
            base = self.loss_fn(preds, targets)
            loss = torch.pow(base, self.focal_tversky_gamma)
            alpha = None
        else:
            # Other losses (dice/dice_alpha/dice_ce/focal/dice_focal/generalized_dice)
            loss_out = self.loss_fn(preds, targets)

            # If reduced already ("dice"), it is scalar.
            if loss_out.dim() == 0:
                loss = loss_out
                alpha = None
            else:
                # Expected shape for unreduced dice: (B, C) but be defensive
                if loss_out.dim() > 2:
                    loss_out = loss_out.view(loss_out.shape[0], -1)

                if self.use_lesion_ratio_weighting:
                    alpha, present = self._compute_lesion_ratio_alpha_and_present_class(targets)
                    B, C = loss_out.shape
                    mult = torch.ones((B, C), device=loss_out.device, dtype=loss_out.dtype)
                    mult[torch.arange(B, device=loss_out.device), present] = alpha
                    loss = (loss_out * mult).mean()
                else:
                    loss = loss_out.mean()
                    alpha = None

        self.log("train/train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if alpha is not None:
            # Useful diagnostics
            self.log("train/lesion_alpha_mean", alpha.mean(), on_step=False, on_epoch=True, prog_bar=False)
            self.log("train/lesion_alpha_max", alpha.max(), on_step=False, on_epoch=True, prog_bar=False)
        return loss


    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, targets = batch["image"], batch["segmentation"]
        images = images.permute(0, 3, 1, 2)
        targets = targets.permute(0, 3, 1, 2)
        preds = self(images).detach()
        preds_bin = self.preds_to_one_hot(preds)
        if self.loss_name == "focal_tversky":
            base = self.loss_fn(preds, targets)
            val_loss = torch.pow(base, self.focal_tversky_gamma)
        else:
            # Other losses (dice/dice_alpha/dice_ce/focal/dice_focal/generalized_dice)
            loss_out = self.loss_fn(preds, targets)

            if loss_out.dim() == 0:
                val_loss = loss_out
            else:
                if loss_out.dim() > 2:
                    loss_out = loss_out.view(loss_out.shape[0], -1)

                if self.use_lesion_ratio_weighting:
                    alpha, present = self._compute_lesion_ratio_alpha_and_present_class(targets)
                    B, C = loss_out.shape
                    mult = torch.ones((B, C), device=loss_out.device, dtype=loss_out.dtype)
                    mult[torch.arange(B, device=loss_out.device), present] = alpha
                    val_loss = (loss_out * mult).mean()
                else:
                    val_loss = loss_out.mean()
        
        targets_one_hot = self.labels_to_one_hot(targets)
        self.val_dice(preds_bin, targets_one_hot)
        self.val_iou(preds_bin, targets_one_hot)

        self.log("val/val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)

        return val_loss
    
    def on_validation_epoch_end(self, outputs=0):
        #mean_dice = self.val_dice.aggregate().item()
        val_dice_means = self.val_dice.aggregate("none")
        class_dices = []
        for i in range(val_dice_means.shape[1]):
            dice_value = val_dice_means[:,i].nanmean().item()
            class_dices.append(dice_value)
            self.log(f"val/val_dice_class_{i+1}", dice_value, on_epoch=True, prog_bar=True)
        mean_dice= sum(class_dices) / len(class_dices)
        self.log("val/val_dice", mean_dice, on_epoch=True, prog_bar=True)

        #mean_iou = self.val_iou.aggregate().item()
        val_iou_means = self.val_iou.aggregate("none")
        class_ious = []
        for i in range(val_iou_means.shape[1]):
            iou_value = val_iou_means[:,i].nanmean().item()
            class_ious.append(iou_value)
            self.log(f"val/val_iou_class_{i+1}", iou_value, on_step=False, on_epoch=True, prog_bar=True)
        mean_iou = sum(class_ious) / len(class_ious)
        self.log("val/val_iou", mean_iou, on_epoch=True, prog_bar=True)

    
    def on_train_epoch_start(self):
        self.val_dice.reset()
        self.val_iou.reset()
    
    def on_validation_epoch_start(self):
        self.val_dice.reset()
        self.val_iou.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # nnU-Net style Polynomial LR schedule:
        # lr(epoch) = lr0 * (1 - epoch/max_epochs) ** power
        if not getattr(self, "use_poly_lr", False):
            return optimizer

        def poly_lr(epoch: int):
            # trainer may not be attached during sanity checks
            max_epochs = getattr(getattr(self, "trainer", None), "max_epochs", None)
            if max_epochs is None or max_epochs <= 0:
                return 1.0
            # clamp to [0, max_epochs]
            e = float(max(0, min(epoch, max_epochs)))
            return (1.0 - (e / float(max_epochs))) ** float(getattr(self, "poly_lr_power", 0.9))

        scheduler = {
            "scheduler": LambdaLR(optimizer, lr_lambda=poly_lr),
            "interval": "epoch",
            "frequency": 1,
            "name": "poly_lr",
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
    
    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, **kwargs):  # TODO: ddim_steps
        log = dict()
        images, targets = batch["image"][:N], batch["segmentation"][:N]
        images = images.permute(0, 3, 1, 2)
        targets = targets.permute(0, 3, 1, 2)
        targets_one_hot = self.labels_to_one_hot(targets)
        preds = self(images).detach()
        
        
        targets_modified = modify_batch_image_channels_for_stroke_dataset(targets_one_hot)
        preds_modified = modify_batch_image_channels_for_stroke_dataset(preds, clip=1.0)
        targets_overlay = overlay_images_batch(images, targets_modified)
        preds_overlay = overlay_images_batch(images, preds_modified)
        preds_bin_overlay = overlay_images_batch(images, preds_modified, labels_to_bin=True)

        log["inputs"] = images
        log["targets"] = targets_overlay
        log["predictions"] = preds_overlay
        log["predictions_bin"] = preds_bin_overlay

        return log
    

    @torch.no_grad()
    def _predict_logits_auto(self, x: torch.Tensor, roi_size=(256, 256), overlap: float = 0.25) -> torch.Tensor:
        """Run model inference.

        - If spatial size is exactly roi_size, run a single forward.
        - Otherwise, use MONAI sliding_window_inference and stitch logits back.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W)
        roi_size : tuple
            Patch size for sliding window.
        overlap : float
            Overlap ratio in [0,1).

        Returns
        -------
        torch.Tensor
            Logits of shape (B, num_classes, H, W)
        """
        if x.dim() != 4:
            raise ValueError(f"Expected input of shape (B,C,H,W), got {tuple(x.shape)}")

        H, W = int(x.shape[-2]), int(x.shape[-1])
        if (H, W) == tuple(roi_size):
            return self(x)

        # Sliding window inference returns stitched logits with the same spatial size
        # Using gaussian blending tends to reduce patch seam artifacts.
        logits = sliding_window_inference(
            inputs=x,
            roi_size=roi_size,
            sw_batch_size=1,
            predictor=self,
            overlap=float(overlap),
            mode="gaussian",
        )
        return logits

    # @torch.no_grad()
    # def log_dice(self, data=None, save_dir=None, ddim_steps=50):
    #     if data is None: # if dataset is not None, means the call comes from inference script.
    #         dataset = self.trainer.datamodule.datasets["validation"]
    #         data = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, collate_fn=list_data_collate)

    #     # self.model.eval()     # ImageLogger will handle this
    #     metrics_dict = dict()
    #     seg_label_dict = dict()

    #     def get_dice(data, save_dir=None):
            
    #         def get_dice_loop(data, save_dir=None):
    #             self.test_dice.reset()
    #             self.test_iou.reset()
    #             self.test_tiou.reset()
    #             pbar = tqdm(data, desc="Validating Segmentation")   # volume-wise
    #             for prompts in pbar:
    #                 image, label = prompts["image"], prompts["segmentation"]
    #                 image = image.permute(0, 3, 1, 2)
    #                 label = label.permute(0, 3, 1, 2)
    #                 label_one_hot=self.labels_to_one_hot(label)
    #                 input = image.cuda()
    #                 # assert image.shape == label_one_hot.shape
    #                 slice_path = prompts["file_path_"]

    #                 # If the incoming slice is larger than the training ROI (e.g., >256x256),
    #                 # run patch-wise inference and stitch logits back.
    #                 #preds = self._predict_logits_auto(input, roi_size=(256, 256), overlap=0.25)
    #                 preds = self(input)
    #                 preds_bin = self.preds_to_one_hot(preds).cpu()
    #                 preds_np = preds.squeeze(0).cpu().permute(1, 2, 0).clamp(min=0, max=1).numpy()
    #                 preds_bin_np = (preds_np > 0.5)
    #                 label_one_hot_np = label_one_hot.squeeze(0).permute(1,2,0).numpy().round().astype(int)
    #                 image_np = image.squeeze(0).permute(1, 2, 0).numpy()

    #                 if save_dir is not None:
    #                     slice_name = slice_path[0].split("/")[-1]
    #                     save_input_path = os.path.join(save_dir, ".".join([slice_name.split(".")[0]+"-input", slice_name.split(".")[-1]]))
    #                     save_gt_path = os.path.join(save_dir, ".".join([slice_name.split(".")[0]+"-gt", slice_name.split(".")[-1]]))
    #                     save_pred_path = os.path.join(save_dir, ".".join([slice_name.split(".")[0]+"-pred", slice_name.split(".")[-1]]))
    #                     save_logits_path = os.path.join(save_dir, ".".join([slice_name.split(".")[0]+"-logits", slice_name.split(".")[-1]]))
    #                     save_all_path = os.path.join(save_dir, ".".join(["all-"+slice_name.split(".")[0], slice_name.split(".")[-1]]))
                        
    #                     save_pred = (preds_bin_np*255).astype(np.uint8)
    #                     save_logits = (preds_np*255).astype(np.uint8)
    #                     save_gt = (label_one_hot_np*255).astype(np.uint8)
    #                     save_input = (image_np*255).astype(np.uint8)
    #                     if save_input.shape[-1] == 1:
    #                         save_input = np.repeat(save_input, 3, axis=-1)

    #                     save_pred = colorize_multichannel_segment(save_pred)
    #                     save_logits = colorize_multichannel_segment(save_logits)
    #                     save_gt = colorize_multichannel_segment(save_gt)
    #                     save_pred = overlay_images(save_input,save_pred)
    #                     save_logits = overlay_images(save_input,save_logits)
    #                     save_gt = overlay_images(save_input,save_gt)

    #                     save_all = np.concatenate((save_input, save_gt, save_pred, save_logits), axis=1)
                        
    #                     Image.fromarray(save_input).save(save_input_path)
    #                     Image.fromarray(save_gt).save(save_gt_path)
    #                     Image.fromarray(save_pred).save(save_pred_path)
    #                     Image.fromarray(save_all).save(save_all_path)
    #                     Image.fromarray(save_logits).save(save_logits_path)
                    

    #                 self.test_dice(preds_bin, label_one_hot)
    #                 self.test_iou(preds_bin, label_one_hot)
    #                 self.test_tiou(preds_bin, label_one_hot)
    #             pbar.close()

    #             test_dice_means = self.test_dice.aggregate("none")
    #             class_dices = []
    #             for idx in range(0, self.num_classes):
    #                 dice_value = test_dice_means[:,idx].nanmean().item()
    #                 class_dices.append(dice_value)
    #                 print(f"\033[31m[Mean Dice][cls {idx}]: {dice_value}\033[0m")
    #             mean_dice= sum(class_dices) / len(class_dices)
    #             print(f"\033[31m[Mean Dice]: {mean_dice}\033[0m")

    #             test_iou_means = self.test_iou.aggregate("none")
    #             class_ious = []
    #             for idx in range(0, self.num_classes):
    #                 iou_value = test_iou_means[:,idx].nanmean().item()
    #                 class_ious.append(iou_value)
    #                 print(f"\033[31m[Mean  IoU][cls {idx}]: {iou_value}\033[0m")
                
    #             mean_iou = sum(class_ious) / len(class_ious)
    #             print(f"\033[31m[Mean IoU]: {mean_iou}\033[0m")
                
    #             test_tiou_means = self.test_tiou.aggregate("none")
    #             class_tious = []
    #             for idx in range(0, self.num_classes):
    #                 tiou_value = test_tiou_means[:,idx].nanmean().item()
    #                 class_tious.append(tiou_value)
    #                 print(f"\033[31m[Mean  tIoU][cls {idx}]: {tiou_value}\033[0m")

    #             mean_tiou = sum(class_tious) / len(class_tious)
    #             print(f"\033[31m[Mean tIoU]: {mean_tiou}\033[0m")

    #             return class_dices, class_ious, class_tious


    #         precision_scope = autocast
    #         with torch.no_grad():
    #             with precision_scope("cuda"):
    #                 dice_list, iou_list, tiou_list= get_dice_loop(data, save_dir=save_dir)
            
    #         return dice_list, iou_list, tiou_list

    #     dice, iou, tiou = get_dice(data, save_dir=save_dir)
    #     multi_dice, multi_iou, multi_tiou = np.array(dice), np.array(iou), np.array(tiou)

    #     # choose one as the segmentation monitor
    #     metrics_dict.update({"val_avg_dice": list(multi_dice)})
    #     metrics_dict.update({"val_avg_iou": list(multi_iou)})
    #     metrics_dict.update({"val_avg_tiou": list(multi_tiou)})

    #     # self.model.train()    # ImageLogger will handle this
    #     return metrics_dict, seg_label_dict
        
    @torch.no_grad()
    def log_dice(self, data=None, save_dir=None, ddim_steps=50, tiou_kernel_size: int = 5):

        if data is None:
            dataset = self.trainer.datamodule.datasets["validation"]
            data = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, collate_fn=list_data_collate)

        metrics_dict = dict()
        seg_label_dict = dict()

        def get_dice_loop(data, save_dir=None):
            self.test_dice.reset()
            self.test_iou.reset()
            self.test_tiou.reset()

            rows = []
            csv_path = None
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                csv_path = os.path.join(save_dir, "metrics_per_slice.csv")

            pbar = tqdm(data, desc="Validating Segmentation")
            for prompts in pbar:
                image, label = prompts["image"], prompts["segmentation"]
                image = image.permute(0, 3, 1, 2)  # (B,C,H,W)
                label = label.permute(0, 3, 1, 2)  # (B,1,H,W) label-map

                #x = image.cuda(non_blocking=True)
                x = image.to(self.device, non_blocking=True)
                slice_path = prompts.get("file_path_", ["unknown"])
                slice_name = os.path.basename(slice_path[0]) if isinstance(slice_path, (list, tuple)) else os.path.basename(str(slice_path))

                # logits
                preds = self._predict_logits_auto(x, roi_size=(256, 256), overlap=0.25)
                preds_bin = self.preds_to_one_hot(preds)  # (B,C,H,W) float 0/1  (CUDA)

                # --- IMPORTANT: move GT onehot to SAME device ---
                label_one_hot = self.labels_to_one_hot(label).to(preds_bin.device, non_blocking=True)

                # per-slice dice/iou for class 1 & 2
                d1, i1 = _dice_iou_onehot(preds_bin, label_one_hot, c=1, ignore_empty=True)
                d2, i2 = _dice_iou_onehot(preds_bin, label_one_hot, c=2, ignore_empty=True)

                d1 = float(d1[0].item())
                i1 = float(i1[0].item())
                d2 = float(d2[0].item())
                i2 = float(i2[0].item())

                # per-slice sensitivity/specificity (one-vs-rest) for class 1 & 2
                sens1_t, spec1_t = sensitivity_specificity_onehot(preds_bin, label_one_hot, c=1, ignore_empty=True)
                sens2_t, spec2_t = sensitivity_specificity_onehot(preds_bin, label_one_hot, c=2, ignore_empty=True)

                sens1 = float(sens1_t[0].item())
                spec1 = float(spec1_t[0].item())
                sens2 = float(sens2_t[0].item())
                spec2 = float(spec2_t[0].item())

                # per-slice HD95 (computed on CPU for stability)
                hd95_bc = hd95_onehot(preds_bin.detach().cpu(), label_one_hot.detach().cpu(), include_background=True)
                hd1 = float(hd95_bc[0, 1].item())
                hd2 = float(hd95_bc[0, 2].item())

                tiou_bc = compute_tolerant_iou(
                    y_pred=preds_bin.to(label_one_hot.device),
                    y=label_one_hot,
                    include_background=True,
                    ignore_empty=True,
                    kernel_size=int(tiou_kernel_size),
                    threshold=0.5,
                )
                t1 = float(tiou_bc[0, 1].item())
                t2 = float(tiou_bc[0, 2].item())

                dice_mean = _nanmean_safe([d1, d2])
                iou_mean  = _nanmean_safe([i1, i2])
                tiou_mean = _nanmean_safe([t1, t2])
                sens_mean = _nanmean_safe([sens1, sens2])
                spec_mean = _nanmean_safe([spec1, spec2])
                hd95_mean = _nanmean_safe([hd1, hd2])

                rows.append({
                    "file": slice_name,
                    "dice_1": d1, "iou_1": i1, "tiou_1": t1,
                    "sens_1": sens1, "spec_1": spec1, "hd95_1": hd1,
                    "dice_2": d2, "iou_2": i2, "tiou_2": t2,
                    "sens_2": sens2, "spec_2": spec2, "hd95_2": hd2,
                    "dice_mean_1_2": dice_mean,
                    "iou_mean_1_2": iou_mean,
                    "tiou_mean_1_2": tiou_mean,
                    "sens_mean_1_2": sens_mean,
                    "spec_mean_1_2": spec_mean,
                    "hd95_mean_1_2": hd95_mean,
                })

                # --- Optional image dumping (same as you had) ---
                if save_dir is not None:
                    preds_np = preds.squeeze(0).detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
                    preds_bin_np = (preds_np > 0.5)
                    label_one_hot_np = label_one_hot.squeeze(0).detach().cpu().permute(1, 2, 0).numpy().round().astype(int)
                    image_np = image.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()

                    save_input_path  = os.path.join(save_dir, f"{os.path.splitext(slice_name)[0]}-input.png")
                    save_gt_path     = os.path.join(save_dir, f"{os.path.splitext(slice_name)[0]}-gt.png")
                    save_pred_path   = os.path.join(save_dir, f"{os.path.splitext(slice_name)[0]}-pred.png")
                    save_logits_path = os.path.join(save_dir, f"{os.path.splitext(slice_name)[0]}-logits.png")
                    save_all_path    = os.path.join(save_dir, f"all-{os.path.splitext(slice_name)[0]}.png")

                    save_pred  = (preds_bin_np * 255).astype(np.uint8)
                    save_logits = (preds_np * 255).astype(np.uint8)
                    save_gt    = (label_one_hot_np * 255).astype(np.uint8)
                    save_input = (image_np * 255).astype(np.uint8)
                    if save_input.shape[-1] == 1:
                        save_input = np.repeat(save_input, 3, axis=-1)

                    save_pred  = colorize_multichannel_segment(save_pred)
                    save_logits = colorize_multichannel_segment(save_logits)
                    save_gt    = colorize_multichannel_segment(save_gt)

                    save_pred  = overlay_images(save_input, save_pred)
                    save_logits = overlay_images(save_input, save_logits)
                    save_gt    = overlay_images(save_input, save_gt)

                    save_all = np.concatenate((save_input, save_gt, save_pred, save_logits), axis=1)

                    Image.fromarray(save_input).save(save_input_path)
                    Image.fromarray(save_gt).save(save_gt_path)
                    Image.fromarray(save_pred).save(save_pred_path)
                    Image.fromarray(save_all).save(save_all_path)
                    Image.fromarray(save_logits).save(save_logits_path)

                # --- Dataset-level MONAI metrics (as before) ---
                self.test_dice(preds_bin.cpu(), label_one_hot.cpu())
                self.test_iou(preds_bin.cpu(), label_one_hot.cpu())
                self.test_tiou(preds_bin.cpu(), label_one_hot.cpu())

            pbar.close()

            # Write CSV once
            if csv_path is not None:
                fieldnames = [
                    "file",
                    "dice_1","iou_1","tiou_1","sens_1","spec_1","hd95_1",
                    "dice_2","iou_2","tiou_2","sens_2","spec_2","hd95_2",
                    "dice_mean_1_2","iou_mean_1_2","tiou_mean_1_2",
                    "sens_mean_1_2","spec_mean_1_2","hd95_mean_1_2",
                ]
                with open(csv_path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    w.writerows(rows)
                print(f"[INFO] Saved per-slice metrics CSV: {csv_path}")

            # --- Aggregated prints (as you had) ---
            test_dice_means = self.test_dice.aggregate("none")
            class_dices = []
            for idx in range(0, self.num_classes):
                dice_value = test_dice_means[:, idx].nanmean().item()
                class_dices.append(dice_value)
                print(f"\033[31m[Mean Dice][cls {idx}]: {dice_value}\033[0m")
            mean_dice = sum(class_dices) / len(class_dices)
            print(f"\033[31m[Mean Dice]: {mean_dice}\033[0m")

            test_iou_means = self.test_iou.aggregate("none")
            class_ious = []
            for idx in range(0, self.num_classes):
                iou_value = test_iou_means[:, idx].nanmean().item()
                class_ious.append(iou_value)
                print(f"\033[31m[Mean  IoU][cls {idx}]: {iou_value}\033[0m")
            mean_iou = sum(class_ious) / len(class_ious)
            print(f"\033[31m[Mean IoU]: {mean_iou}\033[0m")

            test_tiou_means = self.test_tiou.aggregate("none")
            class_tious = []
            for idx in range(0, self.num_classes):
                tiou_value = test_tiou_means[:, idx].nanmean().item()
                class_tious.append(tiou_value)
                print(f"\033[31m[Mean  tIoU][cls {idx}]: {tiou_value}\033[0m")
            mean_tiou = sum(class_tious) / len(class_tious)
            print(f"\033[31m[Mean tIoU]: {mean_tiou}\033[0m")
            # --- Aggregate additional metrics from per-slice rows ---
            def _nanmean_list(vals):
                arr = np.array(vals, dtype=np.float32)
                return float(np.nanmean(arr))

            sens_cls1 = _nanmean_list([r["sens_1"] for r in rows])
            sens_cls2 = _nanmean_list([r["sens_2"] for r in rows])
            spec_cls1 = _nanmean_list([r["spec_1"] for r in rows])
            spec_cls2 = _nanmean_list([r["spec_2"] for r in rows])
            hd95_cls1 = _nanmean_list([r["hd95_1"] for r in rows])
            hd95_cls2 = _nanmean_list([r["hd95_2"] for r in rows])

            # Print in the same style as other metrics
            print(f"\033[31m[Mean Sensitivity][cls 1]: {sens_cls1}\033[0m")
            print(f"\033[31m[Mean Sensitivity][cls 2]: {sens_cls2}\033[0m")
            print(f"\033[31m[Mean Specificity][cls 1]: {spec_cls1}\033[0m")
            print(f"\033[31m[Mean Specificity][cls 2]: {spec_cls2}\033[0m")
            print(f"\033[31m[Mean HD95][cls 1]: {hd95_cls1}\033[0m")
            print(f"\033[31m[Mean HD95][cls 2]: {hd95_cls2}\033[0m")

            # Also compute the mean across stroke classes (1 & 2)
            mean_sens_1_2 = _nanmean_list([r["sens_mean_1_2"] for r in rows])
            mean_spec_1_2 = _nanmean_list([r["spec_mean_1_2"] for r in rows])
            mean_hd95_1_2 = _nanmean_list([r["hd95_mean_1_2"] for r in rows])
            print(f"\033[31m[Mean Sensitivity (cls1+cls2)]: {mean_sens_1_2}\033[0m")
            print(f"\033[31m[Mean Specificity (cls1+cls2)]: {mean_spec_1_2}\033[0m")
            print(f"\033[31m[Mean HD95 (cls1+cls2)]: {mean_hd95_1_2}\033[0m")

            extra = {
                "sens_cls1": sens_cls1,
                "sens_cls2": sens_cls2,
                "spec_cls1": spec_cls1,
                "spec_cls2": spec_cls2,
                "hd95_cls1": hd95_cls1,
                "hd95_cls2": hd95_cls2,
                "sens_mean_1_2": mean_sens_1_2,
                "spec_mean_1_2": mean_spec_1_2,
                "hd95_mean_1_2": mean_hd95_1_2,
            }

            return class_dices, class_ious, class_tious, extra

        precision_scope = autocast
        with torch.no_grad():
            with precision_scope("cuda"):
                dice_list, iou_list, tiou_list, extra = get_dice_loop(data, save_dir=save_dir)

        multi_dice, multi_iou, multi_tiou = np.array(dice_list), np.array(iou_list), np.array(tiou_list)

        metrics_dict.update({"val_avg_dice": list(multi_dice)})
        metrics_dict.update({"val_avg_iou": list(multi_iou)})
        metrics_dict.update({"val_avg_tiou": list(multi_tiou)})

        # Additional clinically relevant metrics (slice-level; class-wise)
        metrics_dict.update({
            "val_sens_class_1": extra["sens_cls1"],
            "val_sens_class_2": extra["sens_cls2"],
            "val_spec_class_1": extra["spec_cls1"],
            "val_spec_class_2": extra["spec_cls2"],
            "val_hd95_class_1": extra["hd95_cls1"],
            "val_hd95_class_2": extra["hd95_cls2"],
            "val_sens_mean_1_2": extra["sens_mean_1_2"],
            "val_spec_mean_1_2": extra["spec_mean_1_2"],
            "val_hd95_mean_1_2": extra["hd95_mean_1_2"],
        })

        # For consistency with existing API style, also expose per-class arrays (index 0 is background placeholder)
        metrics_dict.update({
            "val_avg_sens": [float("nan"), extra["sens_cls1"], extra["sens_cls2"]],
            "val_avg_spec": [float("nan"), extra["spec_cls1"], extra["spec_cls2"]],
            "val_avg_hd95": [float("nan"), extra["hd95_cls1"], extra["hd95_cls2"]],
        })

        return metrics_dict, seg_label_dict
    
    def labels_to_one_hot(self, targets):
        """
        Converts a target tensor to one-hot encoding based on its dimensions.
        
        Parameters:
        - targets (torch.Tensor): The target tensor of shape [N, 1, H, W] or [1, H, W].
        - num_classes (int): The number of classes for one-hot encoding.
        
        Returns:
        - torch.Tensor: One-hot encoded tensor of shape [N, num_classes, H, W] or [num_classes, H, W].
        """
        # Remove the channel dimension if 4D (batch present) or the batch dimension if 3D (no batch)
        targets_squeezed = targets.squeeze(1) if targets.dim() == 4 else targets.squeeze(0)
        
        # Apply one-hot encoding
        one_hot_targets = F.one_hot(targets_squeezed.long(), num_classes=self.num_classes)
        
        # If 4D input, rearrange [N, H, W, num_classes] -> [N, num_classes, H, W]
        # If 3D input, rearrange [H, W, num_classes] -> [num_classes, H, W]
        one_hot_targets = one_hot_targets.permute(0, 3, 1, 2) if targets.dim() == 4 else one_hot_targets.permute(2, 0, 1)
        
        return one_hot_targets.float()

    def preds_to_one_hot(self, preds):
        """
        Converts predictions to a one-hot encoded binary mask.
        
        Args:
            preds (torch.Tensor): Tensor of shape [Batch, Channels, Height, Width].
                                Contains raw prediction scores for each channel.
        
        Returns:
            torch.Tensor: One-hot encoded binary mask of shape [Batch, Channels, Height, Width].
        """
        # Apply softmax across channels (dim=1)
        preds_softmax = torch.softmax(preds, dim=1)
        
        # Find the channel with the maximum value
        preds_argmax = torch.argmax(preds_softmax, dim=1)  # Shape: [Batch, Height, Width]
        
        # Convert to one-hot encoding
        preds_bin = torch.nn.functional.one_hot(preds_argmax, num_classes=preds.shape[1])  # Shape: [Batch, Height, Width, Channels]
        preds_bin = preds_bin.permute(0, 3, 1, 2).float()  # Shape: [Batch, Channels, Height, Width]
        
        return preds_bin

    def _compute_lesion_ratio_alpha_and_present_class(self, targets: torch.Tensor):
        """
        Compute per-slice weighting alpha based on lesion pixel ratio and determine which lesion class is present.

        Assumptions:
        - targets is label-map after permute: shape (B, 1, H, W) with integer labels {0,1,2}
          where 0=background, 1=IS, 2=HS.
        - Typically only one lesion class dominates per slice (your dataset property).

        Returns:
        - alpha: (B,) float tensor
        - present: (B,) long tensor with values 1 or 2 indicating the dominant lesion class
        """
        y = targets.long()
        B = y.shape[0]
        HW = float(y.shape[-2] * y.shape[-1])

        r1 = (y == 1).sum(dim=(1, 2, 3)).float() / HW  # IS ratio
        r2 = (y == 2).sum(dim=(1, 2, 3)).float() / HW  # HS ratio

        present = torch.where(r1 >= r2, torch.ones_like(r1, dtype=torch.long), torch.full_like(r1, 2, dtype=torch.long))
        r = torch.maximum(r1, r2)

        eps = self.lesion_ratio_eps
        alpha = (self.lesion_ratio_r_ref / (r + eps)).pow(self.lesion_ratio_p)
        alpha = torch.clamp(alpha, self.lesion_ratio_a_min, self.lesion_ratio_a_max)
        return alpha, present