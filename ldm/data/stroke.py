import os
import json
import random
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import Dataset
from torchvision import transforms
import monai.transforms as mt

import re

def _extract_case_id_from_path(p: str):
    """
    Dosya yolundan vaka id'si çekmeye çalışır.
    Örn: .../image_png/12345.png  -> "12345"
         .../something_12345_mask.png -> "12345" (son görülen sayı grubu)
    """
    if p is None:
        return None
    base = os.path.basename(p)
    stem = os.path.splitext(base)[0]
    m = re.findall(r"\d+", stem)
    return m[-1] if m else None

def _get_case_id(item: dict):
    # Eğer item içinde açıkça id tutuluyorsa onu kullan
    if "id" in item and item["id"] is not None:
        return str(item["id"])
    # Yoksa path’lerden üret
    for k in ["image_png", "image_dicom", "mask"]:
        if k in item:
            cid = _extract_case_id_from_path(item[k])
            if cid is not None:
                return str(cid)
    return None

def load_split_ids(split_files: str, fold: int = 0):
    """
    split_files içeriği: [ {"train":[...], "val":[...]}, {...}, ... ]
    """
    with open(split_files, "r") as f:
        splits = json.load(f)

    if not isinstance(splits, list) or len(splits) == 0:
        raise ValueError("split_files must be a non-empty JSON list.")

    if fold < 0 or fold >= len(splits):
        raise ValueError(f"fold must be in [0, {len(splits)-1}] for split_files, got {fold}")

    train_ids = set(map(str, splits[fold].get("train", [])))
    val_ids   = set(map(str, splits[fold].get("val", [])))

    if len(train_ids) == 0 or len(val_ids) == 0:
        raise ValueError("split_files fold entry must contain non-empty 'train' and 'val' lists.")

    return train_ids, val_ids


def apply_windowing(image, window_center, window_width):
    min_value = window_center - (window_width / 2)
    max_value = window_center + (window_width / 2)
    windowed_image = np.clip(image, min_value, max_value)
    windowed_image = (windowed_image - min_value) / (max_value - min_value)
    return windowed_image


def apply_multichannel_windowing(image, window_centers, window_widths):
    """
    Applies windowing to the input image for multiple window centers and widths,
    and creates a new multi-channel image where each channel corresponds to a specific window.
    
    Parameters:
    - image: numpy.ndarray
        Input image to be windowed. Expected shape: (H, W) or (1, H, W).
    - window_centers: list or numpy.ndarray
        List of window center values.
    - window_widths: list or numpy.ndarray
        List of window width values.
        
    Returns:
    - numpy.ndarray
        Multi-channel image with shape (num_windows, H, W).
    """
    if len(window_centers) != len(window_widths):
        raise ValueError("window_centers and window_widths must have the same length.")
    
    # Remove the channel dimension if present (assumes 1 channel if shape is (1, H, W))
    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]
    
    # List to store windowed images
    channels = []
    for center, width in zip(window_centers, window_widths):
        min_value = center - (width / 2)
        max_value = center + (width / 2)
        windowed_image = np.clip(image, min_value, max_value)
        windowed_image = (windowed_image - min_value) / (max_value - min_value)
        channels.append(windowed_image)
    
    # Stack all windowed images as channels
    multichannel_image = np.stack(channels, axis=0)
    return multichannel_image

def select_first_channel(keys):
    def _select_first_channel(data):
        for key in keys:
            # Select only the first channel (C, H, W -> 1, H, W)
            data[key] = data[key][:1, :, :]
        return data
    return _select_first_channel

# --- Helper: add_foreground_mask_from_image_png ---
def add_foreground_mask_from_image_png(threshold: float = 0.0, key_in: str = "image_png", key_out: str = "fg_mask"):
    """Create a binary foreground mask from `image_png`.

    Expects `key_in` to be channel-first (C,H,W). Produces `key_out` as (1,H,W) uint8 with values {0,1}.
    """
    def _fn(data):
        img = data[key_in]
        # Ensure channel-first shape
        if img.ndim == 2:
            m = (img > threshold).astype(np.uint8)[None, ...]
        else:
            # use first channel for foreground detection
            m = (img[0] > threshold).astype(np.uint8)[None, ...]
        data[key_out] = m
        return data
    return _fn

def split_train_val(
    nonstr_files,
    isch_files,
    hemorr_files,
    val_ratio=0.1,
    random_seed=0,
    include_background=True,
    fold: int = 0,
    num_folds: int = 5,
):
    """Split into train/val.

    - If `num_folds` >= 2: uses Stratified K-Fold (validation = selected fold).
    - If `num_folds` < 2: falls back to a simple train/val split using `val_ratio`.

    IMPORTANT: Test split is expected to be prepared beforehand (e.g., dataset.json contains test lists).
    """

    # Build the pool that will be split into train/val
    if include_background:
        all_files = list(nonstr_files) + list(isch_files) + list(hemorr_files)
    else:
        all_files = list(isch_files) + list(hemorr_files)

    if len(all_files) == 0:
        return [], []

    # labels for stratification
    y = [int(item.get("class", 0)) for item in all_files]

    # K-Fold path
    if num_folds is not None and int(num_folds) >= 2:
        num_folds = int(num_folds)
        if fold is None:
            fold = 0
        fold = int(fold)
        if fold < 0 or fold >= num_folds:
            raise ValueError(f"fold must be in [0, {num_folds-1}] but got {fold}")

        # If dataset is too small for the requested folds, degrade gracefully
        if len(all_files) < num_folds:
            # fall back to holdout split
            train_files, val_files = train_test_split(
                all_files,
                test_size=val_ratio if val_ratio > 0 else 0.0,
                random_state=random_seed,
                stratify=y if len(set(y)) > 1 else None,
            )
            return train_files, val_files

        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_seed)
        splits = list(skf.split(np.zeros(len(all_files)), y))
        train_idx, val_idx = splits[fold]

        train_files = [all_files[i] for i in train_idx]
        val_files = [all_files[i] for i in val_idx]
        return train_files, val_files

    # Holdout path
    if val_ratio > 0.0:
        train_files, val_files = train_test_split(
            all_files,
            test_size=val_ratio,
            random_state=random_seed,
            stratify=y if len(set(y)) > 1 else None,
        )
    else:
        train_files, val_files = all_files, []

    return train_files, val_files


class StrokeDataset(Dataset):
    def __init__(self,
                 json_file,
                 data_root,
                 size=None,
                 interpolation_mask="nearest",
                 interpolation_image="bilinear",
                 flip_p=0.5,
                 mode="train",
                 data_type="mask",
                 val_ratio=0.2,
                 random_seed=0,
                 split_files: str = None,
                 fold: int = 0,
                 num_folds: int = 5,
                 mask_onehot=False,
                 num_classes=3,
                 include_background=True,
                 mask_scale_dpmm=False,
                 image_scale_dpmm=False,
                 expand_image_channel = False,
                 multichannel_windowing = False,
                 augmentation=False
                 ):
        # JSON dosyasını yükle
        with open(data_root+json_file, "r") as f:
            self.data_info = json.load(f)
        
        self.data_root = data_root
        self.data_type = data_type
        self.mask_onehot = mask_onehot
        self.mask_scale_dpmm = mask_scale_dpmm
        self.image_scale_dpmm = image_scale_dpmm
        self.expand_image_channel = expand_image_channel
        self.num_classes = num_classes
        self.mode = mode
        self.include_background = include_background
        self.augmentation = augmentation
        self.multichannel_windowing = multichannel_windowing
        self.split_files = split_files

        # Verileri sınıflara göre ayır
        nonstroke_files = [
            {
                "image_dicom": os.path.join(data_root, item["image_dicom"]),
                "mask": os.path.join(data_root, item["mask"]),
                "image_png": os.path.join(data_root, item["image_png"]),
                "class": 0
            }
            for item in self.data_info["training_nonstroke"]
        ]

        ischemic_files = [
            {
                "image_dicom": os.path.join(data_root, item["image_dicom"]),
                "mask": os.path.join(data_root, item["mask"]),
                "image_png": os.path.join(data_root, item["image_png"]),
                "class": 1
            }
            for item in self.data_info["training_ischemic"]
        ]

        hemorrhagic_files = [
            {
                "image_dicom": os.path.join(data_root, item["image_dicom"]),
                "mask": os.path.join(data_root, item["mask"]),
                "image_png": os.path.join(data_root, item["image_png"]),
                "class": 2
            }
            for item in self.data_info["training_hemorrhagic"]
        ]

        # Optional pre-split test lists (created by prepare script). If not present, test set is empty.
        test_nonstroke_files = [
            {
                "image_dicom": os.path.join(data_root, item["image_dicom"]),
                "mask": os.path.join(data_root, item["mask"]),
                "image_png": os.path.join(data_root, item["image_png"]),
                "class": 0
            }
            for item in self.data_info.get("test_nonstroke", [])
        ]

        test_ischemic_files = [
            {
                "image_dicom": os.path.join(data_root, item["image_dicom"]),
                "mask": os.path.join(data_root, item["mask"]),
                "image_png": os.path.join(data_root, item["image_png"]),
                "class": 1
            }
            for item in self.data_info.get("test_ischemic", [])
        ]

        test_hemorrhagic_files = [
            {
                "image_dicom": os.path.join(data_root, item["image_dicom"]),
                "mask": os.path.join(data_root, item["mask"]),
                "image_png": os.path.join(data_root, item["image_png"]),
                "class": 2
            }
            for item in self.data_info.get("test_hemorrhagic", [])
        ]

        # --- Split belirleme ---
        if self.split_files is not None and str(self.split_files).strip() != "":
            train_ids, val_ids = load_split_ids(self.split_files, fold=fold)

            # include_background'a göre havuzu oluştur
            if self.include_background:
                all_files = list(nonstroke_files) + list(ischemic_files) + list(hemorrhagic_files)
            else:
                all_files = list(ischemic_files) + list(hemorrhagic_files)

            train_files, val_files = [], []
            missing_train, missing_val = 0, 0

            for it in all_files:
                cid = _get_case_id(it)
                if cid in train_ids:
                    train_files.append(it)
                if cid in val_ids:
                    val_files.append(it)

            # split dosyasında olup dataset'te bulunmayanlar için uyarı amaçlı sayım
            present_ids = set(filter(None, (_get_case_id(x) for x in all_files)))
            missing_train = len(train_ids - present_ids)
            missing_val = len(val_ids - present_ids)

            if len(train_files) == 0 or len(val_files) == 0:
                raise RuntimeError(
                    f"split_files active but train/val became empty. "
                    f"Check ID extraction. train={len(train_files)}, val={len(val_files)}"
                )

            if missing_train > 0 or missing_val > 0:
                print(f"[WARN] split_files: missing ids in dataset: train_missing={missing_train}, val_missing={missing_val}")

        else:
            train_files, val_files = split_train_val(
                nonstr_files=nonstroke_files,
                isch_files=ischemic_files,
                hemorr_files=hemorrhagic_files,
                val_ratio=val_ratio,
                random_seed=random_seed,
                include_background=self.include_background,
                fold=fold,
                num_folds=num_folds,
            )

        # Test set comes from dataset.json (prepared beforehand)
        if self.include_background:
            test_files = test_nonstroke_files + test_ischemic_files + test_hemorrhagic_files
        else:
            test_files = test_ischemic_files + test_hemorrhagic_files

        # Alt küme (train/validation) seçimlerine göre verileri ayır
        if self.mode  == "train":
            self.dataset = train_files
        elif self.mode  == "validation":
            self.dataset = val_files
        elif self.mode  == "test":
            self.dataset = test_files

        self._length = len(self.dataset)

        self.size = size
        self.interpolation_mask = {"nearest": Image.NEAREST,
                              "bilinear": Image.BILINEAR,
                              "bicubic": Image.BICUBIC,
                              "lanczos": Image.LANCZOS,
                              }[interpolation_mask]
        self.interpolation_image = {"nearest": Image.NEAREST,
                              "bilinear": Image.BILINEAR,
                              "bicubic": Image.BICUBIC,
                              "lanczos": Image.LANCZOS,
                              }[interpolation_image]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

        # Define training transformations
        self.train_transforms = mt.Compose([
            mt.LoadImaged(keys=["image_dicom"], reader="PydicomReader", image_only=False),
            mt.LoadImaged(keys=["mask", "image_png"], reader="PILReader"),
            mt.EnsureChannelFirstd(keys=["image_dicom", "image_png", "mask"]),
            mt.ResizeWithPadOrCropd(
                keys=["image_dicom", "image_png", "mask"],
                spatial_size=(512, 512),
            ),
            mt.Lambda(select_first_channel(["image_png"])),
            mt.ScaleIntensityd(keys=["image_png"]),
            # mt.ScaleIntensityRangePercentilesd(
            #     keys=["image_png"],
            #     lower=0.5,
            #     upper=99.5,
            #     b_min=0.0,
            #     b_max=1.0,
            #     clip=True,
            # ),
            *([
                mt.Lambda(func=lambda x: {**x, "image_dicom": apply_multichannel_windowing(x["image_dicom"], [80, 30, 40], [40, 35, 40])}),
            ] if multichannel_windowing else [mt.Lambda(func=lambda x: {**x, "image_dicom": apply_windowing(x["image_dicom"], 50, 300)})]),
            mt.Transposed(keys=["image_dicom","image_png", "mask"], indices=(0, 2, 1)),
            *([
                mt.RandFlipd(keys=["image_dicom","image_png","mask"], spatial_axis=1, prob=0.25),

                mt.RandRotated(
                    keys=["image_dicom","image_png","mask"],
                    range_x=0.1, prob=0.2,
                    mode=[interpolation_image, interpolation_image, interpolation_mask],
                    padding_mode="zeros",
                ),

                #mt.RandAdjustContrastd(keys=["image_dicom","image_png"], prob=0.2, gamma=(0.9, 1.1)),
                #mt.RandGaussianNoised(keys=["image_dicom","image_png"], prob=0.15, mean=0.0, std=0.01),
            ] if self.augmentation else []),
            mt.Resized(keys=["image_dicom", "image_png","mask"], spatial_size=(self.size, self.size), mode=[interpolation_image,interpolation_image, interpolation_mask]),
            mt.Lambdad(keys=["mask"],func=lambda x: mt.AsDiscrete(argmax=False, to_onehot=num_classes)(x) if self.mask_onehot else x),
            mt.Lambdad(keys=["mask"],func=lambda x: mt.ScaleIntensityRange(a_min=0, a_max=1, b_min=-1.0, b_max=1.0, clip=True)(x) if self.mask_scale_dpmm else x),
            mt.Lambdad(keys=["image_dicom", "image_png"], func=lambda x: mt.ScaleIntensityRange(a_min=0, a_max=1, b_min=-1.0, b_max=1.0, clip=True)(x) if self.image_scale_dpmm else x),
            mt.Lambdad(keys=["image_dicom", "image_png"], func=lambda x: x.expand(3, *x.shape[1:]) if self.expand_image_channel and x.shape[0] == 1 else x),
            mt.Transposed(keys=["image_dicom", "image_png", "mask"], indices=(1, 2, 0))
        ])

        # Define validation transformations
        self.val_transforms = mt.Compose([
            mt.LoadImaged(keys=["image_dicom"], reader="PydicomReader", image_only=False),
            mt.LoadImaged(keys=["mask", "image_png"], reader="PILReader"),
            mt.EnsureChannelFirstd(keys=["image_dicom", "image_png", "mask"]),
            mt.ResizeWithPadOrCropd(
                keys=["image_dicom", "image_png", "mask"],
                spatial_size=(512, 512),
            ),
            mt.Lambda(select_first_channel(["image_png"])),
            mt.ScaleIntensityd(keys=["image_png"]),
            # mt.ScaleIntensityRangePercentilesd(
            #     keys=["image_png"],
            #     lower=0.5,
            #     upper=99.5,
            #     b_min=0.0,
            #     b_max=1.0,
            #     clip=True,
            # ),
            *([
                mt.Lambda(func=lambda x: {**x, "image_dicom": apply_multichannel_windowing(x["image_dicom"], [80, 30, 40], [40, 35, 40])}),
            ] if multichannel_windowing else [mt.Lambda(func=lambda x: {**x, "image_dicom": apply_windowing(x["image_dicom"], 50, 300)})]),
            mt.Transposed(keys=["image_dicom", "image_png", "mask"], indices=(0, 2, 1)),
            mt.Resized(keys=["image_dicom", "image_png", "mask"], spatial_size=(self.size, self.size), mode=[interpolation_image, interpolation_image, interpolation_mask]),
            mt.Lambdad(keys=["mask"],func=lambda x: mt.AsDiscrete(argmax=False, to_onehot=num_classes)(x) if self.mask_onehot else x),
            mt.Lambdad(keys=["mask"],func=lambda x: mt.ScaleIntensityRange(a_min=0, a_max=1, b_min=-1.0, b_max=1.0, clip=True)(x) if self.mask_scale_dpmm else x),
            mt.Lambdad(keys=["image_dicom", "image_png"], func=lambda x: mt.ScaleIntensityRange(a_min=0, a_max=1, b_min=-1.0, b_max=1.0, clip=True)(x) if self.image_scale_dpmm else x),
            mt.Lambdad(keys=["image_dicom", "image_png"], func=lambda x: x.expand(3, *x.shape[1:]) if self.expand_image_channel and x.shape[0] == 1 else x),
            mt.Transposed(keys=["image_dicom", "image_png", "mask"], indices=(1, 2, 0))
        ])

        self.test_transforms = mt.Compose([
            mt.LoadImaged(keys=["image_dicom"], reader="PydicomReader", image_only=False),
            mt.LoadImaged(keys=["mask", "image_png"], reader="PILReader"),
            mt.EnsureChannelFirstd(keys=["image_dicom", "image_png", "mask"]),
            mt.ResizeWithPadOrCropd(
                keys=["image_dicom", "image_png", "mask"],
                spatial_size=(512, 512),
            ),
            mt.Lambda(select_first_channel(["image_png"])),
            mt.ScaleIntensityd(keys=["image_png"]),
            # mt.ScaleIntensityRangePercentilesd(
            #     keys=["image_png"],
            #     lower=0.5,
            #     upper=99.5,
            #     b_min=0.0,
            #     b_max=1.0,
            #     clip=True,
            # ),
            *([
                mt.Lambda(func=lambda x: {**x, "image_dicom": apply_multichannel_windowing(x["image_dicom"], [80, 30, 40], [40, 35, 40])}),
            ] if multichannel_windowing else [mt.Lambda(func=lambda x: {**x, "image_dicom": apply_windowing(x["image_dicom"], 50, 300)})]),
            mt.Transposed(keys=["image_dicom", "image_png", "mask"], indices=(0, 2, 1)),
            mt.Resized(keys=["image_dicom", "image_png", "mask"], spatial_size=(self.size, self.size), mode=[interpolation_image, interpolation_image, interpolation_mask]),
            mt.Lambdad(keys=["mask"],func=lambda x: mt.AsDiscrete(argmax=False, to_onehot=num_classes)(x) if self.mask_onehot else x),
            mt.Lambdad(keys=["mask"],func=lambda x: mt.ScaleIntensityRange(a_min=0, a_max=1, b_min=-1.0, b_max=1.0, clip=True)(x) if self.mask_scale_dpmm else x),
            mt.Lambdad(keys=["image_dicom", "image_png"], func=lambda x: mt.ScaleIntensityRange(a_min=0, a_max=1, b_min=-1.0, b_max=1.0, clip=True)(x) if self.image_scale_dpmm else x),
            mt.Lambdad(keys=["image_dicom", "image_png"], func=lambda x: x.expand(3, *x.shape[1:]) if self.expand_image_channel and x.shape[0] == 1 else x),
            mt.Transposed(keys=["image_dicom", "image_png", "mask"], indices=(1, 2, 0))
        ])



    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = self.dataset[i]
        example["file_path_"] = example["image_png"]
        if self.mode == "train":
            example_transformed = self.train_transforms(example)
        elif self.mode == "validation":
            example_transformed = self.val_transforms(example)
        elif self.mode == "test":
            example_transformed = self.test_transforms(example)

        if self.data_type == "mask":
            example["image"] = example_transformed["mask"]
            return example
        
        elif self.data_type == "image_png":
            example["image"] = example_transformed["image_png"]
            return example

        elif self.data_type == "image_dicom":
            example["image"] = example_transformed["image_dicom"]
            return example

        else:  # data_type == "all"
            example["image"] = example_transformed["image_png"]
            example["segmentation"] = example_transformed["mask"]
            example["class_id"] = np.array([example["class"]])
            return example
    def get_class_label(self, idx):
        """Sampler hesaplamasında kullanılacak ham sınıf etiketi."""
        return self.dataset[idx].get("class", 0)

    def compute_class_pixel_counts(self):
        """
        Compute total number of pixels per class over the current split (self.dataset).
        This uses the raw mask files on disk (before any MONAI transforms).
        
        Returns
        -------
        np.ndarray
            Array of length `self.num_classes` with pixel counts for each class id.
        """
        counts = np.zeros(self.num_classes, dtype=np.int64)

        for item in self.dataset:
            mask_path = item["mask"]
            # Load mask as numpy array
            mask = np.array(Image.open(mask_path))

            # If mask has channels (e.g., HWC), assume class is encoded in the first channel
            if mask.ndim == 3:
                mask = mask[..., 0]

            for c in range(self.num_classes):
                counts[c] += np.sum(mask == c)

        return counts

    def compute_class_weights(self, eps=1e-6, normalize=True):
        """
        Compute per-class weights from pixel counts for loss functions (e.g., Dice / CE).
        Uses inverse-frequency weighting by default.

        Parameters
        ----------
        eps : float
            Small constant to avoid division by zero.
        normalize : bool
            If True, rescales weights so that their mean is 1.0.

        Returns
        -------
        np.ndarray
            Array of length `self.num_classes` with weights for each class id.
        """
        counts = self.compute_class_pixel_counts().astype(np.float64)
        inv_freq = 1.0 / (counts + eps)

        if normalize:
            inv_freq = inv_freq * (self.num_classes / inv_freq.sum())

        return inv_freq.astype(np.float32)

class StrokeMaskTrain(StrokeDataset):
    def __init__(self, **kwargs):
        super().__init__(json_file="dataset.json", mode="train", data_type="mask", **kwargs)

class StrokeMaskValidation(StrokeDataset):
    def __init__(self, **kwargs):
        super().__init__(json_file="dataset.json", mode="validation", data_type="mask", **kwargs)

class StrokeMaskTest(StrokeDataset):
    def __init__(self, **kwargs):
        super().__init__(json_file="dataset.json", mode="test", data_type="mask", **kwargs)


class StrokeImageTrain(StrokeDataset):
    def __init__(self, **kwargs):
        super().__init__(json_file="dataset.json", mode="train", data_type="image_png", **kwargs)

class StrokeImageValidation(StrokeDataset):
    def __init__(self, **kwargs):
        super().__init__(json_file="dataset.json", mode="validation", data_type="image_png", **kwargs)

class StrokeImageTest(StrokeDataset):
    def __init__(self, **kwargs):
        super().__init__(json_file="dataset.json", mode="test", data_type="image_png", **kwargs)


class StrokeTrain(StrokeDataset):
    def __init__(self, **kwargs):
        super().__init__(json_file="dataset.json", mode="train", data_type="all", **kwargs)

class StrokeValidation(StrokeDataset):
    def __init__(self, **kwargs):
        super().__init__(json_file="dataset.json", mode="validation", data_type="all", **kwargs)

class StrokeTest(StrokeDataset):
    def __init__(self, **kwargs):
        super().__init__(json_file="dataset.json", mode="test", data_type="all", **kwargs)