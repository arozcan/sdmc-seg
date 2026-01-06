import os
import json
import random
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import Dataset
from torchvision import transforms
import monai.transforms as mt



def select_first_channel(keys):
    def _select_first_channel(data):
        for key in keys:
            # Select only the first channel (C, H, W -> 1, H, W)
            data[key] = data[key][:1, :, :]
        return data
    return _select_first_channel


def split_train_val(
    train_files,
    val_ratio=0.1,
    random_seed=0,
    fold: int = 0,
    num_folds: int = 5,
):
    """Split into train/val.

    - If `num_folds` >= 2: uses Stratified K-Fold (validation = selected fold) *when feasible*.
    - Otherwise: falls back to a simple stratified holdout split using `val_ratio`.

    This is written to be robust when the provided `item["class"]` is not a clean int.

    IMPORTANT: Test split is expected to be prepared beforehand (e.g., dataset.json contains test lists).
    """

    all_files = list(train_files) if train_files is not None else []
    if len(all_files) == 0:
        return [], []

    def _to_int_label(v):
        """Convert common label encodings to int (0/1/2/...).

        Supports:
        - int/np.integer
        - float/np.floating
        - numeric strings ("0", "1", ...)
        - booleans
        - simple textual labels (tumor/non-tumor)
        - otherwise returns 0
        """
        if v is None:
            return 0
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, (float, np.floating)):
            return int(v)
        if isinstance(v, str):
            s = v.strip()
            sl = s.lower()
            if sl in {"tumor", "tumour", "positive", "pos", "true", "yes"}:
                return 1
            if sl in {"nontumor", "non-tumor", "non_tumor", "negative", "neg", "false", "no"}:
                return 0
            try:
                return int(float(s))
            except Exception:
                return 0
        return 0

    # labels for stratification
    y = [_to_int_label(item.get("class", 0)) for item in all_files]

    # If only one class exists, stratification is meaningless.
    unique, counts = np.unique(np.asarray(y), return_counts=True)
    has_multiple_classes = len(unique) > 1

    # K-Fold path (only if feasible)
    if num_folds is not None and int(num_folds) >= 2 and has_multiple_classes:
        num_folds = int(num_folds)
        fold = 0 if fold is None else int(fold)
        if fold < 0 or fold >= num_folds:
            raise ValueError(f"fold must be in [0, {num_folds-1}] but got {fold}")

        # StratifiedKFold requires each class to have at least n_splits samples.
        min_class_count = int(counts.min())
        if min_class_count >= num_folds and len(all_files) >= num_folds:
            skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_seed)
            splits = list(skf.split(np.zeros(len(all_files)), y))
            train_idx, val_idx = splits[fold]
            train_files = [all_files[i] for i in train_idx]
            val_files = [all_files[i] for i in val_idx]
            return train_files, val_files

    # Holdout path
    if val_ratio and float(val_ratio) > 0.0:
        # train_test_split with stratify also requires >=2 samples per class.
        # If that is not the case, fall back to non-stratified split.
        stratify_vec = None
        if has_multiple_classes:
            if int(counts.min()) >= 2:
                stratify_vec = y

        train_files, val_files = train_test_split(
            all_files,
            test_size=float(val_ratio),
            random_state=random_seed,
            stratify=stratify_vec,
        )
        return train_files, val_files

    return all_files, []


class BratsDataset(Dataset):
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
                 fold: int = 0,
                 num_folds: int = 5,
                 mask_onehot=False,
                 num_classes=4,
                 mask_scale_dpmm=False,
                 image_scale_dpmm=False,
                 expand_image_channel = False,
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
        self.augmentation = augmentation

        # Verileri sınıflara göre ayır
        train_files = [
            {
                "mask": os.path.join(data_root, item["label"]),
                "image_png": os.path.join(data_root, item["image_png"]),
                "class": os.path.join(data_root, item["class"])
            }
            for item in self.data_info["training"]
        ]

        # Optional pre-split test lists (created by prepare script). If not present, test set is empty.
        test_files = [
            {
                "mask": os.path.join(data_root, item["label"]),
                "image_png": os.path.join(data_root, item["image_png"]),
                "class": os.path.join(data_root, item["class"])
            }
            for item in self.data_info.get("test", [])
        ]
        train_files, val_files = split_train_val(
            train_files,
            val_ratio=val_ratio,
            random_seed=random_seed,
            fold=fold,
            num_folds=num_folds,
        )


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
            mt.LoadImaged(keys=["mask", "image_png"], reader="PILReader"),
            mt.EnsureChannelFirstd(keys=["image_png", "mask"]),
            mt.Lambda(select_first_channel(["image_png"])),
            mt.ScaleIntensityd(keys=["image_png"]),
            mt.Transposed(keys=["image_png", "mask"], indices=(0, 2, 1)),
            *([
                mt.RandFlipd(keys=["image_png", "mask"], spatial_axis=1, prob=0.2),
                mt.RandRotated(keys=["image_png", "mask"], range_x=0.25, prob=0.2, mode=[interpolation_image, interpolation_mask]),
                #mt.RandZoomd(keys=["image_png", "mask"], min_zoom=1.1, max_zoom=1.3, prob=0.25, mode=[interpolation_image, interpolation_mask]),
                #mt.RandAdjustContrastd(keys=["image_png"], prob=0.5, gamma=(0.75, 1.25)),
            ] if self.augmentation else []),
            mt.Resized(keys=["image_png","mask"], spatial_size=(self.size, self.size), mode=[interpolation_image, interpolation_mask]),
            mt.Lambdad(keys=["mask"],func=lambda x: mt.AsDiscrete(argmax=False, to_onehot=num_classes)(x) if self.mask_onehot else x),
            mt.Lambdad(keys=["mask"],func=lambda x: mt.ScaleIntensityRange(a_min=0, a_max=1, b_min=-1.0, b_max=1.0, clip=True)(x) if self.mask_scale_dpmm else x),
            mt.Lambdad(keys=["image_png"], func=lambda x: mt.ScaleIntensityRange(a_min=0, a_max=1, b_min=-1.0, b_max=1.0, clip=True)(x) if self.image_scale_dpmm else x),
            mt.Lambdad(keys=["image_png"], func=lambda x: x.expand(3, *x.shape[1:]) if self.expand_image_channel and x.shape[0] == 1 else x),
            mt.Transposed(keys=["image_png", "mask"], indices=(1, 2, 0))
            #ScaleIntensityRangePercentilesd(keys="image_dicom", lower=0, upper=99.5, b_min=0, b_max=1),
        ])

        # Define validation transformations
        self.val_transforms = mt.Compose([
            mt.LoadImaged(keys=["mask", "image_png"], reader="PILReader"),
            mt.EnsureChannelFirstd(keys=["image_png", "mask"]),
            mt.Lambda(select_first_channel(["image_png"])),
            mt.ScaleIntensityd(keys=["image_png"]),
            mt.Transposed(keys=["image_png", "mask"], indices=(0, 2, 1)),
            mt.Resized(keys=["image_png", "mask"], spatial_size=(self.size, self.size), mode=[interpolation_image, interpolation_mask]),
            mt.Lambdad(keys=["mask"],func=lambda x: mt.AsDiscrete(argmax=False, to_onehot=num_classes)(x) if self.mask_onehot else x),
            mt.Lambdad(keys=["mask"],func=lambda x: mt.ScaleIntensityRange(a_min=0, a_max=1, b_min=-1.0, b_max=1.0, clip=True)(x) if self.mask_scale_dpmm else x),
            mt.Lambdad(keys=["image_png"], func=lambda x: mt.ScaleIntensityRange(a_min=0, a_max=1, b_min=-1.0, b_max=1.0, clip=True)(x) if self.image_scale_dpmm else x),
            mt.Lambdad(keys=["image_png"], func=lambda x: x.expand(3, *x.shape[1:]) if self.expand_image_channel and x.shape[0] == 1 else x),
            mt.Transposed(keys=["image_png", "mask"], indices=(1, 2, 0))
            #ScaleIntensityRangePercentilesd(keys="image_dicom", lower=0, upper=99.5, b_min=0, b_max=1),
        ])

        self.test_transforms = mt.Compose([
            mt.LoadImaged(keys=["mask", "image_png"], reader="PILReader"),
            mt.EnsureChannelFirstd(keys=["image_png", "mask"]),
            mt.Lambda(select_first_channel(["image_png"])),
            mt.ScaleIntensityd(keys=["image_png"]),
            mt.Transposed(keys=["image_png", "mask"], indices=(0, 2, 1)),
            mt.Resized(keys=["image_png", "mask"], spatial_size=(self.size, self.size), mode=[interpolation_image, interpolation_image, interpolation_mask]),
            mt.Lambdad(keys=["mask"],func=lambda x: mt.AsDiscrete(argmax=False, to_onehot=num_classes)(x) if self.mask_onehot else x),
            mt.Lambdad(keys=["mask"],func=lambda x: mt.ScaleIntensityRange(a_min=0, a_max=1, b_min=-1.0, b_max=1.0, clip=True)(x) if self.mask_scale_dpmm else x),
            mt.Lambdad(keys=["image_png"], func=lambda x: mt.ScaleIntensityRange(a_min=0, a_max=1, b_min=-1.0, b_max=1.0, clip=True)(x) if self.image_scale_dpmm else x),
            mt.Lambdad(keys=["image_png"], func=lambda x: x.expand(3, *x.shape[1:]) if self.expand_image_channel and x.shape[0] == 1 else x),
            mt.Transposed(keys=["image_png", "mask"], indices=(1, 2, 0))
            #ScaleIntensityRangePercentilesd(keys="image_dicom", lower=0, upper=99.5, b_min=0, b_max=1),
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

        else:  # data_type == "all"
            example["image"] = example_transformed["image_png"]
            example["segmentation"] = example_transformed["mask"]
            return example

    def get_class_label(self, idx):
        """Sampler hesaplamasında kullanılacak ham sınıf etiketi."""
        return self.dataset[idx].get("class", 0)

    def compute_class_pixel_counts(self):
        """Compute total number of pixels per class over the current split (self.dataset)."""
        counts = np.zeros(self.num_classes, dtype=np.int64)

        for item in self.dataset:
            mask_path = item["mask"]
            mask = np.array(Image.open(mask_path))
            if mask.ndim == 3:
                mask = mask[..., 0]

            for c in range(self.num_classes):
                counts[c] += np.sum(mask == c)

        return counts

    def compute_class_weights(self, eps=1e-6, normalize=True):
        """Compute per-class weights from pixel counts for loss functions (e.g., Dice / CE)."""
        counts = self.compute_class_pixel_counts().astype(np.float64)
        inv_freq = 1.0 / (counts + eps)

        if normalize:
            inv_freq = inv_freq * (self.num_classes / inv_freq.sum())

        return inv_freq.astype(np.float32)


class BratsMaskTrain(BratsDataset):
    def __init__(self, **kwargs):
        super().__init__(json_file="dataset.json", mode="train", data_type="mask", **kwargs)


class BratsMaskValidation(BratsDataset):
    def __init__(self, **kwargs):
        super().__init__(json_file="dataset.json", mode="validation", data_type="mask", **kwargs)


class BratsMaskTest(BratsDataset):
    def __init__(self, **kwargs):
        super().__init__(json_file="dataset.json", mode="test", data_type="mask", **kwargs)


class BratsImageTrain(BratsDataset):
    def __init__(self, **kwargs):
        super().__init__(json_file="dataset.json", mode="train", data_type="image_png", **kwargs)


class BratsImageValidation(BratsDataset):
    def __init__(self, **kwargs):
        super().__init__(json_file="dataset.json", mode="validation", data_type="image_png", **kwargs)


class BratsImageTest(BratsDataset):
    def __init__(self, **kwargs):
        super().__init__(json_file="dataset.json", mode="test", data_type="image_png", **kwargs)


class BratsTrain(BratsDataset):
    def __init__(self, **kwargs):
        super().__init__(json_file="dataset.json", mode="train", data_type="all", **kwargs)


class BratsValidation(BratsDataset):
    def __init__(self, **kwargs):
        super().__init__(json_file="dataset.json", mode="validation", data_type="all", **kwargs)


class BratsTest(BratsDataset):
    def __init__(self, **kwargs):
        super().__init__(json_file="dataset.json", mode="test", data_type="all", **kwargs)