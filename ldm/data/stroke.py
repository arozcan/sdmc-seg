import os
import json
import random
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
import monai.transforms as mt


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

def split_train_val_test(nonstr_files, isch_files, hemorr_files, val_ratio=0.1, test_ratio=0.1, random_seed=0, include_background=True):
    random.seed(random_seed)

    # Helper function to split files into train, validation, and test sets
    def split_files(files):
        # First, split off the test set if test_ratio > 0.0
        if test_ratio > 0.0:
            train_files, test_files = train_test_split(files, test_size=test_ratio, random_state=random_seed)
        else:
            train_files, test_files = files, []  # if test_ratio is 0.0, set test_files to an empty list

        # Next, split off the validation set if val_ratio > 0.0
        if val_ratio > 0.0:
            train_files, val_files = train_test_split(train_files, test_size=val_ratio / (1 - test_ratio), random_state=random_seed)
        else:
            val_files = []  # if val_ratio is 0.0, set val_files to an empty list

        return train_files, val_files, test_files

    # Apply the split to each class: nonstroke, ischemic, and hemorrhagic
    nonstroke_train, nonstroke_val, nonstroke_test = split_files(nonstr_files)
    ischemic_train, ischemic_val, ischemic_test = split_files(isch_files)
    hemorrhagic_train, hemorrhagic_val, hemorrhagic_test = split_files(hemorr_files)

    # Combine each split according to the include_background flag
    if include_background:
        train_files = nonstroke_train + ischemic_train + hemorrhagic_train
        val_files = nonstroke_val + ischemic_val + hemorrhagic_val
        test_files = nonstroke_test + ischemic_test + hemorrhagic_test
    else:
        train_files = ischemic_train + hemorrhagic_train
        val_files = ischemic_val + hemorrhagic_val
        test_files = ischemic_test + hemorrhagic_test
    
    return train_files, val_files, test_files


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
                 val_ratio=0.1,
                 test_ratio=0.1,
                 random_seed=0,
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

        train_files, val_files, test_files = split_train_val_test(nonstr_files=nonstroke_files, isch_files=ischemic_files, hemorr_files=hemorrhagic_files, val_ratio=val_ratio, test_ratio=test_ratio, random_seed=random_seed, include_background=self.include_background)

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
            *([
                mt.Lambda(func=lambda x: {**x, "image_dicom": apply_multichannel_windowing(x["image_dicom"], [80, 30, 40], [40, 35, 40])}),
            ] if multichannel_windowing else [mt.Lambda(func=lambda x: {**x, "image_dicom": apply_windowing(x["image_dicom"], 50, 300)})]),
            mt.Lambda(select_first_channel(["image_png"])),
            mt.ScaleIntensityd(keys=["image_png"]),
            mt.Transposed(keys=["image_dicom","image_png", "mask"], indices=(0, 2, 1)),
            *([
                mt.RandFlipd(keys=["image_dicom", "image_png", "mask"], spatial_axis=1, prob=0.2),
                mt.RandRotated(keys=["image_dicom", "image_png", "mask"], range_x=0.25, prob=0.2, mode=[interpolation_image, interpolation_image, interpolation_mask]),
                #mt.RandZoomd(keys=["image_dicom", "image_png", "mask"], min_zoom=1.1, max_zoom=1.3, prob=0.25, mode=[interpolation_image, interpolation_image, interpolation_mask]),
                #mt.RandAdjustContrastd(keys=["image_dicom", "image_png"], prob=0.5, gamma=(0.75, 1.25)),
            ] if self.augmentation else []),
            mt.Resized(keys=["image_dicom", "image_png","mask"], spatial_size=(self.size, self.size), mode=[interpolation_image,interpolation_image, interpolation_mask]),
            mt.Lambdad(keys=["mask"],func=lambda x: mt.AsDiscrete(argmax=False, to_onehot=num_classes)(x) if self.mask_onehot else x),
            mt.Lambdad(keys=["mask"],func=lambda x: mt.ScaleIntensityRange(a_min=0, a_max=1, b_min=-1.0, b_max=1.0, clip=True)(x) if self.mask_scale_dpmm else x),
            mt.Lambdad(keys=["image_dicom", "image_png"], func=lambda x: mt.ScaleIntensityRange(a_min=0, a_max=1, b_min=-1.0, b_max=1.0, clip=True)(x) if self.image_scale_dpmm else x),
            mt.Lambdad(keys=["image_dicom", "image_png"], func=lambda x: x.expand(3, *x.shape[1:]) if self.expand_image_channel and x.shape[0] == 1 else x),
            mt.Transposed(keys=["image_dicom", "image_png", "mask"], indices=(1, 2, 0))
            #ScaleIntensityRangePercentilesd(keys="image_dicom", lower=0, upper=99.5, b_min=0, b_max=1),
        ])

        # Define validation transformations
        self.val_transforms = mt.Compose([
            mt.LoadImaged(keys=["image_dicom"], reader="PydicomReader", image_only=False),
            mt.LoadImaged(keys=["mask", "image_png"], reader="PILReader"),
            mt.EnsureChannelFirstd(keys=["image_dicom", "image_png", "mask"]),
            *([
                mt.Lambda(func=lambda x: {**x, "image_dicom": apply_multichannel_windowing(x["image_dicom"], [80, 30, 40], [40, 35, 40])}),
            ] if multichannel_windowing else [mt.Lambda(func=lambda x: {**x, "image_dicom": apply_windowing(x["image_dicom"], 50, 300)})]),
            mt.Lambda(select_first_channel(["image_png"])),
            mt.ScaleIntensityd(keys=["image_png"]),
            mt.Transposed(keys=["image_dicom", "image_png", "mask"], indices=(0, 2, 1)),
            mt.Resized(keys=["image_dicom", "image_png", "mask"], spatial_size=(self.size, self.size), mode=[interpolation_image, interpolation_image, interpolation_mask]),
            mt.Lambdad(keys=["mask"],func=lambda x: mt.AsDiscrete(argmax=False, to_onehot=num_classes)(x) if self.mask_onehot else x),
            mt.Lambdad(keys=["mask"],func=lambda x: mt.ScaleIntensityRange(a_min=0, a_max=1, b_min=-1.0, b_max=1.0, clip=True)(x) if self.mask_scale_dpmm else x),
            mt.Lambdad(keys=["image_dicom", "image_png"], func=lambda x: mt.ScaleIntensityRange(a_min=0, a_max=1, b_min=-1.0, b_max=1.0, clip=True)(x) if self.image_scale_dpmm else x),
            mt.Lambdad(keys=["image_dicom", "image_png"], func=lambda x: x.expand(3, *x.shape[1:]) if self.expand_image_channel and x.shape[0] == 1 else x),
            mt.Transposed(keys=["image_dicom", "image_png", "mask"], indices=(1, 2, 0))
            #ScaleIntensityRangePercentilesd(keys="image_dicom", lower=0, upper=99.5, b_min=0, b_max=1),
        ])

        self.test_transforms = mt.Compose([
            mt.LoadImaged(keys=["image_dicom"], reader="PydicomReader", image_only=False),
            mt.LoadImaged(keys=["mask", "image_png"], reader="PILReader"),
            mt.EnsureChannelFirstd(keys=["image_dicom", "image_png", "mask"]),
            *([
                mt.Lambda(func=lambda x: {**x, "image_dicom": apply_multichannel_windowing(x["image_dicom"], [80, 30, 40], [40, 35, 40])}),
            ] if multichannel_windowing else [mt.Lambda(func=lambda x: {**x, "image_dicom": apply_windowing(x["image_dicom"], 50, 300)})]),
            mt.Lambda(select_first_channel(["image_png"])),
            mt.ScaleIntensityd(keys=["image_png"]),
            mt.Transposed(keys=["image_dicom", "image_png", "mask"], indices=(0, 2, 1)),
            mt.Resized(keys=["image_dicom", "image_png", "mask"], spatial_size=(self.size, self.size), mode=[interpolation_image, interpolation_image, interpolation_mask]),
            mt.Lambdad(keys=["mask"],func=lambda x: mt.AsDiscrete(argmax=False, to_onehot=num_classes)(x) if self.mask_onehot else x),
            mt.Lambdad(keys=["mask"],func=lambda x: mt.ScaleIntensityRange(a_min=0, a_max=1, b_min=-1.0, b_max=1.0, clip=True)(x) if self.mask_scale_dpmm else x),
            mt.Lambdad(keys=["image_dicom", "image_png"], func=lambda x: mt.ScaleIntensityRange(a_min=0, a_max=1, b_min=-1.0, b_max=1.0, clip=True)(x) if self.image_scale_dpmm else x),
            mt.Lambdad(keys=["image_dicom", "image_png"], func=lambda x: x.expand(3, *x.shape[1:]) if self.expand_image_channel and x.shape[0] == 1 else x),
            mt.Transposed(keys=["image_dicom", "image_png", "mask"], indices=(1, 2, 0))
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

        elif self.data_type == "image_dicom":
            example["image"] = example_transformed["image_dicom"]
            return example

        else:  # data_type == "all"
            example["image"] = example_transformed["image_png"]
            example["segmentation"] = example_transformed["mask"]
            example["class_id"] = np.array([example["class"]])
            return example

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