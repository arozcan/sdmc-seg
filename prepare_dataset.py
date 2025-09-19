import os
import numpy as np
from skimage import color
from PIL import Image
import SimpleITK as sitk
import json
from tqdm import tqdm
import shutil
import pydicom

NONSTROKE_IMAGE_FOLDER = 'Training/Non-Stroke/'
ISCHEMIC_IMAGE_FOLDER = 'Training/Ischemic/'
HEMORRHAGIC_IMAGE_FOLDER = 'Training/Hemorrhagic/'

PNG_FOLDER = "PNG/"
OVERLAY_FOLDER = "OVERLAY/"
DICOM_FOLDER = "DICOM/"

NONSTROKE_MASK_VALUE = 0
ISCHEMIC_MASK_VALUE = 1
HEMORRHAGIC_MASK_VALUE = 2

DATASET_TRAINING_IMAGE_FOLDER = 'imageTr'
DATASET_TRAINING_MASK_FOLDER = 'maskTr'

def update_bits_stored(file_path, new_bits_stored=16):
    dataset = pydicom.dcmread(file_path)
    dataset.BitsStored = new_bits_stored
    dataset.save_as(file_path)

def create_masks_from_overlay(source_folder, target_folder, label_value=1, threshold_value=0.5, format="png"):
    """
    This function processes all PNG files in the specified source folder, converts them to HSV format,
    thresholds the saturation channel with the threshold_value, and saves the resulting masks
    with 0, 1, and 2 values in .nii.gz format to the target folder.
    
    Args:
    source_folder (str): Path to the source folder containing the PNG files.
    target_folder (str): Path to the target folder where the masks will be saved.
    mask_value (int): Value of the mask.
    threshold_value (float): Threshold value for the saturation channel.
    format (str): Format in which to save the masks, either "png" or "nifti".
    """
    
    # Create the target folder if it does not exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # List all PNG files in the source folder
    png_files = [f for f in os.listdir(source_folder) if f.endswith('.png') and not f.startswith('.')]

    # Create and save masks using a threshold of 0.4
    for png_file in tqdm(png_files, desc="Creating masks: " + source_folder):
        # Construct the file path
        file_path = os.path.join(source_folder, png_file)
        
        # Load the image in RGBA format
        image_rgba = np.array(Image.open(file_path))
        
        # Convert from RGBA to RGB (remove the alpha channel)
        image_rgb = image_rgba[:, :, :3]
        
        # Convert the image to HSV format
        image_hsv = color.rgb2hsv(image_rgb)
        
        # Extract the saturation channel
        saturation = image_hsv[:, :, 1]
        
        # Apply the threshold to create a mask
        mask = np.where(saturation >= threshold_value, label_value, 0).astype(np.uint8)

        if format == "png":
            # Convert the mask to PIL Image format
            mask_image = Image.fromarray(mask, mode="L")

            # Create a new filename by replacing the original extension
            png_filename = os.path.splitext(png_file)[0] + ".png"
            png_file_path = os.path.join(target_folder, png_filename)
            
            # Save the mask in PNG format
            mask_image.save(png_file_path)
        elif format == "nifti":
            # Convert the mask to SimpleITK Image format
            mask_sitk = sitk.GetImageFromArray(mask)

            # Create a new filename by replacing the original extension
            nii_filename = os.path.splitext(png_file)[0] + ".nii.gz"
            nii_file_path = os.path.join(target_folder, nii_filename)
            
            # Save the mask in .nii.gz format
            sitk.WriteImage(mask_sitk, nii_file_path)

    print("Mask creation and saving process completed.")

def create_masks_from_overlay_and_image(source_image_folder, source_overlay_folder, target_folder, label_value=1, format="png"):
    """
    This function processes all PNG files in the specified source folder, converts them to HSV format,
    thresholds the saturation channel with the threshold_value, and saves the resulting masks
    with 0, 1, and 2 values in .nii.gz format to the target folder.
    
    Args:
    source_folder (str): Path to the source folder containing the PNG files.
    target_folder (str): Path to the target folder where the masks will be saved.
    mask_value (int): Value of the mask.
    threshold_value (float): Threshold value for the saturation channel.
    format (str): Format in which to save the masks, either "png" or "nifti".
    """
    
    # Create the target folder if it does not exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # List all PNG files in the source folder
    png_files = [f for f in os.listdir(source_overlay_folder) if f.endswith('.png') and not f.startswith('.')]

    # Create and save masks using a threshold of 0.4
    for png_file in tqdm(png_files, desc="Creating masks: " + source_overlay_folder):
        # Construct the file path
        file_path_overlay = os.path.join(source_overlay_folder, png_file)
        file_path_image = os.path.join(source_image_folder, png_file)


        overlay_rgba = np.array(Image.open(file_path_overlay))
        image_rgba = np.array(Image.open(file_path_image))
        overlay_rgb = overlay_rgba[:, :, :3]
        image_rgb = image_rgba[:, :, :3]

        # Görüntülerin boyutlarının aynı olduğundan emin olun
        assert image_rgb.shape == overlay_rgb.shape, "Image dimensions must be the same!"

        # Overlay olan pikselleri tespit etme
        mask = np.any(overlay_rgb != image_rgb, axis=-1).astype(np.uint8)
        mask[mask == 1] = label_value

        if format == "png":
            # Convert the mask to PIL Image format
            mask_image = Image.fromarray(mask, mode="L")

            # Create a new filename by replacing the original extension
            png_filename = os.path.splitext(png_file)[0] + ".png"
            png_file_path = os.path.join(target_folder, png_filename)
            
            # Save the mask in PNG format
            mask_image.save(png_file_path)
        elif format == "nifti":
            # Convert the mask to SimpleITK Image format
            mask_sitk = sitk.GetImageFromArray(mask)

            # Create a new filename by replacing the original extension
            nii_filename = os.path.splitext(png_file)[0] + ".nii.gz"
            nii_file_path = os.path.join(target_folder, nii_filename)
            
            # Save the mask in .nii.gz format
            sitk.WriteImage(mask_sitk, nii_file_path)

    print("Mask creation and saving process completed.")

def convert_images_to_nifti(source_folder, target_folder):
    """
    This function processes all PNG files in the specified source folder,
    converts them to NIfTI format, and saves them in the target folder.
    
    Args:
    source_folder (str): Path to the source folder containing PNG files.
    target_folder (str): Path to the target folder where the NIfTI files will be saved.
    """
    
    # Create the target folder if it does not exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # List all PNG files in the source folder
    png_files = [f for f in os.listdir(source_folder) if f.endswith('.png') and not f.startswith('.')]

    image_paths = []

    for png_file in tqdm(png_files, desc="Converting images to NIfTI: " + source_folder):
        # Construct the file path
        file_path = os.path.join(source_folder, png_file)
        
        # Load the image and convert it to grayscale
        image = np.array(Image.open(file_path).convert("L"))

        # Convert the image to SimpleITK Image format
        image_sitk = sitk.GetImageFromArray(image)

        # Create a new filename by replacing the original extension
        nii_filename = os.path.splitext(png_file)[0] + ".nii.gz"
        nii_file_path = os.path.join(target_folder, nii_filename)
        
        # Save the image in NIfTI format
        sitk.WriteImage(image_sitk, nii_file_path)
        image_paths.append(nii_file_path)

    print("Image conversion to NIfTI completed.")
    return image_paths

def copy_images(source_folder, target_folder, format="png"):
    """
    This function copies all PNG or DICOM files from the specified source folder
    to the target folder without opening them, saving them in the specified format.
    
    Args:
    source_folder (str): Path to the source folder containing the images.
    target_folder (str): Path to the target folder where the copied files will be saved.
    format (str): File extension, either "png" or "dicom".
    """
    
    # Create the target folder if it does not exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # List all files in the source folder based on the format
    if format == "png":
        image_files = [f for f in os.listdir(source_folder) if f.endswith('.png') and not f.startswith('.')]
    elif format == "dicom":
        image_files = [f for f in os.listdir(source_folder) if f.endswith('.dcm') and not f.startswith('.')]

    image_paths = []

    for image_file in tqdm(image_files, desc="Copying images: " + source_folder):
        # Construct the file path
        file_path = os.path.join(source_folder, image_file)
        image_file_path = os.path.join(target_folder, image_file)
        
        # Copy the file
        shutil.copy(file_path, image_file_path)
        if format == "dicom":
            update_bits_stored(image_file_path, new_bits_stored=16)
        image_paths.append(image_file_path)
        
    print("Image copying process completed.")
    return image_paths

def prepare_dataset(root_dir, target_dir, create_masks=True):

    if create_masks:
        create_masks_from_overlay_and_image(root_dir+NONSTROKE_IMAGE_FOLDER+PNG_FOLDER, root_dir+NONSTROKE_IMAGE_FOLDER+PNG_FOLDER, target_dir+DATASET_TRAINING_MASK_FOLDER, label_value=NONSTROKE_MASK_VALUE)
        create_masks_from_overlay_and_image(root_dir+ISCHEMIC_IMAGE_FOLDER+PNG_FOLDER, root_dir+ISCHEMIC_IMAGE_FOLDER+OVERLAY_FOLDER,target_dir+DATASET_TRAINING_MASK_FOLDER, label_value=ISCHEMIC_MASK_VALUE)
        create_masks_from_overlay_and_image(root_dir+HEMORRHAGIC_IMAGE_FOLDER+PNG_FOLDER, root_dir+HEMORRHAGIC_IMAGE_FOLDER+OVERLAY_FOLDER,target_dir+DATASET_TRAINING_MASK_FOLDER, label_value=HEMORRHAGIC_MASK_VALUE)
        print("Mask generation process completed.")

    nonstroke_images_png = copy_images(root_dir+NONSTROKE_IMAGE_FOLDER+PNG_FOLDER,target_dir+DATASET_TRAINING_IMAGE_FOLDER,format="png")
    nonstroke_images_dicom = copy_images(root_dir+NONSTROKE_IMAGE_FOLDER+DICOM_FOLDER,target_dir+DATASET_TRAINING_IMAGE_FOLDER,format="dicom")
    ischemic_images_png = copy_images(root_dir+ISCHEMIC_IMAGE_FOLDER+PNG_FOLDER,target_dir+DATASET_TRAINING_IMAGE_FOLDER,format="png")
    ischemic_images_dicom = copy_images(root_dir+ISCHEMIC_IMAGE_FOLDER+DICOM_FOLDER,target_dir+DATASET_TRAINING_IMAGE_FOLDER,format="dicom")
    hemorrhagic_images_png = copy_images(root_dir+HEMORRHAGIC_IMAGE_FOLDER+PNG_FOLDER,target_dir+DATASET_TRAINING_IMAGE_FOLDER,format="png")
    hemorrhagic_images_dicom= copy_images(root_dir+HEMORRHAGIC_IMAGE_FOLDER+DICOM_FOLDER,target_dir+DATASET_TRAINING_IMAGE_FOLDER,format="dicom")

    nonstroke_images_png = sorted(nonstroke_images_png)
    nonstroke_images_dicom = sorted(nonstroke_images_dicom)
    ischemic_images_png = sorted(ischemic_images_png)
    ischemic_images_dicom = sorted(ischemic_images_dicom)
    hemorrhagic_images_png = sorted(hemorrhagic_images_png)
    hemorrhagic_images_dicom = sorted(hemorrhagic_images_dicom)
    all_images_png = sorted(nonstroke_images_png + ischemic_images_png + hemorrhagic_images_png)
    all_images_dicom = sorted(nonstroke_images_dicom + ischemic_images_dicom + hemorrhagic_images_dicom)
    anomaly_images_png = sorted(ischemic_images_png + hemorrhagic_images_png)
    anomaly_images_dicom = sorted(ischemic_images_dicom + hemorrhagic_images_dicom)

    # Create JSON File
    data = {
        "name": "Stroke Data Set",
        "description": "Artificial Intelligence in Healthcare Competition (TEKNOFEST-2021): Stroke Data Set",
        "licence": "CC-BY-SA 4.0",
        "release": "1.0 28/07/2022",
        "tensorImageSize": "2D",
        "modality": {
            "0": "PNG",
            "1": "DICOM"
        },
        "labels": {
            "0": "nonstroke",
            "1": "ischemic",
            "2": "hemorrhagic"
        },
        "numTraining": len(all_images_png),
        "numTrainingNonstroke": len(nonstroke_images_png),
        "numTrainingIschemic": len(ischemic_images_png),
        "numTrainingHemorrhagic": len(hemorrhagic_images_png),
        "training": [{"image_png": os.path.relpath(img_png, target_dir), "image_dicom": os.path.relpath(img_dicom, target_dir), "mask": os.path.relpath(img_png.replace(DATASET_TRAINING_IMAGE_FOLDER, DATASET_TRAINING_MASK_FOLDER), target_dir)} for (img_png,img_dicom) in zip(all_images_png, all_images_dicom)],
        "training_nonstroke": [{"image_png": os.path.relpath(img_png, target_dir), "image_dicom": os.path.relpath(img_dicom, target_dir), "mask": os.path.relpath(img_png.replace(DATASET_TRAINING_IMAGE_FOLDER, DATASET_TRAINING_MASK_FOLDER), target_dir)} for (img_png,img_dicom) in zip(nonstroke_images_png, nonstroke_images_dicom)],
        "training_anomaly": [{"image_png": os.path.relpath(img_png, target_dir), "image_dicom": os.path.relpath(img_dicom, target_dir), "mask": os.path.relpath(img_png.replace(DATASET_TRAINING_IMAGE_FOLDER, DATASET_TRAINING_MASK_FOLDER), target_dir)} for (img_png,img_dicom) in zip(anomaly_images_png, anomaly_images_dicom)],
        "training_ischemic": [{"image_png": os.path.relpath(img_png, target_dir), "image_dicom": os.path.relpath(img_dicom, target_dir), "mask": os.path.relpath(img_png.replace(DATASET_TRAINING_IMAGE_FOLDER, DATASET_TRAINING_MASK_FOLDER), target_dir)} for (img_png,img_dicom) in zip(ischemic_images_png, ischemic_images_dicom)],
        "training_hemorrhagic": [{"image_png": os.path.relpath(img_png, target_dir), "image_dicom": os.path.relpath(img_dicom, target_dir), "mask": os.path.relpath(img_png.replace(DATASET_TRAINING_IMAGE_FOLDER, DATASET_TRAINING_MASK_FOLDER), target_dir)} for (img_png,img_dicom) in zip(hemorrhagic_images_png, hemorrhagic_images_dicom)],
    }

    DATASET_JSON_PATH = os.path.join(target_dir, "dataset.json")

    # Save JSON File
    with open(DATASET_JSON_PATH, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print("JSON file creation process completed.")


if __name__ == "__main__":

    dataset_dir = "downloads/stroke2021/"
    target_dir = "dataset/stroke2021/"

    

    prepare_dataset(dataset_dir, target_dir, create_masks=True)
