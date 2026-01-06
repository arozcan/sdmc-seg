import numpy as np
import torch


# Color map for 14 channels
COLOR_MAP = np.array([
    [0., 0., 0.],        # Channel 0 (black)
    [0., 0., 255.],      # Channel 1 (blue)
    [0., 255., 0.],      # Channel 2 (green)
    [255., 0., 0.],      # Channel 3 (red)
    [255., 255., 0.],    # Channel 4 (yellow)
    [0., 255., 255.],    # Channel 5 (cyan)
    [255., 0., 255.],    # Channel 6 (magenta)
    [255., 239., 213.],  # Channel 7 (beige)
    [0., 0., 205.],      # Channel 8 (medium blue)
    [205., 133., 63.],   # Channel 9 (peru)
    [210., 180., 140.],  # Channel 10 (tan)
    [102., 205., 170.],  # Channel 11 (aquamarine)
    [0., 0., 128.],      # Channel 12 (navy)
    [0., 139., 139.],    # Channel 13 (dark cyan)
])

def colorize_multichannel_segment(image):
    """
    Assigns colors to a multi-channel image based on a color map.

    Args:
        image (numpy.ndarray): Input image with shape (Height, Width, Channels).
                               Each pixel indicates active channels.
        color_map (numpy.ndarray): Color map with shape (Channels, 3), where each
                                    row defines an RGB color for a channel.

    Returns:
        numpy.ndarray: Output image with shape (Height, Width, 3), representing the
                       colorized image.

    Raises:
        ValueError: If the input image is not 3-dimensional or if the number of
                    channels in the image exceeds the rows in the color map.
    """
    # Check if the input image has 3 dimensions
    if image.ndim != 3:
        raise ValueError("Input image must be 3-dimensional: [Height, Width, Channels].")
    
    # Check if the color map has enough entries for all channels
    if image.shape[2] > COLOR_MAP.shape[0]:
        raise ValueError("Color map must have at least as many entries as the number of channels.")

    # Determine the active channel for each pixel
    channel_indices = np.argmax(image, axis=2)  # Shape: (Height, Width)

    # Get the maximum values for each pixel
    max_values = np.max(image, axis=2)  # Shape: (Height, Width)

    # Map the active channel indices to their corresponding colors
    base_colors = COLOR_MAP[channel_indices]  # Shape: (Height, Width, 3)

    # Scale the colors based on the maximum values
    output_image = (base_colors * (max_values[:, :, None] / 255)).astype(np.uint8)

    return output_image

def modify_image_channels_for_stroke_dataset(image):
    modified_image = image.copy()
    modified_image[:, :, 0] = 0
    modified_image[:, :, [1, 2]] = modified_image[:, :, [2, 1]]

    return modified_image


def modify_batch_image_channels_for_stroke_dataset(images,clip=None):
    modified_images = images.clone()
    modified_images[:, 0, :, :] = 0
    modified_images[:, [1, 2], :, :] = modified_images[:, [2, 1], :, :]
    if clip:
        modified_images = torch.clamp(modified_images, 0, clip)
    return modified_images


def overlay_images(image, label, alpha=0.5, clip=255):

    label = alpha * label
    overlay = image + label

    overlay = np.clip(overlay, 0, clip)
    return overlay.astype(np.uint8)

def overlay_images_batch(images, labels, alpha=0.5, clip=1.0, labels_to_bin=False):
    """
    Applies overlay operation for a batch of images and labels using PyTorch.
    
    Parameters:
    - images (torch.Tensor): Batch of images with shape [batch_size, channels, height, width].
    - labels (torch.Tensor): Batch of labels with shape [batch_size, channels, height, width].
    - alpha (float): Transparency factor for the labels.
    - clip (int): Value to clip the overlay results.
    
    Returns:
    - torch.Tensor: Batch of overlay images with shape [batch_size, channels, height, width].
    """
    # Ensure images are 3-channel by expanding if single-channel
    if images.shape[1] == 1:
        images = images.expand(-1, 3, -1, -1)  # [batch_size, 3, height, width]

    if labels_to_bin:
        labels = torch.where(labels > 0.5, 1.0, 0.0)
        
    # Ensure images and labels are float tensors for calculation
    images = images.float()
    labels = labels.float()
    
    # Apply alpha to the labels
    labels = alpha * labels
    
    # Combine images and labels with overlay
    overlay = images + labels
    
    # Clip values to the specified range
    overlay = torch.clamp(overlay, 0, clip)
    
    return overlay
