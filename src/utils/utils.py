import h5py
import yaml
import cv2
import os
import requests
import numpy as np
import rasterio
import matplotlib.pyplot as plt

from PIL import Image
from skimage.transform import resize
from typing import Any, Dict, Tuple


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute sigmoid activation."""
    return 1 / (1 + np.exp(-x))


def visualize(**images: np.ndarray) -> None:
    """
    Display multiple images side by side.

    Args:
        **images: Keyword arguments where key is the title and value is the image array.
    """
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        if name == "image":
            plt.imshow(image.transpose(1, 2, 0))  # convert CHW to HWC
        else:
            plt.imshow(image)
    plt.show()


def preprocess_image(image: Image.Image, expected_size: Tuple[int, int]) -> np.ndarray:
    """
    Convert PIL image to normalized numpy array (CHW) for model input.

    Args:
        image: PIL Image object
        expected_size: Tuple (width, height) for resizing

    Returns:
        np.ndarray: shape (1, C, H, W)
    """
    image = image.convert("RGB")
    image = image.resize(expected_size)
    image_array = np.array(image).astype(np.float32)
    image_array = np.transpose(image_array, (2, 0, 1))
    return np.expand_dims(image_array, axis=0)


def preprocess_tif(file: str) -> Image.Image:
    """
    Load a .tif file and resize to (512, 512) preserving range.

    Args:
        file: Path to .tif file

    Returns:
        PIL.Image: resized RGB image
    """
    with rasterio.open(file) as src:
        image = src.read([1, 2, 3])
        image = np.transpose(image, (1, 2, 0))
        image = resize(image, (512, 512), preserve_range=True, anti_aliasing=True).astype(np.uint8)
    return Image.fromarray(image)


def preprocess_h5(h5_file: str) -> np.ndarray:
    """
    Load an HDF5 file and return image array without resizing or normalization.

    Args:
        h5_file: path or file-like object of HDF5 file

    Returns:
        np.ndarray: image array of shape (H, W, C)
    """
    with h5py.File(h5_file, 'r') as hf:
        if 'img' not in hf:
            raise KeyError("'img' dataset not found in HDF5 file")
        image = hf['img'][:]
    return image.astype(np.float32)


def create_rgb_composite(img_np: np.ndarray) -> np.ndarray:
    """
    Create RGB composite from 14-channel HDF5 image.

    Args:
        img_np: Input image array (H, W, C)

    Returns:
        np.ndarray: RGB image (H, W, 3) uint8
    """
    red = img_np[:, :, 3]   # B4
    green = img_np[:, :, 2] # B3
    blue = img_np[:, :, 1]  # B2

    def norm(x: np.ndarray) -> np.ndarray:
        return (x - x.min()) / (x.max() - x.min() + 1e-6)

    rgb = np.stack([norm(red), norm(green), norm(blue)], axis=-1)
    return (rgb * 255).astype(np.uint8)


def overlay_mask(rgb_img: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Overlay binary mask on an RGB image.

    Args:
        rgb_img: RGB image array (H, W, 3)
        mask: binary mask (H, W) or (1, H, W)
        alpha: blending factor for overlay

    Returns:
        np.ndarray: RGB image with mask overlay
    """
    mask = np.squeeze(mask)  # ensure (H, W)
    overlay = rgb_img.copy()
    overlay[mask > 0] = [255, 0, 0]  # red overlay
    return cv2.addWeighted(rgb_img, 1 - alpha, overlay, alpha, 0)


def load_config() -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Returns:
        dict: configuration dictionary
    """
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)


def download_models(save_dir="assets"):
    """
    Downloads multiple files from given URLs into the specified directory.
    Skips files that already exist.

    Args:
        urls (list of str): List of file URLs to download
        save_dir (str): Directory to save downloaded files
    """
    model_urls = [
        "https://github.com/surajkarki66/AI-Powered_Landslide_Region_Mapping/releases/download/LandslideSegmentationModels_v1.0/cas_landslide_satellite_model_unet_densenet161.onnx",
        "https://github.com/surajkarki66/AI-Powered_Landslide_Region_Mapping/releases/download/LandslideSegmentationModels_v1.0/cas_landslide_uav_model_unet++_resnet50.onnx",
        "https://github.com/surajkarki66/AI-Powered_Landslide_Region_Mapping/releases/download/LandslideSegmentationModels_v1.0/landslide4sense_model_unet_mobilenetv2.onnx",
    ]
    for url in model_urls:
        filename = url.split("/")[-1]  # Keep original filename
        save_path = os.path.join(save_dir, filename)

        # Skip if file already exists
        if os.path.exists(save_path):
            print(f"✅ Skipped (already exists): {save_path}")
            continue

        print(f"⬇️ Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Write file in chunks for memory efficiency
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"✅ Downloaded to {save_path}")
