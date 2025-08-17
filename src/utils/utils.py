import h5py
import yaml
import cv2
import numpy as np
import rasterio
import matplotlib.pyplot as plt

from PIL import Image
from skimage.transform import resize


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def visualize(**images):
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        if name == "image":
            plt.imshow(image.transpose(1, 2, 0))
        else:
            plt.imshow(image)
    plt.show()
    
def preprocess_image(image, expected_size):
    image = image.convert("RGB")
    image = image.resize(expected_size)
    image_array = np.array(image).astype(np.float32)
    image_array = np.transpose(image_array, (2, 0, 1))
    return np.expand_dims(image_array, axis=0)

def preprocess_tif(file):
    with rasterio.open(file) as src:
        image = src.read([1, 2, 3])
        image = np.transpose(image, (1, 2, 0))
        image = resize(image, (512, 512), preserve_range=True, anti_aliasing=True).astype(np.uint8)
    return Image.fromarray(image)


def preprocess_h5(h5_file):
    """
    Load an HDF5 file and return the image as a float32 numpy array
    without resizing or normalization.
    
    Parameters
    ----------
    h5_file : str or file-like
        Path or file object of the HDF5 file.
    
    Returns
    -------
    np.ndarray
        Image array of shape (H, W, C)
    """
    # Load image from HDF5
    with h5py.File(h5_file, 'r') as hf:
        if 'img' not in hf:
            raise KeyError("'img' dataset not found in HDF5 file")
        image = hf['img'][:]

    return image.astype(np.float32)


def create_rgb_composite(img_np):
    """Create RGB composite from HDF5 14-channel input."""
    red = img_np[:, :, 3]   # B4
    green = img_np[:, :, 2] # B3
    blue = img_np[:, :, 1]  # B2

    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-6)

    rgb = np.stack([norm(red), norm(green), norm(blue)], axis=-1)
    return (rgb * 255).astype(np.uint8)


def overlay_mask(rgb_img, mask, alpha=0.4):
    """Overlay binary mask on RGB image."""
    mask = np.squeeze(mask)  # ensure (H, W)
    overlay = rgb_img.copy()
    overlay[mask > 0] = [255, 0, 0]  # red overlay
    return cv2.addWeighted(rgb_img, 1 - alpha, overlay, alpha, 0)

def load_config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)