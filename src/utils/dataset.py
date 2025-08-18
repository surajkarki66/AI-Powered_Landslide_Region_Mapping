import os
import h5py
import rasterio
import numpy as np

from typing import Optional, List, Tuple
from torch.utils.data import Dataset as BaseDataset


class Landslide4SenseDataset(BaseDataset):
    """
    Dataset class for binary segmentation of landslide-affected areas using multidimensional geospatial HDF5 data.

    Args:
        images_dir (str): Path to folder containing input HDF5 image files.
        masks_dir (str): Path to folder containing corresponding HDF5 binary mask files (0 = background, 255 = landslide area).
    """

    def __init__(self, images_dir: str, masks_dir: str):
        # Get list of all file IDs from the images directory
        self.ids: List[str] = os.listdir(images_dir)
        
        # Construct full paths for images and masks
        self.images_fps: List[str] = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps: List[str] = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # Precomputed mean and std for each of the 14 image channels
        self.mean: List[float] = [-0.4914, -0.3074, -0.1277, -0.0625, 0.0439, 0.0803,
                                  0.0644, 0.0802, 0.3000, 0.4082, 0.0823, 0.0516, 0.3338, 0.7819]
        self.std: List[float] = [0.9325, 0.8775, 0.8860, 0.8869, 0.8857, 0.8418,
                                 0.8354, 0.8491, 0.9061, 1.6072, 0.8848, 0.9232, 0.9018, 1.2913]

    def __getitem__(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and return one image-mask pair with preprocessing.

        Returns:
            image (np.ndarray): Normalized input image, shape (C, H, W).
            label (np.ndarray): Binary mask, shape (1, H, W).
        """

        # Load image from HDF5 file
        with h5py.File(self.images_fps[i], 'r') as hf:
            image = hf['img'][:]

        # Load mask from HDF5 file
        with h5py.File(self.masks_fps[i], 'r') as hf:
            label = hf['mask'][:]

        # Convert to float32
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        # Change image shape from (H, W, C) to (C, H, W)
        image = image.transpose((-1, 0, 1))

        # Add channel dimension to mask
        label = np.expand_dims(label, axis=0)

        # Normalize each channel
        for c in range(len(self.mean)):
            image[c, :, :] -= self.mean[c]
            image[c, :, :] /= self.std[c]

        return image.copy(), label.copy()

    def __len__(self) -> int:
        return len(self.ids)


class Landslide4SenseDataset_CrossValidation(BaseDataset):
    """
    Cross-validation version of Landslide4SenseDataset.
    
    Args:
        images_dir (str): Path to folder containing input HDF5 images.
        masks_dir (str): Path to folder containing masks.
        indices (Optional[List[int]]): List of indices to include in this dataset split.
    """

    def __init__(self, images_dir: str, masks_dir: str, indices: Optional[List[int]] = None):
        all_ids: List[str] = sorted(os.listdir(images_dir))
        self.ids: List[str] = [all_ids[i] for i in indices] if indices is not None else all_ids

        self.images_fps: List[str] = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps: List[str] = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        self.mean: List[float] = [-0.4914, -0.3074, -0.1277, -0.0625, 0.0439, 0.0803,
                                  0.0644, 0.0802, 0.3000, 0.4082, 0.0823, 0.0516, 0.3338, 0.7819]
        self.std: List[float] = [0.9325, 0.8775, 0.8860, 0.8869, 0.8857, 0.8418,
                                 0.8354, 0.8491, 0.9061, 1.6072, 0.8848, 0.9232, 0.9018, 1.2913]

    def __getitem__(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        with h5py.File(self.images_fps[i], 'r') as hf:
            image = hf['img'][:]
        with h5py.File(self.masks_fps[i], 'r') as hf:
            label = hf['mask'][:]

        image = np.asarray(image, np.float32).transpose((-1, 0, 1))
        label = np.expand_dims(np.asarray(label, np.float32), axis=0)

        for c in range(len(self.mean)):
            image[c, :, :] -= self.mean[c]
            image[c, :, :] /= self.std[c]

        return image.copy(), label.copy()

    def __len__(self) -> int:
        return len(self.ids)


class CAS_Landslide_Dataset(BaseDataset):
    """
    Dataset for binary segmentation using .tif images and masks.

    Args:
        images_dir (str): Path to input .tif images.
        masks_dir (str): Path to binary .tif masks.
        augmentation (Optional[object]): Albumentations augmentation pipeline.
    """

    def __init__(self, images_dir: str, masks_dir: str, augmentation: Optional[object] = None):
        self.ids: List[str] = sorted(os.listdir(images_dir))
        self.images_fps: List[str] = [os.path.join(images_dir, fname) for fname in self.ids]
        self.masks_fps: List[str] = [os.path.join(masks_dir, fname) for fname in self.ids]
        self.augmentation: Optional[object] = augmentation

    def __getitem__(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        with rasterio.open(self.images_fps[i]) as src_img:
            image = src_img.read()  # (C, H, W)
        with rasterio.open(self.masks_fps[i]) as src_mask:
            mask = src_mask.read(1)  # (H, W)

        image = np.transpose(image, (1, 2, 0))  # (H, W, C)
        mask = np.expand_dims((mask > 0).astype("float32"), axis=-1)  # (H, W, 1)

        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        image = image.transpose(2, 0, 1)  # (C, H, W)
        mask = mask.transpose(2, 0, 1)    # (1, H, W)

        return image, mask

    def __len__(self) -> int:
        return len(self.ids)


class CAS_Landslide_Dataset_Cross_Validation(BaseDataset):
    """
    Cross-validation version of CAS_Landslide_Dataset.

    Args:
        images_dir (str): Path to input .tif images.
        masks_dir (str): Path to binary .tif masks.
        indices (Optional[List[int]]): List of indices for this split.
        augmentation (Optional[object]): Albumentations augmentation pipeline.
    """

    def __init__(self, images_dir: str, masks_dir: str, indices: Optional[List[int]] = None, augmentation: Optional[object] = None):
        all_ids: List[str] = sorted(os.listdir(images_dir))
        self.ids: List[str] = [all_ids[i] for i in indices] if indices is not None else all_ids
        self.images_fps: List[str] = [os.path.join(images_dir, fname) for fname in self.ids]
        self.masks_fps: List[str] = [os.path.join(masks_dir, fname) for fname in self.ids]
        self.augmentation: Optional[object] = augmentation

    def __getitem__(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        with rasterio.open(self.images_fps[i]) as src_img:
            image = src_img.read()  # (C, H, W)
        with rasterio.open(self.masks_fps[i]) as src_mask:
            mask = src_mask.read(1)  # (H, W)

        image = np.transpose(image, (1, 2, 0))  # (H, W, C)
        mask = np.expand_dims((mask > 0).astype("float32"), axis=-1)  # (H, W, 1)

        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        image = image.transpose(2, 0, 1)  # (C, H, W)
        mask = mask.transpose(2, 0, 1)    # (1, H, W)

        return image, mask

    def __len__(self) -> int:
        return len(self.ids)
