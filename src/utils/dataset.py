import os
import h5py
import rasterio
import numpy as np

from torch.utils.data import Dataset as BaseDataset


class Landslide4SenseDataset(BaseDataset):
    """
    Dataset class for binary segmentation of landslide-affected areas using multidimensional geospatial data.

    Args:
        images_dir (str): Path to the folder containing input HDF5 image files.
        masks_dir (str): Path to the folder containing corresponding HDF5 binary mask files (0 = background, 255 = landslide area).
    """

    def __init__(self, images_dir, masks_dir):
        # Get list of all file IDs from the images directory
        self.ids = os.listdir(images_dir)
        
        # Construct full paths for images and masks
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # Precomputed mean and std for each of the 14 image channels (for normalization)
        self.mean = [-0.4914, -0.3074, -0.1277, -0.0625, 0.0439, 0.0803, 0.0644, 0.0802, 0.3000, 0.4082, 0.0823, 0.0516, 0.3338, 0.7819]
        self.std =  [0.9325, 0.8775, 0.8860, 0.8869, 0.8857, 0.8418, 0.8354, 0.8491, 0.9061, 1.6072, 0.8848, 0.9232, 0.9018, 1.2913]

    def __getitem__(self, i):
        """
        Load and return one image-mask pair with preprocessing.

        Returns:
            image (np.ndarray): Normalized input image with shape (C, H, W).
            label (np.ndarray): Binary mask with shape (1, H, W).
        """

        # Load image data from HDF5 file
        with h5py.File(self.images_fps[i], 'r') as hf:
            image = hf['img'][:]

        # Load corresponding mask data from HDF5 file
        with h5py.File(self.masks_fps[i], 'r') as hf:
            label = hf['mask'][:]

        # Convert image and label to float32 numpy arrays
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        # Rearrange image dimensions from (H, W, C) to (C, H, W) for PyTorch
        image = image.transpose((-1, 0, 1))

        # Add channel dimension to label to make shape (1, H, W)
        label = np.expand_dims(label, axis=0)

        # Normalize each channel of the image using precomputed mean and std
        for i in range(len(self.mean)):
            image[i, :, :] -= self.mean[i]
            image[i, :, :] /= self.std[i]

        return image.copy(), label.copy()

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.ids)




class Landslide4SenseDataset_CrossValidation(BaseDataset):
    """
    Cross-validation version of Landslide4SenseDataset.
    
    Args:
        images_dir (str): Path to folder containing input HDF5 image files.
        masks_dir (str): Path to folder containing corresponding HDF5 mask files.
        indices (list, optional): List of indices to include in this dataset split.
    """

    def __init__(self, images_dir, masks_dir, indices=None):
        # Get list of all file IDs
        all_ids = sorted(os.listdir(images_dir))
        
        # Select subset if indices are provided
        if indices is not None:
            self.ids = [all_ids[i] for i in indices]
        else:
            self.ids = all_ids

        # Construct full paths
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # Precomputed mean and std for each of the 14 image channels
        self.mean = [-0.4914, -0.3074, -0.1277, -0.0625, 0.0439, 0.0803, 0.0644,
                     0.0802, 0.3000, 0.4082, 0.0823, 0.0516, 0.3338, 0.7819]
        self.std =  [0.9325, 0.8775, 0.8860, 0.8869, 0.8857, 0.8418, 0.8354,
                     0.8491, 0.9061, 1.6072, 0.8848, 0.9232, 0.9018, 1.2913]

    def __getitem__(self, i):
        # Load image
        with h5py.File(self.images_fps[i], 'r') as hf:
            image = hf['img'][:]

        # Load mask
        with h5py.File(self.masks_fps[i], 'r') as hf:
            label = hf['mask'][:]

        # Convert to float32
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        # Rearrange dimensions (H, W, C) -> (C, H, W)
        image = image.transpose((-1, 0, 1))

        # Add channel dimension to label
        label = np.expand_dims(label, axis=0)

        # Normalize each channel
        for c in range(len(self.mean)):
            image[c, :, :] -= self.mean[c]
            image[c, :, :] /= self.std[c]

        return image.copy(), label.copy()

    def __len__(self):
        return len(self.ids)



class CAS_Landslide_Dataset(BaseDataset):
    """
    Binary Segmentation Dataset for Landslides using .tif images and masks.
    Args:
        images_dir (str): Path to input .tif images.
        masks_dir (str): Path to binary .tif masks (pixel values: 0 for background, 1 for landslide).
        augmentation (albumentations.Compose): Optional augmentations.
    """

    def __init__(self, images_dir, masks_dir, augmentation=None):
        self.ids = sorted(os.listdir(images_dir))
        self.images_fps = [os.path.join(images_dir, fname) for fname in self.ids]
        self.masks_fps = [os.path.join(masks_dir, fname) for fname in self.ids]
        self.augmentation = augmentation

    def __getitem__(self, i):
        # Read image using rasterio
        with rasterio.open(self.images_fps[i]) as src_img:
            image = src_img.read()  # Shape: (C, H, W)

        # Read mask using rasterio
        with rasterio.open(self.masks_fps[i]) as src_mask:
            mask = src_mask.read(1)  # Read the first band only, shape: (H, W)

        # Convert image to HWC for albumentations
        image = np.transpose(image, (1, 2, 0))  # (H, W, C)

        # Convert mask to binary (0 or 1)
        mask = (mask > 0).astype("float32")
        mask = np.expand_dims(mask, axis=-1)  # (H, W, 1)

        # Apply augmentations
        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        # Convert back to CHW
        image = image.transpose(2, 0, 1)  # (C, H, W)
        mask = mask.transpose(2, 0, 1)    # (1, H, W)

        return image, mask

    def __len__(self):
        return len(self.ids)

class CAS_Landslide_Dataset_Cross_Validation(BaseDataset):
    def __init__(self, images_dir, masks_dir, indices=None, augmentation=None):
        all_ids = sorted(os.listdir(images_dir))
        if indices is not None:
            self.ids = [all_ids[i] for i in indices]
        else:
            self.ids = all_ids
        self.images_fps = [os.path.join(images_dir, fname) for fname in self.ids]
        self.masks_fps = [os.path.join(masks_dir, fname) for fname in self.ids]
        self.augmentation = augmentation

    def __getitem__(self, i):
        with rasterio.open(self.images_fps[i]) as src_img:
            image = src_img.read()  # (C, H, W)
        with rasterio.open(self.masks_fps[i]) as src_mask:
            mask = src_mask.read(1)  # (H, W)

        image = np.transpose(image, (1, 2, 0))  # (H, W, C)
        mask = (mask > 0).astype("float32")
        mask = np.expand_dims(mask, axis=-1)

        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)
        return image, mask

    def __len__(self):
        return len(self.ids)
