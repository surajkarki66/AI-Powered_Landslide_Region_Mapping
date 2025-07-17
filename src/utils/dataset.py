import os
import cv2
import numpy as np

from torch.utils.data import Dataset as BaseDataset


class Landslide_Dataset(BaseDataset):
    """
    Binary Segmentation Dataset for Landslides.
    Args:
        images_dir (str): Path to input images folder.
        masks_dir (str): Path to binary masks folder (pixel values: 0 for background, 1 for landslide).
        augmentation (albumentations.Compose): Optional augmentations.
    """

    def __init__(self, images_dir, masks_dir, augmentation=None):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.augmentation = augmentation

    def __getitem__(self, i):
        # Read image and convert to RGB
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read mask in grayscale
        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)

        # Convert mask to binary (0 or 1)
        mask = (mask > 0).astype("float32")
        mask = np.expand_dims(mask, axis=-1)  # Shape: (H, W, 1)

        # Apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # Transpose image from HWC to CHW
        return image.transpose(2, 0, 1), mask.transpose(2, 0, 1)

    def __len__(self):
        return len(self.ids)