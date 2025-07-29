import numpy as np
import rasterio
import matplotlib.pyplot as plt

from PIL import Image
from skimage.transform import resize

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
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