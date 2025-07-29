import streamlit as st
import onnxruntime as ort
import numpy as np
import yaml
import rasterio
import segmentation_models_pytorch as smp

from PIL import Image
from skimage.transform import resize

from src.utils.utils import sigmoid


def load_config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)

@st.cache_resource
def load_model(model_path):
    return ort.InferenceSession(model_path)

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

st.set_page_config(page_title="Landslide Segmentation", layout="wide")
st.title("Landslide Mapping using UAV Images")

config = load_config()
model_path = config["deployment"]["caslandslide"]["model_path"]
model_session = load_model(model_path)

uploaded_file = st.file_uploader("Upload an Image (JPG, PNG, or TIF)", type=["jpg", "jpeg", "png", "tif", "tiff"])
ground_resolution = st.number_input("Ground resolution (m/pixel)", min_value=0.1, max_value=100.0, value=1.0, step=0.1, format="%.2f")

if uploaded_file:
    if uploaded_file.name.endswith(('.tif', '.tiff')):
        image = preprocess_tif(uploaded_file)
    else:
        image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = preprocess_image(image, (512, 512))
    st.session_state["input_tensor"] = input_tensor
    st.session_state["ground_resolution"] = ground_resolution


if st.button("Run Segmentation"):
    if "input_tensor" not in st.session_state:
        st.warning("⚠️ Please upload an image before running segmentation.")
    else:
        with st.spinner("Running segmentation and calculating area..."):
            img_np = st.session_state["input_tensor"]
            ground_resolution = st.session_state["ground_resolution"]

            config = load_config()
            params = smp.encoders.get_preprocessing_params(config.get("deployment").get("encoder_name"))
            mean = np.array(params["mean"]).reshape((1, 3, 1, 1))
            std = np.array(params["std"]).reshape((1, 3, 1, 1))
            img_tensor = ((img_np - mean) / std).astype(np.float32)

            ort_inputs = {model_session.get_inputs()[0].name: img_tensor}
            ort_outs = model_session.run(None, ort_inputs)
            mask_logits = ort_outs[0][0]

            mask_probs = sigmoid(mask_logits.squeeze())
            mask_binary = (mask_probs > 0.5).astype(np.uint8) * 255
            mask_img = np.expand_dims(mask_binary, axis=-1)

            st.image(mask_img, caption="Predicted Mask", use_container_width=True, clamp=True)

            pixel_area_m2 = ground_resolution ** 2
            landslide_pixel_count = np.sum(mask_binary > 0)
            total_area_m2 = landslide_pixel_count * pixel_area_m2
            total_area_km2 = total_area_m2 / 1e6

            st.success(f"Estimated Landslide Area: {total_area_m2:.2f} m² ({total_area_km2:.6f} km²)")
