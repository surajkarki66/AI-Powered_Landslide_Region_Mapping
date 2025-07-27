import os
import numpy as np
import rasterio
import yaml
import streamlit as st
import segmentation_models_pytorch as smp
import onnxruntime as ort

from PIL import Image
from skimage.transform import resize
from src.pipeline.export import check_with_onnx


def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../configs/config.yaml'))
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


@st.cache_resource
def load_model():
    try:
        config = load_config()
        new_config = {}

        new_config["onnx_model_output_path"] = config.get("deployment").get("caslandslide").get("model_path")
        check_with_onnx(new_config)

        # Load the ONNX model
        session = ort.InferenceSession(new_config["onnx_model_output_path"], providers=["CPUExecutionProvider"])
        return session
    except Exception as e:
        st.error(f"Failed to load ONNX model: {e}")
        return None

st.markdown("<h3 style='text-align: center;'>Landslide Segmentation using UAV Images</h3>", unsafe_allow_html=True)

# Load the trained segmentation model
model_session = load_model()

# Upload UAV image
st.info("Upload a UAV image to perform landslide segmentation.")
uploaded_file = st.file_uploader("Upload UAV Image", type=["png", "jpg", "jpeg", "tif"])

if uploaded_file:
    file_ext = os.path.splitext(uploaded_file.name)[-1].lower()

    if "input_tensor" not in st.session_state or st.session_state.get("filename") != uploaded_file.name:
        try:
            if file_ext == ".tif":
                with rasterio.open(uploaded_file) as src:
                    img_np = src.read()  # shape: [C, H, W]
                    preview_np = img_np[:3].transpose(1, 2, 0)
                    img_np = resize(img_np, (img_np.shape[0], 512, 512), preserve_range=True, anti_aliasing=True)
                    preview_img = (preview_np / preview_np.max() * 255).astype(np.uint8)

            elif file_ext in [".png", ".jpg", ".jpeg"]:
                img = Image.open(uploaded_file).convert("RGB")
                img_np = np.array(img)  # shape: [H, W, C]
                preview_np = img_np.copy()
                img_np = resize(img_np, (512, 512, 3), preserve_range=True, anti_aliasing=True)  # resize [H, W, C]
                img_np = img_np.transpose(2, 0, 1)  # to [C, H, W]
                preview_img = preview_np  # already in [H, W, C]
            else:
                st.error("Unsupported file type. Please upload a .tif, .png, or .jpg image.")
                st.stop()

            # Add batch dimension: [1, C, H, W]
            img_np = np.expand_dims(img_np, axis=0)

            # Store in session state
            st.session_state["input_tensor"] = img_np
            st.session_state["preview_image"] = preview_img
            st.session_state["filename"] = uploaded_file.name
        except Exception as e:
            st.error(f"Failed to process image: {e}")
            st.stop()

    # Show preview
    st.image(st.session_state["preview_image"], caption="Preview UAV Image", use_container_width=True)

    # Segmentation
    if model_session:
        if st.button("Run Segmentation"):
            img_np = st.session_state["input_tensor"]
            config = load_config()
            # Normalization
            params = smp.encoders.get_preprocessing_params(config.get("deployment").get("encoder_name"))
            mean = np.array(params["mean"]).reshape((1, 3, 1, 1))
            std = np.array(params["std"]).reshape((1, 3, 1, 1))

            img_tensor = ((img_np - mean) / std).astype(np.float32)

            ort_inputs = {model_session.get_inputs()[0].name: img_tensor}
            ort_outs = model_session.run(None, ort_inputs)
            mask_logits = ort_outs[0][0]

            # Apply sigmoid using numpy
            def sigmoid(x):
                return 1 / (1 + np.exp(-x))

            mask_probs = sigmoid(mask_logits.squeeze())
            mask_binary = (mask_probs > 0.5).astype(np.uint8) * 255
            mask_img = np.expand_dims(mask_binary, axis=-1)

            # Show mask (single-channel grayscale)
            st.image(mask_img, caption="Predicted Mask", use_container_width=True, clamp=True)
