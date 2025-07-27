import os
import numpy as np
import rasterio
import yaml
import streamlit as st
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

        new_config["onnx_model_output_path"] = config.get("deployment").get("caslandslide", {}).get("model_path")
        check_with_onnx(new_config)

        # Load the ONNX model
        session = ort.InferenceSession(new_config["onnx_model_output_path"], providers=["CPUExecutionProvider"])
        return session
    except Exception as e:
        st.error(f"Failed to load ONNX model: {e}")
        return None


st.markdown("<h3 style='text-align: center;'>Landslide Segmentation using UAV Images</h3>", unsafe_allow_html=True)

# Load the CAS landslide UAV ONNX model
model_session = load_model()

# Upload image
st.info("Upload a UAV image to perform landslide segmentation.")
uploaded_file = st.file_uploader("Upload UAV Image", type=["png", "jpg", "jpeg", "tif"])

if uploaded_file:
    file_ext = os.path.splitext(uploaded_file.name)[-1].lower()

    if "input_tensor" not in st.session_state or st.session_state.get("filename") != uploaded_file.name:
        # Process image
        if file_ext == ".tif":
            with rasterio.open(uploaded_file) as src:
                img_np = src.read().astype(np.float32)
                if img_np.shape[0] == 1:
                    img_np = np.repeat(img_np, 3, axis=0)
                preview_np = img_np.copy()

                img_np = img_np / 255.0 if img_np.max() > 1 else img_np
                img_np = resize(img_np, (img_np.shape[0], 512, 512), preserve_range=True, anti_aliasing=True)
                img_np = np.expand_dims(img_np.astype(np.float32), axis=0)

                preview_img = np.transpose(preview_np[:3], (1, 2, 0))
                preview_img = (preview_img / preview_img.max() * 255).astype(np.uint8)

        elif file_ext in [".png", ".jpg", ".jpeg"]:
            img = Image.open(uploaded_file).convert("RGB")
            preview_img = img.copy()
            img_resized = img.resize((512, 512))
            img_np = np.array(img_resized).astype(np.float32) / 255.0
            img_np = np.transpose(img_np, (2, 0, 1))
            img_np = np.expand_dims(img_np, axis=0)
        else:
            st.error("Unsupported file type. Please upload a .tif, .png, or .jpg image.")
            st.stop()

        # Store in session state
        st.session_state["input_tensor"] = img_np
        st.session_state["preview_image"] = preview_img
        st.session_state["filename"] = uploaded_file.name

    # Show preview
    st.image(st.session_state["preview_image"], caption="Preview UAV Image", use_container_width=True)

    # Segmentation
    if model_session:
        if st.button("Run Segmentation"):
            img_np = st.session_state["input_tensor"]
            ort_inputs = {model_session.get_inputs()[0].name: img_np}
            ort_outs = model_session.run(None, ort_inputs)

            mask = ort_outs[0][0]
            st.success(f"Segmentation completed. Mask shape: {mask.shape}")

            mask_img = Image.fromarray((mask.squeeze() > 0.5).astype(np.uint8) * 255)
            st.image(mask_img, caption="Predicted Mask", use_container_width=True)
