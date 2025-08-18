import streamlit as st
import onnxruntime as ort
import numpy as np
import segmentation_models_pytorch as smp
import cv2
from PIL import Image

from src.utils.utils import sigmoid, preprocess_image, preprocess_tif, overlay_mask, load_config
from src.utils.generate_pdf import generate_landslide_pdf

load_config()

@st.cache_resource
def load_model(model_path):
    return ort.InferenceSession(model_path)

st.set_page_config(page_title="Landslide Mapping", layout="wide")
st.title("Landslide Mapping using CAS Landslide Model (UAV Imagery)")

tab = st.radio("Select Mode", ["📷 Single Prediction", "📁 Batch Prediction"])

with st.spinner("Loading the model..."):
    config = load_config()
    model_path = config["deployment"]["caslandslide"]["uav"]["model_path"]
    model_session = load_model(model_path)

# --- SINGLE PREDICTION ---
if tab == "📷 Single Prediction":
    uploaded_file = st.file_uploader(
        "Upload an Image (JPG, PNG, or TIF)", type=["jpg", "jpeg", "png", "tif", "tiff"]
    )
    ground_resolution = st.number_input(
        "Ground resolution (meters/pixel)", min_value=0.1, max_value=100.0, value=1.0, step=0.1, format="%.2f"
    )

    if uploaded_file:
        if uploaded_file.name.endswith(('.tif', '.tiff')):
            image = preprocess_tif(uploaded_file)
        else:
            image = Image.open(uploaded_file)

        st.success(f"✅ Image '{uploaded_file.name}' loaded.")

        input_tensor = preprocess_image(image, (512, 512))
        st.session_state["input_tensor"] = input_tensor
        st.session_state["ground_resolution"] = ground_resolution
        st.session_state["orig_image"] = np.array(image.convert("RGB"), dtype=np.uint8)

    if st.button("Run Segmentation", key="single_run"):
        if "input_tensor" not in st.session_state:
            st.warning("⚠️ Please upload an image before running segmentation.")
        else:
            with st.spinner("Running segmentation and calculating area..."):
                img_tensor = st.session_state["input_tensor"]
                ground_resolution = st.session_state["ground_resolution"]
                orig_image = st.session_state["orig_image"]

                params = smp.encoders.get_preprocessing_params(
                    config["deployment"]["caslandslide"]["uav"]["encoder_name"]
                )
                mean = np.array(params["mean"]).reshape((1, 3, 1, 1))
                std = np.array(params["std"]).reshape((1, 3, 1, 1))
                img_tensor = ((img_tensor - mean) / std).astype(np.float32)

                ort_inputs = {model_session.get_inputs()[0].name: img_tensor}
                ort_outs = model_session.run(None, ort_inputs)
                mask_logits = ort_outs[0][0]

                mask_probs = sigmoid(mask_logits.squeeze())
                mask_binary = (mask_probs > 0.5).astype(np.uint8) * 255
                mask_img = Image.fromarray(mask_binary).convert("L")

                if mask_binary.shape != orig_image.shape[:2]:
                    mask_for_overlay = cv2.resize(mask_binary, (orig_image.shape[1], orig_image.shape[0]))
                else:
                    mask_for_overlay = mask_binary

                overlay_np = overlay_mask(orig_image, mask_for_overlay)
                overlay_img = Image.fromarray(overlay_np.astype(np.uint8))

                # Only m² area
                pixel_area_m2 = ground_resolution ** 2
                landslide_pixel_count = np.sum(mask_binary > 0)
                total_area_m2 = landslide_pixel_count * pixel_area_m2
                st.success(f"Estimated Landslide Area: {total_area_m2:.2f} m²")

                # Display horizontally
                col1, col2, col3 = st.columns(3)
                col1.image(orig_image, caption="Input Image", use_container_width=True)
                col2.image(mask_img, caption="Predicted Mask", use_container_width=True)
                col3.image(overlay_img, caption="Overlay Image", use_container_width=True)

# --- BATCH PREDICTION ---
elif tab == "📁 Batch Prediction":
    uploaded_files = st.file_uploader(
        "Upload multiple images", type=["jpg", "jpeg", "png", "tif", "tiff"], accept_multiple_files=True
    )
    batch_ground_resolution = st.number_input(
        "Ground resolution for all (meters/pixel)", min_value=0.1, max_value=100.0,
        value=1.0, step=0.1, format="%.2f", key="batch_res"
    )

    if st.button("Run Batch Segmentation"):
        if not uploaded_files:
            st.warning("⚠️ Please upload at least one image.")
        else:
            with st.spinner("Running batch segmentation..."):
                processed_images, processed_masks, overlays, areas_m2, filenames = [], [], [], [], []
                preprocessed_batch = []

                params = smp.encoders.get_preprocessing_params(
                    config["deployment"]["caslandslide"]["uav"]["encoder_name"]
                )
                mean = np.array(params["mean"]).reshape((1, 3, 1, 1))
                std = np.array(params["std"]).reshape((1, 3, 1, 1))

                for file in uploaded_files:
                    try:
                        if file.name.endswith((".tif", ".tiff")):
                            image = preprocess_tif(file)
                        else:
                            image = Image.open(file)

                        resized_image = image.resize((512, 512))
                        input_tensor = preprocess_image(resized_image, (512, 512))
                        normalized_tensor = ((input_tensor - mean) / std).astype(np.float32)
                        preprocessed_batch.append(normalized_tensor)
                        processed_images.append(resized_image)
                        filenames.append(file.name)
                    except Exception as e:
                        st.error(f"❌ Failed to process {file.name}: {e}")

                if not preprocessed_batch:
                    st.warning("⚠️ No valid images to process.")
                else:
                    batch_tensor = np.concatenate(preprocessed_batch, axis=0)
                    ort_inputs = {model_session.get_inputs()[0].name: batch_tensor}
                    ort_outs = model_session.run(None, ort_inputs)
                    mask_logits = ort_outs[0]

                    for i, logit in enumerate(mask_logits):
                        mask_probs = sigmoid(logit.squeeze())
                        mask_binary = (mask_probs > 0.5).astype(np.uint8) * 255
                        mask_img = np.expand_dims(mask_binary, axis=-1)

                        orig_image = np.array(processed_images[i].convert("RGB"), dtype=np.uint8)
                        if mask_binary.shape != orig_image.shape[:2]:
                            mask_for_overlay = cv2.resize(mask_binary, (orig_image.shape[1], orig_image.shape[0]))
                        else:
                            mask_for_overlay = mask_binary

                        overlay_np = overlay_mask(orig_image, mask_for_overlay)
                        overlay_img = Image.fromarray(overlay_np.astype(np.uint8))

                        # Only m² area
                        landslide_pixel_count = np.sum(mask_binary > 0)
                        pixel_area_m2 = batch_ground_resolution ** 2
                        total_area_m2 = landslide_pixel_count * pixel_area_m2

                        processed_masks.append(mask_img)
                        overlays.append(overlay_img)
                        areas_m2.append(total_area_m2)

                    # Generate PDF without km²
                    pdf_bytes = generate_landslide_pdf(
                        images=processed_images,
                        masks=processed_masks,
                        overlays=overlays,
                        filenames=filenames,
                        areas_m2=areas_m2
                    )

                    st.success("✅ Batch segmentation completed!")
                    st.download_button("📄 Download PDF Report", pdf_bytes, file_name="landslide_report.pdf")
