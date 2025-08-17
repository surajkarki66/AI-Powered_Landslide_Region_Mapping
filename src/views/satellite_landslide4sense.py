import streamlit as st
import onnxruntime as ort
import numpy as np

from src.utils.utils import sigmoid, preprocess_h5, load_config, create_rgb_composite, overlay_mask
from src.utils.generate_pdf import generate_landslide_pdf


MEAN = np.array([
    -0.4914, -0.3074, -0.1277, -0.0625,
     0.0439,  0.0803,  0.0644,  0.0802,
     0.3000,  0.4082,  0.0823,  0.0516,
     0.3338,  0.7819
], dtype=np.float32).reshape((1, 14, 1, 1))

STD = np.array([
    0.9325, 0.8775, 0.8860, 0.8869,
    0.8857, 0.8418, 0.8354, 0.8491,
    0.9061, 1.6072, 0.8848, 0.9232,
    0.9018, 1.2913
], dtype=np.float32).reshape((1, 14, 1, 1))

@st.cache_resource
def load_model(model_path):
    return ort.InferenceSession(model_path)


st.set_page_config(page_title="Landslide Mapping", layout="wide")
st.title("Landslide Mapping using Landslide4Sense Model (HDF5 Input)")
tab = st.radio("Select Mode", ["📂 Single Prediction", "📁 Batch Prediction"])

with st.spinner("Loading the model..."):
    config = load_config()
    model_path = config["deployment"]["landslide4sense"]["model_path"]
    model_session = load_model(model_path)

if tab == "📂 Single Prediction":
    uploaded_file = st.file_uploader("Upload an HDF5 File", type=["h5"])
    ground_resolution = st.number_input("Ground resolution (meters/pixel)", min_value=0.1,
                                        max_value=100.0, value=1.0, step=0.1, format="%.2f")

    if uploaded_file:
        try:
            img_np = preprocess_h5(uploaded_file)
            img_tensor = np.expand_dims(np.transpose(img_np, (2, 0, 1)), axis=0)
            img_tensor = ((img_tensor - MEAN) / STD).astype(np.float32)
            st.session_state["input_array"] = img_tensor
            st.session_state["ground_resolution"] = ground_resolution
            st.session_state["img_np"] = img_np
            st.success(f"✅ File '{uploaded_file.name}' loaded.")
        except Exception as e:
            st.error(f"❌ Failed to load HDF5 file: {e}")

    if st.button("Run Segmentation", key="single_run_h5"):
        if "input_array" not in st.session_state:
            st.warning("⚠️ Please upload an HDF5 file before running segmentation.")
        else:
            with st.spinner("Running segmentation and calculating area..."):
                ort_inputs = {model_session.get_inputs()[0].name: st.session_state["input_array"]}
                ort_outs = model_session.run(None, ort_inputs)
                mask_logits = ort_outs[0][0]
                mask_probs = sigmoid(mask_logits)
                mask_binary = (mask_probs > 0.5).astype(np.uint8)
                rgb_img = create_rgb_composite(st.session_state["img_np"])
                overlay_img = overlay_mask(rgb_img, mask_binary)
                pixel_area_m2 = st.session_state["ground_resolution"] ** 2
                landslide_pixel_count = np.sum(mask_binary > 0)
                total_area_m2 = landslide_pixel_count * pixel_area_m2
                total_area_km2 = total_area_m2 / 1e6

                st.success(f"Estimated Landslide Area: {total_area_m2:.2f} m² ({total_area_km2:.6f} km²)")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(rgb_img, caption="RGB of Input", use_container_width=True)
                with col2:
                    st.image((mask_binary*255).squeeze(), caption="Predicted Mask", use_container_width=True, clamp=True)
                with col3:
                    st.image(overlay_img, caption="Overlay", use_container_width=True)
                

elif tab == "📁 Batch Prediction":
    uploaded_files = st.file_uploader("Upload multiple HDF5 files", type=["h5"], accept_multiple_files=True)
    batch_ground_resolution = st.number_input("Ground resolution for all (meters/pixel)", min_value=0.1,
                                             max_value=100.0, value=1.0, step=0.1, format="%.2f", key="batch_res_h5")

    if st.button("Run Batch Segmentation"):
        if not uploaded_files:
            st.warning("⚠️ Please upload at least one HDF5 file.")
        else:
            with st.spinner("Running batch segmentation..."):
                processed_masks, overlay_images, areas_m2, areas_km2, filenames, rgb_images = [], [], [], [], [], []
                preprocessed_batch = []

                for file in uploaded_files:
                    try:
                        img_np = preprocess_h5(file)
                        img_tensor = np.expand_dims(np.transpose(img_np, (2, 0, 1)), axis=0)
                        img_tensor = ((img_tensor - MEAN) / STD).astype(np.float32)
                        preprocessed_batch.append(img_tensor)
                        filenames.append(file.name)
                        rgb_images.append(create_rgb_composite(img_np))
                    except Exception as e:
                        st.error(f"❌ Failed to process {file.name}: {e}")

                if not preprocessed_batch:
                    st.warning("⚠️ No valid files to process.")
                else:
                    batch_tensor = np.concatenate(preprocessed_batch, axis=0)
                    ort_inputs = {model_session.get_inputs()[0].name: batch_tensor}
                    ort_outs = model_session.run(None, ort_inputs)
                    mask_logits = ort_outs[0]

                    for i, logit in enumerate(mask_logits):
                        mask_probs = sigmoid(logit)
                        mask_binary = (mask_probs > 0.5).astype(np.uint8)
                        overlay_img = overlay_mask(rgb_images[i], mask_binary)
                        landslide_pixel_count = np.sum(mask_binary > 0)
                        pixel_area_m2 = batch_ground_resolution ** 2
                        total_area_m2 = landslide_pixel_count * pixel_area_m2
                        total_area_km2 = total_area_m2 / 1e6
                        processed_masks.append((mask_binary*255).squeeze())
                        overlay_images.append(overlay_img)
                        areas_m2.append(total_area_m2)
                        areas_km2.append(total_area_km2)

                    pdf_bytes = generate_landslide_pdf(
                        images=rgb_images,
                        masks=processed_masks,
                        overlays=overlay_images,
                        filenames=filenames,
                        areas_m2=areas_m2,
                        areas_km2=areas_km2
                    )

                    st.success("✅ Batch segmentation completed!")
                    st.download_button("📄 Download PDF Report", pdf_bytes, file_name="landslide_report.pdf")
