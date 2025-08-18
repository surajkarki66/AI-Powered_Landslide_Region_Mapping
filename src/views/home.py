import streamlit as st

# --- HERO SECTION ---
# Relative widths: col1 (image) = 1, col2 (text) = 2
col1, col2 = st.columns([1, 2], gap="large", vertical_alignment="center")

with col1:
    st.image("./assets/logo/landslide.png", use_container_width=True)

with col2:
    st.markdown(
        """
        <div style='display: flex; flex-direction: column; justify-content: center; height: 100%;'>
            <h3>AI-Powered Landslide Mapping</h3>
            <p>
            This application leverages the power of Artificial Intelligence and remote sensing to identify and map landslide-affected areas with high precision.
            </p>
            <ul>
                <li>🧠 Uses deep learning-based semantic segmentation</li>
                <li>🗺️ Visualizes both the input image and predicted landslide area</li>
                <li>📏 Calculates the estimated landslide area based on ground resolution</li>
                <li>⚠️ Each model expects specific input data:
                    <ul>
                        <li>CAS Landslide Segmentation (Satellite): Test Data -> <a href="https://github.com/surajkarki66/AI-Powered_Landslide_Region_Mapping_using_Semantic_Segmentation/tree/main/data/test_data/CAS_Landslide_Satellite_Test_Data" target="_blank">here</a></li>
                        <li>CAS Landslide Segmentation (UAV): Test Data -> <a href="https://github.com/surajkarki66/AI-Powered_Landslide_Region_Mapping_using_Semantic_Segmentation/tree/main/data/test_data/CAS_Landslide_UAV_Test_Data" target="_blank">here</a></li>
                        <li>Landslide4Segmentation (Satellite): Test Data -> <a href="https://github.com/surajkarki66/AI-Powered_Landslide_Region_Mapping_using_Semantic_Segmentation/tree/main/data/test_data/Landslide4Sense_Test_Data" target="_blank">here</a></li>
                    </ul>
                </li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
