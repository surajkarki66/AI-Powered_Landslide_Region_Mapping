import streamlit as st


# --- MAIN TITLE ---
#st.title("Landslide Mapping", anchor=False)

# --- HERO SECTION ---
col1, col2 = st.columns(2, gap="large", vertical_alignment="center")
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
                <li>📡 Works with satellite and drone imagery (H5, TIF, JPG, PNG formats)</li>
                <li>🧠 Uses deep learning-based semantic segmentation</li>
                <li>🗺️ Visualizes both the input image and predicted landslide area</li>
                <li>📏 Calculates the estimated landslide area based on ground resolution</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

