import streamlit as st


# --- PAGE SETUP ---
home_page = st.Page(
    "src/views/home.py",
    title="Home",
    icon=":material/home:",
    default=True,
)

uav_landslide_segmentation_page = st.Page(
    "src/views/uav_landslide_segmentation.py",
    title="Segmentation (using UAV)",
    icon="🚁",
)

satellite_landslide_segmentation_page = st.Page(
    "src/views/satellite_landslide_segmentation.py",
    title="Segmentation (using Satellite)",
    icon="🛰️",
)


# --- NAVIGATION SETUP [WITH SECTIONS]---
pg = st.navigation(
    {
        "Info": [home_page],
        "Landslide Mapping": [uav_landslide_segmentation_page, satellite_landslide_segmentation_page],
    }
)

# --- SHARED ON ALL PAGES ---
st.sidebar.markdown("Made with ❤️ by Suraj Karki")


# --- RUN NAVIGATION ---
pg.run()