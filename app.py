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
    title="UAV Landslide Segmentation",
    icon="🪨",
)


# --- NAVIGATION SETUP [WITH SECTIONS]---
pg = st.navigation(
    {
        "Info": [home_page],
        "Mapping": [uav_landslide_segmentation_page],
    }
)

# --- SHARED ON ALL PAGES ---
st.sidebar.markdown("Made with ❤️ by Suraj Karki")


# --- RUN NAVIGATION ---
pg.run()