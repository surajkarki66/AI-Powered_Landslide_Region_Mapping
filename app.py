import streamlit as st


# --- PAGE SETUP ---
home_page = st.Page(
    "views/home.py",
    title="Home",
    icon=":material/home:",
    default=True,
)

landslide_segmentation_page = st.Page(
    "views/landslide_segmentation.py",
    title="Landslide Segmentation",
    icon="🪨",
)


# --- NAVIGATION SETUP [WITH SECTIONS]---
pg = st.navigation(
    {
        "Info": [home_page],
        "Mapping": [landslide_segmentation_page],
    }
)

# --- SHARED ON ALL PAGES ---
st.sidebar.markdown("Made with ❤️ by Suraj Karki")


# --- RUN NAVIGATION ---
pg.run()