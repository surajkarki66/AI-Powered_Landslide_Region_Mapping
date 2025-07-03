import streamlit as st


# --- MAIN TITLE ---
st.title("Landslide Mapping", anchor=False)

# --- HERO SECTION ---
col1, col2 = st.columns(2, gap="large", vertical_alignment="center")
with col1:
    #st.image("./assets/omdena_jaipur_chapter.jpeg", use_container_width=True)
    st.write("Image")

with col2:
    st.write("### Title")
    st.write(
        """
        Content
        """
    )
