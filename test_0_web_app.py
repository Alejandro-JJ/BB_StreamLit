import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.title("Click on the Image — Get Coordinates")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded:
    # Load image
    img = Image.open(uploaded)

    # Show in a nice size
    st.subheader("Click on the image:")
    coords = streamlit_image_coordinates(img)

    # Print click coordinates
    if coords is not None:
        st.write(f"**You clicked at:** x = {coords['x']}, y = {coords['y']}")

    st.write("(Click anywhere on the image to get coordinates)")


