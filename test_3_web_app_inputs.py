import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import pyclesperanto_prototype as cle
import numpy as np

# --- Page config ---
st.set_page_config(layout="wide")
TARGET_WIDTH = 600  # fixed width for canvas

# --- Helper: resize image keeping aspect ratio ---
def resize_image_keep_aspect(img, target_width=TARGET_WIDTH):
    w, h = img.size
    scale = target_width / w
    new_h = int(h * scale)
    return img.resize((target_width, new_h))

# --- GPU Selection ---
st.sidebar.header("GPU Selection")
GPUs = cle.available_device_names()
if len(GPUs) == 0:
    st.sidebar.error("No GPU detected! Using CPU.")
else:
    GPU_selected = st.sidebar.selectbox("Select GPU to use:", GPUs)
    cle.select_device(GPU_selected)
    st.sidebar.success(f"GPU selected: {GPU_selected}")

# --- Filter Inputs ---
st.sidebar.header("Filter Parameters")
threshold = st.sidebar.number_input("Threshold", 0.0, 255.0, 128.0, step=1.0)
pixel_size = st.sidebar.number_input("Pixel size (um)", 0.0, 100.0, 1.0, step=0.1)
brightness = st.sidebar.number_input("Brightness", -100.0, 100.0, 0.0, step=1.0)
contrast = st.sidebar.number_input("Contrast", -100.0, 100.0, 0.0, step=1.0)
smoothing = st.sidebar.number_input("Smoothing (sigma)", 0.0, 10.0, 1.0, step=0.1)
radius = st.sidebar.number_input("Radius", 0.0, 50.0, 5.0, step=0.1)

params = {
    "threshold": threshold,
    "pixel_size": pixel_size,
    "brightness": brightness,
    "contrast": contrast,
    "smoothing": smoothing,
    "radius": radius
}

# --- Layout: controls left, canvas right ---
col_left, col_right = st.columns([1, 3])

with col_left:
    st.header("Controls")
    uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

with col_right:
    st.header("Image Viewer")

    # Placeholder black canvas
    if uploaded is None:
        canvas_img = Image.new("RGB", (TARGET_WIDTH, TARGET_WIDTH), "black")
        coords = streamlit_image_coordinates(canvas_img, key="placeholder")
    else:
        img = Image.open(uploaded)
        canvas_img = resize_image_keep_aspect(img)
        coords = streamlit_image_coordinates(canvas_img, key="realimage")
        
# --- Apply Filters button ---
if st.button("Apply Filters") and uploaded is not None:
    st.write("Applying filters with parameters:")
    st.json(params)
    # Here you would call your processing function, e.g.:
    # output_img = process_image(img, **params)
    # st.image(output_img, caption="Processed Image")
    

# --- Status bar at bottom (text only) ---
st.markdown("<br><br>", unsafe_allow_html=True)
center_col = st.columns([1, 2, 1])[1]

with center_col:
    st.subheader("Status:")
    if uploaded is None:
        st.write("No coords!")
    else:
        if coords is None:
            st.write("Click on the image to get coordinates…")
        else:
            st.write(f"Clicked at: (x={coords['x']}, y={coords['y']})")



