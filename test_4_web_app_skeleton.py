import streamlit as st
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import plotly.express as px

st.set_page_config(layout="wide")
TARGET_WIDTH = 300  # fixed width for canvas display

# -------------------------------
# Mock helper functions
# -------------------------------
def load_3d_tiff(file):
    """Replace with your 3D TIFF loader"""
    # For now, create random 3D volume
    return np.random.randint(0, 255, size=(20, 256, 256), dtype=np.uint8)

def segment_volume(volume, params):
    """Mock segmentation: assign slice-based labels"""
    segmented = np.zeros_like(volume, dtype=np.uint8)
    for i in range(volume.shape[0]):
        segmented[i] = (i % 5) + 1  # labels 1..5
    return segmented

def get_3d_label_block(segmented_volume, label):
    """Return 3D coordinates of the given label"""
    z, y, x = np.where(segmented_volume == label)
    return np.stack([x, y, z], axis=1)

def resize_image_keep_aspect(img, target_width=TARGET_WIDTH):
    w, h = img.size
    scale = target_width / w
    new_h = int(h * scale)
    return img.resize((target_width, new_h))

# -------------------------------
# Layout: Left panel for controls
# -------------------------------
col_controls, col_canvas = st.columns([1, 3])

with col_controls:
    st.header("Controls")

    # File uploader
    uploaded = st.file_uploader("Upload 3D TIFF", type=["tif","tiff"])

    # Filter parameters
    st.subheader("Filter parameters")
    threshold = st.number_input("Threshold", 0, 255, 128)
    pixel_size = st.number_input("Pixel size", 0.1, 10.0, 1.0)
    params = {"threshold": threshold, "pixel_size": pixel_size}

    # GPU selection (optional)
    import pyclesperanto_prototype as cle
    GPUs = cle.available_device_names()
    if len(GPUs) > 0:
        GPU_selected = st.selectbox("Select GPU", GPUs)
        cle.select_device(GPU_selected)

# -------------------------------
# Main canvases
# -------------------------------
with col_canvas:
    st.header("3D Image Viewer")

    # Placeholder 3D volumes if no file uploaded
    if uploaded is None:
        depth, height, width = 20, 256, 256
        volume = np.zeros((depth, height, width), dtype=np.uint8)
        segmented = np.zeros((depth, height, width), dtype=np.uint8)
    else:
        volume = load_3d_tiff(uploaded)
        segmented = segment_volume(volume, params)
        depth, height, width = volume.shape

    # Z-slice slider (shared for original & segmented)
    z_index = st.slider("Z-slice", 0, depth-1, 0)

    # Three columns for canvases
    col1, col2, col3 = st.columns([1, 1, 1])

    # --- Left: Original ---
    with col1:
        st.subheader("Original")
        img_left = Image.fromarray(volume[z_index])
        coords_left = streamlit_image_coordinates(img_left, key=f"left_{z_index}")
        st.image(img_left, caption=f"Slice {z_index}", use_column_width=True)

    # --- Middle: Segmented ---
    with col2:
        st.subheader("Segmented")
        img_middle = Image.fromarray(segmented[z_index])
        coords_middle = streamlit_image_coordinates(img_middle, key=f"middle_{z_index}")
        st.image(img_middle, caption=f"Slice {z_index}", use_column_width=True)

    # --- Right: 3D scatter plot ---
    with col3:
        st.subheader("3D Label View")
        if coords_middle is not None:
            clicked_label = segmented[z_index, coords_middle['y'], coords_middle['x']]
            if clicked_label > 0:
                points = get_3d_label_block(segmented, clicked_label)
                fig3d = px.scatter_3d(x=points[:,0], y=points[:,1], z=points[:,2], size_max=2)
                st.plotly_chart(fig3d, use_container_width=True)
            else:
                st.write("Background clicked — no label")
        else:
            st.write("Click on a segmented region to see its 3D view")
