import streamlit as st
import numpy as np
import tifffile
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import pyclesperanto_prototype as cle # for GPU selection
from bb_funcs.Segmentation import Threshold_Im, MasterSegmenter
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import colorcet as cc


xkcd = mcolors.XKCD_COLORS # for xkcd colors, e.g. xkcd['xkcd:light teal']

CANVAS_DISPLAY_WIDTH = 400
SLIDER_WIDTH = 400

st.set_page_config(layout="wide")

# -------------------------------------------------------
# CSS: slider width only
# -------------------------------------------------------
st.markdown(
    f"""
    <style>
    div[data-testid="stSlider"] > div {{
        width: {SLIDER_WIDTH}px !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------
# Helper functions
# -------------------------------------------------------
def load_3d_tiff(file):
    volume = tifffile.imread(file)
    if volume.ndim != 3:
        st.error(f"Invalid TIFF shape: {volume.shape}. Must be 3D (Z,Y,X).")
        return None
    return volume

def normalize_for_display(slice_2d):
    """Scale 2D slice to uint8 [0,255] for safe display.
    The images are not modified for analysis, since
    we need to take proper pixel values"""
    slice_min = slice_2d.min()
    slice_max = slice_2d.max()
    if slice_max > slice_min:
        slice_norm = (slice_2d - slice_min) / (slice_max - slice_min)
    else:
        slice_norm = slice_2d * 0
    return (slice_norm * 255).astype(np.uint8)

@st.cache_data
def prepare_slices_for_display(volume):
    """
    Takes whole volume, normalizes to (0,255)
    and resizes to respect canvas dimensions
    """
    slices_display = []
    for z in range(volume.shape[0]):
        slice_disp = normalize_for_display(volume[z])
        img = Image.fromarray(slice_disp)
        w, h = img.size
        scale = CANVAS_DISPLAY_WIDTH / w
        img_resized = img.resize((CANVAS_DISPLAY_WIDTH, int(h*scale)))
        slices_display.append(np.array(img_resized))
    return np.array(slices_display)

def get_pixel_value(volume, z, x, y):
    if 0 <= z < volume.shape[0] and 0 <= y < volume.shape[1] and 0 <= x < volume.shape[2]:
        return volume[z, y, x]
    return None

# Custom glasbey colormap
mycmap = cc.glasbey_bw_minc_20_minl_30
mycmap[0]=[0,0,0] # black as first value
cmap_glasbey = LinearSegmentedColormap.from_list('cmap_glasbey', mycmap)


def prepare_slices_for_display_and_color(volume, colormap='cmap_glasbey'):
    """
    Takes whole volume, normalizes to (0,255)
    and resizes to respect canvas dimensions.
    If colormap is provided, it is applied to normalized slices.
    
    colormap: None or name of a matplotlib colormap (e.g. "magma", "viridis")
    """
    slices_display = []

    for z in range(volume.shape[0]):
        slice_2d = volume[z]

        # ----- Normalize to 0–255 -----
        slice_min = slice_2d.min()
        slice_max = slice_2d.max()
        if slice_max > slice_min:
            norm = (slice_2d - slice_min) / (slice_max - slice_min)
        else:
            norm = np.zeros_like(slice_2d, dtype=float)

        # ----- Apply colormap or grayscale -----
        if colormap is not None:
            cmap = cm.get_cmap(colormap)
            colored = cmap(norm)[:, :, :3]               # ignore alpha
            slice_disp = (colored * 255).astype(np.uint8)
        else:
            slice_disp = (norm * 255).astype(np.uint8)

        # ----- Resize -----
        img = Image.fromarray(slice_disp)
        w, h = img.size
        scale = CANVAS_DISPLAY_WIDTH / w
        img_resized = img.resize((CANVAS_DISPLAY_WIDTH, int(h * scale)))

        slices_display.append(np.array(img_resized))

    return np.array(slices_display)

# -------------------------------------------------------
# Sidebar: Upload 3D TIFF
# -------------------------------------------------------
st.sidebar.header("TIFF Upload")
uploaded = st.sidebar.file_uploader("Upload 3D TIFF", type=["tif", "tiff"])

if uploaded is not None:
    volume = load_3d_tiff(uploaded)
    if volume is None:
        st.stop()
    depth, height, width = volume.shape
    slices_display = prepare_slices_for_display(volume)
else:
    depth, height, width = 20, 256, 256
    volume = np.zeros((depth, height, width), dtype=np.uint16)
    slices_display = np.zeros((depth, CANVAS_DISPLAY_WIDTH, CANVAS_DISPLAY_WIDTH), dtype=np.uint8)

st.write("Streamlit version:", st.__version__)
# -------------------------------------------------------
# Sidebar: GPU Selection
# -------------------------------------------------------
st.sidebar.header("GPU Selection")
GPUs = cle.available_device_names()
if len(GPUs) == 0:
    st.sidebar.error("No GPU detected. Using CPU.")
else:
    GPU_selected = st.sidebar.selectbox("Select GPU:", GPUs)
    cle.select_device(GPU_selected)
    st.sidebar.write(f"GPU selected: {GPU_selected}")

# -------------------------------------------------------
# Sidebar: Parameters
# ADD HELP FEATURE!!    
# -------------------------------------------------------
st.sidebar.header("Segmentation Parameters")
SB_colA, SB_colB = st.sidebar.columns(2)
with SB_colA:
    BackGroundNoise = st.number_input("Background noise", 0, 100, 20, step=10)
    Threshold = st.number_input("Threshold", 0, 10**8, 1000, step=10)
with SB_colB:
    SpotSize = st.number_input("Spot size", 0, 100, 1, step=1)
    Outline = st.number_input("Outline", 0, 100, 0, step=1)

st.sidebar.header("Camera parameters")
SB_colC, SB_colD = st.sidebar.columns(2)
with SB_colC:
    Pixel_XY = st.number_input("Pixel size XY (µm)", 0.0, 10.0, 1.0, step=0.1)
with SB_colD:
    Pixel_Z = st.number_input("Pixel size Z (µm)", 0.0, 10.0, 1.0, step=0.1)

st.sidebar.header("Elastic problem")
SB_colE, SB_colF = st.sidebar.columns(2)
with SB_colE:
    G = st.number_input("G [kPa]", 0, 10**8, 1000, step=1000)
with SB_colF:
    Poisson = st.number_input(r"$\nu$", 0.0, 1.0, 0.5, step=0.1)

SH_order = st.sidebar.number_input("Spherical Harmonics solution order", 1, 10, 4)

# -------------------------------------------------------
# Main Layout
# -------------------------------------------------------
col1, col2, col3 = st.columns([1,1,1])

# LEFT: original
with col1:
    st.subheader("Fluorescence image")
    left_placeholder = st.empty() # will be filled later with im
    z_index = st.slider("Z-slice", 0, depth-1, 0, key="left_slider")
    run_segment = st.button("SEGMENT", disabled=(uploaded is None)) # not active until image

# PROCESS SEGMENTATION
if run_segment:
    # Add a spinner status bar
    with st.spinner("Running segmentation..."): # argument show_time=True  will be included in future release
        #segmented_volume = Threshold_Im(volume, Threshold, snooze=3) # change snooze
        segmented_volume, N_beads = MasterSegmenter(volume, timepoint=0, 
                                                    backg_r=20, 
                                                    threshold=100, 
                                                    spot_sigma=1, 
                                                    outline_sigma=1, 
                                                    perc_int=100, snooze=0)
        st.session_state["segmented_volume"] = segmented_volume
    st.success(f"Segmentation complete! \nFound {N_beads} beads.")

# MIDDLE: segmented
with col2:
    st.subheader("Segmented image")
    if "segmented_volume" in st.session_state:
        #segmented_display = prepare_slices_for_display(st.session_state["segmented_volume"])
        segmented_display = prepare_slices_for_display_and_color(st.session_state["segmented_volume"], 
                                                                 colormap=cmap_glasbey)
        display_image = segmented_display[z_index]
    else:
        display_image = np.zeros((CANVAS_DISPLAY_WIDTH, CANVAS_DISPLAY_WIDTH), dtype=np.uint8)

    coords = streamlit_image_coordinates(
        display_image,
        key=f"middle_canvas_slice_{z_index}",
        width=CANVAS_DISPLAY_WIDTH
    )

# RIGHT: placeholder
with col3:
    st.subheader("Segmented objects")
    st.write("3D view will appear here.")

# Update left canvas
left_placeholder.image(
    slices_display[z_index],
    caption=f"Slice {z_index} / {depth}",
    width=CANVAS_DISPLAY_WIDTH
)

# Handle clicks
if coords is not None:
    x_click, y_click = coords["x"], coords["y"]
    x_orig = int(x_click * width / slices_display[z_index].shape[1])
    y_orig = int(y_click * height / slices_display[z_index].shape[0])
    value = get_pixel_value(
        st.session_state.get("segmented_volume", volume),
        z_index, x_orig, y_orig
    )
    st.write(f"Clicked pixel: x={x_orig}, y={y_orig}, z={z_index} value={value}")
