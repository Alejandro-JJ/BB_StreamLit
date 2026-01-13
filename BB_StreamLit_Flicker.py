import streamlit as st
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates
import pyclesperanto_prototype as cle # for GPU selection
from bb_funcs.Segmentation import Threshold_Im, MasterSegmenter
from bb_funcs.Tools import load_3d_tiff, prepare_slices_for_display_and_color, get_pixel_value
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import colorcet as cc

#"""
#TODO:
#    - write helper info buttons
#    - integrate a single segment+slice prep function
#    - cache all functions to rerun only once
#    - markdown style sheet
#"""

xkcd = mcolors.XKCD_COLORS # for xkcd colors, e.g. xkcd['xkcd:light teal']

CANVAS_DISPLAY_WIDTH = 400
SLIDER_WIDTH = 400

st.set_page_config(layout="wide")

# -------------------------------------------------------
# CSS: slider width only, can add colors later
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
# Cached wrapper 
# -------------------------------------------------------

# Custom glasbey colormap
mycmap = cc.glasbey_bw_minc_20_minl_30
mycmap[0]=[0,0,0] # black as first value
cmap_glasbey = LinearSegmentedColormap.from_list('cmap_glasbey', mycmap)



# -------------------------------------------------------
# CACHED WRAPPER FOR THE COLORED SLICES (handles unhashable colormap)
# -------------------------------------------------------
@st.cache_data
def prepare_slices_for_display_and_color_cached(volume, canvas_display_width, _colormap):
    return prepare_slices_for_display_and_color(volume, canvas_display_width, _colormap)


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
    slices_display = prepare_slices_for_display_and_color_cached(volume, canvas_display_width=CANVAS_DISPLAY_WIDTH,
                                                                 _colormap='gray')
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
    BackGroundNoise = st.number_input("Background noise", 0, 100, 20, step=10, help='INput here to change whatever')
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
    # A Button returns True only in the run when it was clicked, never again.
    # So the segmenter runs only once!
    
    # Add a spinner status bar
    with st.spinner("Running segmentation..."):
        #segmented_volume = Threshold_Im(volume, Threshold, snooze=3) # debugging
        segmented_volume, N_beads = MasterSegmenter(volume, timepoint=0, 
                                                    backg_r=BackGroundNoise, 
                                                    threshold=Threshold, 
                                                    spot_sigma=SpotSize, 
                                                    outline_sigma=Outline, 
                                                    perc_int=100, snooze=5)
        st.session_state["segmented_volume"] = segmented_volume
    st.success(f"Segmentation complete! \nFound {N_beads} beads.")

# MIDDLE: segmented
with col2:
    st.subheader("Segmented image")
    if "segmented_volume" in st.session_state:
        # This is cached, will only run once when button clicked
        segmented_display = prepare_slices_for_display_and_color_cached(
            st.session_state["segmented_volume"],
            canvas_display_width=CANVAS_DISPLAY_WIDTH,
            _colormap=cmap_glasbey
        )
        display_image = segmented_display[z_index] # This is the only thing that changes
    else:
        display_image = np.zeros((CANVAS_DISPLAY_WIDTH, CANVAS_DISPLAY_WIDTH), dtype=np.uint8)

    coords = streamlit_image_coordinates(
        display_image,
        key=f"middle_canvas_slice_{z_index}",
        width=CANVAS_DISPLAY_WIDTH
    )
    
    # Later on we can decide based on pixel value
#    extract_disabled = False 
    extract = st.button("Extract!", disabled=False)
    # This will return True only once, when clicked

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
#    st.write(f"Clicked pixel: x={x_orig}, y={y_orig}, z={z_index} value={value}")
    st.write(f"Clicked bead with pixel value: {value}")
    
# Save last click in session_state for later extraction
    st.session_state["last_click"] = {
        "x": x_orig,
        "y": y_orig,
        "z": z_index,
        "value": int(value) if value is not None else None}

# Handle button workings  
if extract:
    last = st.session_state["last_click"]
    label = last["value"]
    seg_vol = st.session_state["segmented_volume"]
    
    if label == 0 or label is None:
        st.warning("Clicked background (label 0) — nothing to extract.")
    else:
        with st.spinner("Extracting object coordinates and plotting..."):
            # Option 1: use your custom function (preferred if you have it)
            # pts = my_extract_function(seg_vol, label)    # pts shape (N,3) with (z,y,x) or (x,y,z)
    
            # Option 2: simple numpy extract -> returns (z_idx, y_idx, x_idx)
            z_idxs, y_idxs, x_idxs = np.where(seg_vol == label)
            if z_idxs.size == 0:
                st.error("No voxels found for that label.")
            else:
                # assemble into Nx3 array in preferred plotting order (x,y,z)
                pts = np.column_stack((x_idxs, y_idxs, z_idxs))
    
                # OPTIONAL: subsample if too many points
                max_points = 20000
                if pts.shape[0] > max_points:
                    step = int(np.ceil(pts.shape[0] / max_points))
                    pts = pts[::step]
    
                # make matplotlib 3D scatter
                import matplotlib.pyplot as plt
                fig = plt.figure(figsize=(5,5))
                ax = fig.add_subplot(projection='3d')
                ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=1, alpha=0.8)
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.set_title(f"Object label {label} — {pts.shape[0]} points (plotted {pts.shape[0]})")
                # optional: improve view
                ax.view_init(elev=20, azim=120)
    
                # Display the figure in the right column
                # (Make sure you are at top-level, not inside `with col2:`)
                with col3:
                    st.pyplot(fig)
