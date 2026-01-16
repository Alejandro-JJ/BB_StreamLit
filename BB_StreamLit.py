import streamlit as st
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates
import pyclesperanto_prototype as cle # for GPU selection
from bb_funcs.Segmentation import Threshold_Im, MasterSegmenter, ExtractSurface, Analysis
from bb_funcs.Tools import load_3d_tiff, prepare_slices_for_display_and_color, get_pixel_value
from bb_funcs.Plotter import Plotter_MapOnMap_Plotly
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import colorcet as cc
import io
import tifffile # to save fillfiles

xkcd = mcolors.XKCD_COLORS

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
# Cached wrapper 
# -------------------------------------------------------
mycmap = cc.glasbey_bw_minc_20_minl_30
mycmap[0] = [0,0,0]
cmap_glasbey = LinearSegmentedColormap.from_list('cmap_glasbey', mycmap)

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
    slices_display = prepare_slices_for_display_and_color_cached(
        volume, canvas_display_width=CANVAS_DISPLAY_WIDTH, _colormap='gray'
    )
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
# -------------------------------------------------------
st.sidebar.header("Segmentation Parameters")
SB_colA, SB_colB = st.sidebar.columns(2)
with SB_colA:
    BackGroundNoise = st.number_input("Background noise", 0, 100, 20, step=10)
    Threshold = st.number_input("Threshold", 0, 10**8, 100, step=10)
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
    left_placeholder = st.empty()
    z_index = st.slider("Z-slice", 0, depth-1, 0, key="left_slider")
    run_segment = st.button("SEGMENT", disabled=(uploaded is None))

# PROCESS SEGMENTATION
if run_segment:
    with st.spinner("Running segmentation..."):
        segmented_volume, N_beads = MasterSegmenter(
            volume, timepoint=0,
            backg_r=BackGroundNoise,
            threshold=Threshold,
            spot_sigma=SpotSize,
            outline_sigma=Outline,
            perc_int=100, snooze=5
        )
        st.session_state["segmented_volume"] = segmented_volume
    st.success(f"Segmentation complete! \nFound {N_beads} beads.")

# MIDDLE: segmented
with col2:
    st.subheader("Segmented image")
    if "segmented_volume" in st.session_state:
        segmented_display = prepare_slices_for_display_and_color_cached(
            st.session_state["segmented_volume"],
            canvas_display_width=CANVAS_DISPLAY_WIDTH,
            _colormap=cmap_glasbey
        )
        display_image = segmented_display[z_index]
    else:
        display_image = np.zeros((CANVAS_DISPLAY_WIDTH, CANVAS_DISPLAY_WIDTH), dtype=np.uint8)

    coords = streamlit_image_coordinates(
        display_image,
        key=f"middle_canvas_slice_{z_index}",
        width=CANVAS_DISPLAY_WIDTH
    )

    extract = st.button("Extract!", disabled=False)
    
    # Same structure as extract button for download: always active, complains if not image in buffer
    # DOWNLOAD BUTTON
#    image = st.session_state.get("extracted_image", None)
#    if image is None:
#        st.download_button(
#            label="Download extracted TIFF",
#            data=b"",
#            file_name="extracted_object.tif",
#            mime="image/tiff",
#            disabled=True,
#            key="download_tiff"
#        )
#    else:
#        buffer = io.BytesIO()
#        tifffile.imwrite(buffer, image)
#        buffer.seek(0)
#    
#        st.download_button(
#            label="Download extracted TIFF",
#            data=buffer,
#            file_name="extracted_object.tif",
#            mime="image/tiff",
#            disabled=False,
#            key="download_tiff"
#        )





# RIGHT: placeholder
with col3:
    st.subheader("Tension map of surface")
#    st.write("3D view will appear here.")

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
    st.write(f"Clicked bead with pixel value: {value}")

    st.session_state["last_click"] = {
        "x": x_orig, "y": y_orig, "z": z_index,
        "value": int(value) if value is not None else None
    }
    

# -------------------------------------------------------
# EXTRACT BUTTON HANDLER — WITH PLOTLY 3D SCATTER
# -------------------------------------------------------
if extract:
    last = st.session_state.get("last_click", None)
    if last is None:
        st.warning("Click on an object first.")
    else:
        label = last["value"]
        seg_vol = st.session_state["segmented_volume"]

        if label == 0 or label is None:
            st.warning("Clicked background (label 0) — nothing to extract.")
        else:
            with st.spinner("Extracting object coordinates and plotting..."):
                # Get voxel coordinates
                #z_idxs, y_idxs, x_idxs = np.where(seg_vol == label)
                #x, y, z, binary_surface = ExtractSurface(seg_vol, label, Pixel_XY, Pixel_Z, buffer=0)
                x, y, z, binary_surface, map_r_R, map_T_R = Analysis(seg_vol, label, Pixel_XY, Pixel_Z, ExpDegree=SH_order, buffer=5)
                
                # Here I load trhe image in disk
                st.session_state["extracted_image"] = binary_surface
                fig = Plotter_MapOnMap_Plotly(map_r_R, map_T_R, title="")
#                fig = Plotter_FlatShade(map_r_R, map_T_R, title="Tension map on surface")
                fig.update_traces(hoverinfo="skip", hovertemplate=None)
                # No hover enevts
                fig.update_layout(
                    hovermode=False,
                    scene=dict(
                        xaxis=dict(showspikes=False),
                        yaxis=dict(showspikes=False),
                        zaxis=dict(showspikes=False),
                    )
                )


                

                if z.size == 0:
                    st.error("No voxels found for that label.")
                else:
#                    pts = np.column_stack((x, y, z))
#
#                    # Subsample if too large
#                    max_points = 5000
#                    if pts.shape[0] > max_points:
#                        step = int(np.ceil(pts.shape[0] / max_points))
#                        pts = pts[::step]
#
#                    # ---- PLOTLY 3D SCATTER ----
#                    fig = go.Figure(data=[
#                        go.Scatter3d(
#                            x=pts[:,0], y=pts[:,1], z=pts[:,2],
#                            mode="markers",
#                            marker=dict(size=2, opacity=0.7)
#                        )
#                    ])
#
#                    fig.update_layout(
#                        width=500, height=500,
#                        scene=dict(
#                            xaxis_title="X",
#                            yaxis_title="Y",
#                            zaxis_title="Z",
#                            aspectmode="data" # to scale
#                        ),
#                        title=f"x_span = {max(x)-min(x)}, y_span = {max(y)-min(y)}, z_span = {max(z)-min(z)}"
#                    )

                    with col3:
                        st.plotly_chart(fig, use_container_width=True)
                        
# DOWNLOAD BUTTON HANDLER (only after extract, so that it gets activated directly, without the necessity to rerun)
with col2:
    image = st.session_state.get("extracted_image", None)
    if image is None:
        st.download_button(
            label="Download extracted TIFF",
            data=b"",
            file_name="extracted_object.tif",
            mime="image/tiff",
            disabled=True,
            key="download_tiff"
        )
    else:
        buffer = io.BytesIO()
        tifffile.imwrite(buffer, image)
        buffer.seek(0)

        st.download_button(
            label="Download extracted TIFF",
            data=buffer,
            file_name="extracted_object.tif",
            mime="image/tiff",
            disabled=False,
            key="download_tiff"
        )

