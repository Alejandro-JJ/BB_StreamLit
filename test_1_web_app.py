import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(layout="wide")

TARGET_WIDTH = 600  # <- fixed width for canvas

# --- helper: resize preserving aspect ratio ---
def resize_image_keep_aspect(img, target_width=600):
    w, h = img.size
    scale = target_width / w
    new_h = int(h * scale)
    return img.resize((target_width, new_h))

# ----- Layout -----
col_left, col_right = st.columns([1,3])

# ----- LEFT: upload -----
with col_left:
    st.header("Controls")
    uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# ----- RIGHT: canvas -----
with col_right:
    st.header("Image Viewer")

    if uploaded is None:
        # black placeholder, fixed width, auto height (square for simplicity)
        placeholder_img = Image.new("RGB", (TARGET_WIDTH, TARGET_WIDTH), "black")
        coords = streamlit_image_coordinates(placeholder_img, key="placeholder")

    else:
        img = Image.open(uploaded)

        # resize uploaded image to fixed width
        img_resized = resize_image_keep_aspect(img, TARGET_WIDTH)

        coords = streamlit_image_coordinates(img_resized, key="realimage")

# ----- STATUS BAR -----
st.markdown("<br><br>", unsafe_allow_html=True)
center_col = st.columns([1,2,1])[1]

with center_col:
    st.subheader("Status:")
    if uploaded is None:
        st.success("No coords!")
    else:
        if coords is None:
            st.warning("Click on the image…")
        else:
            st.success(f"Clicked at: (x={coords['x']}, y={coords['y']})")
