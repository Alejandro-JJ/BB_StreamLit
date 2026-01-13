import streamlit as st
import pyclesperanto_prototype as cle
import numpy as np

# Scan for available GPUs
GPUs = cle.available_device_names()

st.subheader("Available Graphics Cards")

if len(GPUs) == 0:
    st.error("No GPU detected! Falling back to CPU.")
else:
    # Let user select GPU
    GPU_selected = st.selectbox("Select a GPU to use:", GPUs)

    # When the user selects a GPU, configure it
    cle.select_device(GPU_selected)
    st.success(f"The GPU '{GPU_selected}' has been selected for processing")
