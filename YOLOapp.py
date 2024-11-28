# Ultralytics YOLO 🚀, AGPL-3.0 license

import io
import time
from PIL import Image 
import cv2
import torch
import numpy as np

from ultralytics.utils.checks import check_requirements

def inference(model_path):
    """Performs object detection on uploaded images using a custom YOLO model in a Streamlit web application."""
    check_requirements("streamlit>=1.29.0")  # scope imports for faster ultralytics package load speeds
    import streamlit as st
    from ultralytics import YOLO

    # Hide main menu style
    menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""

    # Main title of streamlit application
    main_title_cfg = """<div><h1 style="color:#FF64DA; text-align:center; font-size:40px; 
                             font-family: 'Archivo', sans-serif; margin-top:-50px;margin-bottom:20px;">
                    Custom YOLO Image Classification
                    </h1></div>"""

    # Subtitle of streamlit application
    sub_title_cfg = """<div><h4 style="color:#042AFF; text-align:center; 
                    font-family: 'Archivo', sans-serif; margin-top:-15px; margin-bottom:50px;">
                    Upload an image to classify objects with your custom YOLO model! 🚀</h4>
                    </div>"""

    # Set html page configuration
    st.set_page_config(page_title="Waste image segregation, using YOLO11n-cls.", layout="wide", initial_sidebar_state="auto")

    # Append the custom HTML
    st.markdown(menu_style_cfg, unsafe_allow_html=True)
    st.markdown(main_title_cfg, unsafe_allow_html=True)
    st.markdown(sub_title_cfg, unsafe_allow_html=True)

    # Add ultralytics logo in sidebar
    with st.sidebar:
        logo = "https://cdn.pixabay.com/photo/2015/08/14/19/50/maple-888807_1280.jpg"
        st.image(logo, width=250)

    # Load the custom YOLO model
    with st.spinner("Loading model..."):
        model = YOLO(model_path)  # Load your custom YOLO model
        class_names = list(model.names.values())  # Convert dictionary to list of class names
    st.success("Model loaded successfully!")

    # Upload image file
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        image_np = np.array(image)  # Convert to NumPy array

        # Perform inference
        results = model(image_np)  # Perform detection
        annotated_image = results[0].plot()  # Annotate the image

        # Display the original and annotated images
        col1, col2 = st.columns(2)
        col1.image(image, caption="Original Image", use_column_width=True)
        col2.image(annotated_image, caption="Annotated Image", use_column_width=True)

        # Display detected classes
        detected_classes = results[0].names
        st.write("Detected Classes:")
        st.write(detected_classes)

# Main function call
if __name__ == "__main__":
    model_path = r"C:\Users\QUYNH HUONG\Downloads\best.pt"  # Specify your custom model path here
    inference(model_path)