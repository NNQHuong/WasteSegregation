import streamlit as st
from ultralytics import solutions 
solutions.inference(model="C:\Users\QUYNH HUONG\Downloads\best.pt")
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import cv2
import numpy as np 
import os
import requests

def download_image(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return np.array(image)
    except Exception as e:
        st.error(f'Error downloading image: {e}')
    return None

def preprocess_image(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (640,640))
    return image

def classify_image(image):
    results = solutions.inference()
    if results is None:
        return "No classification made."
    labels = results[0].names
    detections = results[0].boxes
    detected_labels = []
    for box in detections:
        class_id = int(box.cls)
        confidence = box.conf
        label = labels[class_id]
        detected_labels.append(f'{label}: {confidence:.2f}')
    return detected_labels

def main():
    st.title("Waste Image Segregation using YOLO11n-cls")
    st.markdown("<style>/body</style>", unsafe_allow_html = True)
    st.header("Upload an Image for Classification")
    upload_option = st.radio("Select upload method:", ["Upload from computer"])

    if upload_option == "Upload from computer":
        uploaded_file = st.file_uploader("Choose an image", type=['jpg','jpeg','png'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption = 'Uploaded Image', use_column_width=True)
    
    if 'image' in locals() and st.button("classify_image"):
        if image is not None:
            try:
                preprocessed_image = preprocess_image(image)
                label_text = classify_image(preprocessed_image)
                st.success(label_text)
            except Exception as e:
                st.error(f"Error classifying image: {e}")
        else:
            st.error('Error: Image not found for classification.')

    st.markdown('---')
    st.subheader('About Us')
    st.write('This project aims to classify and segregate waste images using the YOLO11n-cls model.')
    st.image("C:\Users\QUYNH HUONG\Documents\STUDY\PTDL\authorproject.jpg",
             caption = 'Honorable mention :3', use_container_width=True)
    st.write("""**YOLO11n-cls** is a state-of-the-art model designed for waste image classification.
    It utilizes deep learning techniques to accurately classify various types of waste,
    helping in effective waste management and segregation.
    """)
    st.image("C:\Users\QUYNH HUONG\Documents\STUDY\PTDL\modelcomparison.png", 
             caption="Comparison between YOLO11n-cls and EfficientNet_B7", use_container_width=True)

if __name__ == "__main__":
    main()