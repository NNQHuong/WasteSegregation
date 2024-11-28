import streamlit as st
import cv2
import numpy as np
import os
from ultralytics import YOLO
from ultralytics import solutions
from PIL import Image
import requests
from io import BytesIO

def download_file(file_id, destination):
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        directory = os.path.dirname(destination)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    else:
        print(f"Failed to download file with ID {file_id}. Status code: {response.status_code}")
        return False

file_ids = [
    "1OsAa-MryW-4lNvzweL2xxw2qjn4yyUk4",
    "1HZlxtAUO1utwc2p9Rl5g5a54GJpNsdiS",
    "1BgyfGvFiqOpIWqHDH-FPnlbA5L-1TR7B"]
destination_paths = [
    "best.pt",
    "authorproject.jpg",
    "modelcomparison.jpg"]
for file_id, destination in zip(file_ids, destination_paths):
    if download_file(file_id, destination):
        st.success(f"Downloaded {destination} successfully.")
    else:
        st.error(f"Failed to download {destination}.")

if os.path.exists("best.pt"):
    model=YOLO("best.pt")
    if model is not None:
        st.success("Model loaded successfully.")
else:
    st.error("Model file does not exist. Please check the download process.")

def download_image(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return np.array(image)
    except Exception as e:
        st.error(f"Error downloading image: {e}")
        return None


def classify_image(image):
    results = model(image)
    if results is None:
        return "No classification made."
    labels = results[0].names
    detections = results[0].boxes
    detected_labels = []
    for box in detections:
        class_id = int(box.cls)
        confidence = box.conf
        label = labels[class_id]
        eco = 'biodegradable' if label in ['food_waste', 'wood_waste', 'paper_waste', 'leaf_waste'] else 'non-biodegradable'
        detected_labels.append(f"{label}: {confidence:.2f}, {eco}.")

    return detected_labels


def main():
    st.title("Waste Image Segregation using YOLO11n - cls")
    st.markdown("<style>/body{background-color: #17462C;}</style>", unsafe_allow_html=True)
    st.header("Upload an Image for Classification")
    upload_option = st.radio("Select Upload Method:", ["Upload from computer", "Input image URL."])

    if upload_option == "Upload from computer":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption='Uploaded Image', use_container_width=True)
    elif upload_option == "Input image URL":
        image_url = st.text_input("Enter the URL of the image:")
        if st.button("Download Image"):
            image = download_image(image_url)
            if image is not None:
                st.image(image, caption='Downloaded Image', use_container_width=True)

    if 'image' in locals() and st.button("Classify Image"):
        if image is not None:
            try:
                label_text = classify_image(image)
                st.success(label_text)
            except Exception as e:
                st.error(f"Error classifying image: {e}")
        else:
            st.error('Error: Image not found for classification.')
    st.markdown("---")
    st.subheader("About Us")
    st.write("This project aims to classify and segregate waste images using the YOLO11n-cls model.")
    st.image("authorproject.jpg", caption="Honorable mention :3", use_container_width=True)
    st.write("""**YOLO11n-cls** is a state-of-the-art model designed for waste image classification.
    It utilizes deep learning techniques to accurately classify various types of waste,
    helping in effective waste management and segregation.
    """)
    st.image("modelcomparison.jpg", caption="Comparison between YOLO11n-cls and EfficientNet_B7", use_container_width=True)

if __name__ == "__main__":
    main()