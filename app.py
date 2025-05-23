import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load the YOLOv8 model
MODEL_PATH = "best.pt"  # Update with your model path
model = YOLO(MODEL_PATH)

def process_image(uploaded_image):
    # Convert the uploaded image to a format suitable for YOLO
    image = np.array(uploaded_image)
    results = model(image)
    
    # Annotate the image with detections
    annotated_image = results[0].plot()
    return annotated_image, results[0].boxes

# Streamlit App
st.title("Fortifications Detections using YOLOv8 ")


# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    
    # Process the image
    with st.spinner("Processing..."):
        annotated_image, detections = process_image(image)

    # Display the images side by side
    col1, col2 = st.columns(2)

    with col1:
       
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        
        st.image(annotated_image, caption="Detection Results", use_container_width=True)
    
    # Display detections
    
    if len(detections) > 0:
         st.write("Fortifications Detected")
            
    else:
        st.write("No Fortifications Detected")

st.write("Developed with ❤️ using YOLOv8 and Streamlit")
