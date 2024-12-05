import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
import numpy as np
import cv2
import pandas as pd
import altair as alt
import folium
from streamlit_folium import st_folium
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from PIL import Image
import io
import base64

# Sidebar Navigation using Dropdown
st.sidebar.title("Marine AI Dashboard")
navigation = st.sidebar.selectbox("Navigate", ["Home", "Prediction", "Webcam", "Graphs with Map", "History"])

# Function to create the AI model
def create_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load or create the model
def load_or_create_model():
    try:
        model = load_model('marine_ai_model.h5')
        print("Loaded existing model.")
    except:
        model = create_model()
        model.save('marine_ai_model.h5')
        print("New model created.")
    return model

model = load_or_create_model()

# Preprocess image for prediction
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# Predict the uploaded image
def predict_image(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    classes = ['Clean Ocean', 'Polluted Ocean']
    return classes[np.argmax(predictions)], np.max(predictions)

# Save predictions to CSV
def save_to_csv(data):
    df = pd.DataFrame(data, columns=["Image_Name", "Prediction", "Confidence"])
    df.to_csv("predictions.csv", index=False, mode='a', header=False)

# Display predictions history from CSV
def display_prediction_history():
    try:
        df = pd.read_csv("predictions.csv", names=["Image_Name", "Prediction", "Confidence"])
        st.dataframe(df)
    except FileNotFoundError:
        st.write("No predictions saved yet.")

# Function to handle webcam stream
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = load_or_create_model()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = preprocess_image(img)
        prediction, confidence = predict_image(img)
        label = f"Prediction: {prediction}, Confidence: {confidence * 100:.2f}%"
        return img, label

# Function to display tips on every page
def display_tips():
    st.sidebar.title("Tips")
    st.sidebar.write("""
        **Tip 1:** Use the webcam to get real-time predictions.
        **Tip 2:** Upload clear images for better prediction accuracy.
        **Tip 3:** Use the 'Graphs with Map' section for data visualization.
    """)

# Home Page
if navigation == "Home":
    st.title("JALA JEEVA")
    st.subheader("Marine Ecosystem Monitoring")
    st.write("""
        This app uses AI to detect plastic pollution in ocean images.
        It leverages a pre-trained MobileNetV2 model to classify images
        as either "Clean Ocean" or "Polluted Ocean."
    """)
    display_tips()

# Prediction Page (Image Upload)
elif navigation == "Prediction":
    st.title("Upload an Image for Prediction")
    uploaded_file = st.file_uploader("Upload an Ocean Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        label, confidence = predict_image(image)
        st.write(f"Prediction: **{label}**")
        st.write(f"Confidence: **{confidence * 100:.2f}%**")
        
        if st.button("Save Prediction"):
            save_to_csv([[uploaded_file.name, label, f"{confidence * 100:.2f}%"]])
            st.success("Prediction saved to predictions.csv")
    display_tips()

# Webcam Page (Capture Image from Webcam)
elif navigation == "Webcam":
    st.title("Capture an Image from Webcam")
    
    # Use st.camera_input for capturing the image
    image = st.camera_input("Capture Image")
    
    if image:
        # Convert the base64 image to a format usable by OpenCV
        img_bytes = base64.b64decode(image)
        img = Image.open(io.BytesIO(img_bytes))
        img = np.array(img)
        
        # Display the captured image
        st.image(img, caption="Captured Image", use_column_width=True)
        
        # Get the prediction for the captured image
        prediction, confidence = predict_image(img)
        
        # Display the prediction
        st.write(f"Prediction: {prediction}")
        st.write(f"Confidence: {confidence * 100:.2f}%")
        
        # Optionally save the prediction
        if st.button("Save Prediction"):
            save_to_csv([["Captured Image", prediction, f"{confidence * 100:.2f}%"]])
            st.success("Prediction saved to predictions.csv")
        
        # Display tips after prediction
        display_tips()

# Graphs and Maps Page
elif navigation == "Graphs with Map":
    st.title("Analysis with Graphs and Maps")
    
    # Example of displaying a basic chart with Altair
    chart = alt.Chart(pd.DataFrame({
        'x': [1, 2, 3, 4],
        'y': [10, 20, 30, 40]
    })).mark_line().encode(
        x='x',
        y='y'
    )
    st.altair_chart(chart, use_container_width=True)

    # Example of displaying a map with folium
    map = folium.Map(location=[37.7749, -122.4194], zoom_start=12)
    st_folium(map, width=700, height=500)
    
    display_tips()

# History Page (Display Past Predictions)
elif navigation == "History":
    st.title("Prediction History")
    st.subheader("View your past predictions")
    display_prediction_history()
    display_tips()
