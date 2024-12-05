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
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from PIL import Image

# Tips to display
TIPS = [
    " Use the webcam to get real-time predictions.",
    " Upload clear images for better prediction accuracy.",
    "Use the 'Graphs with Map' section for data visualization."
]

# Set the background image
def set_background():
    background_style = f"""
    <style>
    .stApp {{
        background-image: url("https://rare-gallery.com/uploads/posts/1140797-digital-art-simple-background-water-minimalism-fish-blue-waves-gradient-underwater-bubbles-biology-drop-wave-computer-wallpaper-macro-photography-marine-biology-deep-se.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)

# Function to display rolling tips
def display_tips():
    tips_style = """
    <style>
    .rolling-tips {
        font-size: 10px;
        color: #FFFFFF;
        padding: 10px;
        margin-bottom: 15px;
        animation: scroll-tips 10s linear infinite;
        white-space: nowrap;
        overflow: hidden;
        display: block;
        text-align: center;
        border-radius: 5px;
    }

    @keyframes scroll-tips {
        from {
            transform: translateX(100%);
        }
        to {
            transform: translateX(-100%);
        }
    }
    </style>
    <div class="rolling-tips">
        """ + " | ".join(TIPS) + """
    </div>
    """
    st.markdown(tips_style, unsafe_allow_html=True)

# Sidebar Navigation using Dropdown
st.sidebar.title("Marine AI Dashboard")
navigation = st.sidebar.selectbox(
    "Navigate",
    ["Home", "Prediction", "Webcam", "Graphs with Map", "History", "Business Idea", "Marine Ecosystem VR/AR"]
)

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
    except:
        model = create_model()
        model.save('marine_ai_model.h5')
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
def save_to_csv(data, filename="predictions.csv"):
    df = pd.DataFrame(data, columns=["Image_Name", "Prediction", "Confidence"])
    df.to_csv(filename, index=False, mode='a', header=False)

# Display predictions history from CSV
def display_prediction_history(filename="predictions.csv"):
    try:
        df = pd.read_csv(filename, names=["Image_Name", "Prediction", "Confidence"])
        st.dataframe(df)
    except FileNotFoundError:
        st.write("No predictions saved yet.")

# Page: Home
set_background()
if navigation == "Home":
    display_tips()
    st.title("JALA JEEVA")
    st.subheader("Marine Ecosystem Monitoring")
    st.write("""
        This app uses AI to detect plastic pollution in ocean images.
        It leverages a pre-trained MobileNetV2 model to classify images
        as either "Clean Ocean" or "Polluted Ocean."
    """)
    
    # Page: History
elif navigation == "History":
    display_tips()
    st.title("Prediction History")
    try:
        # Try to read the predictions.csv file
        df = pd.read_csv("predictions.csv", names=["Image_Name", "Prediction", "Confidence"])
        if df.empty:
            st.info("No predictions saved yet. Upload or capture an image to start.")
        else:
            st.dataframe(df)
            st.download_button(
                label="Download History",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="predictions_history.csv",
                mime="text/csv",
            )
    except FileNotFoundError:
        st.warning("No history file found. Predictions will be saved here once made.")

# Page: Prediction
elif navigation == "Prediction":
    display_tips()
    st.title("Upload an Image for Prediction")
    uploaded_file = st.file_uploader("Upload an Ocean Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        label, confidence = predict_image(image)
        st.write(f"Prediction: {label}")
        st.write(f"Confidence: {confidence * 100:.2f}%")
        if st.button("Save Prediction"):
            save_to_csv([[uploaded_file.name, label, f"{confidence * 100:.2f}%"]])
            st.success("Prediction saved to predictions.csv")

# Page: Webcam
elif navigation == "Webcam":
    display_tips()
    st.title("Capture an Image from Webcam")
    image = st.camera_input("Capture Image")
    if image:
        img = Image.open(image)
        img = np.array(img)
        st.image(img, caption="Captured Image", use_column_width=True)
        prediction, confidence = predict_image(img)
        st.write(f"Prediction: {prediction}")
        st.write(f"Confidence: {confidence * 100:.2f}%")
        save_to_csv([["Captured Image", prediction, f"{confidence * 100:.2f}%"]])
        st.success("Prediction saved to predictions.csv")

# Page: Graphs with Map
elif navigation == "Graphs with Map":
    display_tips()
    st.title("Analysis with Graphs and Maps")
    pollution_data = [
        [37.7749, -122.4194, 0.6],
        [34.0522, -118.2437, 0.8],
        [40.7128, -74.0060, 0.5],
        [51.5074, -0.1278, 0.7],
        [35.6762, 139.6503, 0.9]
    ]
    df_pollution = pd.DataFrame(pollution_data, columns=["Latitude", "Longitude", "Intensity"])
    map = folium.Map(location=[20.0, 0.0], zoom_start=2)
    heat_data = [[row["Latitude"], row["Longitude"], row["Intensity"]] for _, row in df_pollution.iterrows()]
    HeatMap(heat_data).add_to(map)
    st_folium(map, width=700, height=500)

# Page: Business Idea
elif navigation == "Business Idea":
    display_tips()
    st.title("Business Idea: Marine Ecosystem Conservation")
    st.write("Detailed business plan details...")
    with st.form(key='company_form'):
        company_name = st.text_input("Company Name")
        contact_name = st.text_input("Contact Person Name")
        contact_email = st.text_input("Contact Email")
        phone_number = st.text_input("Phone Number")
        cleaning_price = st.number_input("Price for Ocean Cleaning (per km²)", min_value=0.0, format="%.2f")
        recyclable_price = st.number_input("Price for Taking Recyclable Items (per kg)", min_value=0.0, format="%.2f")
        submit_button = st.form_submit_button("Submit")
        if submit_button:
            if company_name and contact_name and contact_email and phone_number:
                company_data = {
                    "Company Name": company_name,
                    "Contact Person": contact_name,
                    "Contact Email": contact_email,
                    "Phone Number": phone_number,
                    "Ocean Cleaning Price (per km²)": cleaning_price,
                    "Recyclable Items Price (per kg)": recyclable_price
                }
                df = pd.DataFrame([company_data])
                try:
                    df_existing = pd.read_csv("company_partners.csv")
                    df_existing = pd.concat([df_existing, df], ignore_index=True)
                    df_existing.to_csv("company_partners.csv", index=False)
                except FileNotFoundError:
                    df.to_csv("company_partners.csv", index=False)
                st.success("Your credentials have been submitted.")

# Page: Marine Ecosystem VR/AR
elif navigation == "Marine Ecosystem VR/AR":
    display_tips()
    st.title("Marine Ecosystem VR/AR Experience")
    st.write("Explore the beauty of marine ecosystems through virtual reality.")
    html_code = """
    <iframe src="https://thehydro.us/xr-experiences" width="100%" height="600" frameborder="0"></iframe>
    """
    st.markdown(html_code, unsafe_allow_html=True)