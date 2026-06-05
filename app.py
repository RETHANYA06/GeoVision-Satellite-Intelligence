import streamlit as st
from PIL import Image

from src.predict import predict_image

st.set_page_config(
    page_title="GeoVision Satellite Intelligence",
    layout="wide"
)

st.title("🛰️ GeoVision Satellite Intelligence")

st.write(
    "AI-powered satellite image classification using deep learning."
)

uploaded_file = st.file_uploader(
    "Upload a satellite image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image")

    prediction, confidence = predict_image(image)

    st.success(
        f"Prediction: {prediction}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )