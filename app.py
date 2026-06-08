import streamlit as st
from PIL import Image

from src.predict import predict_image



st.set_page_config(
    page_title="GeoVision Satellite Intelligence",
    page_icon="🛰️",
    layout="wide"
)



st.sidebar.title("🛰️ GeoVision")

st.sidebar.markdown("### Model Information")

st.sidebar.info("""
**Dataset:** EuroSAT

**Classes:** 10

**Model:** CNN

**Accuracy:** 85.83%
""")

st.sidebar.markdown("---")

st.sidebar.markdown("""
### Applications

- Earth Observation
- Environmental Monitoring
- Agriculture Analysis
- Urban Planning
- Remote Sensing
""")



st.title("🛰️ GeoVision Satellite Intelligence")

st.caption(
    "AI-Powered Earth Observation and Land Cover Classification"
)

st.markdown("---")



with st.expander("ℹ️ About This Project"):
    st.write("""
    GeoVision Satellite Intelligence uses Deep Learning and Computer Vision
    to classify satellite imagery into different land-cover categories.

    The model is trained on the EuroSAT dataset and can identify:

    - Forest
    - River
    - Residential
    - Highway
    - Industrial
    - Agricultural Areas
    - And more...

    This project demonstrates practical applications of AI in
    Earth Observation and Remote Sensing.
    """)



uploaded_file = st.file_uploader(
    "📤 Upload a Satellite Image",
    type=["jpg", "jpeg", "png"]
)



if uploaded_file:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:

        st.subheader("🖼 Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.subheader("🧠 Prediction Results")

        prediction, confidence, top3 = predict_image(image)

        st.success(
            f"Prediction: {prediction}"
        )

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )
        st.markdown("### 🏆 Top 3 Predictions")

        for rank, (cls, score) in enumerate(top3, start=1):
            st.write(
                f"{rank}. {cls} — {score:.2f}%"
    
        )

        st.progress(min(int(confidence), 100))

        st.markdown("### 📊 Model Output")

        st.write(
            f"The model predicts that this image belongs to the **{prediction}** class."
        )

        st.write(
            f"Prediction confidence: **{confidence:.2f}%**"
        )

    st.markdown("---")

    st.subheader("📌 Analysis Summary")

    st.info(
        f"This satellite image has been classified as **{prediction}** "
        f"with a confidence score of **{confidence:.2f}%**."
    )


st.markdown("---")

st.caption(
    "GeoVision Satellite Intelligence | Deep Learning for Earth Observation"
)