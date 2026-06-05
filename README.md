# 🛰️ GeoVision Satellite Intelligence

AI-powered satellite image classification system using deep learning and computer vision.

---

## 📌 Project Overview

GeoVision Satellite Intelligence is a machine learning system that analyzes satellite images and classifies land cover types such as forests, rivers, residential areas, highways, and agricultural land.

It demonstrates the application of **Computer Vision + Remote Sensing + Deep Learning** for Earth observation tasks.

---

## 🚀 Features

- 🧠 CNN-based image classification model
- 🛰️ Satellite image analysis (EuroSAT dataset)
- 🌍 10 land-cover categories prediction
- 🖥️ Streamlit web interface
- 📊 Confidence score output
- ⚡ Real-time image inference

---

## 🧠 Tech Stack

- Python
- PyTorch
- Torchvision
- OpenCV
- Streamlit
- Scikit-learn

---

## 📊 Dataset

Used the EuroSAT dataset:

- AnnualCrop
- Forest
- HerbaceousVegetation
- Highway
- Industrial
- Pasture
- PermanentCrop
- Residential
- River
- SeaLake

---

## 🎯 Model Performance

- Model: Custom CNN
- Accuracy: **85.83%**
- Input: 64x64 RGB satellite images
- Loss Function: CrossEntropyLoss
- Optimizer: Adam

---

## 🖥️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
📷 Demo

Upload a satellite image and get predictions like:

Prediction: River
Confidence: 71.95%
📁 Project Structure
geovision-satellite-intelligence/
│
├── app.py
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│
├── data/
├── models/
└── README.md

🔬 Future Improvements
🔥 Grad-CAM explainability (highlight image regions)
🚀 Transfer learning (ResNet, EfficientNet)
📈 Accuracy improvement (>92%)
🌍 Geo-spatial mapping integration
🛰️ Multi-spectral satellite image support
👨‍💻 Author

Built by a Computer Science student exploring AI applications in Aerospace 
