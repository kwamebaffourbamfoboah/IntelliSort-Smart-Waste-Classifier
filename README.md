♻️ IntelliSort – Smart Waste Classifier

IntelliSort is an **AI-powered smart waste sorting system** that uses **computer vision and embedded systems** to automatically classify and sort waste into Plastic, Paper, and Glass in real time.

Built as a final-year Computer Science project, IntelliSort combines **deep learning, edge computing, and human-centered design** to promote sustainability and support smarter waste management.

---

## 🚀 Features
- **CNN-based image classification** (98% accuracy)
- **Real-time inference** via webcam
- **Automated sorting** using Raspberry Pi + Arduino
- **Feedback-driven retraining pipeline**
- **Streamlit web interface**
- Context-aware **sustainability tips**

---

## 🧠 AI/ML Architecture
- **Model:** Convolutional Neural Network (CNN)
- **Framework:** TensorFlow / Keras
- **Training Data:** 6,000+ images (Kaggle dataset)
- **Input:** 224×224 RGB images
- **Classes:** Plastic, Paper, Glass
- **Accuracy:** 98%

---

## 🖥️ System Architecture
1. Webcam captures image  
2. CNN predicts waste class  
3. Optional user feedback captured for retraining  

---

## 🛠️ Tech Stack
- **Languages:** Python
- **AI/ML:** TensorFlow, Keras, OpenCV, NumPy
- **Web App:** Streamlit
- **Deployment:** Edge-based real-time inference

---



---

## 📦 Installation
```bash
git clone https://github.com/kwamebaffourbamfoboah/IntelliSort-Smart-Waste-Classifier.git
cd IntelliSort-Smart-Waste-Classifier
pip install -r requirements.txt
streamlit run app/main.py
