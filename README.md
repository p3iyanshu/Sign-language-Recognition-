# 🤟 SignSense AI — Real-Time ASL Fingerspelling Recognition Studio

[![Python](https://img.shields.io/badge/Python-3.10%2B-14B8A6?style=flat-square)](https://www.python.org/)
X![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-FF6F00?style=flat-square)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38%2B-FF4B4B?style=flat-square)](https://streamlit.io/)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-99.85%25-10B981?style=flat-square)](#key-capabilities)

**SignSense AI** is a real-time American Sign Language (ASL) fingerspelling translation platform supporting all **26 English alphabet characters (`A–Z`)**. Built to bridge the domain gap between clean benchmark datasets and cluttered real-world webcam feeds, it combines a **Regularized Deep CNN** with **MediaPipe 21-point hand landmark isolation**, **GrabCut studio background normalization**, **CLAHE contrast equalization**, and **Grad-CAM explainability**.

---

## Key Capabilities

* **Full 26-Class (`A—Z`) Vocabulary:** Multi-dataset fusion pipeline overcoming standard 24-class Sign Language MNIST limitations (`J` and `Z` support included) with **99.85% test accuracy** and **0.9984 Macro F1**.
* **Domain-Adaptive Computer Vision Pipeline (`src/inference.py`):**
  1. **21-Point Hand Landmarker (MediaPipe Tasks API):** Locates exact hand bounding boxes and suppresses background faces via Haar cascades.
  2. **GrabCut Studio Isolation:** Segments foreground hand contours onto a standardized studio background (`RGB 215`).
  3. **CLAHE Normalization:** Equalizes uneven indoor lighting before `28×28` grayscale tensor inference.
* **Commercial SaaS Studio (`app/streamlit_app.py`):**
  * **Live Vision Studio:** Real-time webcam capture and image upload with sub-millisecond CNN inference latency and Top-3 probability distribution.
  * **Sentence & Voice Studio:** Interactive word builder with browser-native Web Speech API text-to-speech synthesis.
  * **Explainable AI (Grad-CAM) & Diagnostics:** Visualizes finger-articulation saliency maps alongside convergence curves and confusion matrices.

---

## Quick Start (Local)

```bash
git clone https://github.com/p3iyanshu/Sign-language-Recognition-.git
cd Sign-language-Recognition-
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

---

## Academic Credits

* **Course:** CAI2500 — Machine Learning Mini Project (B.Tech CSE – Data Science, Semester V)
* **Institution:** Presidency University, Bengaluru
* **Team (Group 16):** Priyanshu Choudhary (`20241CSD0050`), Jetty Dheeraj Reddy (`20241CSD0052`), Hrithik C K (`20241CSD0054`)
* **Supervisor:** Ms. Radha Kanakaraj, Assistant Professor
