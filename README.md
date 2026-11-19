# 🤟 American Sign Language (ASL) Recognition using Deep Learning

[![CI](https://github.com/p3iyanshu/Sign-language-Recognition-/actions/workflows/ci.yml/badge.svg)](https://github.com/p3iyanshu/Sign-language-Recognition-/actions)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)

An end-to-end, production-structured **Computer Vision & Deep Learning** system for static American Sign Language (ASL) alphabet recognition (`A–Y`, excluding dynamic signs `J` and `Z`), complete with **Grad-CAM visual explainability**, **CLAHE real-world image normalization**, **interactive Streamlit UI**, and **Docker** containerization.

---

## 🏗️ System Architecture & Engineering Highlights

1. **Domain-Aware Preprocessing (`src/data_loader.py`):**
   - Maps the 24 non-contiguous Sign Language MNIST labels (`0–8, 10–24`) cleanly to contiguous indices (`0–23`).
   - Applies **rotation, zoom, and spatial translation augmentations** while intentionally disabling horizontal flips to preserve ASL handedness semantics.
2. **Regularized CNN Architecture (`src/model.py`):**
   - 3-stage Convolutional feature extractor (`32 -> 64 -> 128` filters) with `BatchNormalization`, `Dropout`, and `GlobalAveragePooling2D` to prevent parameter bloat and overfitting on `28x28` grayscale inputs.
3. **Explainable AI (`src/evaluate.py`):**
   - Computes **Grad-CAM heatmaps** over the final convolutional activation layer (`last_conv_activation`) to verify that predictions rely on finger articulation rather than background artifacts.
4. **Interactive Deployment (`app/streamlit_app.py`):**
   - Supports live webcam snapshots, external image uploads (with **CLAHE** contrast equalization), random test-set benchmarking, and an interactive **Word Builder**.

---

## 📁 Project Structure

```text
Sign-language-Recognition-/
├── .github/workflows/ci.yml    # GitHub Actions CI pipeline
├── configs/config.yaml         # Centralized hyperparameters & paths
├── data/raw/                   # Sign Language MNIST CSV files (gitignored)
├── models/best_asl_cnn.keras   # Trained model artifact
├── assets/                     # Auto-generated training curves, confusion matrix, Grad-CAM
├── src/
│   ├── data_loader.py          # CSV parsing, stratified split & augmentation
│   ├── model.py                # Regularized CNN architecture
│   ├── train.py                # Training pipeline with LR scheduling & checkpointing
│   ├── evaluate.py             # Test metrics, latency benchmark & Grad-CAM generator
│   └── inference.py            # CLAHE preprocessing & prediction wrapper
├── app/streamlit_app.py        # Interactive Streamlit web application
├── Dockerfile                  # Production container specification
└── requirements.txt