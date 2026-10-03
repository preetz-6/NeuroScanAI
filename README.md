---
title: NeuroScan AI - Brain Tumor Detection
emoji: 🧠
colorFrom: gray
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 🧠 NeuroScan AI

Brain tumor classification from MRI scans using a fine-tuned **ResNet-50** model with **Grad-CAM++** visual explainability.

Classifies scans into **4 categories**: Glioma · Meningioma · Pituitary Tumor · No Tumor

🔗 **Live Demo**: [NeuroScan AI on Hugging Face Spaces](https://huggingface.co/spaces/Preetm-666/NeuroScan-AI)

---

## Features

- **Drag-and-drop** MRI image upload (JPG, PNG, BMP, TIFF, WebP — max 10 MB)
- **Real-time inference** with confidence scores and per-class probability breakdown
- **Grad-CAM++ heatmaps** — multi-layer fusion (layer3 + layer4) for explainable predictions
- **Scan history** — tracks up to 20 predictions per session
- **Responsive dark UI** — works on desktop and mobile

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Model** | PyTorch, TorchVision, ResNet-50 (transfer learning) |
| **Backend** | Flask, Pillow, NumPy, Matplotlib |
| **Frontend** | HTML5, CSS3, Vanilla JS |
| **Deployment** | Docker, Hugging Face Spaces, Git LFS |

---

## Project Structure

```
├── app.py                  # Flask server & API routes
├── model.py                # ResNet-50 inference + Grad-CAM++ generation
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container config for deployment
├── model/
│   └── brain_tumor_resnet50.pth   # Trained weights (~90 MB, Git LFS)
├── templates/
│   └── index.html          # Main page template
├── static/
│   ├── style.css           # Dark theme styles
│   └── script.js           # Frontend logic
└── uploads/                # Temp upload dir (gitignored)
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Git with Git LFS

### Run Locally

```bash
git clone https://github.com/preetz-6/NeuroScanAI.git
cd NeuroScanAI

python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

pip install -r requirements.txt
python app.py
```

Open **http://localhost:7860**

### Run with Docker

```bash
docker build -t neuroscan-ai .
docker run -p 7860:7860 neuroscan-ai
```

---

## How It Works

1. Upload an MRI scan → image is resized to 224×224 and normalized
2. ResNet-50 forward pass → softmax probabilities for 4 classes
3. Grad-CAM++ generates a heatmap showing which brain regions influenced the prediction
4. Results displayed with confidence ring, probability bars, and heatmap overlay

---

## Dataset

Trained on the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (~7,023 images across 4 classes).

---

## License

This project is for educational and research purposes. Not intended for clinical diagnosis.
