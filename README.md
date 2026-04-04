# 🚗 Smart Parking Detection System

AI-powered parking analysis for both images and videos, with smart space ranking and a parking assistant chatbot.

🔥 Live demo: https://huggingface.co/spaces/omerfarooq223/parking-detection-system

## 📌 What This Project Does

The app detects two classes in parking-lot scenes:
- occupied spots (cars)
- free spots (empty spaces)

After detection, it:
- assigns zone-based spot IDs
- calculates occupancy and availability metrics
- ranks the best empty spaces using a multi-factor scoring system
- visualizes recommendations, zones, and walking paths from a user-defined entrance

The interface is built with Gradio and includes three tabs:
- Image Analysis
- Video Analysis
- Parking Assistant Chatbot

## ✨ Core Features

- Image parking-space detection with adjustable confidence and IoU
- Optional aggressive detection mode for crowded/unsymmetric layouts
- Zone overlay (3x3 grid) with availability state per zone
- Top 5 recommended spots with score and distance estimate
- Entrance-point click support (for distance and ranking context)
- Video processing with frame skipping for speed/quality control
- Chatbot that answers parking-status and model/training questions
- Optional Hugging Face API integration for LLM-powered chatbot responses

## 🎯 How Recommendations Are Ranked

Each empty spot is scored out of 100 using:
- detection confidence
- distance to entrance
- cluster bonus (nearby empty spots via DBSCAN)
- relative spot size
- local accessibility (occupied spots nearby)
- edge penalty for low-quality edge positions

Top spots are shown both visually and in a detailed text panel.

## 🛠️ Tech Stack

- Python
- Ultralytics YOLO
- OpenCV
- NumPy, SciPy, scikit-learn
- Matplotlib
- Gradio
- python-dotenv
- huggingface-hub

## 📂 Project Structure

parking-detection-system/
- app.py: main app, detection pipeline, visualization, chatbot, video workflow
- requirements.txt: Python dependencies
- README.md: documentation
- samples/: sample media for testing
- screenshots/: UI and result images

## ✅ Requirements

- Python 3.10+ recommended
- model weights file named best.pt in project root (required)
- optional second model file best2.pt for ensemble mode
- optional HF_TOKEN environment variable for Hugging Face chatbot API

If best2.pt is missing, the app runs in single-model mode.

## ▶️ Setup

1. Install dependencies:
	pip install -r requirements.txt

2. Add model files to the repository root:
	- best.pt (required)
	- best2.pt (optional)

3. Optional chatbot API setup:
	create a .env file in project root with:
	HF_TOKEN=your_huggingface_token

4. Run:
	python app.py

The app launches on port 7860.

## 📖 Usage Guide

### 📷 Image Analysis Tab

1. Upload an image.
2. Click on the image to set entrance location (optional; default is bottom-center).
3. Configure:
	- Detection Confidence
	- IOU Threshold (NMS)
	- Aggressive Detection Mode
	- Show Zone Analysis
	- Show Top Recommendations
4. Click Analyze Image.

Outputs:
- annotated image with occupied/free spots
- detailed analysis report
- ranked recommendation list

### 🎥 Video Analysis Tab

1. Upload a video.
2. Configure detection parameters.
3. Set Process Every N Frames:
	- lower value: better temporal detail, slower
	- higher value: faster processing, less detail
4. Click Process Video.

Outputs:
- annotated video
- processing summary and optimization notes

### 💬 Chatbot Tab

1. Analyze an image first.
2. Ask availability, recommendation, occupancy, model, or training questions.

Behavior:
- if HF_TOKEN is configured and valid, responses use Hugging Face LLM API
- otherwise, chatbot uses built-in rule-based fallback responses

## 🧠 Model and Training Information Exposed In App

The app includes and can report:
- YOLOv8m model metadata
- training configuration (epochs, batch size, image size)
- dataset split counts
- reported precision/recall/mAP metrics
- model size and inference speed estimates

## ⚠️ Notes and Limitations

- Detection quality depends heavily on camera angle and image quality.
- Very dense or irregular parking scenes may require aggressive mode.
- Video analysis can be slow for long files or low frame-skip settings.
- Recommendations are heuristic scores, not guaranteed ground truth.

## 📷 Screenshots

Interface:
![UI](screenshots/ui.png)

Detection Example:
![Image Result](screenshots/result1.png)

## 🚀 Future Improvements

- real-time CCTV stream support
- explicit REST API endpoint for external consumers
- multi-lot management dashboard
- model versioning and evaluation report export
- optional tracking across video frames for more stable IDs

## 👤 Author

Muhammad Umar Farooq
