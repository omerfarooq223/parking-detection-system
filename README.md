# Smart Parking Detection System

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00CFFD?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![Gradio](https://img.shields.io/badge/Gradio-UI-FF7C00?style=flat-square&logo=gradio&logoColor=white)](https://gradio.app/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-yellow?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/spaces/omerfarooq223/parking-detection-system)

AI-powered parking lot analysis that detects occupied and free spaces, ranks the best available spots, and answers parking-status queries via a built-in chatbot. Works on both images and video.

---

## Features

- **Parking detection** — identifies occupied and free spots using YOLOv8, with adjustable confidence and IoU thresholds
- **Smart recommendations** — ranks the top 5 empty spots using a multi-factor score: detection confidence, distance to entrance, cluster density (DBSCAN), spot size, local accessibility, and edge penalty
- **Zone analysis** — overlays a 3×3 grid showing per-zone availability
- **Entrance-aware ranking** — click to set a custom entrance point; distance is factored into scores
- **Aggressive detection mode** — for crowded or irregular layouts
- **Video processing** — frame-skip control to balance speed vs. temporal resolution
- **Parking assistant chatbot** — answers availability, occupancy, and model questions; uses Hugging Face LLM API if a token is provided, otherwise falls back to rule-based responses

---

## Requirements

- Python 3.10+
- `best.pt` — YOLOv8 model weights (required, place in project root)
- `best2.pt` — second model for ensemble mode (optional)
- `HF_TOKEN` — Hugging Face API token for LLM-powered chatbot (optional)

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add model weights to the project root
#    best.pt     → required
#    best2.pt    → optional (enables ensemble mode)

# 3. (Optional) Configure chatbot API
echo "HF_TOKEN=your_huggingface_token" > .env

# 4. Run
python app.py
# App starts at http://localhost:7860
```

---

## Usage

### Image Analysis

1. Upload a parking lot image
2. Optionally click the image to set an entrance point (default: bottom-center)
3. Adjust confidence, IoU, and detection mode as needed
4. Click **Analyze Image**

**Output:** annotated image with spot labels, occupancy metrics, zone overlay, and ranked recommendations.

### Video Analysis

1. Upload a video file
2. Set **Process Every N Frames** — lower values give more detail, higher values run faster
3. Click **Process Video**

**Output:** annotated video with per-frame detection and a processing summary.

### Chatbot

Analyze an image first, then ask questions about availability, occupancy, or model details. The chatbot uses the most recent analysis as context.

---

## How Spot Scoring Works

Each empty spot is scored out of 100 using a weighted combination of:

| Factor | Description |
|---|---|
| Detection confidence | Model certainty for that spot |
| Distance to entrance | Closer spots score higher |
| Cluster bonus | More empty neighbors → higher score (via DBSCAN) |
| Spot size | Larger spots ranked up |
| Local accessibility | Penalizes spots surrounded by occupied spaces |
| Edge penalty | Reduces score for low-quality edge detections |

---

## Tech Stack

Python · Ultralytics YOLOv8 · OpenCV · NumPy · SciPy · scikit-learn · Gradio · Hugging Face Hub

---

## Limitations

- Detection quality is sensitive to camera angle and image resolution
- Dense or irregular lots may need aggressive mode enabled
- Video processing can be slow on long files with low frame-skip values
- Spot scores are heuristic — not ground truth

---

## Planned Improvements

- Real-time CCTV stream support
- REST API endpoint for external consumers
- Multi-lot management dashboard
- Cross-frame tracking for stable spot IDs in video

---

## Screenshots

![Interface](screenshots/ui.png)
*Main interface with image upload, detection controls, and zone overlay*

![Detection Result](screenshots/result1.png)
*Annotated output showing occupied/free spots, zone analysis, and ranked recommendations*

---

## Author

**Muhammad Umar Farooq** — [GitHub](https://github.com/omerfarooq223)