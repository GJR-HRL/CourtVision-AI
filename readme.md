<div align="center">

# 🏀 CourtVision AI
### Turning Raw Basketball Footage into Structured Game Intelligence

<p align="center">
  <b>Full-Pipeline Basketball Video Understanding System</b><br>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.x-red?style=flat&logo=pytorch" />
  <img src="https://img.shields.io/badge/Python-green" />
  <img src="https://img.shields.io/badge/Ultralytics--YOLO-blue" />
  <img src="https://img.shields.io/badge/HuggingFace--Transformers-green" />
  <img src="https://img.shields.io/badge/SuperVision-orange" />
  <img src="https://img.shields.io/badge/OpenCV-blue" />
  <img src="https://img.shields.io/badge/NumPy-green" />
  <img src="https://img.shields.io/badge/Pillow-green" />
  <img src="https://img.shields.io/badge/Roboflow-blue" />
  <img src="https://img.shields.io/badge/Pandas-green" />
</p>
<br>

[Demo](#-demo) • 
[Features](#-features) • 
[Architecture](#architecture) • 
[How It Works](#how-it-works) • 
[Metrics](#metrics-generated) • 
[Usage](#usage) • 
[Training](#-training-custom-models)

</div>



# 🎬 Demo

<p align="center">

https://github.com/user-attachments/assets/6b4c26bd-a28b-4316-99de-ebfc986eade5


</p>

<p align="center">
  Full demo video available in <code>assets/basket_ball_analysis_final_compressed.mp4</code>
</p>



# 🧭 Overview

**CourtVision AI** transforms raw basketball footage into structured, analyzable game intelligence.

Instead of just detecting players, this system understands:

- 🏀 Who has possession  
- 🔁 How many passes occurred  
- ❌ When interceptions happened  
- 📈 Which team controlled the game  


# ✨ Features

### Vision Layer
- Player detection (YOLO v11)
- Ball detection (YOLOv5)
- Court keypoint detection (YOLOv8)

### 🧠 Intelligence Layer
- Multi-object tracking (persistent IDs)
- Zero-shot team classification (CLIP)
- Possession logic engine
- Pass detection
- Interception detection
- Real-time event logging

### 🎥 Output Layer
- Fully annotated output video
- Game metrics aggregation
- CPU & GPU inference support


# Architecture

CourtVision AI is built as a clean, modular pipeline:
```
Input Video
↓
Frame Extraction
↓
Object Detection (YOLO Models)
↓
Multi-Object Tracking
↓
Jersey Cropping
↓
Zero-Shot Team Classification (CLIP)
↓
Ball–Player Interaction Logic
↓
Event Detection (Pass / Interception)
↓
Metrics Aggregation
↓
Annotated Output Video
```


Each block is independently replaceable.

We can:
- Swap detectors  
- Change tracker  
- Replace classifier  
- Improve logic engine  

Without breaking the system.



# How It Works

## 1️⃣ Detection

Three independent detectors operate per frame:

| Component | Model |
|-----------|--------|
| Players | YOLO v11 |
| Ball | YOLOv5 |
| Court Keypoints | YOLOv8 |

All trained using Roboflow pipelines and exported as `.pt` weights.


## 2️⃣ Multi-Object Tracking

Tracking assigns **persistent IDs** to:

- Players
- Ball

This enables:
- Identity continuity across frames
- Accurate possession logic
- Reliable event tracking over time


## 3️⃣ Team Assignment (Zero-Shot CLIP)

Instead of fragile HSV color thresholds:

1. Player bounding box is cropped  
2. Jersey region extracted  
3. Embedded using CLIP  
4. Zero-shot classification performed  

Model used:
[patrickjohncyh/fashion-clip](https://huggingface.co/patrickjohncyh/fashion-clip)



Why:
- No hard-coded color rules  
- Generalizes across games  
- Works under lighting variations  
- Avoids manual tuning  


## 4️⃣ Possession Logic Engine

Possession is determined using spatial interaction rules:

- If ball bounding box overlaps player bounding box for consecutive frames → possession assigned  
- If ball transitions to opposing team → interception recorded  
- Frame-wise accumulation → total possession time  

This logic converts detection into structured game events.



## 5️⃣ Event Detection

### 🔁 Pass
- Ball moves from Player A → Player B  
- Both belong to same team  

###  Interception
- Ball transitions between opposing teams  

Events can be logged in real-time during inference.


# Metrics Generated

From frame-level logs:

- Total passes per team  
- Total interceptions  
- Possession time per team  
- Possession transitions  

All computed directly from tracked interactions.

No manual annotation required.

## 6️⃣ Caching
Implemented cahcing using .pkl files 
- Improves performance
- Reduce Redundant Compute
- Resume whenever needed 


# Usage

Run using the CLI:

```bash
python main.py --input path/to/video.mp4 --output annotated_output.mp4
```
Run using Docker:

Build the container if not built already:
```
docker build -t basketball-analysis .
```
Run the container, mounting your local input video folder:
```
docker run \
  -v $(pwd)/videos:/app/videos \
  -v $(pwd)/output_videos:/app/output_videos \
  basketball-analysis \
  python main.py videos/input_video.mp4 --output_video output_videos/output_result.avi
```

Supported Modes:

✅ CPU inference

✅ GPU inference (recommended)



📁 Project Structure
```
CourtVision-AI/
│
├── models/                              # Trained YOLO weights
├── training_notebooks/                  # Model training notebooks
├── assets/                              # Demo media
├── plotters/                            # Classes and function that overlay info on Frames 
├── ball_aquisition/                     # Logic for identifying which player is in possession of the ball
├── pass_interception_detection/         # Identifies passing events and interceptions.
├── court_keypoint_detector/             # Detects lines and keypoints on the court using the specified model.
├── speed_distance_calculator/           # Calculates speed and distance of each player.
├── top_view_converter/                  # Converts player position into Top View using Homography.
├── team_assigner/                       # Uses zero-shot classification (Hugging Face or similar) to assign players to teams based on jersey color
├── tracking/                            # All Tracking logic for Bounding Boxes
├── utils/                               # Helper utilities
└── main.py                              # CLI entry point
```

## 🏋️ Training Custom Models

Training notebooks available in:
```
training_notebooks/
```
Includes:
```
basketball_player_detection_training.ipynb

basketball_ball_training.ipynb

basketball_court_keypoint_training.ipynb
```
## Dataset Workflow

Prepare dataset using Roboflow

Train with Ultralytics YOLO

Export ```.pt``` weights

Place inside ```models/```

> Update configuration in main.py

## 🧩 Design Philosophy

- Replaceable components

- Clear separation of concerns

- Minimal heuristics

- Logic-driven event extraction

- Built for extensibility

## ⚠️ Limitations
Occlusion may affect possession accuracy

Very fast ball motion may reduce detection confidence

Similar jersey colors can confuse classifier

Dependent on camera angle

## 🔮 Future Extensions

- Shot detection & trajectory modeling

- Player heatmaps

- Real-time streaming pipeline

- OpenVINO optimization

- Web dashboard

- REST API layer

- Cloud deployment

## 👤 Built By

An 18-year-old engineer building systems using first principles.


## ⭐ If You Found This Useful

Star the repo.

Fork it.

Break it.

Improve it.

And build something bigger.
<div align="center">
🏀 Built as a Complete Video Intelligence System
Not Just a Detection Demo
</div>