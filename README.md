<div align="center">

<br/>

# 🏀 Basketball Intelligence System

### AI-Powered Game Analysis — From Raw Video to Full Tactical Intelligence

<br/>

[!\[Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge\&logo=python\&logoColor=white)](https://python.org)
[!\[YOLOv5](https://img.shields.io/badge/YOLOv5-Fine--Tuned-FF6B35?style=for-the-badge)](https://github.com/ultralytics/yolov5)
[!\[YOLOv8](https://img.shields.io/badge/YOLOv8-Pose--Model-00B4D8?style=for-the-badge)](https://github.com/ultralytics/ultralytics)
[!\[OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge\&logo=opencv\&logoColor=white)](https://opencv.org)
[!\[HuggingFace](https://img.shields.io/badge/FashionCLIP-Zero--Shot-FFD21E?style=for-the-badge\&logo=huggingface\&logoColor=black)](https://huggingface.co)
[!\[Supervision](https://img.shields.io/badge/ByteTrack-Tracking-E63946?style=for-the-badge)](https://supervision.roboflow.com)
[!\[License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
\[!\[Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen?style=for-the-badge)]()

<br/>

> A professional-grade, multi-model computer vision pipeline that extracts complete tactical intelligence from basketball game footage — no manual tracking, no per-game calibration required.

<br/>

</div>

\---

## 📖 Overview

The Basketball Intelligence System takes raw `.mp4` game footage through a sequential chain of purpose-built AI models. Starting from individual video frames, the system progressively enriches its understanding of the scene — detecting players and the ball, identifying teams by jersey color, tracking possession, recognizing game events, and finally converting the distorted camera perspective into a precise top-down tactical map with real-world coordinates.

The result is a fully annotated output video with rich per-frame analytics covering player kinematics, team possession dynamics, and game event detection — all extracted automatically.

<br/>

## 🎯 Extracted Analytics

|Metric|Method|Output|
|-|-|-|
|**Player Detection \& Tracking**|Fine-tuned YOLOv5 + ByteTrack|Bounding boxes with persistent IDs across all frames|
|**Ball Detection**|Dedicated fine-tuned YOLOv5 + Interpolation|Ball position every frame, interpolated when occluded|
|**Team Assignment**|FashionCLIP zero-shot classification|Each player labeled Team 1 or Team 2|
|**Ball Possession**|Containment overlap + proximity distance|Controlling player, frame-by-frame|
|**Ball Acquisition %**|Running possession time ratio|Cumulative possession percentage per team|
|**Passes**|Intra-team possession change|Count of successful team ball transfers|
|**Interceptions**|Cross-team possession change|Count of ball steals|
|**Tactical Court View**|Court keypoint detection + Homography|Top-down 2D map with player positions|
|**Player Speed**|Sliding-window kinematics on real coordinates|Speed in km/h per player|
|**Distance Covered**|Cumulative path length in meters|Total meters per player|

<br/>

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   BASKETBALL INTELLIGENCE SYSTEM                 │
└─────────────────────────────────────────────────────────────────┘

   Input Video (.mp4)
         │
         ▼  \[OpenCV — Frame Extraction]
   ┌─────────────────────────────────────────────────┐
   │            DETECTION \& TRACKING LAYER            │
   │  ┌─────────────────┐   ┌──────────────────────┐  │
   │  │  Player Tracker  │   │    Ball Tracker       │  │
   │  │  YOLOv5l6 +      │   │    YOLOv5l6 +         │  │
   │  │  ByteTrack       │   │    Interpolation      │  │
   │  └────────┬─────────┘   └────────────┬─────────┘  │
   └───────────┼────────────────────────────┼───────────┘
               │                            │
               ▼                            ▼
   ┌─────────────────────────────────────────────────┐
   │               ANALYTICS LAYER                    │
   │  ┌──────────────┐  ┌──────────────┐  ┌────────┐  │
   │  │Team Assigner  │  │Ball Possess. │  │Court   │  │
   │  │FashionCLIP    │  │Detector      │  │Keypoint│  │
   │  │Zero-shot      │  │Overlap+Dist  │  │YOLOv8x │  │
   │  └──────┬────────┘  └──────┬───────┘  └───┬────┘  │
   │         └──────────────────┼──────────────┘        │
   │                            ▼                        │
   │              ┌───────────────────────┐             │
   │              │ Pass \& Interception   │             │
   │              │ Detector              │             │
   │              └───────────┬───────────┘             │
   └──────────────────────────┼─────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────┐
   │           SPATIAL TRANSFORM LAYER                │
   │  ┌────────────────┐    ┌──────────────────────┐  │
   │  │Keypoint         │    │Tactical View          │  │
   │  │Validator        │───►│Converter              │  │
   │  │Proportionality  │    │Homography/OpenCV      │  │
   │  └────────────────┘    └──────────┬────────────┘  │
   │                                   │               │
   │  ┌────────────────────────────────┘               │
   │  ▼                                               │
   │  Speed \& Distance Calculator                     │
   │  (Pixel → Meter, Sliding-window kinematics)      │
   └──────────────────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────┐
   │              VISUALIZATION LAYER                 │
   │  7 Independent Drawer Modules                   │
   │  \[Ellipses] \[Ball] \[Possession%] \[Events]       │
   │  \[Keypoints] \[Tactical Map] \[Speed+Distance]    │
   └──────────────────────────────────────────────────┘
                              │
                              ▼
         Annotated Output Video (.avi)
```

<br/>

## 🤖 AI Models

### Three Purpose-Built Models

|Model|Base Architecture|Training|Purpose|
|-|-|-|-|
|**Player Detector**|YOLOv5l6|100 epochs, 165 court images|On-court player detection, spectator suppression|
|**Ball Detector**|YOLOv5l6|250 epochs, same dataset|Dedicated ball detection with higher recall|
|**Court Keypoint Detector**|YOLOv8x-pose|500 epochs, 56 court images|18 court landmark keypoints per frame|

### Why Fine-Tune?

Vanilla YOLOv8 (pretrained on COCO) fails on basketball footage in two critical ways:

1. **Detects spectators as players** — every person in the crowd registers as a detection, corrupting all downstream analytics
2. **Misses the basketball** — detected in roughly 40% of frames, frequently misclassified as "sports ball glove" or missed entirely

Fine-tuning on court-specific labeled data eliminates both problems by teaching the model the visual context of on-court activity.

### Why Two Separate Detection Models?

The ball (small, fast, partially occluded) and players (large, slow, always visible) have fundamentally different detection challenges. Dedicated models allow independent epoch tuning (100 vs. 250) and ball-focused training objectives. The cost — maintaining two model files — is easily justified by the accuracy gain.

<br/>

## 📁 Project Structure

```
basketball-analysis/
│
├── 📄 main.py                              ← Orchestration entry point
│
├── 📁 configs/                             ← Centralized configuration
│   └── configs.py                          ← All paths, thresholds, constants
│
├── 📁 models/                              ← Fine-tuned model weights
│   ├── player\_detector.pt
│   ├── ball\_detector.pt
│   └── court\_keypoint\_detector.pt
│
├── 📁 training\_notebooks/                  ← Reproducible Colab training
│   ├── player\_detection\_training.ipynb
│   ├── ball\_detection\_training.ipynb
│   └── court\_keypoint\_detection\_training.ipynb
│
├── 📁 trackers/                            ← Detection + multi-object tracking
│   ├── player\_tracker.py
│   └── ball\_tracker.py
│
├── 📁 team\_assigner/                       ← Zero-shot jersey classification
│   └── team\_assigner.py
│
├── 📁 ball\_acquisition/                    ← Possession detection engine
│   └── ball\_acquisition\_detector.py
│
├── 📁 pass\_interception/                   ← Game event detection
│   └── pass\_interception\_detector.py
│
├── 📁 court\_keypoint\_detector/             ← Court landmark detection
│   ├── court\_keypoint\_detector.py
│   └── homography.py
│
├── 📁 tactical\_view\_converter/             ← Camera → court coordinate mapping
│   └── tactical\_view\_converter.py
│
├── 📁 speed\_distance\_calculator/           ← Player kinematics
│   └── speed\_distance\_calculator.py
│
├── 📁 drawers/                             ← Visualization layer (7 modules)
│   ├── player\_tracks\_drawer.py
│   ├── ball\_tracks\_drawer.py
│   ├── team\_ball\_control\_drawer.py
│   ├── pass\_interception\_drawer.py
│   ├── court\_keypoints\_drawer.py
│   ├── tactical\_view\_drawer.py
│   └── speed\_distance\_drawer.py
│
├── 📁 utils/                               ← Shared utilities
│   ├── video\_utils.py
│   ├── bbox\_utils.py
│   └── stub\_utils.py
│
├── 📁 stubs/                               ← Cached inference outputs (auto-generated)
├── 📁 input\_videos/                        ← Input footage
├── 📁 output\_videos/                       ← Annotated output
└── 📄 requirements.txt
```

<br/>

## 🛠️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR\_USERNAME/basketball-analysis.git
cd basketball-analysis

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\\Scripts\\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place model weights in models/
#    (download links in releases)
```

**`requirements.txt`**

```
ultralytics>=8.0.0
supervision>=0.16.0
opencv-python>=4.8.0
transformers>=4.35.0
torch>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
Pillow>=10.0.0
roboflow
```

<br/>

## 🚀 Usage

```bash
# Basic run
python main.py input\_videos/video1.mp4

# Full options
python main.py input\_videos/video1.mp4 \\
  --stub\_path stubs/ \\
  --output\_path output\_videos/annotated\_game.avi
```

|Argument|Required|Default|Description|
|-|-|-|-|
|`input\_video`|Yes|—|Path to input `.mp4` file|
|`--stub\_path`|No|`stubs/`|Checkpoint directory|
|`--output\_path`|No|`output\_videos/output.avi`|Output video path|

**Stub system:** First run processes all frames (slow). Subsequent runs load from `stubs/\*.pkl` (seconds). Delete stubs to force re-inference.

<br/>

## ⚙️ Configuration

**Adapting to different jersey colors** — no retraining needed:

```python
# team\_assigner/team\_assigner.py
self.team\_one\_class\_name = "white shirt"
self.team\_two\_class\_name = "dark blue shirt"
```

**Possession sensitivity:**

```python
# ball\_acquisition/ball\_acquisition\_detector.py
self.possession\_threshold   = 50    # Max pixel distance for proximity
self.min\_frames             = 13    # Consecutive frames to confirm possession
self.containment\_threshold  = 0.80  # Required bounding box overlap ratio
```

<br/>

## 🧠 Training Your Own Models

```python
# Player Detector — Google Colab (T4 GPU, \~2 hrs)
!yolo task=detect mode=train model=yolov5l6u.pt \\
      data=basketball\_dataset/data.yaml \\
      epochs=100 imgsz=640 batch=8

# Ball Detector — same, epochs=250
# Court Keypoint Detector
!yolo task=pose mode=train model=yolov8x-pose.pt \\
      data=court\_keypoint\_dataset/data.yaml \\
      epochs=500 imgsz=640 batch=16
```

Full reproducible notebooks in `training\_notebooks/`.

<br/>

## 🔑 Design Principles

|Principle|Implementation|
|-|-|
|**Modularity**|Every module has one responsibility and a clean interface — swap any module without touching others|
|**Checkpointing**|All inference outputs persisted via stub system — development decoupled from model runtime|
|**Graceful degradation**|Missing data (undetected ball, occluded player) propagates as empty collections, never crashes|
|**Configuration centralization**|Zero hardcoded values in module logic — all parameters in `configs/`|

<br/>

## ⚠️ Known Limitations

* **2D depth ambiguity:** Ball airborne above player causes overlapping bounding boxes and false possession signals. Mitigated by 13-frame confirmation window.
* **Jersey color similarity:** Zero-shot team assignment degrades with visually similar jerseys.
* **Single camera:** Multi-camera setups require camera calibration and fusion logic.
* **Small training datasets:** 165 player images, 56 keypoint images. More data improves cross-court robustness.

<br/>

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

\---

<div align="center">

*Object Detection · Multi-Object Tracking · Zero-Shot Classification
Perspective Transformation · Sliding-Window Kinematics*

**End-to-end computer vision engineering — from raw video to full tactical intelligence.**

</div>

