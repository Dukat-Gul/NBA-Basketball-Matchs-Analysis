# SOFTWARE ENGINEERING REPORT
## Basketball Intelligence System

---

```
Document Title    : Software Engineering Technical Report
System Name       : Basketball Intelligence System
Version           : 1.0.0
Document Type     : Technical Design & Architecture Report
Classification    : Engineering Reference
```

---

## Abstract

This report presents the complete software engineering analysis of the Basketball Intelligence System — an end-to-end computer vision pipeline that extracts professional-grade tactical analytics from raw basketball game footage. The system chains three purpose-built fine-tuned AI models through a sequential processing architecture to deliver player tracking, team identification, ball possession analytics, game event detection, perspective transformation, and player kinematics in a single automated pipeline.

This document covers the system architecture, module-level design, data flow specifications, design patterns applied, algorithmic analysis, performance characteristics, and software quality considerations. It serves as the primary engineering reference for understanding, maintaining, and extending the system.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Design](#2-architecture-design)
3. [Module Specifications](#3-module-specifications)
4. [Data Flow & Interface Contracts](#4-data-flow--interface-contracts)
5. [AI Model Architecture](#5-ai-model-architecture)
6. [Design Patterns](#6-design-patterns)
7. [Algorithm Analysis](#7-algorithm-analysis)
8. [Performance Analysis](#8-performance-analysis)
9. [Error Handling Strategy](#9-error-handling-strategy)
10. [Configuration Management](#10-configuration-management)
11. [Testing Strategy](#11-testing-strategy)
12. [Dependency Map](#12-dependency-map)
13. [Extension Points](#13-extension-points)

---

## 1. System Overview

### 1.1 Purpose

The Basketball Intelligence System automates the extraction of game analytics from raw basketball video. Traditional sports analytics require trained analysts watching footage manually — a time-intensive, expensive, and non-scalable process. This system replaces manual analysis with a chain of computer vision AI models that operate automatically on any input video.

### 1.2 Scope

The system processes single-camera basketball game footage and produces:
- A fully annotated output video with visual overlays
- Per-frame structured analytics (player positions, possession, events)
- Cumulative statistics (team possession %, pass/interception counts)

The system is out-of-scope for: live/streaming video, multi-camera fusion, audio analysis, and game clock tracking.

### 1.3 Technology Stack

```
Layer               Technology              Purpose
─────────────────────────────────────────────────────────────────
AI Framework        Ultralytics YOLO        Object detection + pose
                    HuggingFace Transformers Zero-shot classification
                    PyTorch                 Neural network backend
─────────────────────────────────────────────────────────────────
Computer Vision     OpenCV (cv2)            Video I/O, homography, drawing
                    Supervision             ByteTrack, YOLO ↔ detection bridge
─────────────────────────────────────────────────────────────────
Data Processing     NumPy                   Array math, keypoint handling
                    Pandas                  Ball position interpolation
─────────────────────────────────────────────────────────────────
Persistence         Pickle                  Stub (checkpoint) serialization
─────────────────────────────────────────────────────────────────
Interface           argparse                CLI argument parsing
                    Python 3.10+            Runtime
─────────────────────────────────────────────────────────────────
Training Platform   Google Colab            GPU-backed model training
                    Roboflow                Dataset management
─────────────────────────────────────────────────────────────────
```

---

## 2. Architecture Design

### 2.1 High-Level Architecture

The system follows a **Sequential Pipeline Architecture** — a linear chain of processing stages where each stage consumes the output of the previous stage and produces enriched output for the next.

```
┌─────────────────────────────────────────────────────────────────────┐
│                   SEQUENTIAL PIPELINE ARCHITECTURE                   │
│                                                                       │
│  Stage 1        Stage 2         Stage 3          Stage 4             │
│  ─────────      ─────────       ──────────        ─────────          │
│  VIDEO I/O  ──► DETECTION   ──► ANALYTICS    ──► SPATIAL            │
│                 & TRACKING      LAYER             TRANSFORM          │
│                                                       │              │
│                                               Stage 5  ▼             │
│                                               ─────────────          │
│                                               KINEMATICS             │
│                                                   │                  │
│                                           Stage 6  ▼                 │
│                                           ─────────────              │
│                                           VISUALIZATION              │
│                                               │                      │
│                                       Stage 7  ▼                     │
│                                       ─────────────                  │
│                                       VIDEO OUTPUT                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Layer Architecture Detail

```
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1: VIDEO I/O                                                  │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  utils/video_utils.py                                          │  │
│  │  OpenCV VideoCapture → List[np.ndarray] (BGR frames)          │  │
│  └────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                         List[np.ndarray] (frames)
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 2: DETECTION & TRACKING                                       │
│  ┌─────────────────────────┐    ┌────────────────────────────────┐  │
│  │  trackers/              │    │  court_keypoint_detector/       │  │
│  │  player_tracker.py      │    │  court_keypoint_detector.py     │  │
│  │  ball_tracker.py        │    │                                 │  │
│  │                         │    │  YOLOv8x-pose → 18 keypoints   │  │
│  │  YOLOv5 + ByteTrack     │    │  per frame                     │  │
│  │  YOLOv5 + Interpolation │    │                                 │  │
│  └─────────────────────────┘    └────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
              player_tracks, ball_tracks, court_keypoints
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 3: ANALYTICS                                                  │
│  ┌──────────────────┐  ┌─────────────────┐  ┌──────────────────┐   │
│  │ team_assigner/   │  │ ball_acquisition/│  │ pass_interception/│  │
│  │                  │  │                  │  │                   │  │
│  │ FashionCLIP      │  │ Containment +    │  │ Possession diff   │  │
│  │ Zero-shot        │  │ Proximity +      │  │ + Team check      │  │
│  │ jersey classify  │  │ 13-frame window  │  │                   │  │
│  └──────────────────┘  └─────────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
       player_assignment, ball_acquisition, passes, interceptions
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 4: SPATIAL TRANSFORM                                          │
│  ┌──────────────────────────┐    ┌──────────────────────────────┐   │
│  │ tactical_view_converter/ │    │ court_keypoint_detector/      │   │
│  │                          │    │ homography.py                 │   │
│  │ Keypoint validation      │───►│                               │   │
│  │ Proportionality check    │    │ cv2.findHomography            │   │
│  │                          │    │ cv2.perspectiveTransform      │   │
│  └──────────────────────────┘    └──────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    tactical_player_positions (meter coords)
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 5: KINEMATICS                                                 │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  speed_distance_calculator/speed_distance_calculator.py        │  │
│  │  Pixel→Meter conversion + Sliding-window speed computation     │  │
│  └────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
               player_distances_per_frame, player_speeds_per_frame
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 6: VISUALIZATION (7 Drawer Modules)                           │
│  ┌────────────┐ ┌─────────────┐ ┌──────────────┐ ┌──────────────┐  │
│  │  Player    │ │    Ball     │ │  Team Ball   │ │   Pass /     │  │
│  │  Tracks    │ │   Tracks   │ │   Control    │ │ Interception │  │
│  │  Drawer    │ │   Drawer   │ │   Drawer     │ │   Drawer     │  │
│  └────────────┘ └─────────────┘ └──────────────┘ └──────────────┘  │
│  ┌────────────┐ ┌─────────────┐ ┌──────────────┐                   │
│  │   Court    │ │  Tactical  │ │   Speed &    │                   │
│  │ Keypoints  │ │    View    │ │   Distance   │                   │
│  │   Drawer   │ │   Drawer   │ │    Drawer    │                   │
│  └────────────┘ └─────────────┘ └──────────────┘                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                         Annotated output frames
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 7: VIDEO OUTPUT                                               │
│  utils/video_utils.py → OpenCV VideoWriter → .avi file              │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3 Stub (Checkpoint) System Architecture

The stub system is a cross-cutting concern that wraps the two most computationally expensive stages.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         STUB SYSTEM FLOW                             │
│                                                                       │
│  get_object_tracks(frames, read_from_stub=True, stub_path="...")     │
│              │                                                        │
│              ▼                                                        │
│   ┌──────────────────────┐                                           │
│   │  read_stub(path)     │                                           │
│   │  stub exists?        │──── YES ──► Load pickle ──► Return data  │
│   │  len(stub)==len(frames)?                                         │
│   └──────────┬───────────┘                                           │
│              │ NO / Invalid                                           │
│              ▼                                                        │
│   ┌──────────────────────┐                                           │
│   │  Run YOLO inference  │ (slow: minutes)                           │
│   │  Run ByteTrack       │                                           │
│   └──────────┬───────────┘                                           │
│              │                                                        │
│              ▼                                                        │
│   ┌──────────────────────┐                                           │
│   │  save_stub(path, data)│ → stubs/player_track_stubs.pkl          │
│   └──────────┬───────────┘                                           │
│              │                                                        │
│              ▼                                                        │
│           Return data                                                 │
│                                                                       │
│  Modules with stub support:                                           │
│  ● PlayerTracker.get_object_tracks()                                 │
│  ● BallTracker.get_object_tracks()                                   │
│  ● CourtKeypointDetector.get_court_keypoints()                      │
│  ● TeamAssigner.get_player_teams_across_frames()                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Module Specifications

### 3.1 Module Dependency Graph

```
main.py
  ├── utils/video_utils.py
  ├── configs/configs.py
  │
  ├── trackers/player_tracker.py
  │     ├── ultralytics.YOLO
  │     ├── supervision.ByteTrack
  │     └── utils/stub_utils.py
  │
  ├── trackers/ball_tracker.py
  │     ├── ultralytics.YOLO
  │     ├── supervision.Detections
  │     ├── pandas (interpolation)
  │     └── utils/stub_utils.py
  │
  ├── team_assigner/team_assigner.py
  │     ├── transformers.CLIPModel (FashionCLIP)
  │     ├── transformers.CLIPProcessor
  │     ├── PIL.Image
  │     └── utils/stub_utils.py
  │
  ├── ball_acquisition/ball_acquisition_detector.py
  │     └── utils/bbox_utils.py
  │
  ├── pass_interception/pass_interception_detector.py
  │     └── (no external dependencies)
  │
  ├── court_keypoint_detector/court_keypoint_detector.py
  │     ├── ultralytics.YOLO
  │     └── utils/stub_utils.py
  │
  ├── court_keypoint_detector/homography.py
  │     ├── cv2.findHomography
  │     ├── cv2.perspectiveTransform
  │     └── numpy
  │
  ├── tactical_view_converter/tactical_view_converter.py
  │     ├── court_keypoint_detector/homography.py
  │     ├── utils/bbox_utils.py
  │     └── copy.deepcopy
  │
  ├── speed_distance_calculator/speed_distance_calculator.py
  │     └── utils/bbox_utils.py
  │
  └── drawers/*.py
        └── cv2, numpy, supervision
```

### 3.2 Module Interface Summary

```
Module                          Primary Interface
─────────────────────────────────────────────────────────────────────────
PlayerTracker                   get_object_tracks(frames, ...) → List[Dict]
BallTracker                     get_object_tracks(frames, ...) → List[Dict]
                                interpolate_ball_positions(tracks) → List[Dict]
                                remove_wrong_detections(tracks) → List[Dict]
TeamAssigner                    get_player_teams_across_frames(frames, tracks, ...) → List[Dict]
BallAcquisitionDetector         detect_ball_possession(player_tracks, ball_tracks) → List[int]
PassInterceptionDetector        detect_passes(ball_acquisition, player_assignment) → List[int]
                                detect_interceptions(ball_acquisition, player_assignment) → List[int]
CourtKeypointDetector           get_court_keypoints(frames, ...) → List[Detection]
TacticalViewConverter           validate_keypoints(keypoint_list) → List[Detection]
                                transform_players_to_tactical_view(keypoints, tracks) → List[Dict]
SpeedDistanceCalculator         calculate_distance(tactical_positions) → List[Dict]
                                calculate_speed(distances, fps) → List[Dict]
[All Drawers]                   draw(video_frames, data, ...) → List[np.ndarray]
─────────────────────────────────────────────────────────────────────────
```

---

## 4. Data Flow & Interface Contracts

### 4.1 Core Data Structures

All modules communicate through a small set of well-defined data structures.

#### Player Tracks (List of Dicts)

```python
player_tracks: List[Dict[int, Dict[str, Any]]]

# Structure:
[
    # Frame 0
    {
        4:  {"bbox": [x1, y1, x2, y2]},  # Player ID 4
        7:  {"bbox": [x1, y1, x2, y2]},  # Player ID 7
        12: {"bbox": [x1, y1, x2, y2]},  # Player ID 12
    },
    # Frame 1
    {
        4:  {"bbox": [x1, y1, x2, y2]},
        7:  {"bbox": [x1, y1, x2, y2]},
    },
    # ... N frames total
]

# Constraints:
# - len(player_tracks) == len(video_frames)
# - Player IDs are persistent across frames (ByteTrack guarantee)
# - A player absent from a frame is simply not a key in that frame's dict
# - bbox format: [x1, y1, x2, y2] in pixel coordinates, all integers
```

#### Ball Tracks (List of Dicts)

```python
ball_tracks: List[Dict[int, Dict[str, Any]]]

# Structure:
[
    {1: {"bbox": [x1, y1, x2, y2]}},  # Ball detected, track ID always = 1
    {},                                  # Ball not detected this frame
    {1: {"bbox": [x1, y1, x2, y2]}},
]

# Constraints:
# - len(ball_tracks) == len(video_frames)
# - Track ID is always 1 (single ball)
# - Empty dict {} means ball was not detected this frame
```

#### Player Assignment (List of Dicts)

```python
player_assignment: List[Dict[int, int]]

# Structure:
[
    # Frame 0
    {4: 1, 7: 2, 12: 1, 28: 2},  # player_id → team_id (1 or 2)
    # Frame 1
    {4: 1, 7: 2, 12: 1, 28: 2},
    # ...
]
```

#### Ball Acquisition (List of ints)

```python
ball_acquisition: List[int]

# Structure:
[-1, -1, 28, 28, 28, -1, -1, 7, 7, 7, 7, ...]

# -1   = no player has the ball (airborne, out of bounds, etc.)
# N    = player ID N has confirmed ball possession (≥ 13 consecutive frames)
```

#### Passes & Interceptions (List of ints)

```python
passes:        List[int]  # -1 = no pass this frame; 1 or 2 = team that made the pass
interceptions: List[int]  # -1 = no interception; 1 or 2 = team that made the steal
```

#### Tactical Player Positions (List of Dicts)

```python
tactical_positions: List[Dict[int, List[float]]]

# Structure:
[
    # Frame 0
    {4: [148.3, 72.1], 7: [89.4, 120.7]},  # player_id → [x_pixels, y_pixels]
                                              # in tactical view (flat court) space
    # ...
]
```

#### Player Distances / Speeds (List of Dicts)

```python
distances: List[Dict[int, float]]  # player_id → meters covered this frame
speeds:    List[Dict[int, float]]  # player_id → km/h this window
```

### 4.2 Bounding Box Convention

```
Pixel coordinate system (OpenCV standard):
  (0,0) ──────────────────────────► x
    │    x1,y1 ┌──────────┐
    │           │  object  │
    │           └──────────┘ x2,y2
    ▼
    y

bbox format: [x1, y1, x2, y2]
  x1 = left edge
  y1 = top edge
  x2 = right edge
  y2 = bottom edge

Derived properties:
  center    = ((x1+x2)/2, (y1+y2)/2)
  foot_pos  = ((x1+x2)/2, y2)          ← used for court position
  width     = x2 - x1
  height    = y2 - y1
```

---

## 5. AI Model Architecture

### 5.1 Player & Ball Detectors — YOLOv5l6

```
INPUT
  Video frame: H × W × 3 (BGR, uint8)
       │
       ▼
  Preprocessing (Ultralytics)
  • Letterbox resize to 640×640
  • BGR → RGB
  • Normalize to [0, 1]
  • Add batch dimension: 1×3×640×640
       │
       ▼
  YOLOv5l6 Backbone (CSP + Focus layers)
  • Multi-scale feature extraction
  • Feature Pyramid Network (FPN)
       │
       ▼
  YOLOv5 Head
  • 3 detection scales (small, medium, large objects)
  • Per anchor: [x, y, w, h, objectness, class_0, ..., class_N]
       │
       ▼
  NMS Post-processing
  • IoU threshold: 0.45
  • Confidence threshold: 0.50 (set in tracker code)
       │
       ▼
OUTPUT per frame:
  List of: [x1, y1, x2, y2, confidence, class_id]
  class_id mapping: {0: player, 1: referee, 2: ball, 3: hoop, 4: clock, 5: scoreboard}
```

**Fine-tuning approach:**  
Transfer learning from COCO-pretrained weights. The backbone (feature extractor) retains COCO weights. The detection head is retrained from scratch on the basketball dataset. This preserves general visual features while specializing output heads for basketball-specific classes.

### 5.2 Court Keypoint Detector — YOLOv8x-pose

```
INPUT
  Video frame: H × W × 3 (BGR, uint8)
       │
       ▼
  Preprocessing
  • Resize/pad to 640×640
  • Normalize
       │
       ▼
  YOLOv8x Backbone (C2f blocks + SPPF)
  • Deeper than YOLOv5, better feature representation
       │
       ▼
  YOLOv8 Pose Head
  • Outputs bounding box (court area) + 18 keypoints
  • Each keypoint: (x, y, visibility_score)
       │
       ▼
OUTPUT per frame:
  • 1 detection (the court area bounding box)
  • 18 keypoints, each: (pixel_x, pixel_y)
  • Undetected/out-of-frame keypoints → (0, 0)
```

### 5.3 Team Assigner — FashionCLIP (Zero-Shot)

```
INPUT
  Player crop image (BGR, cropped from bbox)
  Candidate text labels: ["white shirt", "dark blue shirt"]
       │
       ▼
  BGR → RGB conversion (OpenCV)
  RGB array → PIL Image
       │
       ▼
  CLIP Preprocessor
  • Resize image to 224×224
  • Normalize with ImageNet stats
  • Tokenize text labels
       │
       ▼
  FashionCLIP Dual Encoder
  ┌─────────────────┐    ┌─────────────────┐
  │   Image Encoder  │    │   Text Encoder   │
  │   (ViT backbone) │    │  (Transformer)   │
  │   → 512-dim vec  │    │  → 512-dim vec   │
  └────────┬─────────┘    └────────┬─────────┘
           │                       │
           └───────────┬───────────┘
                       │ Cosine similarity
                       ▼
  Logits: [sim(image, "white shirt"), sim(image, "dark blue shirt")]
       │
       ▼
  Softmax → Probabilities
       │
       ▼
OUTPUT: argmax → team_id (0 → team 1, 1 → team 2)
```

**Why zero-shot works here:**  
CLIP was trained on 400 million image-text pairs. It has a rich understanding of what clothing colors look like and how they're described in text. "White shirt" and "dark blue shirt" are natural language descriptions that map well to the visual feature space. For a simple binary color classification task, zero-shot capability is sufficient.

### 5.4 Perspective Transformation — Homography

```
Given:
  src_points: N×2 array of detected court keypoint positions (pixels, camera view)
  dst_points: N×2 array of known keypoint positions (pixels, flat tactical view)
  
  where N ≥ 4 (minimum for homography)

OpenCV findHomography:
  Solves for 3×3 matrix H such that:
  
  ┌dst_x┐       ┌h00 h01 h02┐   ┌src_x┐
  │dst_y│  ∝    │h10 h11 h12│ × │src_y│
  └ 1   ┘       └h20 h21 h22┘   └  1  ┘
  
  Using RANSAC for robustness to outlier keypoints.

Application:
  For any player foot position (px, py) in camera view:
  
  tactical_pos = cv2.perspectiveTransform([[px, py]], H)
  
  Result is the player's position in flat court coordinates (pixels).

Meter conversion:
  meter_x = (tactical_x / TACTICAL_WIDTH_PIXELS)  × COURT_WIDTH_METERS   (28m)
  meter_y = (tactical_y / TACTICAL_HEIGHT_PIXELS) × COURT_HEIGHT_METERS  (15m)
```

---

## 6. Design Patterns

### 6.1 Pipeline Pattern

The top-level architecture is a **Pipeline Pattern** — a sequence of processing stages where each stage has a defined input type and output type.

```
Stage Interface:
  Input:  Data from previous stage
  Output: Enriched data for next stage
  Side effects: None (pure transformations)

Benefits:
  ✓ Each stage independently testable
  ✓ Stages can be reordered without global refactoring
  ✓ New stages can be inserted anywhere
  ✓ Failed stages can be debugged in isolation
```

### 6.2 Strategy Pattern (Drawers)

All seven drawer modules implement the same interface — `draw(video_frames, data, ...) → List[np.ndarray]`. This is the **Strategy Pattern**: multiple algorithms (different visualization strategies) sharing a common interface, allowing them to be called uniformly in `main.py`.

```python
# main.py — uniform drawer invocation
output_frames = player_drawer.draw(output_frames, player_tracks, player_assignment, ball_acquisition)
output_frames = ball_drawer.draw(output_frames, ball_tracks)
output_frames = team_control_drawer.draw(output_frames, player_assignment, ball_acquisition)
output_frames = pass_interception_drawer.draw(output_frames, passes, interceptions)
output_frames = keypoint_drawer.draw(output_frames, court_keypoints)
output_frames = tactical_drawer.draw(output_frames, court_keypoints, tactical_positions, ...)
output_frames = speed_distance_drawer.draw(output_frames, player_tracks, distances, speeds)
```

Any drawer can be added, removed, or replaced by one with the same interface.

### 6.3 Decorator Pattern (Stub System)

The stub system acts as a **Decorator** around expensive inference functions — augmenting them with caching behavior without modifying the underlying inference logic.

```python
# Without stub decorator (raw inference)
def get_object_tracks(frames):
    return run_expensive_yolo_inference(frames)

# With stub decorator pattern
def get_object_tracks(frames, read_from_stub=False, stub_path=None):
    # Attempt to load from cache
    tracks = read_stub(read_from_stub, stub_path)
    if tracks is not None and len(tracks) == len(frames):
        return tracks  # Cache hit — return immediately
    
    # Cache miss — run actual inference
    tracks = run_expensive_yolo_inference(frames)
    
    # Save for next time
    save_stub(stub_path, tracks)
    return tracks
```

### 6.4 Facade Pattern (Module `__init__.py` files)

Each module folder exposes a clean public interface via its `__init__.py` — a **Facade** that hides the internal file structure from consumers.

```python
# team_assigner/__init__.py
from .team_assigner import TeamAssigner

# main.py — consumer sees only TeamAssigner, not internal file structure
from team_assigner import TeamAssigner
```

### 6.5 Null Object Pattern (Missing Data)

Rather than returning `None` or raising exceptions for missing data (undetected ball, out-of-frame keypoint), the system uses the **Null Object Pattern**: empty collections that behave like real data without requiring null checks.

```python
# Ball not detected → empty dict (not None)
ball_tracks[frame] = {}

# Keypoint out of frame → (0, 0) coordinates (not None)
keypoint_list[frame][i] = [0.0, 0.0]

# No player possesses ball → -1 (not None)
possession_list[frame] = -1
```

Downstream code iterates normally and simply produces no output for empty/zero/negative-one entries.

---

## 7. Algorithm Analysis

### 7.1 Ball Possession Detection Algorithm

```
Algorithm: FindBallPossession
Input:  ball_center (x, y), player_tracks_frame (Dict), ball_bbox ([x1,y1,x2,y2])
Output: player_id (int) or -1

Step 1: CONTAINMENT CHECK (Priority 1)
  For each player in player_tracks_frame:
    containment = Intersection(player_bbox, ball_bbox).area / ball_bbox.area
    if containment >= 0.80:
      add player to high_containment_candidates

  if high_containment_candidates is not empty:
    return player with MAXIMUM containment ratio

Step 2: PROXIMITY CHECK (Priority 2)
  For each player in player_tracks_frame:
    Generate 14 candidate comparison points around player bbox:
      - 4 corners: (x1,y1), (x2,y1), (x1,y2), (x2,y2)
      - 4 edge midpoints: (cx,y1), (cx,y2), (x1,cy), (x2,cy)
      - Up to 4 boundary projections (where ball x or y is within bbox bounds)
    
    min_distance = min(euclidean(ball_center, point) for point in 14_points)
    
    if min_distance <= 50 pixels:
      add player to proximity_candidates

  if proximity_candidates is not empty:
    return player with MINIMUM distance to ball

Step 3: RETURN NO POSSESSION
  return -1

Step 4: CONFIRMATION WINDOW (Applied in detect_ball_possession)
  candidate_player = FindBallPossession(...)
  consecutive_count[candidate_player] += 1
  
  if consecutive_count[candidate_player] >= 13:
    possession_list[frame] = candidate_player  # Confirmed
  
  if candidate_player == -1:
    consecutive_count = {}  # Reset all counters

Complexity: O(P × K) per frame
  P = number of players per frame (typically 10-12)
  K = 14 candidate comparison points
  Total: O(168) per frame → effectively O(1) per frame
```

### 7.2 Keypoint Validation Algorithm

```
Algorithm: ValidateKeypoints
Input:  keypoint_list (List[frame_keypoints])
Output: validated_keypoint_list (same structure, invalid → (0,0))

For each frame in keypoint_list:
  detected_indices = [i for i, kp in enumerate(frame_kps) if kp != (0,0)]
  
  if len(detected_indices) < 3: continue  # Not enough for validation
  
  For each keypoint i in detected_indices:
    if frame_kps[i] == (0,0): continue
    
    other_indices = [j for j in detected_indices if j != i and j not in invalid]
    if len(other_indices) < 2: continue
    
    j, k = other_indices[0], other_indices[1]
    
    # Compute distance ratios in camera view
    d_ij_camera = euclidean(frame_kps[i], frame_kps[j])
    d_ik_camera = euclidean(frame_kps[i], frame_kps[k])
    ratio_camera = d_ij_camera / d_ik_camera  (if d_ik_camera > 0)
    
    # Compute distance ratios in tactical view (ground truth)
    d_ij_tactical = euclidean(tactical_kps[i], tactical_kps[j])
    d_ik_tactical = euclidean(tactical_kps[i], tactical_kps[k])
    ratio_tactical = d_ij_tactical / d_ik_tactical  (if d_ik_tactical > 0)
    
    # Compute normalized error
    error = |ratio_camera - ratio_tactical| / ratio_tactical
    
    if error > 0.80:  # 80% proportionality error threshold
      frame_kps[i] = (0, 0)  # Suppress this keypoint
      invalid.append(i)

Complexity: O(F × K²) where F = frames, K = keypoints per frame (18)
  = O(F × 324) → O(F) effectively (constant keypoint count)
```

### 7.3 Speed Calculation Algorithm

```
Algorithm: CalculateSpeed
Input:  distances (List[Dict[player_id, meters]]), fps=30
Output: speeds (List[Dict[player_id, km/h]])

window_frames = 15  (5 × 3, empirically calibrated)

For each frame_index in range(len(distances)):
  For each player_id with data at frame_index:
    
    start_frame = max(0, frame_index - window_frames)
    
    # Aggregate distance and frame count in window
    total_distance = 0
    frames_present = 0
    
    For i in range(start_frame, frame_index):
      if player_id in distances[i] and last_frame_seen is not None:
        total_distance += distances[i][player_id]
        frames_present += 1
    
    if frames_present >= 5:  # Minimum data for reliable speed
      time_seconds = frames_present / fps
      time_hours   = time_seconds / 3600
      speed_kmh    = (total_distance / 1000) / time_hours
      speeds[frame_index][player_id] = speed_kmh
    else:
      speeds[frame_index][player_id] = 0
```

---

## 8. Performance Analysis

### 8.1 Per-Frame Processing Budget

```
Component                       Approx. Runtime (CPU)    Notes
─────────────────────────────────────────────────────────────────────
Player YOLO inference           80–150ms/frame           Batched (20 frames)
Ball YOLO inference             80–150ms/frame           Batched (20 frames)
Court keypoint inference        100–200ms/frame          Batched (20 frames)
ByteTrack update                < 1ms/frame              Near-instant
FashionCLIP team assign.        200–500ms/player         Cached per player ID
Ball possession detection       < 1ms/frame              Pure Python math
Pass/interception detection     < 1ms/frame              Pure Python math
Homography computation          < 5ms/frame              OpenCV optimized
Speed/distance calculation      < 1ms/frame              NumPy math
Drawing (all 7 drawers)         5–15ms/frame             OpenCV operations
─────────────────────────────────────────────────────────────────────
Total (first run):              ~500ms–1000ms/frame
Total (stub run):               ~10ms/frame (drawing only)
```

### 8.2 Batch Processing

Detection modules process frames in batches of 20 to amortize GPU/CPU overhead:

```python
for i in range(0, len(frames), batch_size):  # batch_size = 20
    batch = frames[i:i + batch_size]
    detections = model.predict(batch, conf=0.5)
```

**Why 20?** Empirically: smaller batches (5, 10) don't fully utilize GPU memory bandwidth. Larger batches (50+) can exhaust RAM for high-resolution video. 20 frames at 1080p ≈ 125MB RAM usage, well within typical limits.

### 8.3 Memory Profile

```
Component                   Memory Usage (approx.)
────────────────────────────────────────────────────
Frame buffer (200 frames)   ~500MB (1080p, RGB)
Player YOLO model           ~300MB
Ball YOLO model             ~300MB
Court keypoint model        ~250MB
FashionCLIP model           ~600MB
All stubs loaded            ~10MB
────────────────────────────────────────────────────
Total peak (first run):     ~2GB RAM
Total (stub run, no models):~600MB RAM
```

GPU memory (if available): YOLOv5l6 requires ~4GB VRAM for batch=20.

### 8.4 Stub System Performance Impact

```
Scenario                    Runtime         Notes
───────────────────────────────────────────────────────────────
No stubs (first run)        5–15 minutes    All models run
Player stub only            3–10 minutes    Ball + keypoint still run
All stubs present           10–30 seconds   Drawing pipeline only
───────────────────────────────────────────────────────────────
```

---

## 9. Error Handling Strategy

### 9.1 Failure Modes and Responses

```
Failure Mode                        Response                        Severity
─────────────────────────────────────────────────────────────────────────────
Ball not detected this frame        Empty dict {} propagated        Low
Player exits camera frame           Player absent from frame dict   Low
<4 court keypoints detected         Skip homography this frame      Medium
Homography computation fails        Skip tactical view this frame   Medium
FashionCLIP crop empty/invalid      Return default team (1)         Low
Stub frame count mismatch           Delete stub, re-run inference   Medium
Model file not found                FileNotFoundError (crash)       High
Input video unreadable              cv2.error (crash)               High
─────────────────────────────────────────────────────────────────────────────
```

### 9.2 Graceful Degradation Architecture

```python
# Pattern used across all analytical modules

# Example: tactical view converter
for frame_index, (frame_keypoints, frame_tracks) in enumerate(zip(keypoints, tracks)):
    
    if frame_keypoints is None or len(frame_keypoints) == 0:
        tactical_positions.append({})  # Empty — no tactical data this frame
        continue
    
    detected = [i for i, kp in enumerate(frame_keypoints) if kp != (0,0)]
    
    if len(detected) < 4:
        tactical_positions.append({})  # Not enough keypoints for homography
        continue
    
    try:
        homography = Homography(source_points, target_points)
        # ... transform players
    except (ValueError, cv2.error):
        tactical_positions.append({})  # Homography failed — skip frame
        continue
    
    # ... process normally
```

Every failure produces an empty collection — never an exception that crashes the pipeline. The output video may have missing overlay elements for some frames but always completes.

---

## 10. Configuration Management

### 10.1 Configuration Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CONFIGURATION HIERARCHY                          │
│                                                                       │
│  Level 1: configs/configs.py (static defaults)                       │
│    PLAYER_DETECTOR_PATH = "models/player_detector.pt"                │
│    BALL_DETECTOR_PATH   = "models/ball_detector.pt"                  │
│    ...                                                                │
│             │                                                         │
│             │ imported by                                             │
│             ▼                                                         │
│  Level 2: main.py → argparse (runtime overrides)                     │
│    python main.py video.mp4 --stub_path /custom/path                 │
│    python main.py video.mp4 --output_path /custom/output.avi         │
│             │                                                         │
│             │ passed to                                               │
│             ▼                                                         │
│  Level 3: Module-level parameters (hardcoded tuning constants)       │
│    possession_threshold   = 50                                        │
│    min_frames             = 13                                        │
│    containment_threshold  = 0.80                                      │
│    batch_size             = 20                                        │
│    cache_flush_interval   = 50                                        │
└─────────────────────────────────────────────────────────────────────┘
```

### 10.2 Configurable Parameters Reference

```
Parameter                   Location                        Default   Purpose
──────────────────────────────────────────────────────────────────────────────────
possession_threshold        ball_acquisition_detector.py    50px      Max dist for proximity possession
min_frames                  ball_acquisition_detector.py    13        Frames to confirm possession
containment_threshold       ball_acquisition_detector.py    0.80      Overlap ratio threshold
batch_size                  player_tracker.py               20        Frames per inference batch
cache_flush_interval        team_assigner.py                50        Frames between jersey re-classify
detection_confidence        player_tracker.py               0.50      Min YOLO confidence
keypoint_error_threshold    tactical_view_converter.py      0.80      Max proportionality error
speed_window_size           speed_distance_calculator.py    5×3=15    Frames for speed window
speed_fps                   speed_distance_calculator.py    30        Input video FPS assumed
distance_penalty            speed_distance_calculator.py    0.40      Distance calibration factor
team_one_class              team_assigner.py                "white shirt"  Jersey color description
team_two_class              team_assigner.py                "dark blue shirt" Jersey color description
──────────────────────────────────────────────────────────────────────────────────
```

---

## 11. Testing Strategy

### 11.1 Test Video Coverage

Three purpose-selected test videos exercise different pipeline paths:

```
Video           Primary Test Focus          Secondary
─────────────────────────────────────────────────────────────────
video1.mp4      Player speed & distance     Ball possession
                (player runs full court)    Team assignment
                                            Tactical view

video2.mp4      Pass detection              Team ball control
                (multiple passes)           Possession percentage

video3.mp4      Interception detection      Pass detection
                (cross-team ball steal)     Ball acquisition
```

### 11.2 Module Isolation Testing Pattern

Each module was validated in isolation before pipeline integration:

```python
# Example: Validating player tracker in isolation
from trackers import PlayerTracker
from utils import read_video

frames = read_video("input_videos/video1.mp4")
tracker = PlayerTracker("models/player_detector.pt")
tracks = tracker.get_object_tracks(frames, read_from_stub=False)

# Validate structure
assert len(tracks) == len(frames)              # Frame count matches
assert all(isinstance(t, dict) for t in tracks) # All frames are dicts
for frame_tracks in tracks:
    for player_id, data in frame_tracks.items():
        assert isinstance(player_id, int)
        assert "bbox" in data
        assert len(data["bbox"]) == 4          # [x1, y1, x2, y2]
```

### 11.3 Visual Validation Protocol

For each module, intermediate outputs were visualized before pipeline integration:

| Module | Visual Validation |
|---|---|
| Player Tracker | Draw raw bounding boxes, verify no crowd detections |
| Ball Tracker | Draw triangle on ball position, verify tracking continuity |
| Team Assigner | Draw colored ellipses, verify consistent team colors |
| Ball Possession | Draw red triangle on possessing player, verify accuracy |
| Court Keypoints | Draw keypoint dots with labels, verify correct positions |
| Tactical View | Overlay mini-map, verify player positions match camera view |
| Speed/Distance | Display text under players, sanity-check km/h values |

---

## 12. Dependency Map

### 12.1 External Library Dependencies

```
Library                 Version     Used By                             Purpose
────────────────────────────────────────────────────────────────────────────────
ultralytics             ≥8.0.0      trackers/, court_keypoint_detector/ YOLO inference
supervision             ≥0.16.0     trackers/                           ByteTrack, Detections
opencv-python           ≥4.8.0      All drawers, tactical_view_converter Video I/O, homography, drawing
transformers            ≥4.35.0     team_assigner/                      FashionCLIP
torch                   ≥2.0.0      team_assigner/, ultralytics          Neural network backend
pandas                  ≥2.0.0      trackers/ball_tracker.py            Interpolation
numpy                   ≥1.24.0     All modules                         Array math
Pillow                  ≥10.0.0     team_assigner/                      Image format conversion
roboflow                any         training_notebooks/                  Dataset download
────────────────────────────────────────────────────────────────────────────────
```

### 12.2 Internal Module Dependencies (Import Graph)

```
utils/bbox_utils.py         ← (no internal deps)
utils/stub_utils.py         ← (no internal deps)
utils/video_utils.py        ← (no internal deps)

trackers/player_tracker.py  ← utils/stub_utils.py
trackers/ball_tracker.py    ← utils/stub_utils.py

team_assigner/*.py          ← utils/stub_utils.py

ball_acquisition/*.py       ← utils/bbox_utils.py

pass_interception/*.py      ← (no internal deps)

court_keypoint_detector/
  court_keypoint_detector.py ← utils/stub_utils.py
  homography.py              ← (no internal deps)

tactical_view_converter/*.py ← court_keypoint_detector/homography.py
                               utils/bbox_utils.py

speed_distance_calculator/*.py ← utils/bbox_utils.py

drawers/*.py                ← utils/bbox_utils.py (some)

main.py                     ← all modules + configs/
```

---

## 13. Extension Points

### 13.1 Adding a New Analytics Module

The pipeline is designed to accept new modules with minimal friction:

```
1. Create folder: new_module/
   ├── __init__.py
   └── new_module.py (class with primary method)

2. Implement the processing method:
   def process(self, video_frames, existing_data, ...) → new_data

3. Add to main.py after the appropriate stage:
   new_data = new_module_instance.process(video_frames, player_tracks, ...)

4. Optionally create a drawer in drawers/new_drawer.py
   output_frames = new_drawer.draw(output_frames, new_data)
```

### 13.2 Adding a New Drawer

```
1. Create drawers/new_drawer.py
2. Implement: class NewDrawer:
     def draw(self, video_frames, data, ...) → List[np.ndarray]:
         output = []
         for i, frame in enumerate(video_frames):
             annotated = frame.copy()
             # ... draw on annotated
             output.append(annotated)
         return output

3. Export from drawers/__init__.py:
   from .new_drawer import NewDrawer

4. Add to main.py drawing chain:
   output_frames = new_drawer.draw(output_frames, new_data)
```

### 13.3 Upgrading a Model

Replace any of the three model files in `models/` and update the path in `configs/configs.py`. No other code changes required. Delete the corresponding stub file to force re-inference with the new model.

### 13.4 Supporting a New Sport

The architecture is sport-agnostic. To adapt for soccer, for example:
1. Fine-tune YOLO on soccer player/ball images
2. Update court keypoint model on soccer field landmarks
3. Update real-world court dimensions in `tactical_view_converter.py`
4. Update jersey color descriptions in `team_assigner.py`

---

## Appendix A — Glossary

| Term | Definition |
|---|---|
| **ByteTrack** | A multi-object tracking algorithm that uses IoU matching and motion prediction to assign persistent IDs to detections across frames |
| **Homography** | A 3×3 projective transformation matrix that maps points from one plane to another — used here to convert camera view coordinates to flat court coordinates |
| **FashionCLIP** | A CLIP (Contrastive Language-Image Pretraining) model fine-tuned on clothing images, used for zero-shot jersey color classification |
| **Stub** | A cached pickle file containing the output of an expensive computation, loaded on subsequent runs to avoid re-computation |
| **Containment ratio** | The fraction of the ball's bounding box area that overlaps a player's bounding box area — used as a possession signal |
| **YOLOv5l6** | The large-6 variant of YOLOv5, a high-capacity object detection model; "6" refers to the P6 anchor configuration handling larger objects |
| **Keypoint** | A specific (x, y) coordinate location in an image, here used for court landmark positions detected by the pose model |
| **Perspective distortion** | The optical phenomenon where parallel lines appear to converge and distant objects appear smaller in camera footage |
| **Zero-shot classification** | Classifying an image into a category defined only by a text label, without training on labeled examples of that category |
| **Sliding window** | A computation technique where a fixed-size window moves across a sequence, computing a statistic over each window position — used here for speed smoothing |

---

## Appendix B — File Count Summary

```
Category                    Files    Description
────────────────────────────────────────────────────────────
Analytics modules           5        trackers, team_assigner, ball_acquisition,
                                     pass_interception, speed_distance_calculator
Spatial modules             2        court_keypoint_detector, tactical_view_converter
Visualization               8        7 drawers + drawers/__init__.py
Utilities                   4        video_utils, bbox_utils, stub_utils, __init__
Configuration               2        configs.py, __init__.py
Orchestration               1        main.py
Training                    3        Colab notebooks
────────────────────────────────────────────────────────────
Total Python files:         ~35
Total folders:              11
Total lines of code:        ~2,000
────────────────────────────────────────────────────────────
```

---

*End of Software Engineering Report*

---

```
Document : SOFTWARE_ENGINEERING_REPORT.md
System   : Basketball Intelligence System
Version  : 1.0.0
Status   : Final
```
