# 📓 PROJECT ENGINEERING JOURNAL
## Basketball Intelligence System

---

```
Author        : [Your Name]
Project       : Basketball Intelligence System
Repository    : basketball-analysis
Started       : [Start Date]
Language      : Python 3.10+
Document Type : Engineering Journal — Phase-by-Phase Development Log
```

---

> This journal documents every phase of the project's construction — the problems encountered, the decisions made, the dead ends explored, and the reasoning behind every architectural choice. It is written as a first-person engineering log and serves as the authoritative record of *how* and *why* this system was built the way it was.

---

## Table of Contents

- [Phase 0 — Project Scoping & Architecture Planning](#phase-0--project-scoping--architecture-planning)
- [Phase 1 — Video I/O Infrastructure (`utils/`)](#phase-1--video-io-infrastructure-utils)
- [Phase 2 — Object Detection Baseline (`trackers/`)](#phase-2--object-detection-baseline-trackers)
- [Phase 3 — Model Fine-Tuning (`training_notebooks/` + `models/`)](#phase-3--model-fine-tuning-training_notebooks--models)
- [Phase 4 — Multi-Object Tracking (`trackers/`)](#phase-4--multi-object-tracking-trackers)
- [Phase 5 — Checkpoint System (`utils/stub_utils.py`)](#phase-5--checkpoint-system-utilsstub_utilspy)
- [Phase 6 — Visualization Foundation (`drawers/`)](#phase-6--visualization-foundation-drawers)
- [Phase 7 — Team Assignment (`team_assigner/`)](#phase-7--team-assignment-team_assigner)
- [Phase 8 — Ball Possession Detection (`ball_acquisition/`)](#phase-8--ball-possession-detection-ball_acquisition)
- [Phase 9 — Game Event Detection (`pass_interception/`)](#phase-9--game-event-detection-pass_interception)
- [Phase 10 — Court Keypoint Detection (`court_keypoint_detector/`)](#phase-10--court-keypoint-detection-court_keypoint_detector)
- [Phase 11 — Perspective Transformation (`tactical_view_converter/`)](#phase-11--perspective-transformation-tactical_view_converter)
- [Phase 12 — Player Kinematics (`speed_distance_calculator/`)](#phase-12--player-kinematics-speed_distance_calculator)
- [Phase 13 — Packaging & Configuration (`configs/` + `main.py`)](#phase-13--packaging--configuration-configs--mainpy)
- [Cross-Cutting Lessons](#cross-cutting-lessons)

---

## Phase 0 — Project Scoping & Architecture Planning

**Status:** Complete  
**Files Produced:** Architecture sketch, folder structure plan

---

### What I Was Trying to Solve

The starting question was: *can a computer watch a basketball game and tell you everything that happened, automatically?*

The ambition was to extract the kind of metrics that sports analysts spend hours calculating manually — possession time, pass counts, player speed, tactical positions — by feeding the system only raw video.

### Planning the Architecture

Before writing a single line of code, I spent time mapping out what the system would need to do and in what order. The dependency chain was immediately obvious:

```
You cannot measure ball possession → without knowing where the ball is
You cannot assign possession →      without knowing who each player is and what team they're on
You cannot measure speed →          without knowing real-world coordinates
You cannot get real-world coords →  without knowing court geometry
You cannot know court geometry →    without court landmark detection
```

This dependency chain dictated the module order from the very first architecture sketch. Every module is downstream of the one before it. This meant the system had to be built as a sequential pipeline, not a parallel one.

### Key Architectural Decision: Strict Module Separation

I made an early decision to separate every analytical concern into its own folder/module, even though it would mean more boilerplate (`__init__.py` files everywhere). The reasoning:

- Each module would be independently testable
- Swapping one model (e.g., upgrading YOLO version) wouldn't ripple through unrelated code
- The drawing code (visualization) would be completely separate from analytical code — you could change how something looks without touching the logic that computes it

**Tradeoff accepted:** More files, more `sys.path` management. Worth it.

### Folder Structure Decision

```
Each analytical concern gets its own top-level folder:
  trackers/           — detection + tracking
  team_assigner/      — jersey classification
  ball_acquisition/   — possession detection
  pass_interception/  — event detection
  court_keypoint_detector/ — landmark detection
  tactical_view_converter/ — coordinate mapping
  speed_distance_calculator/ — kinematics
  drawers/            — visualization (one sub-module per metric)
  utils/              — shared math and I/O helpers
  configs/            — all constants in one place
```

This structure means any developer can open the project and immediately understand what each folder does, without reading a single line of code inside it.

---

## Phase 1 — Video I/O Infrastructure (`utils/`)

**Status:** Complete  
**Files:** `utils/video_utils.py`, `utils/bbox_utils.py`, `utils/stub_utils.py`, `utils/__init__.py`

---

### What I Built

Before doing anything with AI models, I needed the ability to read video files into frames and write frames back to video files. This is foundational — everything else depends on it.

### `utils/video_utils.py`

OpenCV's `VideoCapture` reads a video file frame-by-frame. The choice to read all frames into a Python list upfront (rather than streaming) was deliberate:

**Why load all frames into memory?**
- Processing is done in multiple passes (detection, then drawing, then saving). Streaming would require re-opening the video each pass.
- For game clips of a few hundred frames, memory is not a bottleneck.
- The entire frame buffer can be passed to any module without I/O overhead.

```python
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames
```

**Frame format:** OpenCV reads frames as `numpy.ndarray` in shape `(height, width, 3)` with BGR channel ordering. This is important — PIL and PyTorch models expect RGB. Every module that uses both libraries needs an explicit `BGR → RGB` conversion.

### `utils/bbox_utils.py`

Bounding boxes appear in every single module. Rather than duplicating math everywhere, I centralized the helpers here:

- `get_center_of_bbox(bbox)` — returns `(cx, cy)` as integers
- `get_bbox_width(bbox)` — returns `x2 - x1`
- `get_foot_position(bbox)` — returns `(cx, y2)` — the center-bottom point, used as a player's ground contact point for distance calculations
- `measure_distance(p1, p2)` — Euclidean distance via the Pythagorean theorem

**Why `get_foot_position` specifically?**  
When measuring where a player is on the court, we want their feet — the point actually touching the ground — not their body center. Using body center would be physically incorrect (a player's center of mass floats above the court). The foot position `(cx, y2)` is the best proxy for ground contact in 2D.

### `utils/stub_utils.py`

This module deserves its own section (see Phase 5 below).

---

## Phase 2 — Object Detection Baseline (`trackers/`)

**Status:** Complete  
**Files:** `trackers/player_tracker.py`, `trackers/ball_tracker.py`, `trackers/__init__.py`

---

### First Attempt: Vanilla YOLO

My first experiment was vanilla YOLOv8x (the largest pretrained COCO model) applied directly to `video1.mp4`:

```python
from ultralytics import YOLO
model = YOLO("yolov8x.pt")
results = model.predict("input_videos/video1.mp4", save=True)
```

The output was immediately instructive about what *wouldn't* work. The model:
- Detected every spectator in the stands as a "person"
- Frequently missed the basketball, classifying it as "glove" or not at all
- Had no concept of "on-court player" vs "off-court spectator"

This confirmed the need for fine-tuning (Phase 3). But while waiting to train models, I used the vanilla model to design the data structures that downstream modules would consume.

### Data Structure Design

I settled on this structure for player tracks:

```python
player_tracks = [
    # Frame 0
    {
        4: {"bbox": [x1, y1, x2, y2]},
        7: {"bbox": [x1, y1, x2, y2]},
        12: {"bbox": [x1, y1, x2, y2]},
    },
    # Frame 1
    {
        4: {"bbox": [x1, y1, x2, y2]},
        ...
    },
    ...
]
```

**Why a list of dicts?** List index = frame number. Dict key = player ID. Dict value = another dict (not a bare bbox list) because I anticipated needing to add more properties per player later (team, possession status, speed). The nested dict gives room to grow without changing the outer structure.

**Why not a numpy array?** Players enter and leave the frame. A numpy array would need a fixed second dimension (number of players), which isn't known in advance and varies frame to frame. A list of dicts handles variable-length player sets naturally.

### Ball Track Structure

```python
ball_tracks = [
    # Frame 0 — ball detected
    {1: {"bbox": [x1, y1, x2, y2]}},
    # Frame 1 — ball not detected
    {},
    # Frame 2 — ball detected
    {1: {"bbox": [x1, y1, x2, y2]}},
    ...
]
```

The ball always has track ID `1` (there is only one ball). An empty dict `{}` means the ball was not detected that frame. This is intentional — it's better to have an explicit "not detected" marker than to skip the frame entirely, because downstream modules iterate over all frames and need a stable list length.

---

## Phase 3 — Model Fine-Tuning (`training_notebooks/` + `models/`)

**Status:** Complete  
**Files:** `training_notebooks/*.ipynb`, `models/*.pt`

---

### The Problem With COCO Models

The core issue: COCO models are trained on general internet images. Basketball footage has very specific characteristics that COCO never saw:
- Players wearing numbered jerseys inside hard boundary lines
- A specific orange ball against a hardwood court
- Camera angles tilted above the court

The model needs to learn these domain-specific priors. Fine-tuning gives it that knowledge by showing it labeled examples of exactly what we want.

### Dataset: Roboflow Basketball Dataset

The dataset I found had 165 labeled images with the following classes:
- `player` — on-court players only (no spectators)
- `referee` — officials
- `ball` — basketball
- `hoop` — basket/hoop
- `clock` — shot clock
- `scoreboard` — scoreboard display

**The critical difference from COCO:** In this dataset, people sitting in the stands are simply not labeled. The model learns "unlabeled persons in bleachers are not players" by seeing hundreds of examples where only on-court athletes receive the `player` label.

### Why YOLOv5 Instead of YOLOv8?

This was counterintuitive. YOLOv8 is newer and generally more capable. But after testing both:

- YOLOv8 fine-tuned on this 165-image dataset overfit more aggressively
- YOLOv5l6 produced more consistent detections and better generalization

The likely explanation: the dataset is small (165 images). YOLOv8's larger capacity works against it on small datasets — it has more parameters to overfit with. YOLOv5l6 is "big enough" to learn the domain but "constrained enough" not to memorize training images.

**Lesson:** More powerful model ≠ better results on small datasets. Always empirically test both.

### Two Models, Not One

The original plan was one model handling both players and balls. I changed this after observing that:
- Optimal training duration for player detection: ~100 epochs
- Optimal training duration for ball detection: ~250 epochs

A single model trained to 100 epochs had poor ball recall. Trained to 250 epochs, it started overfitting on players. The detection challenges are simply different enough that separate models are the correct solution.

At inference time, I use the player model's `player` predictions and the ball model's `ball` predictions, ignoring each model's secondary outputs.

### Training Configuration

```
Player Detector:
  Base model:   yolov5l6u.pt
  Epochs:       100
  Batch size:   8
  Image size:   640 × 640
  Hardware:     Google Colab T4 GPU
  Dataset:      165 images (train/val/test split)

Ball Detector:
  Base model:   yolov5l6u.pt
  Epochs:       250  ← doubled for better ball recall
  Batch size:   8
  Image size:   640 × 640

Court Keypoint Detector:
  Base model:   yolov8x-pose.pt  ← pose backbone for keypoint regression
  Epochs:       500  ← small dataset needs longer training
  Batch size:   16
  Image size:   640 × 640
  Dataset:      56 court images, 18 keypoints each
```

### Why a Pose Model for Court Keypoints?

YOLO's pose models (the `-pose` variant) output both bounding boxes and keypoint coordinates — `(x, y)` pixel positions for each landmark. This is exactly the format needed for court landmark positions. By reusing the pose backbone's architecture (which already knows how to regress point coordinates from image features), I skip designing a custom keypoint regression head from scratch.

The transfer works because: the court has clear geometric structure (painted lines) that creates consistent local image features around each keypoint — similar to how human joints create consistent local features in human pose estimation.

---

## Phase 4 — Multi-Object Tracking (`trackers/`)

**Status:** Complete  
**Files:** `trackers/player_tracker.py`, `trackers/ball_tracker.py`

---

### The Tracking Problem

YOLO processes each frame independently. Frame 0 detects "player at (100, 200)." Frame 1 detects "player at (105, 198)." There's nothing in the detection output connecting these as the same person. Without tracking, we can't measure how far anyone moved or analyze individual player behavior.

**Tracking** adds persistent IDs by solving the assignment problem: which detection in frame N corresponds to which detection in frame N-1?

### ByteTrack via Supervision

I used ByteTrack, accessed through the `supervision` library. ByteTrack works by:

1. **Prediction:** Using a Kalman filter, predict where each tracked object should be in the next frame based on its velocity
2. **High-confidence matching:** Match new high-confidence detections to predictions by IoU (intersection-over-union)
3. **Low-confidence matching:** Match remaining predictions to low-confidence detections (ByteTrack's key innovation over simpler trackers)
4. **New object creation:** Unmatched detections spawn new tracks

```python
self.tracker = sv.ByteTrack()
detections_sv = sv.Detections.from_ultralytics(detection)
detections_with_tracks = self.tracker.update_with_detections(detections_sv)
```

### Why Not Track the Ball With ByteTrack?

ByteTrack relies on IoU overlap between consecutive frames to match detections. The basketball moves extremely fast — it's common for the ball to have zero pixel overlap between frame N and frame N+1 when it's in flight. ByteTrack breaks down under these conditions.

**Ball tracking strategy instead:**
1. Take the single highest-confidence "ball" detection per frame
2. Apply **outlier suppression**: reject any detection more than `25 × frame_gap` pixels from the last known position
3. Apply **linear interpolation**: fill in frames where the ball wasn't detected by interpolating between known positions

```python
# Outlier suppression
max_allowed_distance = 25 * frame_gap
if distance > max_allowed_distance:
    ball_positions[i] = {}  # Suppress this detection

# Interpolation via pandas
df_ball = pd.DataFrame(ball_positions_list, columns=["x1","y1","x2","y2"])
df_ball.interpolate().bfill()
```

**Why pandas for interpolation?** Pandas `DataFrame.interpolate()` handles missing values (`NaN`) with linear interpolation in a single line. Writing the equivalent loop-based interpolation would be ~20 lines of error-prone code. Using the right tool for the job is good engineering.

---

## Phase 5 — Checkpoint System (`utils/stub_utils.py`)

**Status:** Complete  
**Files:** `utils/stub_utils.py`

---

### The Development Velocity Problem

During development, I found myself running the full pipeline repeatedly — tweaking the possession threshold, adjusting drawing colors, testing edge cases — but having to wait 3–5 minutes each run for YOLO inference to finish.

The solution was a checkpoint (stub) system. After the first run, every intermediate output is serialized with `pickle` and saved to disk. On subsequent runs, the pipeline checks if a stub exists and loads it instead of re-running inference.

### Implementation

```python
def save_stub(stub_path, obj):
    if stub_path is None:
        return
    os.makedirs(os.path.dirname(stub_path), exist_ok=True)
    with open(stub_path, "wb") as f:
        pickle.dump(obj, f)

def read_stub(read_from_stub, stub_path):
    if read_from_stub and stub_path and os.path.exists(stub_path):
        with open(stub_path, "rb") as f:
            return pickle.load(f)
    return None
```

### Safety Check

There's a subtle bug that the stub system guards against: if you run on video1 and save a stub, then switch to video2, the stub would load the wrong data. The guard:

```python
if tracks is not None and len(tracks) == len(frames):
    return tracks  # Stub is valid
# Otherwise, re-run inference
```

If the stub's frame count doesn't match the current video's frame count, the stub is treated as invalid and inference runs fresh. This prevents silently wrong results when switching input videos.

### Impact on Development Velocity

**Before stubs:** Every code change → 3-5 minute wait for inference  
**After stubs:** Every code change → run in seconds (downstream only)

This is the single change that made iterating on the possession logic, drawing code, and kinematics calculations practical. It's a pattern I'll use in every future ML project.

---

## Phase 6 — Visualization Foundation (`drawers/`)

**Status:** Complete  
**Files:** `drawers/*.py`

---

### Design Philosophy: Drawers as Thin Renderers

A key architectural decision: the drawing code never computes analytics. Drawers are thin renderers — they receive already-computed data and translate it into pixels.

This separation means:
- You can run the pipeline without saving video (for testing)
- You can change how something looks without touching the logic that computes it
- Each drawer can be independently toggled by commenting out one line in `main.py`

### The Ellipse Annotation

Rather than using YOLO's default bounding box (a clunky rectangle with text), I drew ellipses at each player's feet — a design inspired by sports video games like FIFA and NBA 2K.

```
Default YOLO box:                 Custom ellipse annotation:
┌──────────────────┐
│  person  0.94   │                    [Player 4]
│                  │                  ╭─────────────╮  ← ellipse at feet
│  [player figure] │              [player figure]
│                  │
└──────────────────┘
```

The ellipse is drawn using `cv2.ellipse()` with a 45°–235° arc (creating the open-top appearance used in games). The track ID is displayed in a small filled rectangle beneath.

```python
cv2.ellipse(
    frame,
    center=(x_center, y2),           # Foot position
    axes=(width, int(0.35 * width)),  # Proportional to player width
    angle=0,
    startAngle=45,
    endAngle=235,
    color=color,
    thickness=2,
)
```

### One Drawer Per Metric

I created seven separate drawer classes rather than one monolithic drawing function:

| Drawer | Metric |
|---|---|
| `PlayerTracksDrawer` | Player ellipses, team colors, track IDs |
| `BallTracksDrawer` | Inverted triangle pointer above ball |
| `TeamBallControlDrawer` | Possession percentage stats box |
| `PassInterceptionDrawer` | Pass/interception counters box |
| `CourtKeypointsDrawer` | Keypoint dots + index labels |
| `TacticalViewDrawer` | Court mini-map overlay |
| `SpeedDistanceDrawer` | Per-player speed/distance text |

Each drawer follows the same interface:
```python
output_frames = drawer.draw(video_frames, data, ...)
```

This uniformity means `main.py`'s drawing section is a clean sequential chain — readable and easily extensible.

### Transparency Effect

For stats boxes, I used OpenCV's `addWeighted` to create a semi-transparent white overlay:

```python
overlay = frame.copy()
cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), -1)
cv2.addWeighted(overlay, alpha=0.8, frame, 1 - alpha, 0, frame)
```

The alpha of 0.8 keeps the box visible while allowing the court to show through — a subtle but important polish detail.

---

## Phase 7 — Team Assignment (`team_assigner/`)

**Status:** Complete  
**Files:** `team_assigner/team_assigner.py`, `team_assigner/__init__.py`

---

### The Problem

To detect passes and interceptions, I need to know which team each player belongs to. The teams wear different jerseys, so this should be solvable from visual information. But how?

### Options Considered

**Option A: HSV Color Segmentation**  
Segment the jersey area of each crop, compute the dominant HSV color, and threshold for team.  
*Why rejected:* Lighting variation across the court changes perceived HSV values dramatically. Blue under bright court lights and blue in shadow are HSV-different but semantically the same.

**Option B: Train a Binary Classifier**  
Collect and label jersey crop images, train a CNN for two-class output.  
*Why rejected:* Requires labeled data. The whole point was to avoid labeling data per game. Also: what if jersey colors change next game?

**Option C: Zero-Shot CLIP Classification**  
Use a pretrained vision-language model that understands natural language color descriptions — no training data needed.  
*Why chosen:* No labeled data required. Works with natural language jersey descriptions that can be changed in `configs/` without touching model weights.

### Why FashionCLIP Specifically?

Standard CLIP (OpenAI) was tested first and produced acceptable results. FashionCLIP, a CLIP model fine-tuned specifically on clothing and fashion images, produced noticeably better results. This makes intuitive sense: FashionCLIP's embedding space has been specialized to represent garment characteristics (fabric texture, pattern, color in clothing context), whereas standard CLIP balances its embedding space across all image types.

The test:
- Image: player crop wearing a white jersey
- Candidates: `["white shirt", "dark blue shirt"]`
- FashionCLIP confidence for "white shirt": 93%
- Standard CLIP confidence for "white shirt": 71%

FashionCLIP is the clear winner for this use case.

### The 50-Frame Cache Flush

A critical optimization: player team assignments are cached per player ID. Once a player has been assigned to a team, we don't re-classify them on every frame — the jersey doesn't change mid-game.

```python
if player_id in self.player_team_dict:
    return self.player_team_dict[player_id]  # Cache hit
```

However: when two players from opposing teams stand close together, their bounding boxes may partially overlap. The crop passed to FashionCLIP captures both jerseys, confusing the classifier. Without a correction mechanism, this misclassification would persist for the rest of the video.

The fix: flush the cache every 50 frames. This forces periodic re-classification with hopefully a better crop (when players are separated), self-correcting any earlier mistakes.

**Tradeoff:** 50-frame re-classification adds computational overhead vs. caching forever. But the accuracy benefit outweighs it. 50 frames ≈ ~1.7 seconds at 30fps — frequent enough to self-correct without being so frequent it negates the cache benefit.

---

## Phase 8 — Ball Possession Detection (`ball_acquisition/`)

**Status:** Complete  
**Files:** `ball_acquisition/ball_acquisition_detector.py`, `ball_acquisition/__init__.py`

---

### The Core Problem

Given: the ball's bounding box and all players' bounding boxes, per frame.  
Question: which player has the ball right now?

### Two-Strategy Approach

**Strategy 1 — Containment (Priority)**  
If the ball's bounding box overlaps the player's bounding box by ≥80%, the player is "containing" the ball and is likely dribbling or holding it.

```python
# Intersection area calculation
ix1 = max(px1, bx1); iy1 = max(py1, by1)
ix2 = min(px2, bx2); iy2 = min(py2, by2)

if ix2 < ix1 or iy2 < iy1:
    return 0.0  # No overlap

intersection_area = (ix2 - ix1) * (iy2 - iy1)
ball_area = (bx2 - bx1) * (by2 - by1)
containment_ratio = intersection_area / ball_area
```

**Strategy 2 — Proximity (Secondary)**  
If no player's bounding box contains the ball, find the closest player using 14 candidate comparison points around the player bounding box:

```
 ●─────────●─────────●    ← 3 top edge points (corners + midpoint)
 │                   │
 ●         +         ●    ← 2 side midpoints
 │                   │
 ●─────────●─────────●    ← 3 bottom edge points

 + 4 interior projection points (where ball's x or y is within box bounds)
```

The 14-point approach instead of just "distance to bounding box center" is because of dribbling: a player dribbling in front of them has the ball near their feet, not their center. The closest point of the bounding box to the ball is the correct proximity metric.

### The 13-Frame Confirmation Window

**The trickiest design challenge:** when the ball is thrown into the air (a shot, a high pass), it arcs over players' heads. In 2D projection, the ball appears to "pass through" players' bounding boxes even though it's 3 meters above them in 3D space. Without correction, the system would falsely assign possession to whoever the ball happens to pass over.

**Solution:** possession is only officially assigned after a player maintains "candidate possession" for 13+ consecutive frames.

```python
self.consecutive_possession_count[player_id] += 1
if self.consecutive_possession_count[player_id] >= self.min_frames:
    possession_list[frame_num] = player_id  # Official possession
```

**Why 13 specifically?** Empirically tested. 10 frames produced false positives during high-arc passes. 15 frames was too conservative — legitimate fast possession changes (quick catches) were being missed. 13 was the sweet spot for the test clips.

A ball truly held by a player will stay near them for many frames. A ball flying overhead will typically only be in bounding box overlap for 3–8 frames. The 13-frame window reliably discriminates between these cases.

---

## Phase 9 — Game Event Detection (`pass_interception/`)

**Status:** Complete  
**Files:** `pass_interception/pass_interception_detector.py`, `pass_interception/__init__.py`

---

### Building on Possession Data

With reliable possession data from Phase 8, event detection becomes straightforward. A "possession change" — player A controlled the ball in frame N, player B controls it in frame N+1 — is either a pass or an interception. Which one depends on team membership.

```python
if previous_team == current_team:
    passes[frame] = current_team      # Intra-team transfer → PASS
elif previous_team != current_team:
    interceptions[frame] = current_team  # Cross-team transfer → INTERCEPTION
```

### Edge Cases Handled

**Between-possession gaps:** When no one has the ball (ball is in the air, out of bounds), the possession list contains `-1`. The detector ignores these frames and only compares frames where both previous and current possession are assigned.

**First frame:** The very first frame where possession is assigned has no "previous" to compare to. The detector initializes `previous_holder = -1` and skips comparison until a valid previous exists.

**Team assignment failure:** If `player_id` isn't in `player_assignment_frame` (edge case where team classification failed), the event is treated as neither pass nor interception and the frame is skipped.

### Testing Across All Three Videos

Each test video was selected to exercise a specific module:
- `video1.mp4` — Dribbling → tests possession detection, speed/distance
- `video2.mp4` — Passing sequences → tests pass detection
- `video3.mp4` — Interception → tests both event types

Testing all three before moving to Phase 10 confirmed the event detection was reliable before adding more complexity.

---

## Phase 10 — Court Keypoint Detection (`court_keypoint_detector/`)

**Status:** Complete  
**Files:** `court_keypoint_detector/court_keypoint_detector.py`, `court_keypoint_detector/homography.py`, `court_keypoint_detector/__init__.py`

---

### Why Court Keypoints?

Speed and distance in pixel space are meaningless for sports analysis. A player moving 50 pixels in the far half of the court (where pixels are compressed by perspective) covers more physical distance than a player moving 50 pixels in the near half.

To convert pixels to meters, we need to understand the geometry of the court in each frame — which requires detecting visible court landmarks and knowing their real-world positions.

### The 18-Keypoint Model

The fine-tuned YOLOv8x-pose model outputs 18 (x, y) coordinate pairs per frame, corresponding to specific locations on the court:

```
 0    1    2    3    4    5
 ●────●────●────●────●────●   ← Baseline (near)
 │              │            
 6    7         8    9   10
 ●────●─────────●────●────●   ← Mid-court region
 │              │            
11   12        13   14   15
 ●────●─────────●────●────●   ← Mid-court region
 │              │            
16   17                       ← Baseline (far)
 ●────●───────────────────── 
```

Undetected keypoints (outside the camera frame) are output as `(0, 0)`. The homography computation ignores these.

### Keypoint Validation

After detection, I observed that keypoint 12 was frequently misplaced — appearing in a geometrically incorrect location relative to its neighbors. Rather than live with this error, I added a proportionality validation pass:

**Core idea:** The ratio of distances should be consistent between the camera view and the flat tactical view.

```
If: distance(13→14) / distance(13→12) in camera view
≈  distance(13→14) / distance(13→12) in tactical view

Then: keypoint 12 is geometrically plausible
```

Any keypoint whose proportionality error exceeds 80% is set to `(0, 0)` (treated as undetected). This self-correction mechanism runs on every frame, meaning a temporarily misplaced keypoint doesn't corrupt the entire video.

---

## Phase 11 — Perspective Transformation (`tactical_view_converter/`)

**Status:** Complete  
**Files:** `tactical_view_converter/tactical_view_converter.py`, `tactical_view_converter/__init__.py`

---

### The Distortion Problem Quantified

I measured the distortion empirically by pixel-counting the court halves in `video1.mp4`:

- Near half of court: **~600 pixels** vertically
- Far half of court: **~300 pixels** vertically

A 2:1 pixel ratio for two halves that cover the same physical distance. Any player moving in the far half covers twice the real-world distance per pixel compared to a player in the near half. Speed calculations in pixel space would have systematic 2× errors depending on court position.

### Homography: The Correct Solution

Homography (also called projective transformation) is the standard computer vision technique for correcting perspective distortion. OpenCV's `findHomography()` takes point correspondences and computes the 3×3 transformation matrix that maps between them:

```python
H, _ = cv2.findHomography(source_points, target_points)
# source_points: detected keypoint pixel positions in camera frame
# target_points: known keypoint positions in flat tactical view (in pixels, scaled from meters)
```

Once we have `H`, transforming any pixel coordinate to tactical view coordinates is one function call:

```python
tactical_position = cv2.perspectiveTransform(player_foot_position, H)
```

**Real-world coordinate conversion:** The tactical view image is 300×161 pixels representing a 28×15 meter court. Converting pixel positions to meters is simple cross-multiplication:

```
meters_x = (tactical_pixel_x / 300) × 28
meters_y = (tactical_pixel_y / 161) × 15
```

### Homography Failure Handling

`findHomography` requires at least 4 point correspondences and can fail if the input points are degenerate (colinear, or too few). The `homography.py` wrapper handles failures gracefully:

```python
try:
    self.homography = Homography(source_points, target_points)
    tactical_pos = self.homography.transform_points(player_position)
except (ValueError, cv2.error):
    pass  # Skip this frame — no tactical view for this frame
```

Frames with insufficient keypoints produce an empty tactical view entry (no players plotted that frame). This is correct behavior — it's better to have a missing tactical marker than a wrong one.

---

## Phase 12 — Player Kinematics (`speed_distance_calculator/`)

**Status:** Complete  
**Files:** `speed_distance_calculator/speed_distance_calculator.py`, `speed_distance_calculator/__init__.py`

---

### Distance Calculation

With players in real-world meter coordinates, per-frame distance is straightforward:

```python
distance_meters = measure_distance(previous_pos_meters, current_pos_meters) × 0.4
```

The `0.4 penalty factor` deserves explanation. Initial measurements showed all distances were systematically overestimated by approximately 2.5×. The sources of overestimation:

1. **Interpolated ball positions** contribute noise to apparent player positions (indirect effect)
2. **Camera motion / panning** adds apparent pixel movement that isn't player movement
3. **Keypoint jitter** — small frame-to-frame variation in keypoint positions propagates through homography

Rather than trying to model each source of error separately, I calibrated empirically: ran the system on a clip where a player clearly runs the full court length (28m) and adjusted the penalty factor until the measured distance matched. `0.4` produced results within ~10% of ground truth.

### Speed Calculation: Sliding Window Approach

Computing speed as `distance / 1 frame` produces extremely noisy values — any tiny keypoint jitter or tracking wobble shows up as an apparent velocity spike.

**Solution:** compute speed over a 15-frame sliding window (≈0.5 seconds at 30fps):

```python
window_size = 5  # Used to scale frame gap
for frame_index in range(len(distances)):
    start_frame = max(0, frame_index - window_size * 3)  # 15 frames back
    
    total_distance = 0
    frames_present = 0
    
    for i in range(start_frame, frame_index):
        if player_id in distances[i]:
            total_distance += distances[i][player_id]
            frames_present += 1
    
    if frames_present >= window_size:
        time_in_seconds = frames_present / fps
        time_in_hours = time_in_seconds / 3600
        speed_kmh = (total_distance / 1000) / time_in_hours
        speeds[frame_index][player_id] = speed_kmh
```

**Why 15 frames?** Less than 10 frames: too noisy. More than 20 frames: speed values lag too far behind actual movement (a player who stops shows "still moving" for too long). 15 frames provides good balance.

---

## Phase 13 — Packaging & Configuration (`configs/` + `main.py`)

**Status:** Complete  
**Files:** `configs/configs.py`, `configs/__init__.py`, `main.py`

---

### The Hardcoding Problem

By Phase 12, the codebase had model paths scattered across multiple files, and changing which video to process required editing source code. This is bad engineering — a user shouldn't need to touch implementation files to use the tool.

### Centralizing Configuration

All constants moved to `configs/configs.py`:

```python
STUB_DEFAULT_PATH           = "stubs"
PLAYER_DETECTOR_PATH        = "models/player_detector.pt"
BALL_DETECTOR_PATH          = "models/ball_detector.pt"
COURT_KEYPOINT_DETECTOR_PATH = "models/court_keypoint_detector.pt"
OUTPUT_VIDEO_PATH           = "output_videos/output.avi"
```

Any path change is now a one-file edit. Any developer picking up the project knows exactly where to look for configuration.

### CLI with argparse

The input video path and optional overrides are accepted via command line:

```python
parser = argparse.ArgumentParser(description="Basketball Video Analysis")
parser.add_argument("input_video", type=str, help="Path to input video file")
parser.add_argument("--stub_path", type=str, default=STUB_DEFAULT_PATH)
parser.add_argument("--output_path", type=str, default=OUTPUT_VIDEO_PATH)
args = parser.parse_args()
```

This makes the tool usable without source code changes:

```bash
# Process any game clip
python main.py /path/to/any/game.mp4 --output_path /path/to/output.avi
```

### `main.py` Structure

The orchestration file follows a strict sequential structure that mirrors the dependency chain:

```python
def main():
    args = parse_args()
    
    # 1. I/O
    video_frames = read_video(args.input_video)
    
    # 2. Detection & Tracking
    player_tracks = player_tracker.get_object_tracks(video_frames, ...)
    ball_tracks = ball_tracker.get_object_tracks(video_frames, ...)
    
    # 3. Analytics
    player_assignment = team_assigner.get_player_teams_across_frames(...)
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)
    ball_acquisition = ball_acquisition_detector.detect_ball_possession(...)
    passes = pass_interception_detector.detect_passes(...)
    interceptions = pass_interception_detector.detect_interceptions(...)
    
    # 4. Spatial Transform
    court_keypoints = court_keypoint_detector.get_court_keypoints(...)
    court_keypoints = tactical_view_converter.validate_keypoints(court_keypoints)
    tactical_player_positions = tactical_view_converter.transform_players_to_tactical_view(...)
    
    # 5. Kinematics
    player_distances = calculator.calculate_distance(tactical_player_positions)
    player_speeds = calculator.calculate_speed(player_distances)
    
    # 6. Drawing
    output_frames = player_drawer.draw(...)
    output_frames = ball_drawer.draw(...)
    output_frames = team_control_drawer.draw(...)
    output_frames = pass_interception_drawer.draw(...)
    output_frames = keypoint_drawer.draw(...)
    output_frames = tactical_drawer.draw(...)
    output_frames = speed_distance_drawer.draw(...)
    
    # 7. Save
    save_video(output_frames, args.output_path)
```

Each step feeds cleanly into the next. Reading `main.py` is reading the system architecture.

---

## Cross-Cutting Lessons

### 1. Always Validate Assumptions Visually

Every module was tested with visual output (drawing intermediate results) before being connected to the next module. This caught bugs that would have been nearly impossible to debug if the entire pipeline ran end-to-end first.

**Example:** The `draw_keypoints` drawer revealed that keypoint 12 was systematically misplaced. If I had gone straight to the homography without visualization, the perspective transform would have been silently wrong.

### 2. Empirical Calibration Beats Analytical Exactness

Several parameters in this system — the 13-frame possession window, the 0.4 distance penalty factor, the 80% keypoint proportionality threshold — were calibrated empirically, not derived analytically. This is appropriate: the "correct" values depend on the specific basketball footage, camera setup, and YOLO model performance, none of which are easily modeled analytically.

### 3. Graceful Degradation Over Exceptions

Every module that processes per-frame data explicitly handles the "missing" case:
- Ball not detected → empty dict `{}`
- Keypoint outside frame → `(0, 0)` coordinates
- Insufficient keypoints for homography → skip frame, no error

This design means the pipeline never crashes on edge cases. It may produce incomplete results for some frames, but it always completes.

### 4. Separate Analytical Logic From Visualization

This principle paid dividends throughout development. When a metric looked wrong in the output video, I could immediately answer: "Is this a computation bug or a drawing bug?" by checking the raw data before it entered the drawers. Without this separation, debugging would require tracing through interleaved compute-and-draw code.

### 5. The Stub System Is Not Optional

Development without the stub system would have been 10× slower. For any ML pipeline where inference is slow, checkpointing intermediate results is not a nice-to-have — it's a prerequisite for practical development velocity.

---

*End of Engineering Journal*

---

```
Document: PROJECT_JOURNAL.md
Project:  Basketball Intelligence System
Phases:   13 complete
Status:   Production-ready
```
