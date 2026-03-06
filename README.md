# LiDAR Object Detection and Tracking — Case Study
## DLMDSEAAD02 · Localization, Motion Planning and Sensor Fusion

**Author:** Kalpana Abhiseka Maddi | **Registration:** 10249408 | **Tutor:** Florian Simroth

---

## Overview

A full autonomous-driving perception pipeline applied to a real stationary **Blickfeld Cube 1** solid-state LiDAR sensor recording (718 CSV frames, ~72 seconds).

```
Raw LiDAR CSV (718 frames)
        │
        ▼
   Preprocessing          ← range filter · RANSAC ground removal · SOR · voxel downsample
        │
        ▼
  DBSCAN Clustering       ← eps = 0.8 m · min_samples = 8
        │
        ▼
Rule-Based Classification ← VEHICLE / PEDESTRIAN / STATIC_STRUCTURE / UNKNOWN
        │
        ▼
 Kalman Filter Tracking   ← constant-velocity model · Hungarian assignment
        │
        ▼
  Verification Report     ← proxy metrics (no ground truth available)
        │
        ▼
  Cinematic MP4 Video     ← jet colourmap · neon bounding boxes · 1920×1080
```

> **Note:** No ground-truth annotations are available. All performance metrics are
> verification-oriented proxy metrics (detection stability, track length distribution,
> temporal stability index). MOTA, precision, recall and accuracy (%) are **not reported**
> because they cannot be computed without ground truth.

---

## Project Structure

```
Lidar case study/
├── data_loader.py              Load Blickfeld Cube 1 CSV frames
├── preprocessing.py            Range filter · voxel downsample · RANSAC ground · SOR
├── clustering.py               DBSCAN object segmentation
├── classification.py           Rule-based classifier (thresholds documented below)
├── tracking.py                 Kalman-filter MOT + velocity-based static reclassification
├── main_pipeline.py            Full pipeline — all 718 frames, prints verification report
├── performance_analysis.py     Honest verification metrics (no fabricated numbers)
├── generate_cinematic_real.py  Cinematic video from REAL sensor data  ← main video
├── generate_3d_video.py        Matplotlib 3-D tracking video (ground-level view)
├── generate_cinematic_video.py Synthetic demo video (urban simulation, not real data)
├── visualization.py            BEV and Open3D helpers
├── lidar_validation.py         Sensor data sanity checks
├── requirements.txt            Python dependencies
│
├── Lider datasets/             Raw Blickfeld Cube 1 recordings (4 ZIP parts)
│   ├── *_frame-1899_part_1/   Frames 1849–1899   ( 51 frames)
│   ├── *_frame-2155_part_2/   Frames 1900–2155   (256 frames)
│   ├── *_frame-2414_part_3/   Frames 2156–2414   (259 frames)
│   └── *_frame-2566_part_4/   Frames 2415–2566   (152 frames)
│
├── lidar_cinematic_real.mp4    OUTPUT — cinematic video using real sensor data ★
├── lidar_cinematic.mp4         OUTPUT — cinematic synthetic demo video
└── lidar_3d_tracking.mp4       OUTPUT — 3-D matplotlib tracking video
```

---

## Requirements

### Python Version
```
Python 3.12.4  (via pyenv)
/Users/abhisekmaddi/.pyenv/versions/3.12.4/bin/python3
```

### Install Dependencies
```bash
/Users/abhisekmaddi/.pyenv/versions/3.12.4/bin/python3 -m pip install \
    numpy pandas scikit-learn scipy filterpy \
    open3d matplotlib opencv-python tqdm
```

Or install from requirements file:
```bash
/Users/abhisekmaddi/.pyenv/versions/3.12.4/bin/python3 -m pip install -r requirements.txt
```

### Required Packages

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥ 1.21 | Array operations |
| `pandas` | ≥ 1.3 | CSV loading |
| `scikit-learn` | ≥ 1.0 | DBSCAN clustering |
| `scipy` | ≥ 1.7 | Hungarian algorithm |
| `filterpy` | ≥ 1.4 | Kalman filter |
| `open3d` | ≥ 0.15 | Voxel downsample · RANSAC |
| `opencv-python` | ≥ 4.5 | Video rendering |
| `matplotlib` | ≥ 3.4 | 3-D video (generate_3d_video) |

---

## How to Run

> All commands below use the pyenv Python. You can set a shorthand:
> ```bash
> PYTHON=/Users/abhisekmaddi/.pyenv/versions/3.12.4/bin/python3
> ```

---

### 1. Full Perception Pipeline (all 718 frames)
Runs the complete pipeline and prints a verification report.
```bash
$PYTHON main_pipeline.py
```
**Expected output:** Detection counts, classification breakdown, track statistics,
proxy metric scores. Takes ~3–5 minutes.

```bash
# Limit to first 200 frames for a quick test
$PYTHON main_pipeline.py --frames 200
```

---

### 2. Cinematic Real-Data Video ★ (recommended)
Generates `lidar_cinematic_real.mp4` — real Blickfeld Cube 1 data rendered with
cinematic style (black background · jet colourmap · neon bounding boxes).

```bash
$PYTHON generate_cinematic_real.py
```

**Output:** `lidar_cinematic_real.mp4`
- Resolution: 1920 × 1080
- Duration: ~72 seconds (718 frames @ 10 fps)
- Size: ~140–210 MB
- Render time: ~2 minutes

**What you see:**
- Real LiDAR point cloud coloured by distance (blue = close, red = far)
- Neon green boxes = PEDESTRIAN detections (labelled **PED**)
- Neon red-orange boxes = VEHICLE detections (labelled **VEH**)
- Active object counter (bottom-right): `VEH: N` and `PED: N`
- Distance colourbar (left) · Spinning LiDAR indicator (top-right)

---

### 3. 3-D Matplotlib Tracking Video
Ground-level 3-D view rendered with matplotlib (Blickfeld viewer style).

```bash
$PYTHON generate_3d_video.py
```

**Output:** `lidar_3d_tracking.mp4` (~31 MB, 718 frames, 10 fps)

```bash
# Quick 50-frame test
$PYTHON generate_3d_video.py --frames 50 --output test_3d.mp4
```

---

### 4. Synthetic Cinematic Demo Video
Urban simulation video (NOT real data — for visual style demonstration only).

```bash
$PYTHON generate_cinematic_video.py
# Default: 60 seconds
$PYTHON generate_cinematic_video.py --duration 15 --output demo_15s.mp4
```

**Output:** `lidar_cinematic.mp4` (~450 MB, 1200 frames @ 20 fps)

---

### 5. Verification / Performance Analysis
Prints honest proxy metrics — no fabricated accuracy numbers.

```bash
$PYTHON performance_analysis.py
```

---

### 6. Data Loader Test
Validates that the CSV files load correctly.

```bash
$PYTHON data_loader.py
```

---

## Execution Order (First Time Setup)

```
Step 1 — Verify data is present
    ls "Lider datasets/"
    → Should show 4 folders (part_1 … part_4)

Step 2 — Install dependencies
    $PYTHON -m pip install -r requirements.txt

Step 3 — Run full pipeline (prints report)
    $PYTHON main_pipeline.py

Step 4 — Generate cinematic real-data video
    $PYTHON generate_cinematic_real.py

Step 5 — (Optional) Generate 3-D tracking video
    $PYTHON generate_3d_video.py
```

---

## Pipeline Parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `min_range` | 2.0 m | Remove near-field sensor artefacts |
| `max_range` | 100.0 m | Effective range for this dataset |
| `voxel_size` | 0.15 m | Uniform spatial resolution |
| `ground_threshold` | 0.25 m | RANSAC ground-plane inlier tolerance |
| `eps` (DBSCAN) | **0.8 m** | Neighbourhood radius validated on dataset |
| `min_samples` | 8 | Minimum dense-region size |
| `sensor_fps` | 10 Hz | Blickfeld Cube 1 frame rate |
| `max_age` | 8 frames | Track kept without update before deletion |
| `min_hits` | 2 frames | Minimum detections before track confirmed |
| `static_speed_threshold` | 0.4 m/s | Speed below which object is reclassified as static |

---

## Classification Thresholds

### VEHICLE
| Dimension | Range |
|-----------|-------|
| Length | 2.0 – 8.0 m |
| Width | 1.3 – 3.0 m |
| Height | 1.0 – 3.5 m |
| Footprint (L × W) | ≤ 18 m² |
| Aspect ratio (L/W) | ≤ 5.0 |
| Height hard cap | ≤ 4.0 m |

### PEDESTRIAN
| Dimension | Range |
|-----------|-------|
| Length | 0.2 – 1.2 m |
| Width | 0.2 – 1.2 m |
| Height | 1.2 – 2.2 m |
| H/W ratio | ≥ 1.5 |

### STATIC_STRUCTURE (triggered if ANY condition holds)
- Footprint > 18 m² (oversized)
- Height > 4.0 m (building)
- L/W ratio > 6.0 (wall / fence shape)

### Velocity-Based Reclassification (in Tracker)
Tracks with average speed < **0.4 m/s** over ≥ 8 consecutive frames are
automatically reclassified from VEHICLE → **STATIC_STRUCTURE**.
This catches bollards, walls, and fences that pass geometric filters.

---

## Issues Fixed from Professor Feedback

| Feedback Issue | Resolution |
|----------------|------------|
| MOTA 97.9% claimed without ground truth | Removed — MOTA is undefined without GT |
| Classification accuracy 97.9% claimed | Removed — replaced with distribution statistics |
| Only 50 of 718 frames processed | Fixed — all 718 frames now processed |
| Walls/buildings mis-labelled as vehicles | Fixed — static pre-filter + velocity reclassification |
| Parameter mismatch (text eps=0.5 vs code eps=0.8) | Fixed — both now use eps=0.8 |
| Numerical thresholds never stated | Fixed — all thresholds documented in code and README |
| Missing 3-D visualisation video | Fixed — three video outputs now available |

---

## Dataset

**Sensor:** Blickfeld Cube 1 solid-state LiDAR
**Recording:** 2020-11-25, 20:01:45 (stationary sensor)
**Format:** CSV with semicolon separator
**Columns:** `X ; Y ; Z ; DISTANCE ; INTENSITY ; POINT_ID ; RETURN_ID ; AMBIENT ; TIMESTAMP`
**Coordinate frame:** Y = forward (depth), X = lateral, Z = vertical
**Total frames:** 718 · **Points per frame:** ~15,000–20,000 · **Range:** 5–100 m
