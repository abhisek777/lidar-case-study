# LiDAR Object Detection and Tracking Pipeline

A complete perception pipeline for autonomous driving that processes Blickfeld Cube 1 LiDAR point cloud data to detect, classify, and track objects (vehicles and pedestrians).

**Course:** Localization, Motion Planning and Sensor Fusion
**Author:** Abhisek Maddi

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Required packages: `numpy`, `pandas`, `open3d`, `scikit-learn`, `filterpy`, `scipy`, `matplotlib`, `tqdm`.

### 2. Run the demo

```bash
python demo_run.py
```

This runs the full pipeline on the 51 CSV frames in `lidar_data/` and saves all output plots to `demo_output/`.

**Custom options:**

```bash
python demo_run.py --data lidar_data --frames 51 --output demo_output
```

| Flag | Description | Default |
|------|-------------|---------|
| `--data` / `-d` | Path to directory with CSV files | `lidar_data` |
| `--frames` / `-n` | Number of frames to process | `51` (all) |
| `--output` / `-o` | Output directory for plots | `demo_output` |

### 3. View results

After running, the `demo_output/` folder contains:

| File | Description |
|------|-------------|
| `detections_frame_000.png` ... | BEV bounding box plots for selected frames |
| `object_trajectories.png` | Trajectories of all tracked objects over time |
| `pipeline_summary.png` | 4-panel summary (detections, tracks, timing, track durations) |

---

## Pipeline Architecture

```
CSV File (Blickfeld Cube 1)
    |
    v
[1] Data Loading          -- data_loader.py
    |                        Reads semicolon-separated CSV (X;Y;Z;DISTANCE;INTENSITY;TIMESTAMP)
    v
[2] Preprocessing         -- preprocessing.py
    |                        Range filtering (5-250 m)
    |                        Voxel grid downsampling (0.1 m)
    |                        RANSAC ground plane removal
    |                        Statistical outlier removal
    v
[3] Clustering (DBSCAN)   -- clustering.py
    |                        eps=0.5 m, min_samples=10
    |                        Groups nearby points into object clusters
    v
[4] Feature Extraction    -- classification.py
    |                        3D bounding box, dimensions, point density
    v
[5] Classification        -- classification.py
    |                        Rule-based: VEHICLE / PEDESTRIAN / UNKNOWN
    |                        Based on bounding box dimensions
    v
[6] Tracking              -- tracking.py
    |                        Kalman Filter multi-object tracker
    |                        Hungarian algorithm for data association
    |                        Consistent object IDs across frames
    v
[7] Visualization         -- visualization.py, demo_run.py
                             Bird's Eye View with bounding boxes
                             Object trajectories
                             Performance statistics
```

---

## Project Structure

```
.
├── demo_run.py                 # <-- MAIN ENTRY POINT: run this
├── main_pipeline.py            # Integrated pipeline class
├── run_complete_pipeline.py    # Extended pipeline with validation
│
├── data_loader.py              # Blickfeld CSV data loading
├── preprocessing.py            # Point cloud preprocessing
├── clustering.py               # DBSCAN object segmentation
├── classification.py           # Rule-based classification
├── tracking.py                 # Kalman filter tracking
├── enhanced_tracking.py        # Enhanced tracking with acceleration model
│
├── visualization.py            # BEV and 3D visualization
├── enhanced_visualization.py   # Extended visualization
├── academic_visualizations.py  # Academic paper figure generation
├── semantic_visualization.py   # Semantic 3D visualization
├── cinematic_visualization.py  # Cinematic video visualization
│
├── lidar_validation.py         # Dataset validation
├── performance_analysis.py     # Performance metrics
├── report_content.py           # Report generation
│
├── requirements.txt            # Python dependencies
├── lidar_data/                 # 51 Blickfeld Cube 1 CSV frames
├── academic_figures/           # Generated academic plots + video
├── semantic_output/            # Semantic visualization output
├── full_pipeline_output/       # Pipeline reports and validation
└── demo_output/                # Output from demo_run.py (generated)
```

---

## CSV Data Format

The LiDAR data comes from a **Blickfeld Cube 1** sensor. Each CSV file represents one frame with semicolon-separated columns:

```
X;Y;Z;DISTANCE;INTENSITY;TIMESTAMP
-0.123;4.567;-0.890;4.612;0.45;1606330905.123
...
```

| Column | Unit | Description |
|--------|------|-------------|
| X | meters | Forward distance |
| Y | meters | Lateral distance |
| Z | meters | Height |
| DISTANCE | meters | Radial distance from sensor |
| INTENSITY | 0-1 | Surface reflectivity |
| TIMESTAMP | seconds | Unix timestamp |

---

## Classification Rules

| Class | Length | Width | Height | Volume |
|-------|--------|-------|--------|--------|
| VEHICLE | 2.0 - 8.0 m | 1.3 - 3.0 m | 1.0 - 3.5 m | >= 3.0 m^3 |
| PEDESTRIAN | 0.2 - 1.2 m | 0.2 - 1.2 m | 1.2 - 2.2 m | <= 2.0 m^3 |
| UNKNOWN | — | — | — | Does not match above |

---

## Alternative Run Methods

### Run the integrated pipeline class directly

```bash
python main_pipeline.py --data lidar_data --frames 10 --output output_dir
```

### Run with full validation and reporting

```bash
python run_complete_pipeline.py --data lidar_data --mode full --output pipeline_output
```

### Generate academic paper figures

```bash
python academic_visualizations.py
```

This generates publication-quality figures in `academic_figures/`.
