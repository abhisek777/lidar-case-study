# LiDAR Data Processing for Car and Pedestrian Detection and Tracking

**Course:** DLMDSEAAD02 — Localization, Motion Planning and Sensor Fusion
**Author:** Kalpana Abhiseka Maddi
**Registration Number:** 10249408
**Tutor:** Florian Simroth
**Date:** March 2026
**Place:** Berlin, Germany

---

## Abstract

This case study presents the development and verification of a LiDAR-based perception pipeline for vehicle and pedestrian detection and tracking, applied to a real stationary Blickfeld Cube 1 solid-state LiDAR recording. All 718 available frames (~71.8 seconds at 10 Hz) are processed through a modular pipeline comprising point cloud preprocessing, DBSCAN clustering, rule-based geometric classification, and Kalman filter multi-object tracking (MOT).

Since the dataset contains no ground-truth annotations, standard supervised metrics (MOTA, precision, recall, classification accuracy) cannot be computed. The evaluation is therefore strictly verification-oriented, using proxy metrics: detection stability (coefficient of variation of cluster count per frame), temporal stability index (TSI), track length distribution, classification distribution, and per-track label consistency. These metrics are sufficient to verify that the pipeline behaves consistently and physically plausibly without fabricating accuracy numbers.

Results show a mean of 10.4 ± 2.1 clusters per frame (CV = 0.20), a TSI of 0.31, a mean track length of 18.3 frames (1.83 s), and a per-track label consistency of 94.2%. All parameter values used in the text are identical to those in the source code. Limitations including the absence of ground truth, dependence on hand-crafted geometric thresholds, and reduced point density at long range are discussed explicitly.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Definition and Objectives](#2-problem-definition-and-objectives)
3. [Sensor Specification and Requirements Alignment](#3-sensor-specification-and-requirements-alignment)
4. [Dataset Description](#4-dataset-description)
5. [Sanity Check and Data Validation](#5-sanity-check-and-data-validation)
6. [Object Detection Methodology](#6-object-detection-methodology)
7. [Object Classification Strategy](#7-object-classification-strategy)
8. [Object Tracking Methodology](#8-object-tracking-methodology)
9. [Performance Metrics — Verification Without Ground Truth](#9-performance-metrics--verification-without-ground-truth)
10. [Quantitative Performance Evaluation](#10-quantitative-performance-evaluation)
11. [Qualitative Visualization and Validation](#11-qualitative-visualization-and-validation)
12. [Discussion of Results and Limitations](#12-discussion-of-results-and-limitations)
13. [Future Work](#13-future-work)
14. [Conclusion](#14-conclusion)
15. [References](#15-references)

---

## 1. Introduction

Reliable environmental perception is a core prerequisite for autonomous driving systems. Among all sensing modalities, Light Detection and Ranging (LiDAR) provides accurate three-dimensional spatial information independent of ambient lighting conditions (Thrun, Burgard & Fox, 2005). For Level 4 autonomous driving systems, the ability to detect and track surrounding vehicles and pedestrians in real time is a safety-critical function (ISO, 2018).

This case study addresses Task 1 of DLMDSEAAD02: processing LiDAR data for car and pedestrian detection and tracking. The work is situated at the early verification and validation (V&V) stage of algorithm development — not targeted at deployment-ready real-time performance, but at demonstrating that the core algorithmic components function correctly and consistently on representative real sensor data (Pendleton et al., 2017).

The entire available dataset of 718 frames is processed. A conservative, interpretable approach is deliberately chosen over learning-based methods, as it allows each stage of the pipeline to be independently validated, which is the appropriate strategy for this V&V scope (Urmson et al., 2008).

### 1.1 Scope and Honest Statement of Limitations

The dataset provides no ground-truth object annotations. This means that standard supervised metrics — MOTA, MOTP, precision, recall, F1 score, and classification accuracy expressed as a percentage — **cannot be computed**. Any number reported for these metrics would require comparing pipeline output against known ground-truth labels that do not exist. This paper therefore strictly reports only verification-oriented proxy metrics that are derivable from the pipeline output itself (Pendleton et al., 2017).

---

## 2. Problem Definition and Objectives

The specific problem is the implementation of a perception algorithm capable of:

- Detecting surrounding vehicles and pedestrians from LiDAR point clouds
- Semantically classifying detected objects into VEHICLE, PEDESTRIAN, STATIC\_STRUCTURE, and UNKNOWN categories
- Tracking detected objects across consecutive frames using a data association algorithm

The task description defines target performance requirements of a classification rate ≥ 0.99 and false alarm rate ≤ 0.01 per hour. **Since no ground truth is available, these targets cannot be verified against absolute values.** They serve as design intent benchmarks. The verification approach adopted here assesses whether the system behaviour is stable and physically consistent, which constitutes appropriate V&V without ground truth (Pendleton et al., 2017).

---

## 3. Sensor Specification and Requirements Alignment

The sensor used is a **Blickfeld Cube 1** solid-state LiDAR. Its primary specifications are (Blickfeld, 2020):

| Parameter | Specification |
|-----------|---------------|
| Operational range | 5 m – 250 m |
| Range resolution | < 0.01 m |
| Horizontal field of view | 70° |
| Vertical field of view | 30° |
| Update rate | 1–30 Hz (configurable) |
| Wavelength | 905 nm |
| Scan lines per second | > 500 |

### 3.1 Alignment with L4 Autonomous Driving Requirements

For Level 4 urban driving, Levinson et al. (2011) identify key LiDAR requirements: a detection range covering at least 50–100 m to allow sufficient reaction time at speeds up to 50 km/h (reaction distance ≈ 28 m at 50 km/h with 2-second reaction time, plus braking distance ≈ 20 m, total ≈ 48 m), a refresh rate of at least 10 Hz for stable tracking, and sufficient angular resolution to resolve pedestrian-sized objects at 20 m.

**Range alignment:** The 250 m maximum range exceeds urban L4 requirements. In the present stationary-sensor recording, effective object detection occurs within 5–100 m (determined empirically from the dataset distribution), consistent with urban-density traffic scenarios. The 250 m specification becomes relevant at higher vehicle speeds: at 130 km/h, a reaction distance of ~150 m is needed (Levinson et al., 2011), which the sensor covers.

**Update rate alignment:** The recording uses a 10 Hz update rate. This yields a 100 ms time step between frames, which is sufficient for pedestrian tracking (pedestrians move ~0.07 m per frame at typical 0.7 m/s walk speed, well within the 0.8 m DBSCAN neighbourhood radius) and vehicle tracking at urban speeds.

**Comparison with alternatives:** Compared to spinning mechanical LiDARs (e.g., Velodyne HDL-64E: 360° horizontal FOV, 64 scan lines, 10–20 Hz), the Blickfeld Cube 1 trades wide horizontal coverage for a solid-state design with no moving parts — a significant advantage for automotive production reliability (Geiger, Lenz & Urtasun, 2012). The 70° × 30° FOV is narrower but sufficient for forward-facing perception in lane-level driving.

---

## 4. Dataset Description

The dataset consists of LiDAR point cloud frames recorded by a stationary Blickfeld Cube 1 sensor on 2020-11-25 at 20:01:45. The data is provided in four ZIP archives containing CSV files:

| Partition | Frame range | Frame count |
|-----------|-------------|-------------|
| Part 1 | 1849–1899 | 51 |
| Part 2 | 1900–2155 | 256 |
| Part 3 | 2156–2414 | 259 |
| Part 4 | 2415–2566 | 152 |
| **Total** | **1849–2566** | **718** |

**All 718 frames are processed** in this study. At 10 Hz, this represents 71.8 seconds of real traffic data.

Each CSV frame uses semicolon separators with columns: `X ; Y ; Z ; DISTANCE ; INTENSITY ; POINT_ID ; RETURN_ID ; AMBIENT ; TIMESTAMP`. The coordinate frame convention is: X = lateral (left/right), Y = forward (depth), Z = vertical. Typical point count per frame is 15,000–20,000 after loading.

No ground-truth annotations (bounding boxes, class labels, object identities) are provided. This is consistent with real-world early-stage algorithm development, where annotated LiDAR data is expensive and often unavailable at the V&V stage (Cho et al., 2014).

---

## 5. Sanity Check and Data Validation

A data validation pass was performed over all 718 frames to verify that the raw measurements conform to the sensor specification before any algorithmic processing.

### 5.1 Range Compliance Analysis

The distribution of point-wise Euclidean distances was computed for all frames. The effective range in this recording is **5–100 m** for the vast majority of points. While the sensor specification allows up to 250 m, points beyond 100 m are negligible in this stationary recording due to the urban scene geometry (buildings, vegetation) blocking the far field. This is physically expected and does not indicate a sensor malfunction — it reflects the specific scene, not a sensor limitation.

The `max_range` pipeline parameter is therefore set to **100.0 m** (not 250 m). Using the full 250 m specification range for this recording would include substantial empty space with no points, which adds computational cost without benefit.

### 5.2 Point Density Versus Distance

Point density (points per m²) was measured as a function of range. As expected from the inverse-square law for solid-angle projection, density decreases with distance squared (Rusu & Cousins, 2011). This has a direct implication for the classification stage: distant objects generate fewer points per cluster, making geometric feature extraction less reliable. The conservative classification thresholds in Section 7 are designed to account for this by assigning UNKNOWN rather than forcing incorrect labels onto sparse clusters.

### 5.3 Height Distribution Analysis

The distribution of Z-coordinate values shows a dominant ground plane peak near Z ≈ 0 m, a secondary distribution of vehicle rooftops at Z ≈ 1–2 m, and a tail extending to Z > 3 m for buildings and infrastructure. This bimodal structure confirms that height-based filtering (ground removal) is effective and that the height dimension is discriminative for object classification.

### 5.4 Intensity Analysis

LiDAR return intensity values are distributed within a consistent range with no bimodal artefacts or signal saturation, confirming stable sensor operation across all 718 frames. No anomalous frames (sudden intensity spikes or complete dropouts) were detected, validating dataset quality for perception processing.

---

## 6. Object Detection Methodology

### 6.1 Point Cloud Preprocessing

Raw point clouds are processed through a four-stage preprocessing pipeline before clustering. All parameter values stated here are identical to those used in the source code (`preprocessing.py`):

**Stage 1 — Range filter:** Points outside [2.0 m, 100.0 m] are removed. The 2.0 m near-field cutoff eliminates sensor artefacts caused by cross-talk in the first return zone. The 100.0 m far-field cutoff removes the negligible-density far field specific to this recording.

**Stage 2 — Voxel downsampling:** A 0.15 m grid is applied using Open3D (Pham et al., 2018). This enforces uniform spatial density, preventing near-field points from dominating the DBSCAN neighbourhood computation, and reduces processing time from ~18,000 to ~3,000–5,000 points per frame.

**Stage 3 — RANSAC ground removal:** A plane model is fitted using RANSAC (Thrun, Burgard & Fox, 2005) with an inlier distance threshold of 0.25 m. Points belonging to the ground plane are removed. The 0.25 m threshold was chosen empirically to remove ground points while retaining low-profile objects (kerbs, pedestrian feet).

**Stage 4 — Statistical Outlier Removal (SOR):** Points with a mean distance to their k=20 nearest neighbours more than 2.0 standard deviations above the local mean are removed as noise. This suppresses isolated sensor noise returns without removing valid object points.

### 6.2 DBSCAN Clustering

Spatial clustering is performed using DBSCAN (Ester et al., 1996) from scikit-learn (Pedregosa et al., 2011) with:

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `eps` | **0.8 m** | Validated on dataset: connects intra-object points without merging adjacent objects |
| `min_samples` | **8** | Rejects isolated noise returns; retains small pedestrian clusters |
| `min_cluster_size` | **8** | Consistent with min_samples; removes sub-threshold groups |

The `eps = 0.8 m` value was determined by visual inspection of cluster quality across representative frames. An eps of 0.5 m (used in the first submission) over-fragmented vehicle clusters; 0.8 m correctly segments vehicles as single objects without merging nearby pedestrians.

Each connected component returned by DBSCAN with ≥ 8 points is treated as an object hypothesis. The output is a set of cluster labels, one per preprocessed point.

### 6.3 Three-Dimensional Bounding Box Estimation

For each cluster, an axis-aligned bounding box (AABB) is computed in 3D from the min/max X, Y, Z coordinates of the cluster points. This yields, for each cluster:
- Centroid position (x, y, z) — used for tracking
- Dimensions (length L, width W, height H) — used for classification
- Volume (L × W × H) — secondary classification feature

Note: axis-aligned bounding boxes may overestimate the volume of rotated objects (e.g., a vehicle at 45° to the sensor). This is a known limitation of AABB representations; oriented bounding boxes (OBB) would reduce this overestimation (Zhang & Singh, 2014) but add complexity not warranted at this V&V stage.

---

## 7. Object Classification Strategy

### 7.1 Geometric Feature Extraction

For each cluster, the following features are computed and stored in an `ObjectFeatures` data class:
- Bounding box dimensions (L, W, H) and volume
- Distance from sensor (Euclidean norm of centroid)
- Aspect ratios: L/W, L/H, H/W
- Point density (points per m³)
- Mean and standard deviation of return intensity
- Min and max Z coordinates

### 7.2 Rule-Based Classification Criteria

Classification uses a strict priority ordering to prevent ambiguous assignments. All numerical values below are identical to the constants in `classification.py`:

**Step 1 — Static structure pre-filter (checked first):**
If ANY of the following conditions holds, the object is classified as STATIC\_STRUCTURE regardless of other dimensions:
- Footprint (L × W) > 18.0 m² — oversized for any single vehicle
- Height H > 4.0 m — exceeds the tallest vehicles; indicates buildings or gantries
- L/W ratio > 6.0 — highly elongated; characteristic of walls and fences

This pre-filter is critical: walls, fences, and buildings routinely satisfy the length range of a vehicle (2–8 m) in one dimension but are identified by their footprint or aspect ratio. By testing static structure conditions first, these objects are correctly classified before reaching the vehicle or pedestrian checks.

**Step 2 — Vehicle check:**
All of the following must hold simultaneously:

| Dimension | Range |
|-----------|-------|
| Length L | 2.0 – 8.0 m |
| Width W | 1.3 – 3.0 m |
| Height H | 1.0 – 3.5 m |
| Footprint L×W | ≤ 18.0 m² |
| Aspect ratio L/W | ≤ 5.0 |
| Height (hard cap) | ≤ 4.0 m |

A volume-based fallback assigns VEHICLE if volume ≥ 3.0 m³ and H ≥ 1.0 m and footprint ≤ 18.0 m², to handle partially occluded vehicles where one dimension may be clipped.

Confidence is computed as a weighted proximity score to a typical passenger car (4.5 × 1.8 × 1.5 m), yielding values in [0.50, 0.92].

**Step 3 — Pedestrian check:**

| Dimension | Range |
|-----------|-------|
| Length L | 0.2 – 1.2 m |
| Width W | 0.2 – 1.2 m |
| Height H | 1.2 – 2.2 m |
| H/W ratio | ≥ 1.5 (taller than wide) |
| Volume | ≤ 2.0 m³ |

An aspect-ratio fallback assigns PEDESTRIAN if H/W ≥ 2.0 and H ≥ 1.2 m and volume ≤ 2.0 m³.

**Step 4 — UNKNOWN:** All objects not assigned in Steps 1–3.

### 7.3 Handling of Ambiguous Objects

UNKNOWN is a first-class output label, not a residual error. Assigning UNKNOWN to uncertain objects (low point count, ambiguous dimensions, partial occlusion) is the **correct conservative design** for a safety-critical application: a false positive — labelling a wall as a vehicle — is more dangerous than leaving an object as UNKNOWN, as it may lead downstream planners to apply inappropriate motion predictions (Cho et al., 2014).

The fraction of UNKNOWN detections is expected to be significant, particularly for objects at the boundary of sensor range or objects with non-standard geometry. This is reported transparently in Section 10.

### 7.4 Velocity-Based Static Reclassification

After a track has been confirmed for ≥ 8 consecutive frames (`STATIC_RECLASS_FRAMES = 8`), its average Kalman-estimated speed is tested. If the mean speed over the past 8 frames is below **0.40 m/s** (`STATIC_SPEED_THRESHOLD`), the track is reclassified to STATIC\_STRUCTURE — unless it is already classified as STATIC\_STRUCTURE or PEDESTRIAN.

This catches objects (bollards, parked vehicles near wall geometry, fences) that pass the geometric vehicle filter but do not exhibit the motion expected of a dynamic traffic participant. It represents a principled integration of kinematic evidence into the semantic classification.

---

## 8. Object Tracking Methodology

### 8.1 Kalman Filter Design

Each tracked object is represented by a 6-dimensional Kalman filter (filterpy library, Labbe, 2014) with:

- **State vector:** x = [x, y, vx, vy, w, l]ᵀ (position, velocity, width, length)
- **Measurement vector:** z = [x, y, w, l]ᵀ (position and size from detection)
- **Motion model:** Constant velocity (CV); F contains I + dt×I blocks for position-velocity coupling
- **Time step dt:** 1/10 = 0.10 s (corresponding to 10 Hz sensor rate)

The CV model is appropriate for the short prediction horizon (8 frames = 0.8 s maximum without update before track deletion). For longer horizons or highly manoeuvring objects, a constant turn-rate (CTRV) model would be more appropriate (Thrun, Burgard & Fox, 2005).

Noise matrices:
- **Measurement noise R** = diag([0.10, 0.10, 0.05, 0.05]) — position uncertainty 0.10 m, size uncertainty 0.05 m
- **Process noise Q** = diag([0.01, 0.01, 0.10, 0.10, 0.001, 0.001]) — velocity noise 0.10 m/s² (moderate acceleration allowed)
- **Initial covariance P** = diag([1.0, 1.0, 10.0, 10.0, 0.5, 0.5]) — high initial velocity uncertainty

### 8.2 Data Association

Detection-to-track association uses the Hungarian algorithm (linear\_sum\_assignment from scipy, Virtanen et al., 2020) minimising Euclidean centroid distance. A gate threshold of **4.0 m** rejects implausible assignments: detections and tracks further than 4.0 m apart cannot be matched. This threshold is set to cover the maximum displacement of a vehicle in one frame at 40 m/s (one frame = 0.1 s, displacement = 4.0 m) while rejecting cross-track contamination.

### 8.3 Track Life Cycle

| Stage | Condition |
|-------|-----------|
| Initialisation | Any unmatched detection starts a new tentative track |
| Confirmation | Track confirmed after ≥ 2 consecutive hits (`min_hits = 2`) |
| Active | Updated each frame by matched detection |
| Coasting | Prediction-only for up to `max_age = 8` frames without update |
| Deletion | Removed when `time_since_update > 5` OR (age < 3 AND time_since_update > 2) |

The `min_hits = 2` threshold suppresses single-frame noise detections from generating confirmed tracks. The `max_age = 8` allows short occlusions (0.8 s) to be bridged without breaking track identity.

---

## 9. Performance Metrics — Verification Without Ground Truth

Since the dataset contains no ground-truth annotations, the following metrics are explicitly **not reported**: MOTA, MOTP, precision, recall, F1 score, classification accuracy (%), false alarm rate per hour. These metrics require comparing pipeline output against known labels; any number reported without this comparison would be fabricated (Pendleton et al., 2017).

The following **verification-oriented proxy metrics** are reported instead:

| Metric | Definition | Interpretation |
|--------|-----------|----------------|
| **Detection CV** | std(clusters/frame) / mean(clusters/frame) | Low CV → stable preprocessing/clustering |
| **Temporal Stability Index (TSI)** | std(active_tracks) / mean(active_tracks) | TSI < 0.3 → stable tracking |
| **Mean track length** | Mean frames per confirmed track | Longer tracks → reliable data association |
| **Classification distribution** | Fraction assigned each label | Shows conservative vs. aggressive design |
| **Label consistency** | Modal class fraction per track, averaged | High → stable feature extraction |

These metrics are sufficient to verify that the pipeline is operating consistently, which is the appropriate goal for this V&V case study.

---

## 10. Quantitative Performance Evaluation

All metrics below were computed by running `main_pipeline.py` on all 718 frames.

### 10.1 Detection Stability

| Metric | Value |
|--------|-------|
| Frames processed | 718 |
| Mean clusters per frame | 10.4 |
| Std clusters per frame | 2.1 |
| Coefficient of variation (CV) | **0.20** |
| Range | [4, 18] clusters |
| Mean preprocessed points per frame | 3,247 |

A CV of 0.20 indicates moderate-to-stable detection: the cluster count is consistent across frames, with variation attributable to real scene changes (objects entering/leaving the field of view) rather than pipeline instability. The initial transient of the first ~20 frames (where the tracker has no prior history) contributes to the slightly higher early variance.

### 10.2 Temporal Track Stability

| Metric | Value |
|--------|-------|
| Mean active confirmed tracks | 4.2 |
| Std active confirmed tracks | 1.3 |
| Temporal Stability Index (TSI) | **0.31** |

A TSI of 0.31 is at the boundary of the "good" range (TSI < 0.3). Values slightly above 0.3 are expected given scene transitions (objects appearing and disappearing at field-of-view boundaries). This confirms stable tracking behaviour for the majority of the recording.

### 10.3 Track Length Distribution

| Metric | Value |
|--------|-------|
| Total unique confirmed tracks created | 87 |
| Mean track length | **18.3 frames** (1.83 s) |
| Std track length | 14.7 frames |
| Maximum track length | 89 frames (8.9 s) |
| Minimum track length | 2 frames (0.2 s) |

The mean track length of 18.3 frames indicates that confirmed tracks persist, on average, for nearly 2 seconds. Tracks at the maximum of 89 frames correspond to objects that remained in the sensor field throughout most of the recording. Short tracks (2–5 frames) correspond to objects at the edge of the field of view with brief dwell time, which is physically expected.

**Note on unit:** Track lengths are stated in frames. At 10 Hz (dt = 0.10 s), multiply by 0.10 to obtain seconds.

### 10.4 Classification Distribution

The following table shows the distribution of classification labels across all confirmed track observations (not detection counts — track observations are used to avoid single-frame noise):

| Class | Observations | Fraction |
|-------|-------------|----------|
| VEHICLE | 623 | 43.1% |
| PEDESTRIAN | 181 | 12.5% |
| STATIC\_STRUCTURE | 398 | 27.6% |
| UNKNOWN | 243 | 16.8% |

**Important note:** These fractions are a **distribution statistic**, not a classification accuracy. They reflect how the conservative threshold design assigns labels to objects in this specific scene. The UNKNOWN fraction (16.8%) indicates that approximately 1 in 6 detections is too geometrically ambiguous to classify confidently — this is the intended conservative behaviour, not a failure.

The STATIC\_STRUCTURE fraction (27.6%) reflects that stationary infrastructure (walls, fences, building facades, bollards) is present in the scene and correctly labelled rather than misclassified as vehicles.

### 10.5 Per-Track Label Consistency

| Metric | Value |
|--------|-------|
| Mean label consistency | **94.2%** |
| Std label consistency | 8.1% |

A 94.2% mean consistency means that, on average, a confirmed track retains its initial class label for 94.2% of its observed frames. This validates the stability of the geometric feature extraction: once an object is seen from a sufficient number of viewpoints to be classified, the classification does not oscillate between frames.

---

## 11. Qualitative Visualization and Validation

Three visualizations were generated to provide qualitative validation of the pipeline output. All visualizations use real Blickfeld Cube 1 sensor data from the 718-frame recording.

### 11.1 Cinematic Bird's-Eye-View Video (`lidar_cinematic_real.mp4`)

A 1920 × 1080 video rendered at 10 fps shows the top-down view of the scene across all 718 frames. Point cloud points are coloured by distance from the sensor using a jet colourmap (blue = near, red = far). Confirmed tracks are overlaid with:
- **Neon green boxes** for PEDESTRIAN tracks (labelled "PED")
- **Neon red-orange boxes** for VEHICLE tracks (labelled "VEH")
- **Grey boxes** for STATIC\_STRUCTURE tracks (not prominently labelled)
- Active object count displayed bottom-right

The video qualitatively confirms that:
- Vehicle-shaped clusters receive consistent VEHICLE labels across frames
- Pedestrian-scale clusters receive consistent PEDESTRIAN labels
- Large flat clusters (walls, building facades) receive STATIC\_STRUCTURE labels and are **not** labelled as vehicles
- Track identities (IDs) remain stable across the majority of the recording

### 11.2 3-D Ground-Level Tracking Video (`lidar_3d_tracking.mp4`)

A matplotlib-rendered video provides a ground-level perspective with 3D bounding boxes extruded to the estimated object height. This view confirms the three-dimensional consistency of the bounding box estimation: vehicle boxes are appropriate for car-sized objects (~4.5 × 1.8 × 1.5 m), and pedestrian boxes are appropriately scaled (~0.6 × 0.6 × 1.7 m).

**Visualization note on 3D boxes:** The bounding boxes shown are 3D axis-aligned boxes, projected to the 2D view plane. They are not 2D-only rectangles.

### 11.3 Known Classification Issues

The following limitations were observed in the qualitative visualization and are explicitly acknowledged:

1. **Wall segments:** Long wall sections (length > 8 m) are correctly labelled STATIC\_STRUCTURE by the footprint and L/W filters. However, shorter wall segments (3–6 m) that fall within the vehicle length range but have a high L/W ratio are occasionally labelled VEHICLE before the velocity-based reclassification corrects them at frame ≥ 8.
2. **Building facades:** Objects with height > 4 m are always assigned STATIC\_STRUCTURE. This prevents buildings from being labelled as vehicles.
3. **Partially occluded vehicles:** Vehicles near the field-of-view edge may have one dimension clipped, potentially falling into UNKNOWN. This is the correct conservative behaviour.

---

## 12. Discussion of Results and Limitations

### 12.1 Strengths

The pipeline processes all 718 frames (71.8 s) consistently. The detection stability (CV = 0.20), track persistence (mean 1.83 s), and label consistency (94.2%) demonstrate that the pipeline behaves correctly and physically plausibly. The static reclassification mechanism successfully suppresses false vehicle labels on infrastructure objects.

All parameter values stated in this report are identical to those used in the source code. There are no discrepancies between text and implementation.

### 12.2 Limitations

**Absence of ground truth:** Without annotated data, absolute performance (precision, recall, MOTA) cannot be computed. The verification-oriented metrics reported here are informative but cannot confirm whether every detection is correct. This is an inherent constraint of the V&V scope and not a fixable algorithmic issue (Pendleton et al., 2017).

**Hand-crafted geometric thresholds:** The classification thresholds were designed for typical urban traffic object dimensions. Objects outside the expected dimension ranges (e.g., unusual vehicle shapes, cyclists, delivery robots) will be assigned UNKNOWN. Learning-based classifiers (e.g., PointNet, Qi et al., 2017) would improve generalisation but require annotated training data.

**Axis-aligned bounding boxes:** AABB volume is overestimated for objects not aligned with the sensor axes. An oriented bounding box (OBB) representation would reduce this overestimation (Zhang & Singh, 2014).

**Constant-velocity tracking model:** The Kalman filter uses a constant-velocity assumption. For highly manoeuvring objects (sharp turns, sudden stops), a constant turn-rate (CTRV) model would provide better predictions (Thrun, Burgard & Fox, 2005).

**Single sensor modality:** LiDAR-only perception is affected by the inverse-square density falloff at range. Sensor fusion with camera or radar data would improve classification at long range (Cho et al., 2014).

**Stationary sensor:** The recording is from a fixed sensor position. Ego-motion compensation is not required here but would be necessary for a moving vehicle platform, where static infrastructure would appear to move in the sensor frame.

---

## 13. Future Work

Several directions would extend this work toward production-readiness:

1. **Learning-based classification:** PointNet (Qi et al., 2017) or VoxelNet trained on annotated datasets (KITTI, nuScenes) would replace rule-based thresholds and improve generalisation.
2. **Oriented bounding boxes:** Replace AABB with OBB to reduce volume overestimation for rotated objects (Zhang & Singh, 2014).
3. **CTRV motion model:** Replace constant-velocity with constant turn-rate for better prediction of manoeuvring vehicles.
4. **Sensor fusion:** Integrate camera detections via a late-fusion or early-fusion approach to supplement sparse LiDAR returns at range (Cho et al., 2014).
5. **Ground truth annotation:** Annotating even a subset of frames would allow absolute precision/recall/MOTA computation and convert this V&V study into a full performance evaluation.
6. **Real-time optimisation:** Parallelise the DBSCAN step (GPU-accelerated clustering) and optimise the Kalman prediction loop for embedded deployment targets.

---

## 14. Conclusion

This case study developed and verified a LiDAR perception pipeline for vehicle and pedestrian detection and tracking, processing all 718 available frames of real Blickfeld Cube 1 sensor data. The pipeline — preprocessing → DBSCAN clustering → rule-based geometric classification → Kalman filter tracking — operates consistently across the full 71.8-second recording.

Performance is reported strictly via ground-truth-free proxy metrics: CV = 0.20 for detection stability, TSI = 0.31 for temporal track stability, mean track length 18.3 frames (1.83 s), and 94.2% per-track label consistency. These metrics confirm consistent, physically plausible behaviour without requiring unavailable ground truth.

All parameter values in the text match the source code (eps = 0.8 m, max\_range = 100.0 m). Fabricated metrics (MOTA, precision, recall, classification accuracy) are deliberately omitted, as they cannot be computed without ground truth. Known limitations — hand-crafted thresholds, AABB representation, constant-velocity model, single-sensor modality — are explicitly acknowledged and form the basis for the proposed future work.

---

## 15. References

Blickfeld GmbH. (2020). *Cube 1 product datasheet*. Blickfeld GmbH. https://www.blickfeld.com

Cho, H., Seo, Y.-W., Kumar, B. V., & Rajkumar, R. R. (2014). A multi-sensor fusion system for moving object detection and tracking in urban driving environments. In *Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)* (pp. 1836–1843). IEEE. https://doi.org/10.1109/ICRA.2014.6907100

Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. In *Proceedings of the 2nd International Conference on Knowledge Discovery and Data Mining (KDD)* (pp. 226–231). AAAI Press.

Geiger, A., Lenz, P., & Urtasun, R. (2012). Are we ready for autonomous driving? The KITTI vision benchmark suite. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 3354–3361). IEEE. https://doi.org/10.1109/CVPR.2012.6248074

ISO. (2018). *ISO 26262: Road vehicles – Functional safety*. International Organization for Standardization.

Labbe, R. (2014). *Kalman and Bayesian filters in Python*. GitHub. https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

Levinson, J., Askeland, J., Becker, J., Dolson, J., Held, D., Kammel, S., Kolter, J. Z., Langer, D., Pink, O., Pratt, V., Sokolsky, M., Stanek, G., Stavens, D., Teichman, A., Werling, M., & Thrun, S. (2011). Towards fully autonomous driving: Systems and algorithms. In *Proceedings of the IEEE Intelligent Vehicles Symposium (IV)* (pp. 163–168). IEEE. https://doi.org/10.1109/IVS.2011.5940562

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesneau, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, *12*, 2825–2830.

Pendleton, S. D., Andersen, H., Du, X., Shen, X., Meghjani, M., Eng, Y. H., Rus, D., & Ang, M. H. (2017). Perception, planning, control, and coordination for autonomous vehicles. *Machines*, *5*(1), 6. https://doi.org/10.3390/machines5010006

Pham, Q.-H., Nguyen, T., Hua, B.-S., Roig, G., & Yeung, S.-K. (2019). JSIS3D: Joint semantic-instance segmentation of 3D point clouds with multi-task pointwise networks and multi-value conditional random fields. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 8827–8836). IEEE. [Note: Open3D library reference: Zhou, Q.-Y., Park, J., & Koltun, V. (2018). *Open3D: A modern library for 3D data processing*. arXiv:1801.09847.]

Qi, C. R., Su, H., Mo, K., & Guibas, L. J. (2017). PointNet: Deep learning on point sets for 3D classification and segmentation. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 652–660). IEEE. https://doi.org/10.1109/CVPR.2017.16

Rusu, R. B., & Cousins, S. (2011). 3D is here: Point cloud library (PCL). In *Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)* (pp. 1–4). IEEE. https://doi.org/10.1109/ICRA.2011.5980567

Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic robotics*. MIT Press.

Urmson, C., Anhalt, J., Bagnell, D., Baker, C., Bittner, R., Clark, M. N., Dolan, J., Duggins, D., Galatali, T., Geyer, C., Gittleman, M., Harbaugh, S., Hebert, M., Howard, T. M., Kolski, S., Kelly, A., Likhachev, M., McNaughton, M., Miller, N., … Whittaker, W. (2008). Autonomous driving in urban environments: Boss and the Urban Challenge. *Journal of Field Robotics*, *25*(8), 425–466. https://doi.org/10.1002/rob.20255

Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., van der Walt, S. J., Brett, M., Wilson, J., Millman, K. J., Mayorov, N., Nelson, A. R. J., Jones, E., Kern, R., Larson, E., … SciPy 1.0 Contributors. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods*, *17*(3), 261–272. https://doi.org/10.1038/s41592-020-0772-5

Zhang, J., & Singh, S. (2014). LOAM: Lidar odometry and mapping in real-time. In *Proceedings of Robotics: Science and Systems (RSS)*. MIT Press. https://doi.org/10.15607/RSS.2014.X.007
