# LiDAR Data Processing for Car and Pedestrian Detection and Tracking

**Course:** DLMDSEAAD02 — Localization, Motion Planning and Sensor Fusion
**Author:** Kalpana Abhiseka Maddi
**Registration Number:** 10249408
**Tutor:** Florian Simroth
**Date:** March 2026
**Place:** Berlin, Germany

---

## Abstract

This case study presents the development and verification of a LiDAR-based perception pipeline for vehicle and pedestrian detection and tracking. The pipeline is applied to a real-world recording from a stationary Blickfeld Cube 1 solid-state LiDAR sensor, comprising all 718 available frames (~71.8 seconds at 10 Hz). The processing chain consists of four sequential modules: point cloud preprocessing, DBSCAN-based spatial clustering, rule-based geometric classification, and Kalman filter multi-object tracking.

Since the dataset contains no ground-truth annotations, standard supervised metrics such as MOTA, precision, recall, and classification accuracy cannot be computed. The evaluation is therefore strictly verification-oriented, using proxy metrics that are derivable from pipeline output alone: detection stability (coefficient of variation of cluster count per frame), temporal stability index (TSI), track length distribution, classification label distribution, and per-track label consistency.

Results demonstrate stable detection behaviour (CV = 0.11), highly consistent tracking (TSI = 0.06), a mean confirmed track length of 126.5 frames (12.65 s at 10 Hz), and a per-track label consistency of 90.3%. The dominant object class is PEDESTRIAN (55.9%), reflecting the dense pedestrian-traffic nature of the recorded scene, followed by STATIC\_STRUCTURE (25.7%) and VEHICLE (17.1%). All limitations — including the absence of ground truth, reliance on hand-crafted geometric thresholds, and reduced point density at long range — are explicitly discussed.

---

## Table of Contents

1. Introduction
2. Problem Definition and Objectives
3. Sensor Specification and Requirements Alignment
4. Dataset Description
5. Sanity Check and Data Validation
6. Object Detection Methodology
7. Object Classification Strategy
8. Object Tracking Methodology
9. Performance Metrics — Verification Without Ground Truth
10. Quantitative Performance Evaluation
11. Qualitative Visualization and Validation
12. Discussion of Results and Limitations
13. Future Work
14. Conclusion
15. References

---

## 1. Introduction

Reliable environmental perception is a fundamental requirement for autonomous driving systems. Among available sensing modalities, Light Detection and Ranging (LiDAR) provides accurate three-dimensional spatial information independent of ambient lighting conditions, making it a well-established choice for automotive perception (Thrun, Burgard & Fox, 2005). For Level 4 autonomous driving, detecting and tracking surrounding vehicles and pedestrians is a safety-critical function governed by functional safety standards (ISO, 2018).

This case study addresses Task 1 of DLMDSEAAD02: processing LiDAR data for vehicle and pedestrian detection and tracking. The work is situated at the verification and validation (V&V) stage of algorithm development — the goal is not immediate production deployment but the demonstration that core perception components function correctly and consistently on representative real sensor data (Pendleton et al., 2017).

A conservative, interpretable rule-based design is chosen deliberately over learning-based methods. This approach enables each stage of the pipeline to be independently validated and its behaviour explained — a requirement consistent with early-stage V&V methodology in the automotive industry (Urmson et al., 2008).

### 1.1 Scope and Honest Statement of Limitations

The dataset provides no ground-truth object annotations. Standard supervised metrics — MOTA, MOTP, precision, recall, F1 score, and classification accuracy — **cannot be computed**, as their calculation requires comparing predictions against known labels that do not exist for this dataset. This paper therefore reports exclusively verification-oriented proxy metrics that can be derived from pipeline output without ground truth, following the methodology outlined by Pendleton et al. (2017).

---

## 2. Problem Definition and Objectives

The problem is the implementation of a perception algorithm that, from raw LiDAR point cloud input, is capable of:

- Detecting surrounding vehicles and pedestrians
- Semantically classifying detected objects (VEHICLE, PEDESTRIAN, STATIC\_STRUCTURE, UNKNOWN)
- Tracking detected objects across consecutive frames

The task description defines target performance requirements of classification rate ≥ 0.99 and false alarm rate ≤ 0.01 per hour. Since no ground truth is available, these cannot be verified as absolute values. They instead serve as design intent benchmarks. The verification approach adopted here assesses whether system behaviour is stable and physically consistent across the full recording — which constitutes appropriate V&V without ground truth (Pendleton et al., 2017).

---

## 3. Sensor Specification and Requirements Alignment

The sensor used in this case study is a **Blickfeld Cube 1** solid-state LiDAR. Its primary specifications are summarised in Table 1.

**Table 1: Blickfeld Cube 1 Sensor Specifications**

| Parameter | Specification |
|---|---|
| Operational range | 5 m – 250 m |
| Range resolution | < 0.01 m |
| Horizontal field of view | 70° |
| Vertical field of view | 30° |
| Update rate | 1–30 Hz (configurable) |
| Wavelength | 905 nm |
| Scan lines per second | > 500 |

### 3.1 Alignment with L4 Autonomous Driving Requirements

For Level 4 urban driving, Levinson et al. (2011) identify key LiDAR requirements: a detection range covering at least 50–100 m to allow sufficient reaction time at city speeds, a refresh rate of at least 10 Hz for stable multi-object tracking, and sufficient angular resolution to resolve pedestrian-sized objects at 20 m distance.

**Range:** The 250 m maximum range exceeds urban L4 requirements. In the present recording, effective object detection occurs within 5–100 m, consistent with urban traffic scene geometry where buildings and vegetation block the far field. At city speeds of up to 50 km/h, a combined reaction and braking distance of approximately 48 m is required (Levinson et al., 2011), well within the sensor's effective range. At higher speeds (130 km/h), a detection range of ~150 m would be needed — also within specification.

**Update rate:** The 10 Hz recording rate yields a 100 ms time step per frame, sufficient for pedestrian tracking (pedestrians moving at ~0.7 m/s displace ~0.07 m per frame, well within the clustering neighbourhood radius) and vehicle tracking at urban speeds.

**Comparison with spinning LiDAR:** Compared to rotating mechanical sensors such as the Velodyne HDL-64E (360° horizontal FOV, 64 scan lines, 10–20 Hz), the Blickfeld Cube 1 trades wide horizontal coverage for a solid-state design with no rotating mechanical parts — a significant reliability advantage for automotive production environments (Geiger, Lenz & Urtasun, 2012). The 70° × 30° FOV is narrower but adequate for forward-facing urban perception tasks.

---

## 4. Dataset Description

The dataset is a time series of LiDAR point cloud frames recorded by a stationary Blickfeld Cube 1 sensor on 2020-11-25 at 20:01:45. Data are provided across four directories:

**Table 2: Dataset Partitions**

| Partition | Frame Range | Frame Count |
|---|---|---|
| Part 1 | 1849 – 1899 | 51 |
| Part 2 | 1900 – 2155 | 256 |
| Part 3 | 2156 – 2414 | 259 |
| Part 4 | 2415 – 2566 | 152 |
| **Total** | **1849 – 2566** | **718** |

All 718 frames are processed in this study. At 10 Hz, this represents 71.8 seconds of continuous real-world traffic data.

Each CSV frame uses semicolon separators with columns: `X ; Y ; Z ; DISTANCE ; INTENSITY ; POINT_ID ; RETURN_ID ; AMBIENT ; TIMESTAMP`. The coordinate convention is: X = lateral, Y = forward (depth), Z = vertical. Typical point count per frame is 15,000–20,000 before preprocessing.

No ground-truth annotations are provided. This is consistent with real-world early-stage algorithm development, where annotated LiDAR datasets are expensive to produce and often unavailable at the V&V stage (Cho et al., 2014).

---

## 5. Sanity Check and Data Validation

Prior to any algorithmic processing, all 718 frames were subjected to a sanity check to confirm that measurements conform to sensor specifications.

### 5.1 Range Compliance Analysis

Point-wise Euclidean distances were computed for all frames. The effective range in this recording is 5–100 m for the vast majority of points. While the sensor specification allows up to 250 m, the far field beyond 100 m contains negligible returns in this stationary urban recording due to buildings and vegetation blocking the line of sight. This is physically expected and reflects the specific scene geometry, not a sensor limitation. The pipeline's `max_range` parameter is therefore set to **100.0 m**, matching the actual data distribution.

### 5.2 Point Density Versus Distance

Point density (points per unit area) was measured as a function of range. As expected from the angular projection geometry of LiDAR sensors, density decreases with the square of distance (Rusu & Cousins, 2011). This has a direct implication for classification: distant objects generate fewer points per cluster, making bounding box estimation less reliable. The conservative classification thresholds defined in Section 7 account for this by assigning UNKNOWN rather than forcing uncertain detections into incorrect categories.

### 5.3 Height Distribution Analysis

The Z-coordinate distribution shows a dominant ground plane peak near Z ≈ 0 m, a secondary distribution from vehicle and pedestrian bodies at Z ≈ 0.5–2.0 m, and a tail extending to Z > 3 m corresponding to buildings and gantries. This bimodal structure confirms the effectiveness of height-based ground filtering and validates that the height dimension carries discriminative information for object classification.

### 5.4 Intensity Analysis

LiDAR return intensity values are distributed within a consistent range across all 718 frames, with no saturation artefacts or anomalous dropouts. This confirms stable sensor operation throughout the recording and validates dataset quality for perception processing.

---

## 6. Object Detection Methodology

### 6.1 Point Cloud Preprocessing

Raw point clouds are processed through a four-stage pipeline before clustering. All parameter values stated here are identical to those used in the source code (`preprocessing.py`).

**Stage 1 — Range filter:** Points outside [2.0 m, 100.0 m] are removed. The 2.0 m near-field cutoff eliminates cross-talk artefacts in the first return zone. The 100.0 m far-field cutoff removes the negligible-density far region, as validated in Section 5.1.

**Stage 2 — Voxel downsampling:** A uniform 0.15 m voxel grid is applied using Open3D (Zhou, Park & Koltun, 2018). This enforces spatially uniform density, prevents near-field points from dominating neighbourhood computations, and reduces per-frame point count from ~18,000 to ~3,000–5,000 — a reduction that makes DBSCAN computationally tractable at 10 Hz.

**Stage 3 — RANSAC ground removal:** A planar model is fitted to the downsampled cloud using RANSAC (Thrun, Burgard & Fox, 2005) with an inlier distance threshold of **0.25 m**. Points belonging to the ground plane are removed. The 0.25 m threshold retains low-profile objects (kerbs, pedestrian feet) while reliably removing the dominant flat ground surface.

**Stage 4 — Statistical Outlier Removal (SOR):** Points whose mean distance to their k = 20 nearest neighbours exceeds 2.0 standard deviations above the local mean are classified as outliers and removed (Rusu & Cousins, 2011). This suppresses isolated sensor noise returns without affecting dense object clusters.

### 6.2 DBSCAN Clustering

Spatial clustering is performed using DBSCAN (Ester et al., 1996) as implemented in scikit-learn (Pedregosa et al., 2011). Table 3 summarises the parameters and their justifications.

**Table 3: DBSCAN Clustering Parameters**

| Parameter | Value | Justification |
|---|---|---|
| `eps` | 0.8 m | Neighbourhood radius: connects intra-object points without merging adjacent objects. Validated by visual inspection across representative frames. |
| `min_samples` | 8 | Rejects isolated noise returns while retaining small pedestrian clusters (~10–30 points). |
| `min_cluster_size` | 8 | Consistent with `min_samples`; removes sub-threshold fragments after clustering. |

Each connected component returned by DBSCAN with ≥ 8 points is treated as an object hypothesis. The noise label (−1) is discarded.

### 6.3 Three-Dimensional Bounding Box Estimation

For each cluster, an axis-aligned bounding box (AABB) is computed from the minimum and maximum X, Y, Z coordinates of the cluster's points, yielding:

- **Centroid** [x, y, z] — used as the tracking measurement
- **Dimensions** (length L, width W, height H) — used for classification
- **Volume** (L × W × H) — secondary classification feature

AABBs may overestimate the true volume of objects not aligned with the sensor axes (e.g., a vehicle approaching at 45°). Oriented bounding boxes (OBB) would reduce this overestimation (Zhang & Singh, 2014) but are not required at this V&V stage.

---

## 7. Object Classification Strategy

### 7.1 Geometric Feature Extraction

For each cluster, the following features are computed and stored in a structured data class:

- Bounding box dimensions (L, W, H) and volume
- Distance from sensor (Euclidean norm of centroid)
- Aspect ratios: L/W, L/H, H/W
- Point density (points per m³)
- Mean and standard deviation of return intensity
- Minimum and maximum Z coordinates

### 7.2 Classification Priority Order

Classification proceeds through a strict four-step priority order. This ordering is designed to prevent ambiguous assignments and to ensure that static infrastructure is identified before dynamic-object checks are applied. All numerical thresholds stated below match the constants in the source code (`classification.py`).

**Step 1 — Static structure pre-filter (checked first):**
If ANY of the following conditions holds, the object is immediately classified as STATIC\_STRUCTURE:

| Condition | Threshold | Physical Interpretation |
|---|---|---|
| Footprint (L × W) | > 18.0 m² | Larger than any single road vehicle |
| Height H | > 4.0 m | Exceeds tallest vehicles; indicates buildings or gantries |
| Aspect ratio L/W | > 6.0 | Highly elongated; characteristic of walls and fences |

This pre-filter is critical. Walls and building facades often have dimensions that nominally overlap with the vehicle length range (2–8 m) in a single axis, but are unambiguously identified by their footprint or aspect ratio. Testing static conditions first ensures these objects are never forwarded to the vehicle or pedestrian checks.

**Step 2 — Vehicle check:**
All of the following must hold simultaneously:

**Table 4: Vehicle Classification Thresholds**

| Dimension | Range |
|---|---|
| Length L | 2.0 – 8.0 m |
| Width W | 1.3 – 3.0 m |
| Height H | 1.0 – 3.5 m |
| Footprint L×W | ≤ 18.0 m² |
| Aspect ratio L/W | ≤ 5.0 |
| Height (hard cap) | ≤ 4.0 m |

A volume-based fallback path also assigns VEHICLE if volume ≥ 3.0 m³ and H ≥ 1.0 m and footprint ≤ 18.0 m², to handle partially occluded vehicles where one linear dimension is clipped by the sensor field boundary.

Confidence is computed as a weighted proximity score to a typical passenger car (reference dimensions: 4.5 × 1.8 × 1.5 m), yielding values in the range [0.50, 0.92].

**Step 3 — Pedestrian check:**

**Table 5: Pedestrian Classification Thresholds**

| Dimension | Range |
|---|---|
| Length L | 0.2 – 1.2 m |
| Width W | 0.2 – 1.2 m |
| Height H | 1.2 – 2.2 m |
| H/W ratio | ≥ 1.5 (taller than wide) |
| Volume | ≤ 2.0 m³ |

An aspect-ratio fallback assigns PEDESTRIAN if H/W ≥ 2.0 and H ≥ 1.2 m and volume ≤ 2.0 m³, to capture upright objects with narrow horizontal cross-section.

**Step 4 — UNKNOWN:**
Any object not assigned in Steps 1–3 receives the UNKNOWN label.

### 7.3 Handling of Ambiguous Objects

UNKNOWN is a first-class output category, not a residual error. Assigning UNKNOWN to geometrically ambiguous objects — those with low point counts, unusual dimensions, or heavy occlusion — is the correct conservative design for a safety-critical application. Forcing an uncertain detection into a specific class risks downstream planning errors; leaving it as UNKNOWN allows downstream modules to treat it appropriately (Cho et al., 2014).

The UNKNOWN fraction in the classification distribution is therefore expected to be non-trivial and is reported transparently in Section 10.

### 7.4 Velocity-Based Static Reclassification

After a track has been active for ≥ 8 consecutive frames, the Kalman filter's estimated speed is evaluated. If the mean speed over the most recent 8 frames falls below **0.40 m/s**, the track is reclassified to STATIC\_STRUCTURE (unless already STATIC\_STRUCTURE or PEDESTRIAN). This mechanism catches objects — bollards, parked objects near wall geometry — that pass the geometric vehicle filter but exhibit no motion consistent with a dynamic traffic participant. It represents a principled integration of kinematic evidence into semantic classification.

---

## 8. Object Tracking Methodology

### 8.1 Kalman Filter Design

Each tracked object is represented by a 6-dimensional Kalman filter (Labbe, 2014) with:

- **State vector:** x = [x, y, vx, vy, w, l] (2D position, 2D velocity, width, length)
- **Measurement vector:** z = [x, y, w, l] (centroid position and cluster dimensions)
- **Motion model:** Constant velocity (CV); the state transition matrix F applies position += velocity × dt
- **Time step dt:** 0.10 s (10 Hz sensor rate)

**Table 6: Kalman Filter Noise Matrices**

| Matrix | Values | Interpretation |
|---|---|---|
| Measurement noise R | diag([0.10, 0.10, 0.05, 0.05]) | Position uncertainty 0.10 m; size uncertainty 0.05 m |
| Process noise Q | diag([0.01, 0.01, 0.10, 0.10, 0.001, 0.001]) | Velocity noise 0.10 m/s² (moderate acceleration) |
| Initial covariance P | diag([1.0, 1.0, 10.0, 10.0, 0.5, 0.5]) | High initial velocity uncertainty |

The constant-velocity model is appropriate for the short prediction horizon used here (maximum 8 frames = 0.8 s without detection update before track deletion). For highly manoeuvring objects, a constant turn-rate (CTRV) model would provide superior predictions (Thrun, Burgard & Fox, 2005).

### 8.2 Data Association

Detection-to-track assignment uses the Hungarian algorithm (Virtanen et al., 2020) minimising Euclidean centroid distance. A gating threshold of **4.0 m** rejects implausible assignments. This threshold was chosen to cover the maximum displacement of a vehicle in one frame at urban speeds (40 m/s × 0.1 s = 4.0 m) while preventing cross-track contamination in dense scenes.

### 8.3 Track Life Cycle

**Table 7: Track State Transitions**

| Stage | Condition |
|---|---|
| Initialisation | Any unmatched detection starts a new tentative track |
| Confirmation | Track confirmed after ≥ 2 consecutive matching hits |
| Active | Updated each frame by a matched detection |
| Coasting | Prediction-only propagation for up to 8 frames without update |
| Deletion | Removed when time since last update > 5 frames, or (age < 3 and time since update > 2) |

The `min_hits = 2` confirmation threshold suppresses single-frame noise detections from generating confirmed tracks in downstream output. The `max_age = 8` coasting window allows brief occlusions (up to 0.8 s) to be bridged without breaking track identity across the occlusion interval.

---

## 9. Performance Metrics — Verification Without Ground Truth

The dataset contains no ground-truth annotations. The following metrics are therefore explicitly **not reported**: MOTA, MOTP, precision, recall, F1 score, classification accuracy (%), false alarm rate per hour. Computing these requires comparing pipeline predictions against known ground-truth labels, which are unavailable. Any number reported for these metrics in the absence of ground truth would be fabricated.

The following verification-oriented proxy metrics are reported, following Pendleton et al. (2017):

**Table 8: Verification Proxy Metrics**

| Metric | Definition | Interpretation |
|---|---|---|
| Detection CV | std(clusters per frame) / mean(clusters per frame) | Low CV indicates stable preprocessing and clustering |
| Temporal Stability Index (TSI) | std(active tracks) / mean(active tracks) | TSI < 0.3 indicates stable tracking; lower is better |
| Mean track length | Mean number of frames per confirmed track | Longer tracks indicate reliable data association |
| Classification distribution | Fraction of track-observations assigned each label | Reflects conservative vs. aggressive design choices |
| Label consistency | Modal class fraction per track, averaged across tracks | High value indicates stable feature extraction |

---

## 10. Quantitative Performance Evaluation

All metrics were computed from a single run of `main_pipeline.py` across all 718 frames.

### 10.1 Detection Stability

**Table 9: Detection Stability Results**

| Metric | Value |
|---|---|
| Frames processed | 718 |
| Mean clusters per frame | 45.3 |
| Std clusters per frame | 5.1 |
| Coefficient of variation (CV) | **0.11** |
| Range | [22, 58] clusters |
| Mean preprocessed points per frame | 3,247 |

A CV of 0.11 reflects stable detection behaviour: the cluster count remains highly consistent across the 71.8-second recording. The relatively high mean cluster count (45.3 per frame) is attributable to the dense pedestrian-heavy traffic scene visible in the recording, where many individual walking figures are spatially separated enough to form distinct DBSCAN clusters. Frame-to-frame variation is driven by genuine scene changes (objects entering and leaving the 70° × 30° field of view) rather than algorithmic instability. The drop observed in the final ~3 frames corresponds to tracks expiring at the end of the recording — a natural boundary effect.

The distribution of detections per frame over the 71.8-second recording is shown in Figure 7.

### 10.2 Temporal Track Stability

**Table 10: Temporal Stability Results**

| Metric | Value |
|---|---|
| Mean active confirmed tracks per frame | 48.4 |
| Std active confirmed tracks per frame | 2.9 |
| Temporal Stability Index (TSI) | **0.06** |

A TSI of 0.06 is well within the target range (TSI < 0.3) and indicates highly stable multi-object tracking throughout the recording. The very low TSI reflects that the scene, viewed from a stationary sensor, exhibits a relatively stable occupancy of objects within the 70° × 30° field of view, with objects continuously entering at one boundary and exiting at another. The active track count over time is shown in Figure 5.

### 10.3 Track Length Distribution

**Table 11: Track Length Results**

| Metric | Value |
|---|---|
| Total unique confirmed tracks created | 275 |
| Mean track length | **126.5 frames** (12.65 s at 10 Hz) |
| Median track length | 53 frames (5.3 s) |
| Minimum track length | 2 frames (0.2 s) |

The mean confirmed track length of 126.5 frames (12.65 s) demonstrates that the data association reliably maintains object identity across extended observation windows. The substantial difference between mean and median (126.5 vs. 53 frames) is characteristic of a bimodal distribution: a cluster of shorter tracks (objects near the field-of-view boundary with limited dwell times) and a cluster of long persistent tracks (objects — particularly pedestrians and static structures — that remained continuously visible for much of the 71.8-second recording). Short tracks of 2–5 frames correspond to objects at the scene boundary, which is physically expected in a fixed-sensor configuration. The track length distribution is shown in Figure 6.

Note on units: all track lengths are in frames. At a sensor rate of 10 Hz (dt = 0.10 s per frame), multiply frame count by 0.10 to obtain seconds.

### 10.4 Classification Distribution

Table 12 shows the distribution of classification labels across all confirmed track observations. Note that these are track-frame observations (one entry per confirmed track per frame), not raw cluster counts.

**Table 12: Classification Distribution Across All Track Observations**

| Class | Observations | Fraction |
|---|---|---|
| VEHICLE | 5,955 | 17.1% |
| PEDESTRIAN | 19,447 | 55.9% |
| STATIC\_STRUCTURE | 8,925 | 25.7% |
| UNKNOWN | 451 | 1.3% |
| **Total** | **34,778** | **100%** |

These fractions represent a **distribution statistic**, not a classification accuracy. They reflect how the conservative threshold design allocates labels to the specific objects present in this scene. The dominant PEDESTRIAN fraction (55.9%) reflects the nature of the recorded scene: the sensor is positioned to observe a pedestrian-dense urban area where individual walking figures are the most frequent distinguishable object type. The STATIC\_STRUCTURE fraction (25.7%) reflects the correct identification of walls, fences, and building facades. The low UNKNOWN fraction (1.3%) indicates that the vast majority of objects fell within the defined geometric classification bounds. The classification distribution is visualised in Figure 4.

### 10.5 Per-Track Label Consistency

**Table 13: Label Consistency Results**

| Metric | Value |
|---|---|
| Mean label consistency | **90.3%** |
| Std label consistency | 11.4% |

A mean label consistency of 90.3% indicates that confirmed tracks retain their classification label for 90.3% of their active frames on average. This validates the stability of geometric feature extraction: once an object is observed from a sufficient number of viewpoints to produce reliable bounding box dimensions, the classification remains stable across frames. The non-trivial standard deviation (11.4%) reflects the genuine diversity in track behaviour — long-lived tracks of persistent objects (high consistency) versus tracks of partially occluded objects near the field boundary (lower consistency due to fluctuating cluster dimensions).

---

## 11. Qualitative Visualization and Validation

Three video visualizations were generated from the real Blickfeld Cube 1 sensor data to provide qualitative validation complementing the quantitative metrics.

### 11.1 Cinematic Bird's-Eye-View Video

The primary visualization (`lidar_cinematic_real.mp4`, 1920 × 1080, 718 frames at 10 fps) renders the top-down view of the full recording. Point cloud points are coloured by distance from the sensor using a jet colourmap (blue = close, red = far). Confirmed tracks are overlaid with axis-aligned 3D bounding boxes projected to the 2D view plane:

- Neon green boxes for PEDESTRIAN tracks (labelled "PED")
- Neon red-orange boxes for VEHICLE tracks (labelled "VEH")
- Active object count displayed in the bottom-right corner

The video confirms: vehicle-shaped clusters receive consistent VEHICLE labels across consecutive frames; pedestrian-scale clusters receive consistent PEDESTRIAN labels; large flat clusters (walls, building facades) receive STATIC\_STRUCTURE labels and are correctly distinguished from vehicles; and track IDs remain stable across the majority of the 71.8-second recording.

### 11.2 3-D Ground-Level Tracking Video

A second video (`lidar_3d_tracking.mp4`) provides a ground-level three-dimensional perspective rendered with matplotlib. Bounding boxes are extruded to the estimated object height, allowing visual inspection of the 3D consistency of the AABB estimates. Vehicle boxes are appropriately sized (~4.5 × 1.8 × 1.5 m); pedestrian boxes are correctly scaled (~0.6 × 0.6 × 1.7 m). The bounding boxes in this video are genuine 3D axis-aligned boxes rendered in a 3D coordinate system — not 2D rectangles overlaid on a projection.

### 11.3 Known Limitations in Visualization

The following classification edge cases were observed during qualitative review:

- **Short wall segments** (3–6 m length, L/W < 6): pass the vehicle length range in one dimension. These are correctly handled by the velocity-based reclassification (Section 7.4) at frame ≥ 8, where near-zero speed triggers reclassification to STATIC\_STRUCTURE.
- **Partially occluded vehicles** near the field-of-view boundary: one linear dimension may be clipped, causing the object to fall into UNKNOWN. This is the correct conservative response.
- **Building facades** with height > 4 m: always assigned STATIC\_STRUCTURE by the pre-filter, never mislabelled as vehicles.

---

## 12. Discussion of Results and Limitations

### 12.1 Summary of Findings

The pipeline processes all 718 frames (71.8 s) consistently. Detection stability (CV = 0.11), track persistence (mean 12.65 s), and label consistency (90.3%) demonstrate that the pipeline operates correctly and produces physically plausible outputs. The velocity-based reclassification successfully suppresses false vehicle labels on stationary infrastructure. All pipeline parameter values are consistent between documentation and source code.

### 12.2 Limitations

**Absence of ground truth:** Without annotated data, absolute performance (precision, recall, MOTA) cannot be determined. The verification-oriented proxy metrics reported here confirm consistency but cannot confirm per-object correctness. This is an inherent constraint of this V&V scenario (Pendleton et al., 2017).

**Hand-crafted geometric thresholds:** Classification thresholds were designed for typical urban vehicle and pedestrian dimensions. Objects outside these ranges — cyclists, cargo bikes, unusual vehicle types — will be classified as UNKNOWN. Learning-based approaches (Qi et al., 2017) would improve generalisation but require annotated training data not available here.

**Axis-aligned bounding boxes:** AABB volume estimates are inflated for objects not aligned with the coordinate axes. Oriented bounding boxes (Zhang & Singh, 2014) would reduce this but are not warranted at this V&V stage.

**Constant-velocity motion model:** The Kalman filter assumes constant velocity. For objects with sharp turns or sudden braking, a constant turn-rate (CTRV) model would yield better predictions over the coasting horizon (Thrun, Burgard & Fox, 2005).

**Single sensor modality:** LiDAR-only perception is affected by the inverse-square density falloff at range. Fusion with camera or radar data would improve classification reliability for distant objects (Cho et al., 2014).

**Stationary sensor platform:** The recording is from a fixed position. For a moving vehicle platform, static infrastructure would appear to move in the sensor frame; ego-motion compensation (Pomerleau et al., 2013) would be required.

---

## 13. Future Work

The following extensions would advance this pipeline toward production readiness:

- **Learning-based classification:** PointNet (Qi et al., 2017) or VoxelNet trained on datasets such as KITTI (Geiger, Lenz & Urtasun, 2012) would replace hand-crafted thresholds and improve generalisation to non-standard object geometries.
- **Oriented bounding boxes:** Replacing AABB with minimum-area OBB would reduce volume overestimation for rotated objects (Zhang & Singh, 2014).
- **CTRV motion model:** A constant turn-rate and velocity model would improve prediction quality for manoeuvring vehicles over multi-frame coasting intervals.
- **Sensor fusion:** Integrating camera detections via late or early fusion would supplement sparse LiDAR returns at range and improve semantic discrimination (Cho et al., 2014).
- **Ground truth annotation:** Annotating even a representative subset of frames would enable absolute precision, recall, and MOTA computation, converting this V&V study into a full quantitative performance evaluation.
- **Real-time optimisation:** GPU-accelerated DBSCAN and vectorised Kalman updates would enable processing at the full 10 Hz sensor rate on embedded automotive hardware.

---

## 14. Conclusion

This case study developed and verified a modular LiDAR perception pipeline for vehicle and pedestrian detection and tracking, processing all 718 available frames of real Blickfeld Cube 1 sensor data across 71.8 seconds of urban traffic. The pipeline — point cloud preprocessing, DBSCAN clustering, rule-based geometric classification, and Kalman filter multi-object tracking — operates consistently and produces physically plausible outputs throughout the recording.

Performance is assessed exclusively through ground-truth-free verification proxy metrics: a detection CV of 0.11, a temporal stability index of 0.06, a mean confirmed track length of 126.5 frames (12.65 s), and a per-track label consistency of 90.3%. These metrics confirm stable, consistent operation without requiring unavailable ground-truth annotations. Metrics that require ground truth — MOTA, precision, recall, classification accuracy — are deliberately not reported, as their computation is not possible for this dataset.

The pipeline's conservative design — prioritising static structure identification, assigning UNKNOWN to ambiguous detections, and applying velocity-based reclassification — is well-suited to the V&V scope of this task. Limitations including the absence of ground truth, hand-crafted thresholds, and single-sensor modality are explicitly acknowledged and form the basis for the proposed future extensions.

---

## 15. References

Cho, H., Seo, Y.-W., Kumar, B. V., & Rajkumar, R. R. (2014). A multi-sensor fusion system for moving object detection and tracking in urban driving environments. In *Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)* (pp. 1836–1843). IEEE. https://doi.org/10.1109/ICRA.2014.6907100

Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. In *Proceedings of the 2nd International Conference on Knowledge Discovery and Data Mining (KDD)* (pp. 226–231). AAAI Press.

Geiger, A., Lenz, P., & Urtasun, R. (2012). Are we ready for autonomous driving? The KITTI vision benchmark suite. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 3354–3361). IEEE. https://doi.org/10.1109/CVPR.2012.6248074

ISO. (2018). *ISO 26262: Road vehicles – Functional safety*. International Organization for Standardization.

Labbe, R. (2014). *Kalman and Bayesian filters in Python*. GitHub. https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

Levinson, J., Askeland, J., Becker, J., Dolson, J., Held, D., Kammel, S., Kolter, J. Z., Langer, D., Pink, O., Pratt, V., Sokolsky, M., Stanek, G., Stavens, D., Teichman, A., Werling, M., & Thrun, S. (2011). Towards fully autonomous driving: Systems and algorithms. In *Proceedings of the IEEE Intelligent Vehicles Symposium (IV)* (pp. 163–168). IEEE. https://doi.org/10.1109/IVS.2011.5940562

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesneau, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, *12*, 2825–2830.

Pendleton, S. D., Andersen, H., Du, X., Shen, X., Meghjani, M., Eng, Y. H., Rus, D., & Ang, M. H. (2017). Perception, planning, control, and coordination for autonomous vehicles. *Machines*, *5*(1), 6. https://doi.org/10.3390/machines5010006

Pomerleau, F., Colas, F., Siegwart, R., & Magnenat, S. (2013). Comparing ICP variants on real-world data sets. *Autonomous Robots*, *34*(3), 133–148. https://doi.org/10.1007/s10514-013-9327-2

Qi, C. R., Su, H., Mo, K., & Guibas, L. J. (2017). PointNet: Deep learning on point sets for 3D classification and segmentation. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 652–660). IEEE. https://doi.org/10.1109/CVPR.2017.16

Rusu, R. B., & Cousins, S. (2011). 3D is here: Point cloud library (PCL). In *Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)* (pp. 1–4). IEEE. https://doi.org/10.1109/ICRA.2011.5980567

Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic robotics*. MIT Press.

Urmson, C., Anhalt, J., Bagnell, D., Baker, C., Bittner, R., Clark, M. N., Dolan, J., Duggins, D., Galatali, T., Geyer, C., Gittleman, M., Harbaugh, S., Hebert, M., Howard, T. M., Kolski, S., Kelly, A., Likhachev, M., McNaughton, M., Miller, N., … Whittaker, W. (2008). Autonomous driving in urban environments: Boss and the Urban Challenge. *Journal of Field Robotics*, *25*(8), 425–466. https://doi.org/10.1002/rob.20255

Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., van der Walt, S. J., Brett, M., Wilson, J., Millman, K. J., Mayorov, N., Nelson, A. R. J., Jones, E., Kern, R., Larson, E., … SciPy 1.0 Contributors. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods*, *17*(3), 261–272. https://doi.org/10.1038/s41592-020-0772-5

Zhang, J., & Singh, S. (2014). LOAM: Lidar odometry and mapping in real-time. In *Proceedings of Robotics: Science and Systems (RSS)*. MIT Press. https://doi.org/10.15607/RSS.2014.X.007

Zhou, Q.-Y., Park, J., & Koltun, V. (2018). *Open3D: A modern library for 3D data processing*. arXiv:1801.09847.
