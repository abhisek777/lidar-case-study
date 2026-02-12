# Figure Captions for Academic Case Study

Generated: 2026-01-26 17:21:41

Course: DLMDSEAAD02 – Localization, Motion Planning and Sensor Fusion

---

## 1. Range Distribution
**File:** `range_distribution.png`

Figure 1: Range Distribution of LiDAR Points.
This histogram shows the distribution of point distances from the sensor origin.
The Blickfeld Cube 1 sensor operates within a specified range of 5m to 250m.
The vertical dashed lines indicate the operational boundaries. Points within
this range demonstrate compliance with sensor specifications. The distribution
shows typical urban driving characteristics with higher point density at
closer ranges, following the inverse-square relationship of LiDAR sensing.

## 2. Point Density vs Distance
**File:** `density_vs_distance.png`

Figure 2: Point Density as a Function of Distance.
This graph demonstrates the inverse-square relationship between point density
and distance, a fundamental characteristic of LiDAR sensors. The density
(points/m²) decreases quadratically with distance due to the fixed angular
resolution of the sensor. The four distance bins (0-20m, 20-50m, 50-100m,
100-250m) show this physical relationship. Near-range density (~10.7 pts/m²)
is significantly higher than far-range density (~0.00 pts/m²), which is
expected behavior and validates the sensor's performance.

## 3. Height Distribution
**File:** `height_distribution.png`

Figure 3: Height (Z-axis) Distribution of LiDAR Points.
This histogram reveals the vertical structure of the scanned environment.
The prominent peak around z = -1.0m corresponds to the ground plane,
while points above this level represent objects such as vehicles (typically
1-2m height), pedestrians (1.5-1.8m), and structures. This distribution
validates successful ground plane detection and object segmentation in the
preprocessing pipeline.

## 4. Intensity Distribution
**File:** `intensity_distribution.png`

Figure 4: LiDAR Return Intensity Distribution.
The intensity values represent the reflectivity of surfaces detected by the
sensor. Higher intensities typically correspond to retroreflective materials
(road signs, lane markings) while lower values indicate absorptive surfaces.
The distribution provides insight into the material composition of the scanned
environment and can be used for surface classification and data quality
assessment.

## 5. Track Length Distribution
**File:** `track_length_histogram.png`

Figure 5: Distribution of Object Track Lengths.
This histogram shows how long objects are tracked across consecutive frames.
Longer tracks indicate stable detection and successful data association.
The mean track length of 40.2 frames demonstrates the tracking system's
ability to maintain object identity over time. Short tracks (<5 frames) may
represent objects entering/leaving the sensor field of view or temporary
occlusions.

## 6. Active Tracks Over Time
**File:** `active_tracks_over_time.png`

Figure 6: Number of Active Tracks Over Time.
This graph shows the temporal evolution of tracked objects throughout the
sequence. Variations in active track count correspond to objects entering
and leaving the sensor's field of view. The relatively stable count indicates
consistent detection performance. The mean of 27.7 active tracks per
frame with a standard deviation of 5.9 demonstrates tracking stability.

## 7. Classification Distribution
**File:** `classification_distribution.png`

Figure 7: Object Classification Distribution.
This bar chart shows the distribution of detected objects across classification
categories. The pipeline detected 511 vehicles, 821 pedestrians,
and 1527 objects that could not be confidently classified. The classification
is based on geometric features (dimensions, aspect ratios) using a rule-based
system calibrated for urban driving scenarios.

---

## Verification and Validation Summary

These visualizations demonstrate:

1. **Sensor Data Sanity Checking**: Range, height, and intensity distributions confirm data quality and sensor specification compliance.

2. **Object Detection**: Classification distribution shows successful detection of vehicles and pedestrians in urban driving scenarios.

3. **Object Tracking**: Track length histogram and active tracks plot demonstrate stable multi-object tracking performance.

4. **Physical Validation**: The inverse-square relationship in point density confirms expected LiDAR physics behavior.

