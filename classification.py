"""
LiDAR Feature Extraction and Classification Module
==================================================
Autonomous Driving Perception Pipeline - Object Classification

This module implements:
1. Feature extraction from clustered point clouds
2. Rule-based classification into Vehicle and Pedestrian classes

Classification is based on geometric features:
- Bounding box dimensions (length, width, height)
- Point density
- Shape ratios

Author: Perception Engineer
Course: Localization, Motion Planning and Sensor Fusion
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BoundingBox3D:
    """
    3D Axis-Aligned Bounding Box (AABB) for a cluster.

    Represents the smallest box aligned with coordinate axes
    that contains all points in the cluster.
    """
    min_point: np.ndarray  # [x_min, y_min, z_min]
    max_point: np.ndarray  # [x_max, y_max, z_max]
    center: np.ndarray     # [x_center, y_center, z_center]
    length: float          # X dimension (forward/back)
    width: float           # Y dimension (left/right)
    height: float          # Z dimension (up/down)
    volume: float          # Bounding box volume


@dataclass
class ObjectFeatures:
    """
    Container for all extracted features of a detected object.

    Used for classification and tracking.
    """
    cluster_id: int
    num_points: int

    # Bounding box
    bounding_box: BoundingBox3D
    center: np.ndarray
    length: float
    width: float
    height: float
    volume: float

    # Position relative to sensor
    distance_from_sensor: float

    # Density features
    point_density: float  # points per cubic meter

    # Shape features
    aspect_ratio_lw: float  # length / width
    aspect_ratio_lh: float  # length / height
    aspect_ratio_hw: float  # height / width

    # Intensity features
    mean_intensity: float
    std_intensity: float

    # Ground plane features
    min_z: float
    max_z: float

    # Classification result (set after classification)
    classification: Optional[str] = None
    confidence: float = 0.0


class FeatureExtractor:
    """
    Extracts geometric and statistical features from point cloud clusters.
    """

    def __init__(self):
        """Initialize the feature extractor."""
        pass

    def extract_features(self, points: np.ndarray,
                        cluster_labels: np.ndarray,
                        verbose: bool = True) -> List[ObjectFeatures]:
        """
        Extract features from all clusters.

        Args:
            points: Point cloud (N, 4) [X, Y, Z, INTENSITY]
            cluster_labels: Cluster label for each point (-1 = noise)
            verbose: Print progress information

        Returns:
            List of ObjectFeatures for each cluster
        """
        if verbose:
            print("\n" + "="*60)
            print("FEATURE EXTRACTION")
            print("="*60)

        # Get unique cluster IDs (exclude noise)
        unique_labels = np.unique(cluster_labels)
        cluster_ids = unique_labels[unique_labels != -1]

        if verbose:
            print(f"\nExtracting features from {len(cluster_ids)} clusters...\n")

        features_list = []

        for cluster_id in cluster_ids:
            # Get points for this cluster
            mask = cluster_labels == cluster_id
            cluster_points = points[mask]

            # Extract features
            features = self._extract_single_cluster(cluster_id, cluster_points)
            features_list.append(features)

            if verbose:
                print(f"  Cluster {cluster_id:3d}: {features.num_points:4d} pts | "
                      f"L={features.length:5.2f}m W={features.width:5.2f}m H={features.height:5.2f}m | "
                      f"Dist={features.distance_from_sensor:5.1f}m")

        if verbose:
            print("\n" + "="*60 + "\n")

        return features_list

    def _extract_single_cluster(self, cluster_id: int,
                               points: np.ndarray) -> ObjectFeatures:
        """
        Extract features from a single cluster.

        Args:
            cluster_id: Cluster identifier
            points: Points in the cluster (N, 4)

        Returns:
            ObjectFeatures for this cluster
        """
        xyz = points[:, :3]
        intensity = points[:, 3] if points.shape[1] >= 4 else np.ones(len(points)) * 0.5

        # Compute bounding box
        min_point = np.min(xyz, axis=0)
        max_point = np.max(xyz, axis=0)
        center = (min_point + max_point) / 2.0

        dimensions = max_point - min_point
        length = max(dimensions[0], 0.01)  # X (forward)
        width = max(dimensions[1], 0.01)   # Y (lateral)
        height = max(dimensions[2], 0.01)  # Z (vertical)
        volume = length * width * height

        bbox = BoundingBox3D(
            min_point=min_point,
            max_point=max_point,
            center=center,
            length=length,
            width=width,
            height=height,
            volume=volume
        )

        # Distance from sensor (assumed at origin)
        distance_from_sensor = np.linalg.norm(center)

        # Point density
        point_density = len(points) / max(volume, 0.001)

        # Shape aspect ratios
        aspect_ratio_lw = length / max(width, 0.01)
        aspect_ratio_lh = length / max(height, 0.01)
        aspect_ratio_hw = height / max(width, 0.01)

        # Intensity statistics
        mean_intensity = np.mean(intensity)
        std_intensity = np.std(intensity)

        return ObjectFeatures(
            cluster_id=cluster_id,
            num_points=len(points),
            bounding_box=bbox,
            center=center,
            length=length,
            width=width,
            height=height,
            volume=volume,
            distance_from_sensor=distance_from_sensor,
            point_density=point_density,
            aspect_ratio_lw=aspect_ratio_lw,
            aspect_ratio_lh=aspect_ratio_lh,
            aspect_ratio_hw=aspect_ratio_hw,
            mean_intensity=mean_intensity,
            std_intensity=std_intensity,
            min_z=min_point[2],
            max_z=max_point[2]
        )


class RuleBasedClassifier:
    """
    Rule-based classifier for LiDAR objects.

    Classifies objects into:
    - VEHICLE: Cars, trucks, buses (larger objects)
    - PEDESTRIAN: People (smaller, taller objects)

    Classification Logic:
    ---------------------
    VEHICLE characteristics:
    - Length: 3.5m - 7.0m (compact car to truck)
    - Width: 1.5m - 2.5m (standard vehicle width)
    - Height: 1.2m - 3.0m (car to bus)
    - Shape: Elongated (length > width)

    PEDESTRIAN characteristics:
    - Length: 0.3m - 1.0m
    - Width: 0.3m - 1.0m
    - Height: 1.4m - 2.0m
    - Shape: Tall and narrow (height >> width)
    """

    # Classification thresholds
    VEHICLE_MIN_LENGTH = 2.0
    VEHICLE_MAX_LENGTH = 8.0
    VEHICLE_MIN_WIDTH = 1.3
    VEHICLE_MAX_WIDTH = 3.0
    VEHICLE_MIN_HEIGHT = 1.0
    VEHICLE_MAX_HEIGHT = 3.5
    VEHICLE_MIN_VOLUME = 3.0

    PEDESTRIAN_MIN_LENGTH = 0.2
    PEDESTRIAN_MAX_LENGTH = 1.2
    PEDESTRIAN_MIN_WIDTH = 0.2
    PEDESTRIAN_MAX_WIDTH = 1.2
    PEDESTRIAN_MIN_HEIGHT = 1.2
    PEDESTRIAN_MAX_HEIGHT = 2.2
    PEDESTRIAN_MAX_VOLUME = 2.0

    def __init__(self):
        """Initialize the classifier."""
        pass

    def classify(self, features_list: List[ObjectFeatures],
                verbose: bool = True) -> Dict[int, str]:
        """
        Classify all detected objects.

        Args:
            features_list: List of ObjectFeatures
            verbose: Print progress information

        Returns:
            Dictionary mapping cluster_id to classification label
        """
        if verbose:
            print("\n" + "="*60)
            print("RULE-BASED CLASSIFICATION")
            print("="*60)
            print("\n{:^8} | {:^6} | {:^18} | {:^12} | {:^10}".format(
                "Cluster", "Points", "Dimensions (LxWxH)", "Class", "Confidence"
            ))
            print("-"*60)

        classifications = {}
        counts = {'VEHICLE': 0, 'PEDESTRIAN': 0, 'UNKNOWN': 0}

        for features in features_list:
            label, confidence = self._classify_single(features)

            # Store result
            features.classification = label
            features.confidence = confidence
            classifications[features.cluster_id] = label
            counts[label] += 1

            if verbose:
                dims_str = f"{features.length:.2f}x{features.width:.2f}x{features.height:.2f}"
                print(f"{features.cluster_id:^8} | {features.num_points:^6} | "
                      f"{dims_str:^18} | {label:^12} | {confidence:^10.1%}")

        if verbose:
            print("-"*60)
            print("\n--- Classification Summary ---")
            print(f"  Total objects:  {len(features_list)}")
            print(f"  Vehicles:       {counts['VEHICLE']}")
            print(f"  Pedestrians:    {counts['PEDESTRIAN']}")
            print(f"  Unknown:        {counts['UNKNOWN']}")
            print("="*60 + "\n")

        return classifications

    def _classify_single(self, features: ObjectFeatures) -> Tuple[str, float]:
        """
        Classify a single object.

        Args:
            features: ObjectFeatures for the object

        Returns:
            Tuple of (classification_label, confidence)
        """
        length = features.length
        width = features.width
        height = features.height
        volume = features.volume
        aspect_hw = features.aspect_ratio_hw

        # Rule 1: Check for vehicle (large objects)
        if (self.VEHICLE_MIN_LENGTH <= length <= self.VEHICLE_MAX_LENGTH and
            self.VEHICLE_MIN_WIDTH <= width <= self.VEHICLE_MAX_WIDTH and
            self.VEHICLE_MIN_HEIGHT <= height <= self.VEHICLE_MAX_HEIGHT):
            # Confidence based on how well it fits typical vehicle dimensions
            confidence = self._calculate_vehicle_confidence(features)
            return 'VEHICLE', confidence

        # Rule 2: Volume-based vehicle detection
        if volume >= self.VEHICLE_MIN_VOLUME and height >= 1.0:
            confidence = min(0.8, volume / 20.0)
            return 'VEHICLE', confidence

        # Rule 3: Check for pedestrian (tall, narrow objects)
        if (self.PEDESTRIAN_MIN_LENGTH <= length <= self.PEDESTRIAN_MAX_LENGTH and
            self.PEDESTRIAN_MIN_WIDTH <= width <= self.PEDESTRIAN_MAX_WIDTH and
            self.PEDESTRIAN_MIN_HEIGHT <= height <= self.PEDESTRIAN_MAX_HEIGHT):
            # Confidence based on aspect ratio (pedestrians are tall and narrow)
            if aspect_hw >= 2.0:
                confidence = min(0.95, 0.7 + aspect_hw * 0.05)
            else:
                confidence = 0.6
            return 'PEDESTRIAN', confidence

        # Rule 4: Aspect ratio based pedestrian detection
        if aspect_hw >= 2.5 and height >= 1.4 and volume <= self.PEDESTRIAN_MAX_VOLUME:
            confidence = min(0.85, 0.5 + aspect_hw * 0.1)
            return 'PEDESTRIAN', confidence

        # Rule 5: Large length suggests vehicle
        if length >= 2.5:
            confidence = min(0.7, length / 5.0)
            return 'VEHICLE', confidence

        # Rule 6: Very small objects are unknown (likely noise)
        if volume < 0.1 or features.num_points < 5:
            return 'UNKNOWN', 0.3

        # Default: Unknown
        return 'UNKNOWN', 0.5

    def _calculate_vehicle_confidence(self, features: ObjectFeatures) -> float:
        """
        Calculate confidence score for vehicle classification.

        Args:
            features: Object features

        Returns:
            Confidence score (0-1)
        """
        # Typical car dimensions: 4.5m x 1.8m x 1.5m
        length_score = 1.0 - abs(features.length - 4.5) / 4.5
        width_score = 1.0 - abs(features.width - 1.8) / 1.8
        height_score = 1.0 - abs(features.height - 1.5) / 1.5

        # Clip scores to [0, 1]
        length_score = max(0, min(1, length_score))
        width_score = max(0, min(1, width_score))
        height_score = max(0, min(1, height_score))

        # Weighted average
        confidence = 0.4 * length_score + 0.3 * width_score + 0.3 * height_score

        # Boost confidence for very large objects
        if features.volume > 10:
            confidence = min(0.95, confidence + 0.1)

        return max(0.5, min(0.95, confidence))


def extract_and_classify(points: np.ndarray,
                        cluster_labels: np.ndarray,
                        verbose: bool = True) -> Tuple[List[ObjectFeatures], Dict[int, str]]:
    """
    Convenience function to extract features and classify objects.

    Args:
        points: Point cloud (N, 4)
        cluster_labels: Cluster labels
        verbose: Print progress information

    Returns:
        Tuple of (features_list, classifications)
    """
    # Extract features
    extractor = FeatureExtractor()
    features_list = extractor.extract_features(points, cluster_labels, verbose)

    # Classify objects
    classifier = RuleBasedClassifier()
    classifications = classifier.classify(features_list, verbose)

    return features_list, classifications


# Example usage and testing
if __name__ == "__main__":
    from data_loader import generate_simulated_frame
    from preprocessing import preprocess_point_cloud
    from clustering import cluster_point_cloud

    print("="*70)
    print("FEATURE EXTRACTION & CLASSIFICATION MODULE - TEST")
    print("="*70)

    # Generate, preprocess, and cluster test data
    print("\n[Test 1] Processing simulated data...")
    raw_points = generate_simulated_frame(num_points=15000, seed=42)
    processed_points = preprocess_point_cloud(raw_points, verbose=False)
    labels, num_clusters = cluster_point_cloud(processed_points, verbose=False)
    print(f"  Detected {num_clusters} clusters")

    # Extract features and classify
    print("\n[Test 2] Extracting features and classifying...")
    features_list, classifications = extract_and_classify(processed_points, labels)

    # Print detailed results
    print("\n[Test 3] Detailed Classification Results:")
    for features in features_list:
        print(f"\n  Object {features.cluster_id}:")
        print(f"    Class: {features.classification} (confidence: {features.confidence:.1%})")
        print(f"    Points: {features.num_points}")
        print(f"    Dimensions: {features.length:.2f}m x {features.width:.2f}m x {features.height:.2f}m")
        print(f"    Volume: {features.volume:.2f} m3")
        print(f"    Distance from sensor: {features.distance_from_sensor:.2f} m")
        print(f"    Point density: {features.point_density:.1f} pts/m3")

    print("\n" + "="*70)
    print("Classification module ready for use.")
    print("="*70)
