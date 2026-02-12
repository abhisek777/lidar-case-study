"""
LiDAR Clustering Module
=======================
Autonomous Driving Perception Pipeline - Object Detection

This module implements DBSCAN clustering for detecting objects in LiDAR data.
DBSCAN is the industry-standard algorithm for point cloud object detection
in autonomous driving systems.

Why DBSCAN:
- No need to specify number of objects in advance
- Handles arbitrary cluster shapes (vehicles, pedestrians)
- Robust to noise (explicitly labels outliers)
- Physically meaningful parameters (distance-based)

Author: Perception Engineer
Course: Localization, Motion Planning and Sensor Fusion
"""

import numpy as np
from sklearn.cluster import DBSCAN
import time
from typing import Tuple, List, Dict


class PointCloudClusterer:
    """
    DBSCAN-based clusterer for LiDAR point clouds.

    Detects individual objects (vehicles, pedestrians) in preprocessed
    point cloud data.
    """

    def __init__(self,
                 eps: float = 0.5,
                 min_samples: int = 10,
                 min_cluster_size: int = 5,
                 max_cluster_size: int = 10000):
        """
        Initialize the clusterer.

        Args:
            eps: Maximum distance between points in same cluster (meters)
                 - Too small: Objects split into multiple clusters
                 - Too large: Nearby objects merged together
                 - Typical: 0.3-1.0m for urban driving
            min_samples: Minimum points to form a dense region
                 - Lower: Detects smaller objects, more noise
                 - Higher: Misses small objects, less noise
                 - Typical: 5-20 for LiDAR data
            min_cluster_size: Minimum points to consider valid cluster
            max_cluster_size: Maximum points (filters large structures)
        """
        self.eps = eps
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size

        # Store results from last clustering
        self.labels_ = None
        self.num_clusters_ = 0
        self.noise_count_ = 0
        self.stats = {}

    def cluster(self, points: np.ndarray, verbose: bool = True) -> np.ndarray:
        """
        Apply DBSCAN clustering to point cloud.

        Args:
            points: Preprocessed point cloud (N, 4) [X, Y, Z, INTENSITY]
            verbose: Print progress information

        Returns:
            Cluster labels for each point (-1 = noise)
        """
        start_time = time.time()

        if verbose:
            print("\n" + "="*60)
            print("DBSCAN CLUSTERING")
            print("="*60)
            print(f"\n--- Parameters ---")
            print(f"  eps (neighborhood radius): {self.eps} m")
            print(f"  min_samples:               {self.min_samples}")
            print(f"  min_cluster_size:          {self.min_cluster_size}")
            print(f"  Input points:              {points.shape[0]:,}")

        # Extract XYZ coordinates for clustering (ignore intensity)
        xyz = points[:, :3]

        # Apply DBSCAN
        clustering = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            algorithm='kd_tree',  # Efficient spatial indexing
            metric='euclidean',
            n_jobs=-1  # Use all CPU cores
        )

        labels = clustering.fit_predict(xyz)

        # Post-process: Filter clusters by size
        labels = self._filter_clusters_by_size(labels, verbose)

        # Store results
        self.labels_ = labels
        unique_labels = np.unique(labels)
        self.num_clusters_ = len(unique_labels) - (1 if -1 in unique_labels else 0)
        self.noise_count_ = np.sum(labels == -1)

        elapsed_time = time.time() - start_time

        # Calculate cluster statistics
        self.stats = self._calculate_statistics(labels, points)
        self.stats['processing_time'] = elapsed_time

        if verbose:
            print(f"\n--- Clustering Results ---")
            print(f"  Objects detected:  {self.num_clusters_}")
            print(f"  Noise points:      {self.noise_count_:,} ({100*self.noise_count_/max(1, len(labels)):.1f}%)")
            print(f"  Processing time:   {elapsed_time*1000:.1f} ms")

            if self.num_clusters_ > 0:
                sizes = self.stats['cluster_sizes']
                print(f"\n--- Cluster Size Statistics ---")
                print(f"  Mean size:    {np.mean(sizes):.1f} points")
                print(f"  Min size:     {np.min(sizes)} points")
                print(f"  Max size:     {np.max(sizes)} points")
                print(f"  Median size:  {np.median(sizes):.1f} points")

            print("="*60 + "\n")

        return labels

    def _filter_clusters_by_size(self, labels: np.ndarray,
                                 verbose: bool) -> np.ndarray:
        """
        Filter out clusters that are too small or too large.

        Args:
            labels: Cluster labels from DBSCAN
            verbose: Print progress information

        Returns:
            Filtered cluster labels
        """
        # Get unique cluster labels (excluding noise)
        unique_labels = np.unique(labels)
        cluster_labels = unique_labels[unique_labels != -1]

        # Count clusters removed
        removed_small = 0
        removed_large = 0

        # Create mapping from old labels to new labels
        label_mapping = {-1: -1}
        new_label = 0

        for old_label in cluster_labels:
            cluster_size = np.sum(labels == old_label)

            if cluster_size < self.min_cluster_size:
                # Mark as noise (too small)
                label_mapping[old_label] = -1
                removed_small += 1
            elif cluster_size > self.max_cluster_size:
                # Mark as noise (too large - likely building/structure)
                label_mapping[old_label] = -1
                removed_large += 1
            else:
                # Valid cluster
                label_mapping[old_label] = new_label
                new_label += 1

        # Apply mapping
        new_labels = np.array([label_mapping[l] for l in labels])

        if verbose and (removed_small > 0 or removed_large > 0):
            print(f"\n--- Cluster Filtering ---")
            print(f"  Removed (too small): {removed_small}")
            print(f"  Removed (too large): {removed_large}")
            print(f"  Valid clusters:      {new_label}")

        return new_labels

    def _calculate_statistics(self, labels: np.ndarray,
                             points: np.ndarray) -> Dict:
        """
        Calculate statistics about the clustering results.

        Args:
            labels: Cluster labels
            points: Point cloud

        Returns:
            Dictionary with statistics
        """
        unique_labels = np.unique(labels)
        cluster_labels = unique_labels[unique_labels != -1]

        stats = {
            'num_clusters': len(cluster_labels),
            'noise_points': np.sum(labels == -1),
            'cluster_sizes': [],
            'cluster_centroids': [],
        }

        for label in cluster_labels:
            cluster_mask = labels == label
            cluster_points = points[cluster_mask, :3]

            stats['cluster_sizes'].append(cluster_points.shape[0])
            stats['cluster_centroids'].append(np.mean(cluster_points, axis=0))

        if len(stats['cluster_sizes']) > 0:
            stats['cluster_sizes'] = np.array(stats['cluster_sizes'])
            stats['cluster_centroids'] = np.array(stats['cluster_centroids'])

        return stats

    def get_cluster_points(self, points: np.ndarray,
                          cluster_id: int) -> np.ndarray:
        """
        Get points belonging to a specific cluster.

        Args:
            points: Original point cloud
            cluster_id: Cluster ID (0, 1, 2, ...)

        Returns:
            Points in the specified cluster
        """
        if self.labels_ is None:
            raise ValueError("No clustering results. Call cluster() first.")

        mask = self.labels_ == cluster_id
        return points[mask]

    def get_all_clusters(self, points: np.ndarray) -> List[np.ndarray]:
        """
        Get list of point arrays for all clusters.

        Args:
            points: Original point cloud

        Returns:
            List of point arrays, one per cluster
        """
        if self.labels_ is None:
            raise ValueError("No clustering results. Call cluster() first.")

        clusters = []
        unique_labels = np.unique(self.labels_)

        for label in unique_labels:
            if label == -1:  # Skip noise
                continue
            cluster_points = self.get_cluster_points(points, label)
            clusters.append(cluster_points)

        return clusters

    def get_num_clusters(self) -> int:
        """
        Get number of detected clusters.

        Returns:
            Number of clusters (excluding noise)
        """
        return self.num_clusters_

    def get_statistics(self) -> Dict:
        """
        Get statistics from last clustering.

        Returns:
            Dictionary with clustering statistics
        """
        return self.stats.copy()


def cluster_point_cloud(points: np.ndarray,
                       eps: float = 0.5,
                       min_samples: int = 10,
                       verbose: bool = True) -> Tuple[np.ndarray, int]:
    """
    Convenience function to cluster a point cloud.

    Args:
        points: Preprocessed point cloud (N, 4)
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN min_samples parameter
        verbose: Print progress information

    Returns:
        Tuple of (cluster_labels, num_clusters)
    """
    clusterer = PointCloudClusterer(eps=eps, min_samples=min_samples)
    labels = clusterer.cluster(points, verbose=verbose)
    return labels, clusterer.get_num_clusters()


# Example usage and testing
if __name__ == "__main__":
    from data_loader import generate_simulated_frame, print_frame_statistics
    from preprocessing import preprocess_point_cloud

    print("="*70)
    print("LIDAR CLUSTERING MODULE - TEST")
    print("="*70)

    # Generate and preprocess test data
    print("\n[Test 1] Generating and preprocessing data...")
    raw_points = generate_simulated_frame(num_points=15000, seed=42)
    processed_points = preprocess_point_cloud(raw_points, verbose=False)
    print(f"  Preprocessed points: {processed_points.shape[0]:,}")

    # Create clusterer and cluster
    print("\n[Test 2] Clustering with DBSCAN...")
    clusterer = PointCloudClusterer(
        eps=0.5,
        min_samples=10,
        min_cluster_size=10
    )
    labels = clusterer.cluster(processed_points)

    # Print cluster information
    print(f"\n[Test 3] Cluster Analysis:")
    clusters = clusterer.get_all_clusters(processed_points)
    for i, cluster in enumerate(clusters):
        centroid = np.mean(cluster[:, :3], axis=0)
        print(f"  Cluster {i}: {cluster.shape[0]:4d} points, "
              f"centroid=({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})")

    print("\n" + "="*70)
    print("Clustering module ready for use.")
    print("="*70)
