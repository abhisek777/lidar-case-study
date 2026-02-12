"""
LiDAR Preprocessing Module
==========================
Autonomous Driving Perception Pipeline - Preprocessing

This module implements preprocessing steps for LiDAR point clouds:
1. Range filtering (5-250m as per assignment specification)
2. Ground plane removal using RANSAC
3. Noise filtering using statistical outlier removal
4. Voxel grid downsampling for computational efficiency

Author: Perception Engineer
Course: Localization, Motion Planning and Sensor Fusion
"""

import numpy as np
import open3d as o3d
import time
from typing import Tuple, Optional


class LiDARPreprocessor:
    """
    LiDAR point cloud preprocessor.

    Implements standard preprocessing pipeline for autonomous driving:
    - Range filtering
    - Ground plane removal (RANSAC)
    - Statistical outlier removal
    - Voxel grid downsampling
    """

    def __init__(self,
                 min_range: float = 5.0,
                 max_range: float = 250.0,
                 voxel_size: float = 0.1,
                 ground_threshold: float = 0.2,
                 ransac_iterations: int = 1000):
        """
        Initialize preprocessor with parameters.

        Args:
            min_range: Minimum range in meters (removes near-field noise)
            max_range: Maximum range in meters (removes unreliable distant points)
            voxel_size: Voxel size for downsampling in meters
            ground_threshold: Distance threshold for RANSAC ground fitting
            ransac_iterations: Number of RANSAC iterations
        """
        self.min_range = min_range
        self.max_range = max_range
        self.voxel_size = voxel_size
        self.ground_threshold = ground_threshold
        self.ransac_iterations = ransac_iterations

        # Store statistics from last processing
        self.stats = {}

    def preprocess(self, points: np.ndarray, verbose: bool = True) -> np.ndarray:
        """
        Apply full preprocessing pipeline.

        Pipeline steps:
        1. Range filtering (5-250m)
        2. Voxel grid downsampling
        3. Ground plane removal (RANSAC)
        4. Statistical outlier removal

        Args:
            points: Input point cloud (N, 4) [X, Y, Z, INTENSITY]
            verbose: Print progress information

        Returns:
            Preprocessed point cloud (M, 4) where M < N
        """
        self.stats = {'original_points': points.shape[0]}
        start_time = time.time()

        if verbose:
            print("\n" + "="*60)
            print("PREPROCESSING PIPELINE")
            print("="*60)

        # Step 1: Range filtering
        points = self.filter_by_range(points, verbose)

        # Step 2: Voxel grid downsampling
        points = self.voxel_downsample(points, verbose)

        # Step 3: Ground plane removal
        points, ground_points = self.remove_ground_ransac(points, verbose)

        # Step 4: Statistical outlier removal
        points = self.remove_statistical_outliers(points, verbose)

        elapsed_time = time.time() - start_time
        self.stats['final_points'] = points.shape[0]
        self.stats['processing_time'] = elapsed_time

        if verbose:
            print("\n" + "-"*60)
            print("PREPROCESSING SUMMARY")
            print("-"*60)
            print(f"  Original points:    {self.stats['original_points']:,}")
            print(f"  Final points:       {self.stats['final_points']:,}")
            print(f"  Reduction ratio:    {self.stats['original_points']/max(1, self.stats['final_points']):.2f}x")
            print(f"  Processing time:    {elapsed_time*1000:.1f} ms")
            print("="*60 + "\n")

        return points

    def filter_by_range(self, points: np.ndarray, verbose: bool = True) -> np.ndarray:
        """
        Filter points by distance from sensor.

        Removes:
        - Points too close (< min_range): Near-field noise, sensor artifacts
        - Points too far (> max_range): Unreliable distant measurements

        Args:
            points: Input point cloud (N, 4)
            verbose: Print progress information

        Returns:
            Filtered point cloud
        """
        # Calculate Euclidean distance from sensor (assumed at origin)
        distances = np.linalg.norm(points[:, :3], axis=1)

        # Create valid range mask
        valid_mask = (distances >= self.min_range) & (distances <= self.max_range)

        # Filter points
        filtered_points = points[valid_mask]

        if verbose:
            removed = points.shape[0] - filtered_points.shape[0]
            print(f"\n[Step 1] Range Filtering ({self.min_range}m - {self.max_range}m)")
            print(f"  Points before: {points.shape[0]:,}")
            print(f"  Points after:  {filtered_points.shape[0]:,}")
            print(f"  Removed:       {removed:,} ({100*removed/max(1, points.shape[0]):.1f}%)")

        self.stats['after_range_filter'] = filtered_points.shape[0]
        return filtered_points

    def voxel_downsample(self, points: np.ndarray, verbose: bool = True) -> np.ndarray:
        """
        Downsample point cloud using voxel grid filtering.

        Benefits:
        - Reduces computational load
        - Creates uniform point density
        - Reduces noise through averaging

        Args:
            points: Input point cloud (N, 4)
            verbose: Print progress information

        Returns:
            Downsampled point cloud
        """
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])

        # Store intensity as colors for preservation during downsampling
        intensity_normalized = np.clip(points[:, 3:4], 0, 1)
        intensity_rgb = np.repeat(intensity_normalized, 3, axis=1)
        pcd.colors = o3d.utility.Vector3dVector(intensity_rgb)

        # Apply voxel downsampling
        pcd_down = pcd.voxel_down_sample(voxel_size=self.voxel_size)

        # Extract downsampled points
        points_down = np.asarray(pcd_down.points)
        intensity_down = np.asarray(pcd_down.colors)[:, 0:1]

        # Combine XYZ and intensity
        downsampled = np.hstack([points_down, intensity_down])

        if verbose:
            reduction = points.shape[0] / max(1, downsampled.shape[0])
            print(f"\n[Step 2] Voxel Downsampling (voxel size: {self.voxel_size}m)")
            print(f"  Points before: {points.shape[0]:,}")
            print(f"  Points after:  {downsampled.shape[0]:,}")
            print(f"  Reduction:     {reduction:.2f}x")

        self.stats['after_voxel'] = downsampled.shape[0]
        return downsampled.astype(np.float32)

    def remove_ground_ransac(self, points: np.ndarray,
                             verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove ground plane using RANSAC plane fitting.

        RANSAC (Random Sample Consensus):
        1. Randomly sample 3 points to fit a plane
        2. Count inliers (points within threshold distance)
        3. Repeat and keep best plane model
        4. Points on the plane are ground, others are objects

        Args:
            points: Input point cloud (N, 4)
            verbose: Print progress information

        Returns:
            Tuple of (non_ground_points, ground_points)
        """
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])

        # Fit plane using RANSAC
        try:
            plane_model, inlier_indices = pcd.segment_plane(
                distance_threshold=self.ground_threshold,
                ransac_n=3,
                num_iterations=self.ransac_iterations
            )
        except Exception as e:
            if verbose:
                print(f"\n[Step 3] Ground Removal - RANSAC failed: {e}")
                print("  Skipping ground removal...")
            return points, np.array([])

        # Extract plane equation coefficients
        [a, b, c, d] = plane_model

        # Create masks for ground and non-ground points
        inlier_indices = np.array(inlier_indices)
        ground_mask = np.zeros(points.shape[0], dtype=bool)
        ground_mask[inlier_indices] = True

        # Separate ground and non-ground points
        ground_points = points[ground_mask]
        non_ground_points = points[~ground_mask]

        if verbose:
            print(f"\n[Step 3] Ground Plane Removal (RANSAC)")
            print(f"  Plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
            print(f"  Ground normal:  [{a:.3f}, {b:.3f}, {c:.3f}]")
            print(f"  Points before:  {points.shape[0]:,}")
            print(f"  Ground points:  {ground_points.shape[0]:,} ({100*ground_points.shape[0]/max(1, points.shape[0]):.1f}%)")
            print(f"  Object points:  {non_ground_points.shape[0]:,}")

            # Check if ground is roughly horizontal
            if abs(c) < 0.7:
                print(f"  Warning: Ground plane may not be horizontal (z-component: {c:.3f})")

        self.stats['ground_points'] = ground_points.shape[0]
        self.stats['after_ground_removal'] = non_ground_points.shape[0]
        return non_ground_points, ground_points

    def remove_statistical_outliers(self, points: np.ndarray,
                                    nb_neighbors: int = 20,
                                    std_ratio: float = 2.0,
                                    verbose: bool = True) -> np.ndarray:
        """
        Remove statistical outliers (noise points).

        Points are removed if their average distance to neighbors
        is larger than std_ratio * standard deviation of all distances.

        Args:
            points: Input point cloud (N, 4)
            nb_neighbors: Number of neighbors to analyze
            std_ratio: Standard deviation multiplier for threshold
            verbose: Print progress information

        Returns:
            Filtered point cloud
        """
        # Need at least nb_neighbors + 1 points for statistical outlier removal
        min_points_required = nb_neighbors + 1
        if points.shape[0] < min_points_required:
            if verbose:
                print(f"\n[Step 4] Statistical Outlier Removal - Skipped (too few points: {points.shape[0]})")
            return points

        # Ensure valid parameters
        nb_neighbors = max(1, min(nb_neighbors, points.shape[0] - 1))
        std_ratio = max(0.1, std_ratio)

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])

        try:
            # Apply statistical outlier removal
            filtered_pcd, inlier_indices = pcd.remove_statistical_outlier(
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio
            )
            # Convert to list of integers for indexing
            inlier_indices = list(inlier_indices)
        except Exception as e:
            if verbose:
                print(f"\n[Step 4] Statistical Outlier Removal - Skipped (error: {e})")
            self.stats['after_outlier_removal'] = points.shape[0]
            return points

        # Filter points using valid indices
        if len(inlier_indices) == 0:
            if verbose:
                print(f"\n[Step 4] Statistical Outlier Removal - Skipped (all points are inliers)")
            self.stats['after_outlier_removal'] = points.shape[0]
            return points

        filtered_points = points[inlier_indices]

        if verbose:
            removed = points.shape[0] - filtered_points.shape[0]
            print(f"\n[Step 4] Statistical Outlier Removal")
            print(f"  Points before: {points.shape[0]:,}")
            print(f"  Points after:  {filtered_points.shape[0]:,}")
            print(f"  Outliers:      {removed:,} ({100*removed/max(1, points.shape[0]):.1f}%)")

        self.stats['after_outlier_removal'] = filtered_points.shape[0]
        return filtered_points

    def get_stats(self) -> dict:
        """
        Get statistics from last preprocessing run.

        Returns:
            Dictionary with processing statistics
        """
        return self.stats.copy()


def preprocess_point_cloud(points: np.ndarray,
                          min_range: float = 5.0,
                          max_range: float = 250.0,
                          voxel_size: float = 0.1,
                          ground_threshold: float = 0.2,
                          verbose: bool = True) -> np.ndarray:
    """
    Convenience function to preprocess a point cloud.

    Args:
        points: Input point cloud (N, 4)
        min_range: Minimum range filter
        max_range: Maximum range filter
        voxel_size: Voxel size for downsampling
        ground_threshold: RANSAC threshold for ground fitting
        verbose: Print progress information

    Returns:
        Preprocessed point cloud
    """
    preprocessor = LiDARPreprocessor(
        min_range=min_range,
        max_range=max_range,
        voxel_size=voxel_size,
        ground_threshold=ground_threshold
    )
    return preprocessor.preprocess(points, verbose=verbose)


# Example usage and testing
if __name__ == "__main__":
    from data_loader import generate_simulated_frame, print_frame_statistics

    print("="*70)
    print("LIDAR PREPROCESSING MODULE - TEST")
    print("="*70)

    # Generate test data
    print("\n[Test] Generating simulated LiDAR frame...")
    raw_points = generate_simulated_frame(num_points=15000, seed=42)
    print_frame_statistics(raw_points, frame_index=0)

    # Create preprocessor
    preprocessor = LiDARPreprocessor(
        min_range=5.0,
        max_range=250.0,
        voxel_size=0.1,
        ground_threshold=0.2
    )

    # Apply preprocessing
    processed_points = preprocessor.preprocess(raw_points)

    # Print final statistics
    print_frame_statistics(processed_points, frame_index=0)

    print("\n" + "="*70)
    print("Preprocessing module ready for use.")
    print("="*70)
