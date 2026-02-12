"""
LiDAR Data Loading Module
=========================
Autonomous Driving Perception Pipeline - Data Loading

This module handles loading LiDAR data from Blickfeld Cube 1 sensor.
- Extracts ZIP archives containing CSV files
- Loads CSV files with semicolon separator
- Parses X, Y, Z, DISTANCE, INTENSITY, TIMESTAMP columns

Dataset Format:
- CSV files with semicolon (;) separator
- Columns: X, Y, Z, DISTANCE, INTENSITY, TIMESTAMP
- Coordinates in meters

Author: Perception Engineer
Course: Localization, Motion Planning and Sensor Fusion
"""

import os
import zipfile
import glob
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Generator


class BlickfeldDataLoader:
    """
    Data loader for Blickfeld Cube 1 LiDAR sensor data.

    Handles ZIP extraction and CSV parsing for LiDAR point cloud frames.
    """

    # Expected CSV columns from Blickfeld Cube 1
    EXPECTED_COLUMNS = ['X', 'Y', 'Z', 'DISTANCE', 'INTENSITY', 'TIMESTAMP']

    def __init__(self, data_path: str, extract_dir: Optional[str] = None):
        """
        Initialize the data loader.

        Args:
            data_path: Path to ZIP file or directory containing CSV files
            extract_dir: Directory to extract ZIP files to (default: same as ZIP location)
        """
        self.data_path = data_path
        self.extract_dir = extract_dir
        self.csv_files: List[str] = []
        self.current_frame_index = 0

        # Initialize data source
        self._setup_data_source()

    def _setup_data_source(self) -> None:
        """
        Set up the data source by extracting ZIP or finding CSV files.
        """
        if self.data_path.endswith('.zip'):
            self._extract_zip()
        elif os.path.isdir(self.data_path):
            self._find_csv_files(self.data_path)
        else:
            raise ValueError(f"Invalid data path: {self.data_path}. "
                           f"Expected ZIP file or directory.")

        if len(self.csv_files) == 0:
            print(f"Warning: No CSV files found in {self.data_path}")
            print("Using simulated data for demonstration...")
        else:
            print(f"Found {len(self.csv_files)} LiDAR frames")

    def _extract_zip(self) -> None:
        """
        Extract ZIP archive containing CSV files.
        """
        if not os.path.exists(self.data_path):
            print(f"ZIP file not found: {self.data_path}")
            return

        # Determine extraction directory
        if self.extract_dir is None:
            self.extract_dir = os.path.splitext(self.data_path)[0]

        # Extract if not already done
        if not os.path.exists(self.extract_dir):
            print(f"Extracting ZIP archive to: {self.extract_dir}")
            with zipfile.ZipFile(self.data_path, 'r') as zip_ref:
                zip_ref.extractall(self.extract_dir)
            print("Extraction complete.")
        else:
            print(f"Using existing extracted data: {self.extract_dir}")

        # Find CSV files in extracted directory
        self._find_csv_files(self.extract_dir)

    def _find_csv_files(self, directory: str) -> None:
        """
        Find all CSV files in the given directory.

        Args:
            directory: Directory to search for CSV files
        """
        # Search for CSV files recursively
        csv_pattern = os.path.join(directory, '**', '*.csv')
        self.csv_files = sorted(glob.glob(csv_pattern, recursive=True))

        # Also check for .CSV extension (case insensitive)
        csv_pattern_upper = os.path.join(directory, '**', '*.CSV')
        self.csv_files.extend(sorted(glob.glob(csv_pattern_upper, recursive=True)))

        # Remove duplicates and sort by filename
        self.csv_files = sorted(list(set(self.csv_files)))

        print(f"Found {len(self.csv_files)} CSV files in {directory}")

    def load_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """
        Load a single LiDAR frame from CSV file.

        Args:
            frame_index: Index of the frame to load (0-based)

        Returns:
            Point cloud array of shape (N, 4) with columns [X, Y, Z, INTENSITY]
            Returns None if frame cannot be loaded
        """
        if frame_index < 0 or frame_index >= len(self.csv_files):
            print(f"Frame index {frame_index} out of range [0, {len(self.csv_files)-1}]")
            return None

        csv_path = self.csv_files[frame_index]
        return self._load_csv(csv_path)

    def _load_csv(self, csv_path: str) -> Optional[np.ndarray]:
        """
        Load point cloud from a CSV file.

        Args:
            csv_path: Path to CSV file

        Returns:
            Point cloud array of shape (N, 4) with columns [X, Y, Z, INTENSITY]
        """
        try:
            # Load CSV with semicolon separator (Blickfeld format)
            df = pd.read_csv(csv_path, sep=';')

            # Check for expected columns (case-insensitive)
            df.columns = df.columns.str.upper().str.strip()

            # Extract required columns
            required_cols = ['X', 'Y', 'Z']
            for col in required_cols:
                if col not in df.columns:
                    print(f"Missing required column '{col}' in {csv_path}")
                    return None

            # Extract X, Y, Z coordinates
            x = df['X'].values
            y = df['Y'].values
            z = df['Z'].values

            # Extract intensity if available, otherwise use default
            if 'INTENSITY' in df.columns:
                intensity = df['INTENSITY'].values
            else:
                intensity = np.ones(len(x)) * 0.5  # Default intensity

            # Combine into point cloud array
            points = np.column_stack([x, y, z, intensity])

            # Remove NaN values
            valid_mask = ~np.isnan(points).any(axis=1)
            points = points[valid_mask]

            return points.astype(np.float32)

        except Exception as e:
            print(f"Error loading {csv_path}: {e}")
            return None

    def load_all_frames(self) -> List[np.ndarray]:
        """
        Load all LiDAR frames from CSV files.

        Returns:
            List of point cloud arrays
        """
        frames = []
        for i in range(len(self.csv_files)):
            frame = self.load_frame(i)
            if frame is not None:
                frames.append(frame)
        return frames

    def frame_generator(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Generator that yields frames one at a time.

        Yields:
            Tuple of (frame_index, point_cloud_array)
        """
        for i in range(len(self.csv_files)):
            frame = self.load_frame(i)
            if frame is not None:
                yield i, frame

    def get_num_frames(self) -> int:
        """
        Get total number of available frames.

        Returns:
            Number of CSV files (frames)
        """
        return len(self.csv_files)

    def get_frame_path(self, frame_index: int) -> Optional[str]:
        """
        Get the file path of a specific frame.

        Args:
            frame_index: Index of the frame

        Returns:
            Path to the CSV file
        """
        if 0 <= frame_index < len(self.csv_files):
            return self.csv_files[frame_index]
        return None


def generate_simulated_frame(num_points: int = 15000,
                            seed: Optional[int] = None,
                            frame_index: int = 0) -> np.ndarray:
    """
    Generate a simulated LiDAR frame for testing when real data is unavailable.

    Creates a realistic urban driving scene with:
    - Ground plane
    - Vehicles at various positions
    - Pedestrians
    - Background noise

    Args:
        num_points: Total number of points to generate
        seed: Random seed for reproducibility
        frame_index: Frame index for time-varying simulation

    Returns:
        Point cloud array of shape (N, 4) with columns [X, Y, Z, INTENSITY]
    """
    if seed is not None:
        np.random.seed(seed + frame_index)

    points_list = []

    # 1. Ground plane (40% of points)
    num_ground = int(num_points * 0.4)
    ground_x = np.random.uniform(-30, 50, num_ground)
    ground_y = np.random.uniform(-20, 20, num_ground)
    ground_z = np.random.normal(0, 0.05, num_ground)  # Slight variation around z=0
    ground_intensity = np.random.uniform(0.2, 0.4, num_ground)
    ground_points = np.column_stack([ground_x, ground_y, ground_z, ground_intensity])
    points_list.append(ground_points)

    # 2. Vehicles (35% of points) - 3-5 vehicles
    num_vehicles = np.random.randint(3, 6)
    points_per_vehicle = int(num_points * 0.35) // num_vehicles

    for v in range(num_vehicles):
        # Vehicle position (moves slightly with frame_index)
        base_x = np.random.uniform(10, 40)
        base_y = np.random.uniform(-15, 15)

        # Add motion (vehicles moving forward)
        velocity_x = np.random.uniform(-0.5, 2.0)  # m/frame
        veh_center_x = base_x + velocity_x * frame_index * 0.1
        veh_center_y = base_y

        # Vehicle dimensions (typical car: 4.5m x 1.8m x 1.5m)
        length = np.random.uniform(4.0, 5.5)
        width = np.random.uniform(1.6, 2.0)
        height = np.random.uniform(1.3, 1.8)

        veh_x = np.random.uniform(veh_center_x - length/2, veh_center_x + length/2, points_per_vehicle)
        veh_y = np.random.uniform(veh_center_y - width/2, veh_center_y + width/2, points_per_vehicle)
        veh_z = np.random.uniform(0, height, points_per_vehicle)
        veh_intensity = np.random.uniform(0.5, 0.9, points_per_vehicle)

        vehicle_points = np.column_stack([veh_x, veh_y, veh_z, veh_intensity])
        points_list.append(vehicle_points)

    # 3. Pedestrians (15% of points) - 2-4 pedestrians
    num_pedestrians = np.random.randint(2, 5)
    points_per_pedestrian = int(num_points * 0.15) // num_pedestrians

    for p in range(num_pedestrians):
        # Pedestrian position
        base_x = np.random.uniform(5, 25)
        base_y = np.random.uniform(-10, 10)

        # Pedestrian motion (slow walking)
        velocity = np.random.uniform(-0.2, 0.5)
        ped_center_x = base_x + velocity * frame_index * 0.1
        ped_center_y = base_y

        # Pedestrian dimensions (0.5m x 0.5m x 1.7m)
        ped_x = np.random.uniform(ped_center_x - 0.25, ped_center_x + 0.25, points_per_pedestrian)
        ped_y = np.random.uniform(ped_center_y - 0.25, ped_center_y + 0.25, points_per_pedestrian)
        ped_z = np.random.uniform(0, 1.7, points_per_pedestrian)
        ped_intensity = np.random.uniform(0.4, 0.7, points_per_pedestrian)

        pedestrian_points = np.column_stack([ped_x, ped_y, ped_z, ped_intensity])
        points_list.append(pedestrian_points)

    # 4. Background/noise (10% of points)
    num_noise = num_points - sum(p.shape[0] for p in points_list)
    if num_noise > 0:
        noise_x = np.random.uniform(-50, 80, num_noise)
        noise_y = np.random.uniform(-30, 30, num_noise)
        noise_z = np.random.uniform(-1, 5, num_noise)
        noise_intensity = np.random.uniform(0.1, 0.3, num_noise)
        noise_points = np.column_stack([noise_x, noise_y, noise_z, noise_intensity])
        points_list.append(noise_points)

    # Combine all points
    point_cloud = np.vstack(points_list).astype(np.float32)

    return point_cloud


def print_frame_statistics(points: np.ndarray, frame_index: int = 0) -> None:
    """
    Print statistics about a point cloud frame.

    Args:
        points: Point cloud array of shape (N, 4)
        frame_index: Frame number for display
    """
    print(f"\n--- Frame {frame_index} Statistics ---")
    print(f"  Total points: {points.shape[0]:,}")
    print(f"  X range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}] m")
    print(f"  Y range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}] m")
    print(f"  Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}] m")
    print(f"  Intensity range: [{points[:, 3].min():.2f}, {points[:, 3].max():.2f}]")

    # Calculate distance from sensor
    distances = np.linalg.norm(points[:, :3], axis=1)
    print(f"  Distance range: [{distances.min():.2f}, {distances.max():.2f}] m")
    print(f"  Mean distance: {distances.mean():.2f} m")


# Example usage and testing
if __name__ == "__main__":
    print("="*70)
    print("LIDAR DATA LOADER - TEST")
    print("="*70)

    # Test with simulated data
    print("\n[Test 1] Generating simulated LiDAR frame...")
    frame = generate_simulated_frame(num_points=15000, seed=42, frame_index=0)
    print_frame_statistics(frame, frame_index=0)

    # Test frame generator
    print("\n[Test 2] Testing frame generator with simulated sequence...")
    for i in range(3):
        frame = generate_simulated_frame(num_points=15000, seed=42, frame_index=i)
        print(f"  Frame {i}: {frame.shape[0]:,} points")

    print("\n" + "="*70)
    print("Data loader ready for use with Blickfeld Cube 1 CSV files.")
    print("="*70)
