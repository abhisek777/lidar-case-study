"""
Real LiDAR Data Visualization
=============================
Cinematic visualization for Blickfeld Cube 1 dataset.

Processes real CSV data and creates:
- Dark-themed 3D visualizations
- Frame-by-frame captures
- Video output

Author: Perception Engineer
Course: Localization, Motion Planning and Sensor Fusion
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import subprocess

# Import pipeline modules
from preprocessing import preprocess_point_cloud
from clustering import cluster_point_cloud
from classification import extract_and_classify, ObjectFeatures
from tracking import MultiObjectTracker, KalmanObjectTracker
from cinematic_visualization import (
    colorize_by_distance,
    create_grid_floor,
    create_bounding_box_lines,
    create_sensor_marker,
    CLASS_COLORS,
    BBOX_COLORS
)


class RealDataVisualizer:
    """
    Visualizer for real Blickfeld LiDAR data.
    """

    def __init__(self,
                 data_dir: str,
                 output_dir: str = "real_lidar_output",
                 width: int = 1920,
                 height: int = 1080):
        """
        Initialize visualizer.

        Args:
            data_dir: Directory containing CSV files
            output_dir: Output directory for frames/video
            width: Frame width
            height: Frame height
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.width = width
        self.height = height

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Find CSV files
        self.csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        print(f"Found {len(self.csv_files)} CSV files in {data_dir}")

        # Dark theme settings
        self.bg_color = np.array([0.02, 0.02, 0.05])  # Very dark blue
        self.grid_color = [0.1, 0.1, 0.15]
        self.point_size = 2.0

        # Initialize tracker
        self.tracker = MultiObjectTracker(
            max_age=5,
            min_hits=2,
            association_threshold=3.0
        )

    def load_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Load a single frame from CSV.

        Args:
            frame_idx: Frame index

        Returns:
            Point cloud (N, 4) [X, Y, Z, INTENSITY]
        """
        if frame_idx >= len(self.csv_files):
            return None

        csv_path = self.csv_files[frame_idx]

        try:
            df = pd.read_csv(csv_path, sep=';')

            # Extract columns
            points = df[['X', 'Y', 'Z', 'INTENSITY']].values.astype(np.float32)

            # Normalize intensity to 0-1
            points[:, 3] = np.clip(points[:, 3] / 255.0, 0, 1)

            return points
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")
            return None

    def process_frame(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[ObjectFeatures]]:
        """
        Process a frame through the pipeline.

        Args:
            points: Raw point cloud

        Returns:
            Tuple of (processed_points, labels, features)
        """
        # Preprocess
        processed = preprocess_point_cloud(
            points,
            min_range=5.0,
            max_range=100.0,
            voxel_size=0.1,
            ground_threshold=0.3,
            verbose=False
        )

        # Cluster
        labels, _ = cluster_point_cloud(
            processed,
            eps=0.8,
            min_samples=15,
            verbose=False
        )

        # Classify
        features, _ = extract_and_classify(processed, labels, verbose=False)

        return processed, labels, features

    def create_geometries(self,
                         points: np.ndarray,
                         labels: np.ndarray,
                         features: List[ObjectFeatures],
                         show_grid: bool = True,
                         show_bboxes: bool = True) -> List:
        """
        Create Open3D geometries for visualization.
        """
        geometries = []

        # 1. Grid floor
        if show_grid:
            grid = create_grid_floor(
                size=150.0,
                spacing=10.0,
                height=-0.5,
                color=self.grid_color
            )
            geometries.append(grid)

        # 2. Point cloud with distance coloring
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])

        # Color by distance
        colors = colorize_by_distance(points, max_distance=80.0)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(pcd)

        # 3. Sensor marker
        sensor = create_sensor_marker(
            position=np.array([0, 0, 1.0]),
            color=[0.0, 0.8, 1.0]
        )
        geometries.append(sensor)

        # 4. Bounding boxes
        if show_bboxes:
            for feat in features:
                bbox_color = BBOX_COLORS.get(feat.classification, BBOX_COLORS['UNKNOWN'])

                # Adjust center - Z should be at ground level + half height
                center = np.array([
                    feat.center[0],
                    feat.center[1],
                    feat.height / 2
                ])

                bbox = create_bounding_box_lines(
                    center=center,
                    dimensions=np.array([feat.length, feat.width, feat.height]),
                    color=bbox_color
                )
                geometries.append(bbox)

        return geometries

    def setup_camera(self, vis, view_type: str = "overview"):
        """Set up camera view."""
        ctr = vis.get_view_control()

        if view_type == "overview":
            # Looking at the scene from behind-left
            ctr.set_front([-0.3, -0.8, 0.5])
            ctr.set_lookat([0, 30, 0])
            ctr.set_up([0, 0, 1])
            ctr.set_zoom(0.08)

        elif view_type == "side":
            ctr.set_front([-0.9, -0.3, 0.3])
            ctr.set_lookat([0, 30, 0])
            ctr.set_up([0, 0, 1])
            ctr.set_zoom(0.1)

        elif view_type == "top":
            ctr.set_front([0, 0, 1])
            ctr.set_lookat([0, 30, 0])
            ctr.set_up([0, 1, 0])
            ctr.set_zoom(0.05)

        elif view_type == "front":
            ctr.set_front([0.1, -1.0, 0.2])
            ctr.set_lookat([0, 40, 0])
            ctr.set_up([0, 0, 1])
            ctr.set_zoom(0.1)

    def capture_frame(self,
                     frame_idx: int,
                     view_type: str = "overview",
                     process: bool = True) -> Optional[str]:
        """
        Capture a single frame.

        Args:
            frame_idx: Frame index
            view_type: Camera view type
            process: Whether to run detection pipeline

        Returns:
            Path to saved frame
        """
        # Load frame
        points = self.load_frame(frame_idx)
        if points is None:
            return None

        # Process if requested
        if process:
            processed, labels, features = self.process_frame(points)
        else:
            processed = points
            labels = np.zeros(len(points), dtype=np.int32)
            features = []

        # Create geometries
        geometries = self.create_geometries(processed, labels, features)

        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=self.width, height=self.height, visible=False)

        # Set render options
        opt = vis.get_render_option()
        opt.background_color = self.bg_color
        opt.point_size = self.point_size

        # Add geometries
        for geom in geometries:
            vis.add_geometry(geom)

        # Set camera
        self.setup_camera(vis, view_type)

        # Capture
        filepath = os.path.join(self.output_dir, f"frame_{frame_idx:04d}.png")
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(filepath, do_render=True)
        vis.destroy_window()

        return filepath

    def visualize_interactive(self, frame_idx: int = 0, view_type: str = "overview"):
        """
        Show interactive visualization of a frame.
        """
        print(f"\nLoading frame {frame_idx}...")

        # Load and process
        points = self.load_frame(frame_idx)
        if points is None:
            print("Failed to load frame")
            return

        print(f"  Raw points: {len(points):,}")

        processed, labels, features = self.process_frame(points)
        print(f"  Processed points: {len(processed):,}")
        print(f"  Detected objects: {len(features)}")

        # Count by class
        vehicles = sum(1 for f in features if f.classification == 'VEHICLE')
        pedestrians = sum(1 for f in features if f.classification == 'PEDESTRIAN')
        print(f"  Vehicles: {vehicles}, Pedestrians: {pedestrians}")

        # Create geometries
        geometries = self.create_geometries(processed, labels, features)

        print("\nLaunching interactive visualization...")
        print("  Controls: Left=Rotate, Right=Pan, Scroll=Zoom, Q=Close")

        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=f"LiDAR Frame {frame_idx} - Real Data",
            width=self.width,
            height=self.height
        )

        # Set render options
        opt = vis.get_render_option()
        opt.background_color = self.bg_color
        opt.point_size = self.point_size

        # Add geometries
        for geom in geometries:
            vis.add_geometry(geom)

        # Set camera
        self.setup_camera(vis, view_type)

        # Run
        vis.run()
        vis.destroy_window()

    def generate_video_frames(self,
                             start_frame: int = 0,
                             num_frames: int = 50,
                             view_type: str = "overview") -> List[str]:
        """
        Generate frames for video.
        """
        print(f"\nGenerating {num_frames} frames starting from frame {start_frame}...")

        # Reset tracker
        KalmanObjectTracker._next_id = 0
        self.tracker = MultiObjectTracker(max_age=5, min_hits=2)

        frame_paths = []

        for i in range(num_frames):
            frame_idx = start_frame + i

            if frame_idx >= len(self.csv_files):
                print(f"  Reached end of dataset at frame {frame_idx}")
                break

            filepath = self.capture_frame(frame_idx, view_type)

            if filepath:
                frame_paths.append(filepath)

            if (i + 1) % 10 == 0:
                print(f"  Generated frame {i + 1}/{num_frames}")

        print(f"  Frames saved to: {self.output_dir}")
        return frame_paths

    def create_video(self, output_filename: str = "lidar_real_data.mp4", fps: int = 10) -> Optional[str]:
        """
        Create video from captured frames.
        """
        output_path = os.path.join(self.output_dir, output_filename)
        frame_pattern = os.path.join(self.output_dir, "frame_%04d.png")

        print(f"\nCreating video: {output_path}")

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            output_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  Video saved: {output_path}")
                return output_path
            else:
                print(f"  FFmpeg error: {result.stderr[:200]}")
                return None
        except FileNotFoundError:
            print("  FFmpeg not found. Frames saved in output directory.")
            return None


def run_real_data_demo(data_dir: str, output_dir: str = "real_lidar_output"):
    """
    Run visualization demo on real data.
    """
    print("="*70)
    print("REAL LIDAR DATA VISUALIZATION")
    print("="*70)

    # Create visualizer
    viz = RealDataVisualizer(
        data_dir=data_dir,
        output_dir=output_dir,
        width=1920,
        height=1080
    )

    if len(viz.csv_files) == 0:
        print("No CSV files found!")
        return

    # Show interactive visualization of first frame
    viz.visualize_interactive(frame_idx=0, view_type="overview")


def run_video_generation(data_dir: str,
                        output_dir: str = "real_lidar_video",
                        num_frames: int = 50,
                        fps: int = 10):
    """
    Generate video from real data.
    """
    print("="*70)
    print("GENERATING VIDEO FROM REAL LIDAR DATA")
    print("="*70)

    # Create visualizer
    viz = RealDataVisualizer(
        data_dir=data_dir,
        output_dir=output_dir,
        width=1920,
        height=1080
    )

    if len(viz.csv_files) == 0:
        print("No CSV files found!")
        return

    # Generate frames
    frame_paths = viz.generate_video_frames(
        start_frame=0,
        num_frames=num_frames,
        view_type="overview"
    )

    # Create video
    video_path = viz.create_video(
        output_filename="lidar_detection.mp4",
        fps=fps
    )

    print("\n" + "="*70)
    print("VIDEO GENERATION COMPLETE")
    print("="*70)
    print(f"  Frames: {len(frame_paths)}")
    print(f"  Video: {video_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Real LiDAR Data Visualization")
    parser.add_argument("--data", "-d", type=str, required=True,
                       help="Directory containing CSV files")
    parser.add_argument("--mode", "-m", choices=["interactive", "video", "frames"],
                       default="interactive", help="Visualization mode")
    parser.add_argument("--frames", "-n", type=int, default=50,
                       help="Number of frames for video")
    parser.add_argument("--fps", type=int, default=10, help="Video frame rate")
    parser.add_argument("--output", "-o", type=str, default="real_lidar_output",
                       help="Output directory")

    args = parser.parse_args()

    if args.mode == "interactive":
        run_real_data_demo(args.data, args.output)
    elif args.mode == "video":
        run_video_generation(args.data, args.output, args.frames, args.fps)
    elif args.mode == "frames":
        viz = RealDataVisualizer(args.data, args.output)
        for i in range(min(args.frames, len(viz.csv_files))):
            viz.capture_frame(i)
            if (i + 1) % 10 == 0:
                print(f"Captured frame {i + 1}")
