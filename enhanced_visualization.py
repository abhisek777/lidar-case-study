"""
Enhanced Visualization Module
=============================
3D visualization with tracking features: IDs, velocity arrows, trajectories.

Features:
1. Bounding boxes with object IDs
2. Velocity arrows showing motion direction
3. Trajectory trails for each track
4. Class-based coloring
5. Frame capture and video generation

Course: Localization, Motion Planning and Sensor Fusion
Assignment: LiDAR-based Object Detection and Tracking Pipeline
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from typing import List, Tuple, Optional, Dict
import os
import subprocess

from enhanced_tracking import EnhancedTrackState


# Color scheme
CLASS_COLORS = {
    'VEHICLE': [0.0, 0.8, 0.2],      # Green
    'PEDESTRIAN': [1.0, 0.4, 0.0],   # Orange
    'UNKNOWN': [0.5, 0.5, 0.5]       # Gray
}

BBOX_COLORS = {
    'VEHICLE': [0.0, 1.0, 0.3],
    'PEDESTRIAN': [1.0, 0.5, 0.0],
    'UNKNOWN': [0.7, 0.7, 0.7]
}

TRAJECTORY_COLORS = {
    'VEHICLE': [0.2, 0.8, 0.4],
    'PEDESTRIAN': [1.0, 0.6, 0.2],
    'UNKNOWN': [0.6, 0.6, 0.6]
}


def create_velocity_arrow(start: np.ndarray, velocity: np.ndarray,
                         color: List[float] = [1, 0, 0],
                         scale: float = 2.0) -> o3d.geometry.LineSet:
    """
    Create an arrow geometry representing velocity vector.

    Args:
        start: Arrow start position [x, y, z]
        velocity: Velocity vector [vx, vy]
        color: Arrow color
        scale: Scale factor for arrow length

    Returns:
        Open3D LineSet representing the arrow
    """
    # Arrow end point
    end = start.copy()
    end[0] += velocity[0] * scale
    end[1] += velocity[1] * scale

    # Arrow body
    points = [start, end]

    # Arrow head
    direction = np.array([velocity[0], velocity[1], 0])
    length = np.linalg.norm(direction)

    if length > 0.1:
        direction = direction / length
        head_length = min(0.5, length * scale * 0.3)

        # Perpendicular vector for arrow head
        perp = np.array([-direction[1], direction[0], 0])

        head_left = end - direction * head_length + perp * head_length * 0.5
        head_right = end - direction * head_length - perp * head_length * 0.5

        points.extend([head_left, end, head_right, end])

    # Create line set
    lines = [[0, 1]]
    if len(points) > 2:
        lines.extend([[2, 3], [4, 5]])

    arrow = o3d.geometry.LineSet()
    arrow.points = o3d.utility.Vector3dVector(points)
    arrow.lines = o3d.utility.Vector2iVector(lines)
    arrow.colors = o3d.utility.Vector3dVector([color] * len(lines))

    return arrow


def create_trajectory_line(positions: List[np.ndarray],
                          color: List[float] = [0, 1, 0],
                          fade: bool = True) -> o3d.geometry.LineSet:
    """
    Create a trajectory line from position history.

    Args:
        positions: List of 3D positions
        color: Base color
        fade: Whether to fade older positions

    Returns:
        Open3D LineSet representing the trajectory
    """
    if len(positions) < 2:
        return None

    points = [np.array(p) for p in positions]
    lines = [[i, i+1] for i in range(len(points)-1)]

    trajectory = o3d.geometry.LineSet()
    trajectory.points = o3d.utility.Vector3dVector(points)
    trajectory.lines = o3d.utility.Vector2iVector(lines)

    # Color with fade effect
    if fade:
        colors = []
        for i in range(len(lines)):
            alpha = (i + 1) / len(lines)  # Newer = brighter
            c = [color[j] * alpha for j in range(3)]
            colors.append(c)
        trajectory.colors = o3d.utility.Vector3dVector(colors)
    else:
        trajectory.colors = o3d.utility.Vector3dVector([color] * len(lines))

    return trajectory


def create_bounding_box_with_id(center: np.ndarray,
                                dimensions: np.ndarray,
                                track_id: int,
                                classification: str,
                                velocity: np.ndarray = None) -> List:
    """
    Create bounding box with ID label.

    Args:
        center: Box center [x, y, z]
        dimensions: Box size [length, width, height]
        track_id: Track ID
        classification: Object class
        velocity: Optional velocity for arrow

    Returns:
        List of Open3D geometries
    """
    geometries = []

    # Bounding box
    color = BBOX_COLORS.get(classification, BBOX_COLORS['UNKNOWN'])

    # Create oriented bounding box
    bbox = o3d.geometry.OrientedBoundingBox(
        center=center,
        R=np.eye(3),
        extent=dimensions
    )
    bbox.color = color
    geometries.append(bbox)

    # Create wireframe version
    line_box = o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox)
    line_box.paint_uniform_color(color)
    geometries.append(line_box)

    # Velocity arrow (if moving)
    if velocity is not None and np.linalg.norm(velocity) > 0.3:
        arrow = create_velocity_arrow(
            start=center,
            velocity=velocity,
            color=[1.0, 0.2, 0.2],
            scale=2.0
        )
        geometries.append(arrow)

    return geometries


class EnhancedVisualizer:
    """
    Enhanced 3D visualizer with tracking visualization features.
    """

    def __init__(self,
                 width: int = 1920,
                 height: int = 1080,
                 background_color: np.ndarray = np.array([0.02, 0.02, 0.05])):
        """
        Initialize visualizer.

        Args:
            width: Window width
            height: Window height
            background_color: Background RGB color
        """
        self.width = width
        self.height = height
        self.background_color = background_color
        self.point_size = 2.0

    def visualize_frame(self,
                       points: np.ndarray,
                       tracks: List[EnhancedTrackState],
                       frame_idx: int = 0,
                       show_trajectories: bool = True,
                       show_velocity: bool = True,
                       show_grid: bool = True,
                       interactive: bool = True,
                       save_path: str = None) -> Optional[str]:
        """
        Visualize a single frame with tracking information.

        Args:
            points: Point cloud (N, 4)
            tracks: List of track states
            frame_idx: Frame index
            show_trajectories: Show trajectory trails
            show_velocity: Show velocity arrows
            show_grid: Show ground grid
            interactive: Show interactive window
            save_path: Path to save frame

        Returns:
            Path to saved frame if save_path provided
        """
        geometries = []

        # 1. Ground grid
        if show_grid:
            grid = self._create_grid(size=150, spacing=10, height=-0.5)
            geometries.append(grid)

        # 2. Point cloud with distance coloring
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])

        # Color by distance
        distances = np.linalg.norm(points[:, :3], axis=1)
        max_dist = 80.0
        normalized = np.clip(distances / max_dist, 0, 1)

        # Blue (near) -> Green -> Yellow -> Red (far)
        colors = np.zeros((len(points), 3))
        colors[:, 0] = normalized  # Red increases with distance
        colors[:, 1] = 0.5 + 0.5 * (1 - normalized)  # Green
        colors[:, 2] = 1 - normalized  # Blue decreases

        pcd.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(pcd)

        # 3. Sensor marker
        sensor = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
        sensor.translate([0, 0, 1.0])
        sensor.paint_uniform_color([0.0, 0.8, 1.0])
        geometries.append(sensor)

        # 4. Track visualizations
        for track in tracks:
            # Bounding box
            box_geoms = create_bounding_box_with_id(
                center=track.position,
                dimensions=track.dimensions,
                track_id=track.track_id,
                classification=track.classification,
                velocity=track.velocity if show_velocity else None
            )
            geometries.extend(box_geoms)

            # Trajectory
            if show_trajectories and len(track.trajectory) >= 2:
                traj_color = TRAJECTORY_COLORS.get(
                    track.classification, TRAJECTORY_COLORS['UNKNOWN']
                )
                traj = create_trajectory_line(track.trajectory, color=traj_color)
                if traj:
                    geometries.append(traj)

        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=f"LiDAR Frame {frame_idx}",
            width=self.width,
            height=self.height,
            visible=interactive
        )

        # Render options
        opt = vis.get_render_option()
        opt.background_color = self.background_color
        opt.point_size = self.point_size

        # Add geometries
        for geom in geometries:
            vis.add_geometry(geom)

        # Set camera
        self._setup_camera(vis)

        if interactive:
            vis.run()
        else:
            vis.poll_events()
            vis.update_renderer()

        # Save frame
        saved_path = None
        if save_path:
            vis.capture_screen_image(save_path, do_render=True)
            saved_path = save_path

        vis.destroy_window()

        return saved_path

    def _create_grid(self, size: float = 100, spacing: float = 10,
                    height: float = -0.5) -> o3d.geometry.LineSet:
        """Create ground grid."""
        lines = []
        points = []

        # Create grid lines
        for i in range(int(-size/spacing), int(size/spacing) + 1):
            # Lines parallel to Y axis
            p1 = [i * spacing, -size, height]
            p2 = [i * spacing, size, height]
            idx = len(points)
            points.extend([p1, p2])
            lines.append([idx, idx + 1])

            # Lines parallel to X axis
            p3 = [-size, i * spacing, height]
            p4 = [size, i * spacing, height]
            idx = len(points)
            points.extend([p3, p4])
            lines.append([idx, idx + 1])

        grid = o3d.geometry.LineSet()
        grid.points = o3d.utility.Vector3dVector(points)
        grid.lines = o3d.utility.Vector2iVector(lines)
        grid.paint_uniform_color([0.1, 0.1, 0.15])

        return grid

    def _setup_camera(self, vis):
        """Set up camera view."""
        ctr = vis.get_view_control()
        ctr.set_front([-0.3, -0.8, 0.5])
        ctr.set_lookat([0, 30, 0])
        ctr.set_up([0, 0, 1])
        ctr.set_zoom(0.08)


class BEVVisualizer:
    """
    Bird's Eye View visualizer with tracking information.
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 10)):
        """Initialize BEV visualizer."""
        self.figsize = figsize

    def visualize_frame(self,
                       points: np.ndarray,
                       tracks: List[EnhancedTrackState],
                       frame_idx: int = 0,
                       x_range: Tuple[float, float] = (-50, 50),
                       y_range: Tuple[float, float] = (-10, 100),
                       show_ids: bool = True,
                       show_velocity: bool = True,
                       show_trajectories: bool = True,
                       save_path: str = None) -> Optional[str]:
        """
        Create Bird's Eye View visualization.

        Args:
            points: Point cloud (N, 4)
            tracks: List of track states
            frame_idx: Frame index
            x_range: X axis range
            y_range: Y axis range
            show_ids: Show track IDs
            show_velocity: Show velocity arrows
            show_trajectories: Show trajectories
            save_path: Path to save figure

        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Dark theme
        fig.patch.set_facecolor('#0a0a14')
        ax.set_facecolor('#0a0a14')

        # Point cloud
        x, y = points[:, 0], points[:, 1]
        distances = np.sqrt(x**2 + y**2)
        scatter = ax.scatter(x, y, c=distances, cmap='viridis',
                           s=0.5, alpha=0.6)

        # Sensor position
        ax.plot(0, 0, 'c^', markersize=15, label='Sensor')

        # Track visualizations
        for track in tracks:
            color = CLASS_COLORS.get(track.classification, CLASS_COLORS['UNKNOWN'])
            pos = track.position

            # Bounding box
            half_l = track.dimensions[0] / 2
            half_w = track.dimensions[1] / 2

            rect_x = [pos[0] - half_l, pos[0] + half_l, pos[0] + half_l,
                     pos[0] - half_l, pos[0] - half_l]
            rect_y = [pos[1] - half_w, pos[1] - half_w, pos[1] + half_w,
                     pos[1] + half_w, pos[1] - half_w]

            ax.plot(rect_x, rect_y, color=color, linewidth=2)

            # Track ID
            if show_ids:
                label = f"ID{track.track_id}\n{track.classification[:3]}"
                ax.annotate(label,
                          (pos[0], pos[1] + half_w + 1),
                          color='white',
                          fontsize=8,
                          ha='center',
                          va='bottom',
                          fontweight='bold')

            # Velocity arrow
            if show_velocity and track.is_moving:
                ax.arrow(pos[0], pos[1],
                        track.velocity[0] * 2, track.velocity[1] * 2,
                        head_width=0.5, head_length=0.3,
                        fc='red', ec='red', alpha=0.8)

                # Speed label
                ax.annotate(f"{track.speed:.1f}m/s",
                          (pos[0] + track.velocity[0], pos[1] + track.velocity[1]),
                          color='yellow',
                          fontsize=7)

            # Trajectory
            if show_trajectories and len(track.trajectory) >= 2:
                traj = np.array(track.trajectory)
                ax.plot(traj[:, 0], traj[:, 1],
                       color=color, alpha=0.5, linewidth=1.5,
                       linestyle='--')

        # Axis settings
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_xlabel('X (m)', color='white', fontsize=12)
        ax.set_ylabel('Y (m)', color='white', fontsize=12)
        ax.set_title(f'Bird\'s Eye View - Frame {frame_idx}',
                    color='white', fontsize=14)
        ax.tick_params(colors='white')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2, color='gray')

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Vehicle'),
            Patch(facecolor='orange', label='Pedestrian'),
            plt.Line2D([0], [0], marker='^', color='cyan',
                      label='Sensor', markersize=10, linestyle='None')
        ]
        ax.legend(handles=legend_elements, loc='upper right',
                 facecolor='#1a1a2e', edgecolor='gray',
                 labelcolor='white')

        # Track info panel
        info_text = f"Frame: {frame_idx}\n"
        info_text += f"Tracks: {len(tracks)}\n"
        vehicles = sum(1 for t in tracks if t.classification == 'VEHICLE')
        peds = sum(1 for t in tracks if t.classification == 'PEDESTRIAN')
        info_text += f"Vehicles: {vehicles}\n"
        info_text += f"Pedestrians: {peds}"

        ax.text(0.02, 0.98, info_text,
               transform=ax.transAxes,
               verticalalignment='top',
               fontsize=10,
               color='white',
               bbox=dict(boxstyle='round', facecolor='#1a1a2e',
                        edgecolor='gray', alpha=0.8))

        plt.tight_layout()

        # Save
        saved_path = None
        if save_path:
            plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(),
                       bbox_inches='tight')
            saved_path = save_path

        plt.close()

        return saved_path


def generate_tracking_video(data_dir: str,
                           output_dir: str = "tracking_video",
                           num_frames: int = 50,
                           fps: int = 10) -> str:
    """
    Generate video with tracking visualization.

    Args:
        data_dir: Directory with LiDAR data
        output_dir: Output directory
        num_frames: Number of frames
        fps: Video frame rate

    Returns:
        Path to generated video
    """
    import glob
    import pandas as pd
    from preprocessing import preprocess_point_cloud
    from clustering import cluster_point_cloud
    from classification import extract_and_classify
    from enhanced_tracking import EnhancedMultiObjectTracker, EnhancedKalmanTracker

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating tracking video...")
    print(f"  Data: {data_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Frames: {num_frames}")

    # Find CSV files
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    num_frames = min(num_frames, len(csv_files))

    # Initialize
    EnhancedKalmanTracker._next_id = 0
    tracker = EnhancedMultiObjectTracker(max_age=5, min_hits=2, dt=0.1)
    bev_viz = BEVVisualizer()

    frame_paths = []

    for i in range(num_frames):
        try:
            # Load frame
            df = pd.read_csv(csv_files[i], sep=';')
            points = df[['X', 'Y', 'Z', 'INTENSITY']].values.astype(np.float32)
            points[:, 3] = np.clip(points[:, 3] / 255.0, 0, 1)

            # Process
            processed = preprocess_point_cloud(points, verbose=False)
            labels, _ = cluster_point_cloud(processed, verbose=False)
            features, _ = extract_and_classify(processed, labels, verbose=False)
            tracks = tracker.update(features, verbose=False)

            # Visualize
            save_path = os.path.join(output_dir, f"frame_{i:04d}.png")
            bev_viz.visualize_frame(
                processed, tracks, frame_idx=i,
                show_ids=True, show_velocity=True, show_trajectories=True,
                save_path=save_path
            )
            frame_paths.append(save_path)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{num_frames} frames")

        except Exception as e:
            print(f"  Error on frame {i}: {e}")
            continue

    # Create video
    video_path = os.path.join(output_dir, "tracking_visualization.mp4")
    frame_pattern = os.path.join(output_dir, "frame_%04d.png")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", frame_pattern,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        video_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  Video saved: {video_path}")
        else:
            print(f"  FFmpeg error: {result.stderr[:200]}")
    except FileNotFoundError:
        print("  FFmpeg not found. Frames saved as PNG.")

    return video_path


# Test
if __name__ == "__main__":
    print("="*70)
    print("ENHANCED VISUALIZATION MODULE - TEST")
    print("="*70)

    from data_loader import generate_simulated_frame
    from preprocessing import preprocess_point_cloud
    from clustering import cluster_point_cloud
    from classification import extract_and_classify
    from enhanced_tracking import EnhancedMultiObjectTracker, EnhancedKalmanTracker

    # Reset
    EnhancedKalmanTracker._next_id = 0
    tracker = EnhancedMultiObjectTracker()

    # Process a few frames to build trajectories
    for i in range(5):
        points = generate_simulated_frame(15000, seed=42, frame_index=i)
        processed = preprocess_point_cloud(points, verbose=False)
        labels, _ = cluster_point_cloud(processed, verbose=False)
        features, _ = extract_and_classify(processed, labels, verbose=False)
        tracks = tracker.update(features, verbose=False)

    print(f"\nActive tracks: {len(tracks)}")
    for track in tracks:
        print(f"  ID {track.track_id}: {track.classification}, "
              f"speed={track.speed:.2f}m/s, traj_len={len(track.trajectory)}")

    # BEV visualization
    bev = BEVVisualizer()
    bev.visualize_frame(processed, tracks, frame_idx=4,
                       save_path="test_bev.png")
    print("\nSaved test_bev.png")
