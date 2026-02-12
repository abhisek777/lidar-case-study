"""
Cinematic LiDAR Visualization Module
====================================
Dark-themed 3D visualization for autonomous driving case study.

Features:
- Dark background with grid floor
- Distance-based point coloring (cyan->green->yellow->red)
- Bounding boxes for detected objects
- Object highlighting (vehicles, pedestrians, cyclists)
- Frame-by-frame capture for video generation

Author: Perception Engineer
Course: Localization, Motion Planning and Sensor Fusion
"""

import os
import sys
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import subprocess

# Import pipeline modules
from data_loader import generate_simulated_frame
from preprocessing import preprocess_point_cloud
from clustering import cluster_point_cloud
from classification import extract_and_classify, ObjectFeatures


# ============================================================================
# COLOR SCHEMES
# ============================================================================

def create_distance_colormap():
    """
    Create a distance-based colormap: cyan -> green -> yellow -> red

    Near (0-10m): Cyan/Blue - #00FFFF
    Medium (10-30m): Green - #00FF00
    Far (30-60m): Yellow - #FFFF00
    Very Far (60m+): Red - #FF0000
    """
    colors = [
        (0.0, 1.0, 1.0),   # Cyan (near)
        (0.0, 1.0, 0.5),   # Cyan-Green
        (0.0, 1.0, 0.0),   # Green (medium)
        (0.5, 1.0, 0.0),   # Yellow-Green
        (1.0, 1.0, 0.0),   # Yellow (far)
        (1.0, 0.5, 0.0),   # Orange
        (1.0, 0.0, 0.0),   # Red (very far)
    ]
    return colors


def distance_to_color(distance: float, max_distance: float = 80.0) -> np.ndarray:
    """
    Convert distance to RGB color using the LiDAR colormap.

    Args:
        distance: Distance from sensor in meters
        max_distance: Maximum distance for color scaling

    Returns:
        RGB color as numpy array [r, g, b] in range [0, 1]
    """
    # Normalize distance to [0, 1]
    t = min(distance / max_distance, 1.0)

    # Color stops
    if t < 0.15:  # 0-12m: Cyan
        return np.array([0.0, 1.0, 1.0])
    elif t < 0.30:  # 12-24m: Cyan to Green
        ratio = (t - 0.15) / 0.15
        return np.array([0.0, 1.0, 1.0 - ratio])
    elif t < 0.50:  # 24-40m: Green to Yellow
        ratio = (t - 0.30) / 0.20
        return np.array([ratio, 1.0, 0.0])
    elif t < 0.75:  # 40-60m: Yellow to Orange
        ratio = (t - 0.50) / 0.25
        return np.array([1.0, 1.0 - ratio * 0.5, 0.0])
    else:  # 60m+: Orange to Red
        ratio = (t - 0.75) / 0.25
        return np.array([1.0, 0.5 - ratio * 0.5, 0.0])


def colorize_by_distance(points: np.ndarray, max_distance: float = 80.0) -> np.ndarray:
    """
    Colorize point cloud based on distance from sensor.

    Args:
        points: Point cloud (N, 3) or (N, 4)
        max_distance: Maximum distance for color scaling

    Returns:
        Colors array (N, 3) with RGB values in [0, 1]
    """
    # Calculate distances
    distances = np.linalg.norm(points[:, :3], axis=1)

    # Apply color mapping
    colors = np.zeros((len(points), 3))
    for i, dist in enumerate(distances):
        colors[i] = distance_to_color(dist, max_distance)

    return colors


# ============================================================================
# OBJECT CLASS COLORS
# ============================================================================

CLASS_COLORS = {
    'VEHICLE': np.array([1.0, 0.2, 0.2]),      # Red
    'PEDESTRIAN': np.array([0.2, 1.0, 0.2]),   # Green
    'CYCLIST': np.array([1.0, 1.0, 0.2]),      # Yellow
    'UNKNOWN': np.array([0.7, 0.7, 0.7]),      # Gray
}

BBOX_COLORS = {
    'VEHICLE': np.array([1.0, 0.3, 0.3]),      # Bright Red
    'PEDESTRIAN': np.array([0.3, 1.0, 0.3]),   # Bright Green
    'CYCLIST': np.array([1.0, 1.0, 0.3]),      # Bright Yellow
    'UNKNOWN': np.array([0.8, 0.8, 0.8]),      # Light Gray
}


# ============================================================================
# GEOMETRY CREATION
# ============================================================================

def create_grid_floor(size: float = 100.0,
                     spacing: float = 5.0,
                     height: float = -0.1,
                     color: List[float] = [0.2, 0.2, 0.3]) -> o3d.geometry.LineSet:
    """
    Create a grid floor for the visualization.

    Args:
        size: Total size of the grid (size x size meters)
        spacing: Distance between grid lines
        height: Z-height of the grid (slightly below ground)
        color: RGB color of grid lines

    Returns:
        Open3D LineSet geometry
    """
    lines = []
    points = []

    half_size = size / 2
    num_lines = int(size / spacing) + 1

    point_idx = 0

    # Lines parallel to X-axis
    for i in range(num_lines):
        y = -half_size + i * spacing
        points.append([-half_size, y, height])
        points.append([half_size, y, height])
        lines.append([point_idx, point_idx + 1])
        point_idx += 2

    # Lines parallel to Y-axis
    for i in range(num_lines):
        x = -half_size + i * spacing
        points.append([x, -half_size, height])
        points.append([x, half_size, height])
        lines.append([point_idx, point_idx + 1])
        point_idx += 2

    # Create LineSet
    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(np.array(points))
    grid.lines = o3d.utility.Vector2iVector(np.array(lines))
    grid.colors = o3d.utility.Vector3dVector([color] * len(lines))

    return grid


def create_bounding_box_lines(center: np.ndarray,
                              dimensions: np.ndarray,
                              color: np.ndarray,
                              rotation: float = 0.0) -> o3d.geometry.LineSet:
    """
    Create a 3D bounding box as line geometry.

    Args:
        center: Center position [x, y, z]
        dimensions: Box dimensions [length, width, height]
        color: RGB color [r, g, b]
        rotation: Rotation around Z-axis in radians

    Returns:
        Open3D LineSet geometry
    """
    l, w, h = dimensions[0] / 2, dimensions[1] / 2, dimensions[2] / 2

    # 8 corners of the box (before rotation)
    corners = np.array([
        [-l, -w, 0],
        [l, -w, 0],
        [l, w, 0],
        [-l, w, 0],
        [-l, -w, h * 2],
        [l, -w, h * 2],
        [l, w, h * 2],
        [-l, w, h * 2],
    ])

    # Apply rotation around Z-axis
    cos_r, sin_r = np.cos(rotation), np.sin(rotation)
    rot_matrix = np.array([
        [cos_r, -sin_r, 0],
        [sin_r, cos_r, 0],
        [0, 0, 1]
    ])
    corners = corners @ rot_matrix.T

    # Translate to center position
    corners += center

    # Define edges (12 edges of a box)
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical edges
    ]

    # Create LineSet
    bbox = o3d.geometry.LineSet()
    bbox.points = o3d.utility.Vector3dVector(corners)
    bbox.lines = o3d.utility.Vector2iVector(edges)
    bbox.colors = o3d.utility.Vector3dVector([color] * len(edges))

    return bbox


def create_sensor_marker(position: np.ndarray = np.array([0, 0, 1.5]),
                        color: List[float] = [0.0, 0.5, 1.0]) -> o3d.geometry.TriangleMesh:
    """
    Create a marker for the LiDAR sensor position.

    Args:
        position: Sensor position [x, y, z]
        color: RGB color

    Returns:
        Open3D TriangleMesh (sphere)
    """
    sensor = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
    sensor.translate(position)
    sensor.paint_uniform_color(color)
    sensor.compute_vertex_normals()
    return sensor


def create_direction_arrow(length: float = 5.0,
                          color: List[float] = [0.0, 0.8, 1.0]) -> o3d.geometry.TriangleMesh:
    """
    Create an arrow indicating forward direction.

    Args:
        length: Arrow length
        color: RGB color

    Returns:
        Open3D TriangleMesh (arrow)
    """
    # Create arrow pointing in +X direction
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.1,
        cone_radius=0.2,
        cylinder_height=length * 0.8,
        cone_height=length * 0.2
    )

    # Rotate to point forward (+X)
    R = arrow.get_rotation_matrix_from_xyz((0, np.pi/2, 0))
    arrow.rotate(R, center=[0, 0, 0])
    arrow.translate([0, 0, 0.5])

    arrow.paint_uniform_color(color)
    arrow.compute_vertex_normals()

    return arrow


# ============================================================================
# SCENE GENERATION
# ============================================================================

@dataclass
class SceneObject:
    """Represents an object in the scene for visualization."""
    object_id: int
    classification: str
    center: np.ndarray
    dimensions: np.ndarray
    velocity: np.ndarray = None
    points: np.ndarray = None
    confidence: float = 0.0


def generate_demo_scene(frame_idx: int = 0,
                       num_points: int = 20000,
                       seed: int = 42) -> Tuple[np.ndarray, List[SceneObject]]:
    """
    Generate a demonstration scene with vehicles, pedestrians, and cyclists.

    Args:
        frame_idx: Frame index for animation
        num_points: Total points to generate
        seed: Random seed

    Returns:
        Tuple of (point_cloud, list of SceneObject)
    """
    np.random.seed(seed)

    points_list = []
    objects = []
    object_id = 0

    # Time factor for animation
    t = frame_idx * 0.1

    # 1. Ground plane (40% of points)
    num_ground = int(num_points * 0.4)
    ground_x = np.random.uniform(-40, 60, num_ground)
    ground_y = np.random.uniform(-30, 30, num_ground)
    ground_z = np.random.normal(0, 0.03, num_ground)
    ground_intensity = np.random.uniform(0.1, 0.3, num_ground)
    ground_points = np.column_stack([ground_x, ground_y, ground_z, ground_intensity])
    points_list.append(ground_points)

    # 2. Approaching vehicle (moving towards sensor)
    vehicle1_x = 35.0 - t * 8.0  # Moving at 8 m/s towards sensor
    vehicle1_y = 3.0
    vehicle1_dims = np.array([4.5, 1.8, 1.5])

    if vehicle1_x > 5.0:  # Only show if in range
        num_v1 = int(num_points * 0.08)
        v1_x = np.random.uniform(vehicle1_x - 2.25, vehicle1_x + 2.25, num_v1)
        v1_y = np.random.uniform(vehicle1_y - 0.9, vehicle1_y + 0.9, num_v1)
        v1_z = np.random.uniform(0.1, 1.5, num_v1)
        v1_intensity = np.random.uniform(0.6, 0.9, num_v1)
        v1_points = np.column_stack([v1_x, v1_y, v1_z, v1_intensity])
        points_list.append(v1_points)

        objects.append(SceneObject(
            object_id=object_id,
            classification='VEHICLE',
            center=np.array([vehicle1_x, vehicle1_y, 0.75]),
            dimensions=vehicle1_dims,
            velocity=np.array([-8.0, 0.0]),
            confidence=0.95
        ))
        object_id += 1

    # 3. Parked vehicle (stationary)
    vehicle2_x = 15.0
    vehicle2_y = -8.0
    vehicle2_dims = np.array([5.0, 2.0, 1.6])

    num_v2 = int(num_points * 0.06)
    v2_x = np.random.uniform(vehicle2_x - 2.5, vehicle2_x + 2.5, num_v2)
    v2_y = np.random.uniform(vehicle2_y - 1.0, vehicle2_y + 1.0, num_v2)
    v2_z = np.random.uniform(0.1, 1.6, num_v2)
    v2_intensity = np.random.uniform(0.5, 0.8, num_v2)
    v2_points = np.column_stack([v2_x, v2_y, v2_z, v2_intensity])
    points_list.append(v2_points)

    objects.append(SceneObject(
        object_id=object_id,
        classification='VEHICLE',
        center=np.array([vehicle2_x, vehicle2_y, 0.8]),
        dimensions=vehicle2_dims,
        velocity=np.array([0.0, 0.0]),
        confidence=0.92
    ))
    object_id += 1

    # 4. Cyclist (crossing)
    cyclist_x = 12.0 + t * 3.0  # Moving at 3 m/s
    cyclist_y = -2.0 + t * 1.5
    cyclist_dims = np.array([1.8, 0.6, 1.7])

    if cyclist_x < 40.0:
        num_c = int(num_points * 0.03)
        c_x = np.random.uniform(cyclist_x - 0.9, cyclist_x + 0.9, num_c)
        c_y = np.random.uniform(cyclist_y - 0.3, cyclist_y + 0.3, num_c)
        c_z = np.random.uniform(0.1, 1.7, num_c)
        c_intensity = np.random.uniform(0.4, 0.7, num_c)
        c_points = np.column_stack([c_x, c_y, c_z, c_intensity])
        points_list.append(c_points)

        objects.append(SceneObject(
            object_id=object_id,
            classification='CYCLIST',
            center=np.array([cyclist_x, cyclist_y, 0.85]),
            dimensions=cyclist_dims,
            velocity=np.array([3.0, 1.5]),
            confidence=0.88
        ))
        object_id += 1

    # 5. Pedestrians (walking)
    pedestrian_positions = [
        (8.0, 5.0 + t * 0.8, np.array([0.5, 0.8])),   # Walking forward
        (10.0, -4.0 - t * 0.5, np.array([-0.5, 0.0])), # Walking backward
        (20.0, 8.0, np.array([0.0, 0.0])),             # Standing
    ]

    for px, py, pvel in pedestrian_positions:
        ped_dims = np.array([0.5, 0.5, 1.75])
        num_p = int(num_points * 0.015)

        p_x = np.random.uniform(px - 0.25, px + 0.25, num_p)
        p_y = np.random.uniform(py - 0.25, py + 0.25, num_p)
        p_z = np.random.uniform(0.1, 1.75, num_p)
        p_intensity = np.random.uniform(0.3, 0.6, num_p)
        p_points = np.column_stack([p_x, p_y, p_z, p_intensity])
        points_list.append(p_points)

        objects.append(SceneObject(
            object_id=object_id,
            classification='PEDESTRIAN',
            center=np.array([px, py, 0.875]),
            dimensions=ped_dims,
            velocity=pvel,
            confidence=0.85
        ))
        object_id += 1

    # 6. Background structures (buildings, walls)
    num_bg = int(num_points * 0.15)

    # Left wall
    wall_x = np.random.uniform(5, 50, num_bg // 2)
    wall_y = np.ones(num_bg // 2) * 25.0 + np.random.normal(0, 0.1, num_bg // 2)
    wall_z = np.random.uniform(0, 4.0, num_bg // 2)
    wall_intensity = np.random.uniform(0.2, 0.4, num_bg // 2)
    wall_points = np.column_stack([wall_x, wall_y, wall_z, wall_intensity])
    points_list.append(wall_points)

    # Right structures
    struct_x = np.random.uniform(20, 45, num_bg // 2)
    struct_y = np.ones(num_bg // 2) * -20.0 + np.random.normal(0, 0.3, num_bg // 2)
    struct_z = np.random.uniform(0, 3.0, num_bg // 2)
    struct_intensity = np.random.uniform(0.2, 0.4, num_bg // 2)
    struct_points = np.column_stack([struct_x, struct_y, struct_z, struct_intensity])
    points_list.append(struct_points)

    # 7. Noise/scatter
    num_noise = num_points - sum(p.shape[0] for p in points_list)
    if num_noise > 0:
        noise_x = np.random.uniform(-30, 60, num_noise)
        noise_y = np.random.uniform(-25, 25, num_noise)
        noise_z = np.random.uniform(-0.5, 5, num_noise)
        noise_intensity = np.random.uniform(0.05, 0.2, num_noise)
        noise_points = np.column_stack([noise_x, noise_y, noise_z, noise_intensity])
        points_list.append(noise_points)

    # Combine all points
    point_cloud = np.vstack(points_list).astype(np.float32)

    return point_cloud, objects


# ============================================================================
# VISUALIZATION CLASS
# ============================================================================

class CinematicLiDARVisualizer:
    """
    Creates cinematic dark-themed LiDAR visualizations.
    """

    def __init__(self,
                 width: int = 1920,
                 height: int = 1080,
                 output_dir: str = "lidar_frames"):
        """
        Initialize the visualizer.

        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            output_dir: Directory to save frames
        """
        self.width = width
        self.height = height
        self.output_dir = output_dir

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Dark background color (very dark blue-gray)
        self.bg_color = np.array([0.05, 0.05, 0.08])

        # Grid settings
        self.grid_size = 100.0
        self.grid_spacing = 5.0
        self.grid_color = [0.15, 0.15, 0.2]

        # Point size
        self.point_size = 2.0

    def create_scene_geometries(self,
                               points: np.ndarray,
                               objects: List[SceneObject] = None,
                               show_grid: bool = True,
                               show_sensor: bool = True,
                               show_bboxes: bool = True,
                               color_by_distance: bool = True) -> List:
        """
        Create all geometries for the scene.

        Args:
            points: Point cloud (N, 4)
            objects: List of detected objects
            show_grid: Show grid floor
            show_sensor: Show sensor marker
            show_bboxes: Show bounding boxes
            color_by_distance: Color points by distance

        Returns:
            List of Open3D geometries
        """
        geometries = []

        # 1. Grid floor
        if show_grid:
            grid = create_grid_floor(
                size=self.grid_size,
                spacing=self.grid_spacing,
                height=-0.05,
                color=self.grid_color
            )
            geometries.append(grid)

        # 2. Point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])

        if color_by_distance:
            colors = colorize_by_distance(points, max_distance=60.0)
        else:
            # Use intensity for coloring
            intensity = points[:, 3] if points.shape[1] >= 4 else np.ones(len(points)) * 0.5
            colors = np.column_stack([intensity, intensity, intensity])

        pcd.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(pcd)

        # 3. Sensor marker
        if show_sensor:
            sensor = create_sensor_marker(
                position=np.array([0, 0, 1.5]),
                color=[0.0, 0.7, 1.0]
            )
            geometries.append(sensor)

            # Direction arrow
            arrow = create_direction_arrow(length=3.0, color=[0.0, 0.8, 1.0])
            geometries.append(arrow)

        # 4. Bounding boxes for objects
        if show_bboxes and objects:
            for obj in objects:
                bbox_color = BBOX_COLORS.get(obj.classification, BBOX_COLORS['UNKNOWN'])

                bbox = create_bounding_box_lines(
                    center=obj.center,
                    dimensions=obj.dimensions,
                    color=bbox_color
                )
                geometries.append(bbox)

        return geometries

    def setup_camera(self, vis: o3d.visualization.Visualizer,
                    view_type: str = "overview") -> None:
        """
        Set up camera position for visualization.

        Args:
            vis: Open3D visualizer
            view_type: Camera view type ("overview", "chase", "side", "top")
        """
        ctr = vis.get_view_control()

        if view_type == "overview":
            # Elevated rear-side view
            ctr.set_front([0.5, 0.3, 0.4])
            ctr.set_lookat([15, 0, 1])
            ctr.set_up([0, 0, 1])
            ctr.set_zoom(0.15)

        elif view_type == "chase":
            # Behind vehicle view
            ctr.set_front([0.8, 0.0, 0.2])
            ctr.set_lookat([20, 0, 1])
            ctr.set_up([0, 0, 1])
            ctr.set_zoom(0.2)

        elif view_type == "side":
            # Side view
            ctr.set_front([0.3, 0.8, 0.3])
            ctr.set_lookat([15, 0, 1])
            ctr.set_up([0, 0, 1])
            ctr.set_zoom(0.15)

        elif view_type == "top":
            # Bird's eye view
            ctr.set_front([0.0, 0.0, 1.0])
            ctr.set_lookat([15, 0, 0])
            ctr.set_up([1, 0, 0])
            ctr.set_zoom(0.08)

    def capture_frame(self,
                     points: np.ndarray,
                     objects: List[SceneObject] = None,
                     frame_idx: int = 0,
                     view_type: str = "overview",
                     filename: str = None) -> str:
        """
        Capture a single frame to file.

        Args:
            points: Point cloud
            objects: Detected objects
            frame_idx: Frame number
            view_type: Camera view type
            filename: Output filename (auto-generated if None)

        Returns:
            Path to saved frame
        """
        if filename is None:
            filename = f"frame_{frame_idx:04d}.png"

        filepath = os.path.join(self.output_dir, filename)

        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name="LiDAR Capture",
            width=self.width,
            height=self.height,
            visible=False  # Offscreen rendering
        )

        # Set render options
        opt = vis.get_render_option()
        opt.background_color = self.bg_color
        opt.point_size = self.point_size
        opt.light_on = True

        # Create and add geometries
        geometries = self.create_scene_geometries(points, objects)
        for geom in geometries:
            vis.add_geometry(geom)

        # Set up camera
        self.setup_camera(vis, view_type)

        # Render and capture
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(filepath, do_render=True)

        vis.destroy_window()

        return filepath

    def visualize_interactive(self,
                             points: np.ndarray,
                             objects: List[SceneObject] = None,
                             view_type: str = "overview") -> None:
        """
        Show interactive 3D visualization.

        Args:
            points: Point cloud
            objects: Detected objects
            view_type: Initial camera view
        """
        print("\nLaunching interactive visualization...")
        print("  Controls: Left=Rotate, Right=Pan, Scroll=Zoom, Q=Close")

        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name="LiDAR Visualization - Dark Theme",
            width=self.width,
            height=self.height
        )

        # Set render options
        opt = vis.get_render_option()
        opt.background_color = self.bg_color
        opt.point_size = self.point_size
        opt.light_on = True

        # Create and add geometries
        geometries = self.create_scene_geometries(points, objects)
        for geom in geometries:
            vis.add_geometry(geom)

        # Set up camera
        self.setup_camera(vis, view_type)

        # Run visualization
        vis.run()
        vis.destroy_window()

    def generate_video_frames(self,
                             num_frames: int = 100,
                             fps: int = 30,
                             view_type: str = "overview") -> List[str]:
        """
        Generate frames for video.

        Args:
            num_frames: Number of frames to generate
            fps: Target frame rate
            view_type: Camera view type

        Returns:
            List of frame file paths
        """
        print(f"\nGenerating {num_frames} frames for video...")

        frame_paths = []

        for i in range(num_frames):
            # Generate scene for this frame
            points, objects = generate_demo_scene(frame_idx=i, num_points=25000)

            # Capture frame
            filepath = self.capture_frame(
                points, objects,
                frame_idx=i,
                view_type=view_type
            )

            frame_paths.append(filepath)

            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  Generated frame {i + 1}/{num_frames}")

        print(f"  All frames saved to: {self.output_dir}")
        return frame_paths

    def create_video(self,
                    output_filename: str = "lidar_visualization.mp4",
                    fps: int = 30) -> str:
        """
        Create video from captured frames using ffmpeg.

        Args:
            output_filename: Output video filename
            fps: Frame rate

        Returns:
            Path to output video
        """
        output_path = os.path.join(self.output_dir, output_filename)
        frame_pattern = os.path.join(self.output_dir, "frame_%04d.png")

        print(f"\nCreating video: {output_path}")

        # FFmpeg command
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            output_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  Video saved: {output_path}")
                return output_path
            else:
                print(f"  FFmpeg error: {result.stderr}")
                return None
        except FileNotFoundError:
            print("  FFmpeg not found. Please install ffmpeg to create videos.")
            print("  Frames are saved in:", self.output_dir)
            return None


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def run_interactive_demo():
    """Run interactive visualization demo."""
    print("="*70)
    print("CINEMATIC LIDAR VISUALIZATION - INTERACTIVE DEMO")
    print("="*70)

    # Generate demo scene
    print("\nGenerating demo scene...")
    points, objects = generate_demo_scene(frame_idx=5, num_points=25000)

    print(f"  Points: {len(points):,}")
    print(f"  Objects: {len(objects)}")
    for obj in objects:
        print(f"    - {obj.classification}: pos=({obj.center[0]:.1f}, {obj.center[1]:.1f})")

    # Create visualizer
    viz = CinematicLiDARVisualizer(width=1920, height=1080)

    # Show interactive visualization
    viz.visualize_interactive(points, objects, view_type="overview")


def run_video_generation(num_frames: int = 100, fps: int = 30, output_dir: str = "lidar_frames"):
    """Generate video from LiDAR frames."""
    print("="*70)
    print("CINEMATIC LIDAR VISUALIZATION - VIDEO GENERATION")
    print("="*70)

    # Create visualizer
    viz = CinematicLiDARVisualizer(
        width=1920,
        height=1080,
        output_dir=output_dir
    )

    # Generate frames
    frame_paths = viz.generate_video_frames(
        num_frames=num_frames,
        fps=fps,
        view_type="overview"
    )

    # Create video
    video_path = viz.create_video(
        output_filename="lidar_cinematic.mp4",
        fps=fps
    )

    print("\n" + "="*70)
    print("VIDEO GENERATION COMPLETE")
    print("="*70)
    print(f"  Frames: {len(frame_paths)}")
    print(f"  Output: {video_path or 'Frames only (ffmpeg not available)'}")

    return video_path


def run_single_frame_capture(output_dir: str = "lidar_frames"):
    """Capture a single high-quality frame."""
    print("="*70)
    print("CINEMATIC LIDAR VISUALIZATION - SINGLE FRAME CAPTURE")
    print("="*70)

    # Create visualizer
    viz = CinematicLiDARVisualizer(
        width=1920,
        height=1080,
        output_dir=output_dir
    )

    # Generate scene
    points, objects = generate_demo_scene(frame_idx=5, num_points=30000)

    # Capture different views
    views = ["overview", "chase", "side", "top"]

    for view in views:
        filepath = viz.capture_frame(
            points, objects,
            frame_idx=0,
            view_type=view,
            filename=f"lidar_{view}.png"
        )
        print(f"  Saved: {filepath}")

    print("\nFrame capture complete!")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cinematic LiDAR Visualization"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["interactive", "video", "frame"],
        default="interactive",
        help="Visualization mode"
    )
    parser.add_argument(
        "--frames", "-n",
        type=int,
        default=100,
        help="Number of frames for video"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video frame rate"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="lidar_frames",
        help="Output directory"
    )

    args = parser.parse_args()

    if args.mode == "interactive":
        run_interactive_demo()
    elif args.mode == "video":
        run_video_generation(args.frames, args.fps, args.output)
    elif args.mode == "frame":
        run_single_frame_capture(args.output)
