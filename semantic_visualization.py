"""
Semantic 3D LiDAR Perception Visualization
==========================================
Professional autonomous-driving style visualization for academic case study.

Features:
- Dark-themed 3D scene with ground grid
- Height-based point cloud coloring (blue->green->yellow->red)
- Semantic object segmentation (vehicles, cyclists, pedestrians)
- Floating labels with callout arrows
- Smooth camera motion for video generation

Author: Perception Engineer
Course: Localization, Motion Planning and Sensor Fusion
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import subprocess

# Try to import Open3D
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Warning: Open3D not available, using matplotlib fallback")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

# Import pipeline modules
from preprocessing import preprocess_point_cloud
from clustering import cluster_point_cloud
from classification import extract_and_classify, ObjectFeatures
from tracking import MultiObjectTracker, KalmanObjectTracker


# ============================================================================
# COLOR SCHEMES - Height-based coloring
# ============================================================================

def height_to_color(z: float, z_min: float = -2.0, z_max: float = 3.0) -> np.ndarray:
    """
    Convert height to RGB color.

    Blue (low) -> Cyan -> Green (mid) -> Yellow -> Red (high)

    Args:
        z: Height value
        z_min: Minimum height for color mapping
        z_max: Maximum height for color mapping

    Returns:
        RGB color as numpy array [r, g, b] in range [0, 1]
    """
    # Normalize to [0, 1]
    t = np.clip((z - z_min) / (z_max - z_min), 0, 1)

    if t < 0.25:  # Blue to Cyan
        ratio = t / 0.25
        return np.array([0.0, ratio, 1.0])
    elif t < 0.5:  # Cyan to Green
        ratio = (t - 0.25) / 0.25
        return np.array([0.0, 1.0, 1.0 - ratio])
    elif t < 0.75:  # Green to Yellow
        ratio = (t - 0.5) / 0.25
        return np.array([ratio, 1.0, 0.0])
    else:  # Yellow to Red
        ratio = (t - 0.75) / 0.25
        return np.array([1.0, 1.0 - ratio, 0.0])


def colorize_by_height(points: np.ndarray, z_min: float = -2.0, z_max: float = 3.0) -> np.ndarray:
    """
    Colorize point cloud based on height (Z value).

    Args:
        points: Point cloud (N, 3) or (N, 4)
        z_min: Minimum height for color mapping
        z_max: Maximum height for color mapping

    Returns:
        Colors array (N, 3) with RGB values in [0, 1]
    """
    z_values = points[:, 2]
    colors = np.zeros((len(points), 3))

    for i, z in enumerate(z_values):
        colors[i] = height_to_color(z, z_min, z_max)

    return colors


def colorize_by_height_vectorized(points: np.ndarray, z_min: float = -2.0, z_max: float = 3.0) -> np.ndarray:
    """
    Vectorized height-based coloring for better performance.
    """
    z_values = points[:, 2]
    t = np.clip((z_values - z_min) / (z_max - z_min), 0, 1)

    colors = np.zeros((len(points), 3))

    # Blue to Cyan (t < 0.25)
    mask1 = t < 0.25
    ratio1 = t[mask1] / 0.25
    colors[mask1, 0] = 0.0
    colors[mask1, 1] = ratio1
    colors[mask1, 2] = 1.0

    # Cyan to Green (0.25 <= t < 0.5)
    mask2 = (t >= 0.25) & (t < 0.5)
    ratio2 = (t[mask2] - 0.25) / 0.25
    colors[mask2, 0] = 0.0
    colors[mask2, 1] = 1.0
    colors[mask2, 2] = 1.0 - ratio2

    # Green to Yellow (0.5 <= t < 0.75)
    mask3 = (t >= 0.5) & (t < 0.75)
    ratio3 = (t[mask3] - 0.5) / 0.25
    colors[mask3, 0] = ratio3
    colors[mask3, 1] = 1.0
    colors[mask3, 2] = 0.0

    # Yellow to Red (t >= 0.75)
    mask4 = t >= 0.75
    ratio4 = (t[mask4] - 0.75) / 0.25
    colors[mask4, 0] = 1.0
    colors[mask4, 1] = 1.0 - ratio4
    colors[mask4, 2] = 0.0

    return colors


# ============================================================================
# SEMANTIC OBJECT DATA
# ============================================================================

@dataclass
class SemanticObject:
    """Represents a semantically labeled object in the scene."""
    object_id: int
    classification: str
    label: str  # Display label (e.g., "approaching car")
    center: np.ndarray  # 3D center position
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    points: np.ndarray  # Points belonging to this object
    color: Tuple[float, float, float]  # Object highlight color
    velocity: np.ndarray = None  # For motion tracking


# Semantic colors for different object types
SEMANTIC_COLORS = {
    'VEHICLE': (1.0, 0.3, 0.1),      # Orange-red for vehicles
    'PEDESTRIAN': (0.2, 0.8, 1.0),   # Cyan for pedestrians
    'CYCLIST': (1.0, 1.0, 0.2),      # Yellow for cyclists
    'UNKNOWN': (0.5, 0.5, 0.5),      # Gray for unknown
}

SEMANTIC_LABELS = {
    'VEHICLE': 'approaching car',
    'PEDESTRIAN': 'pedestrian',
    'CYCLIST': 'cyclist',
}


# ============================================================================
# GROUND GRID GENERATION
# ============================================================================

def create_ground_grid(size: float = 50.0,
                       spacing: float = 5.0,
                       z: float = -1.5,
                       color: List[float] = [0.15, 0.15, 0.2]) -> o3d.geometry.LineSet:
    """
    Create a 3D ground grid for spatial reference.

    Args:
        size: Grid extent in meters (half-width)
        spacing: Grid line spacing
        z: Ground height
        color: Grid line color

    Returns:
        Open3D LineSet geometry
    """
    lines = []
    points = []

    # Generate grid lines
    num_lines = int(2 * size / spacing) + 1

    for i in range(num_lines):
        offset = -size + i * spacing

        # Lines along X axis
        p1 = [offset, -size, z]
        p2 = [offset, size, z]
        idx = len(points)
        points.extend([p1, p2])
        lines.append([idx, idx + 1])

        # Lines along Y axis
        p3 = [-size, offset, z]
        p4 = [size, offset, z]
        idx = len(points)
        points.extend([p3, p4])
        lines.append([idx, idx + 1])

    # Create LineSet
    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(np.array(points))
    grid.lines = o3d.utility.Vector2iVector(np.array(lines))
    grid.colors = o3d.utility.Vector3dVector([color] * len(lines))

    return grid


def create_ground_grid_dense(size: float = 60.0,
                             major_spacing: float = 10.0,
                             minor_spacing: float = 2.0,
                             z: float = -1.5) -> o3d.geometry.LineSet:
    """
    Create a dense ground grid with major and minor lines.
    """
    lines = []
    points = []
    colors = []

    major_color = [0.2, 0.2, 0.25]
    minor_color = [0.1, 0.1, 0.12]

    # Minor lines
    num_minor = int(2 * size / minor_spacing) + 1
    for i in range(num_minor):
        offset = -size + i * minor_spacing

        # Skip if it's a major line
        if abs(offset % major_spacing) < 0.01:
            continue

        # Lines along X
        idx = len(points)
        points.extend([[offset, -size, z], [offset, size, z]])
        lines.append([idx, idx + 1])
        colors.append(minor_color)

        # Lines along Y
        idx = len(points)
        points.extend([[-size, offset, z], [size, offset, z]])
        lines.append([idx, idx + 1])
        colors.append(minor_color)

    # Major lines
    num_major = int(2 * size / major_spacing) + 1
    for i in range(num_major):
        offset = -size + i * major_spacing

        # Lines along X
        idx = len(points)
        points.extend([[offset, -size, z], [offset, size, z]])
        lines.append([idx, idx + 1])
        colors.append(major_color)

        # Lines along Y
        idx = len(points)
        points.extend([[-size, offset, z], [size, offset, z]])
        lines.append([idx, idx + 1])
        colors.append(major_color)

    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(np.array(points))
    grid.lines = o3d.utility.Vector2iVector(np.array(lines))
    grid.colors = o3d.utility.Vector3dVector(colors)

    return grid


# ============================================================================
# BOUNDING BOX CREATION
# ============================================================================

def create_semantic_bbox(obj: SemanticObject, line_width: float = 2.0) -> o3d.geometry.LineSet:
    """
    Create a colored bounding box for a semantic object.
    """
    bbox_min = obj.bbox_min
    bbox_max = obj.bbox_max

    # 8 corners of the bounding box
    corners = np.array([
        [bbox_min[0], bbox_min[1], bbox_min[2]],
        [bbox_max[0], bbox_min[1], bbox_min[2]],
        [bbox_max[0], bbox_max[1], bbox_min[2]],
        [bbox_min[0], bbox_max[1], bbox_min[2]],
        [bbox_min[0], bbox_min[1], bbox_max[2]],
        [bbox_max[0], bbox_min[1], bbox_max[2]],
        [bbox_max[0], bbox_max[1], bbox_max[2]],
        [bbox_min[0], bbox_max[1], bbox_max[2]],
    ])

    # 12 edges
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top
        [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical
    ]

    bbox = o3d.geometry.LineSet()
    bbox.points = o3d.utility.Vector3dVector(corners)
    bbox.lines = o3d.utility.Vector2iVector(edges)
    bbox.colors = o3d.utility.Vector3dVector([obj.color] * len(edges))

    return bbox


# ============================================================================
# SEMANTIC VISUALIZER CLASS
# ============================================================================

class SemanticLiDARVisualizer:
    """
    Professional semantic 3D LiDAR visualization for autonomous driving.
    """

    def __init__(self,
                 data_dir: str,
                 output_dir: str = "semantic_output",
                 width: int = 1920,
                 height: int = 1080):
        """
        Initialize the semantic visualizer.

        Args:
            data_dir: Directory containing LiDAR CSV files
            output_dir: Output directory for images/video
            width: Output width
            height: Output height
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.width = width
        self.height = height

        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)

        # Find CSV files
        self.csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        print(f"Found {len(self.csv_files)} CSV files")

        # Dark theme settings
        self.bg_color = np.array([0.02, 0.02, 0.04])  # Very dark blue-black
        self.point_size = 2.0

        # Height coloring range
        self.z_min = -2.0
        self.z_max = 3.0

        # Initialize tracker
        self.tracker = MultiObjectTracker(
            max_age=5,
            min_hits=2,
            association_threshold=3.0
        )

        # Camera parameters for smooth motion
        self.camera_position = np.array([0.0, -25.0, 15.0])
        self.camera_lookat = np.array([0.0, 10.0, 0.0])
        self.camera_up = np.array([0.0, 0.0, 1.0])

        # Track semantic objects across frames
        self.tracked_objects: Dict[int, SemanticObject] = {}

    def load_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Load a single frame from CSV."""
        if frame_idx >= len(self.csv_files):
            return None

        csv_path = self.csv_files[frame_idx]

        try:
            df = pd.read_csv(csv_path, sep=';')
            points = df[['X', 'Y', 'Z']].values.astype(np.float32)

            if 'INTENSITY' in df.columns:
                intensity = df['INTENSITY'].values.astype(np.float32)
                points = np.column_stack([points, intensity / 255.0])
            else:
                points = np.column_stack([points, np.ones(len(points)) * 0.5])

            return points
        except Exception as e:
            print(f"Error loading frame {frame_idx}: {e}")
            return None

    def process_frame(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[ObjectFeatures], Dict]:
        """
        Process point cloud through perception pipeline.

        Returns:
            processed_points, cluster_labels, object_features, classifications
        """
        # Preprocessing
        processed = preprocess_point_cloud(points)

        # Clustering - returns (labels, num_clusters)
        labels, num_clusters = cluster_point_cloud(processed)

        # Classification - returns (features_list, classifications_dict)
        features_list, classifications = extract_and_classify(processed, labels)

        return processed, labels, features_list, classifications

    def create_semantic_objects(self,
                                points: np.ndarray,
                                labels: np.ndarray,
                                features: List[ObjectFeatures],
                                classifications: Dict[int, str]) -> List[SemanticObject]:
        """
        Create semantic object representations from detection results.
        """
        semantic_objects = []

        for feat in features:
            # Get classification from the dictionary
            classification = classifications.get(feat.cluster_id, 'UNKNOWN')
            if classification in ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']:
                # Get points for this cluster
                cluster_mask = labels == feat.cluster_id
                cluster_points = points[cluster_mask]

                if len(cluster_points) < 10:
                    continue

                # Compute bounding box
                bbox_min = np.min(cluster_points[:, :3], axis=0)
                bbox_max = np.max(cluster_points[:, :3], axis=0)
                center = (bbox_min + bbox_max) / 2

                # Get semantic label and color
                label = SEMANTIC_LABELS.get(classification, classification.lower())
                color = SEMANTIC_COLORS.get(classification, (0.5, 0.5, 0.5))

                sem_obj = SemanticObject(
                    object_id=feat.cluster_id,
                    classification=classification,
                    label=label,
                    center=center,
                    bbox_min=bbox_min,
                    bbox_max=bbox_max,
                    points=cluster_points,
                    color=color
                )

                semantic_objects.append(sem_obj)

        return semantic_objects

    def create_point_cloud_geometry(self,
                                    points: np.ndarray,
                                    labels: np.ndarray,
                                    semantic_objects: List[SemanticObject]) -> o3d.geometry.PointCloud:
        """
        Create colorized point cloud with semantic highlighting.
        """
        # Start with height-based coloring
        colors = colorize_by_height_vectorized(points, self.z_min, self.z_max)

        # Highlight semantic objects with their specific colors
        for obj in semantic_objects:
            cluster_mask = labels == obj.object_id
            # Blend object color with height color for visibility
            obj_color = np.array(obj.color)
            colors[cluster_mask] = colors[cluster_mask] * 0.3 + obj_color * 0.7

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def setup_camera(self, vis: o3d.visualization.Visualizer,
                     position: np.ndarray = None,
                     lookat: np.ndarray = None):
        """
        Set up camera for road-view perspective.
        """
        if position is None:
            position = self.camera_position
        if lookat is None:
            lookat = self.camera_lookat

        ctr = vis.get_view_control()

        # Set camera parameters
        params = ctr.convert_to_pinhole_camera_parameters()

        # Compute camera transformation matrix
        forward = lookat - position
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, self.camera_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        # Create rotation matrix
        R = np.array([right, -up, forward]).T

        # Create extrinsic matrix
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = position
        extrinsic = np.linalg.inv(extrinsic)

        params.extrinsic = extrinsic
        ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)

    def render_frame_to_image(self,
                              frame_idx: int,
                              camera_offset: float = 0.0) -> Optional[np.ndarray]:
        """
        Render a single frame to an image array.

        Args:
            frame_idx: Frame index to render
            camera_offset: Camera position offset for smooth motion

        Returns:
            Image as numpy array (H, W, 3) or None if failed
        """
        # Load and process frame
        points = self.load_frame(frame_idx)
        if points is None:
            return None

        processed, labels, features, classifications = self.process_frame(points)
        semantic_objects = self.create_semantic_objects(processed, labels, features, classifications)

        # Create geometries
        pcd = self.create_point_cloud_geometry(processed, labels, semantic_objects)
        grid = create_ground_grid_dense(size=60.0)

        # Create bounding boxes
        bboxes = [create_semantic_bbox(obj) for obj in semantic_objects]

        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=self.width, height=self.height, visible=False)

        # Set render options
        opt = vis.get_render_option()
        opt.background_color = self.bg_color
        opt.point_size = self.point_size
        opt.line_width = 2.0

        # Add geometries
        vis.add_geometry(pcd)
        vis.add_geometry(grid)
        for bbox in bboxes:
            vis.add_geometry(bbox)

        # Set camera with smooth motion
        camera_pos = self.camera_position.copy()
        camera_pos[1] += camera_offset * 0.5  # Slow forward movement
        camera_pos[0] += np.sin(camera_offset * 0.1) * 2  # Slight side movement

        lookat = self.camera_lookat.copy()
        lookat[1] += camera_offset * 0.3

        self.setup_camera(vis, camera_pos, lookat)

        # Render
        vis.poll_events()
        vis.update_renderer()

        # Capture image
        image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        image = (image * 255).astype(np.uint8)

        vis.destroy_window()

        # Add labels overlay
        image = self.add_labels_overlay(image, semantic_objects, camera_pos, lookat)

        return image

    def project_3d_to_2d(self,
                         point_3d: np.ndarray,
                         camera_pos: np.ndarray,
                         lookat: np.ndarray) -> Tuple[int, int]:
        """
        Project 3D point to 2D screen coordinates (approximate).
        """
        # Simple perspective projection
        forward = lookat - camera_pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, self.camera_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        # Vector from camera to point
        to_point = point_3d - camera_pos

        # Project onto camera plane
        depth = np.dot(to_point, forward)
        if depth < 0.1:
            return None

        x = np.dot(to_point, right) / depth
        y = np.dot(to_point, up) / depth

        # Convert to screen coordinates
        fov_scale = 1.5
        screen_x = int(self.width / 2 + x * self.width / fov_scale)
        screen_y = int(self.height / 2 - y * self.height / fov_scale)

        # Check bounds
        if 0 <= screen_x < self.width and 0 <= screen_y < self.height:
            return (screen_x, screen_y)
        return None

    def add_labels_overlay(self,
                           image: np.ndarray,
                           semantic_objects: List[SemanticObject],
                           camera_pos: np.ndarray,
                           lookat: np.ndarray) -> np.ndarray:
        """
        Add floating labels with callout arrows to the image.
        """
        # Convert to PIL for text rendering
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)

        # Try to load a font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
            font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
                font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            except:
                font = ImageFont.load_default()
                font_small = font

        for obj in semantic_objects:
            # Project object center to 2D
            screen_pos = self.project_3d_to_2d(obj.center, camera_pos, lookat)
            if screen_pos is None:
                continue

            obj_x, obj_y = screen_pos

            # Calculate label position (offset above and to the side)
            label_offset_x = 80
            label_offset_y = -60

            # Alternate sides to avoid overlap
            if obj.object_id % 2 == 0:
                label_offset_x = -label_offset_x - 150

            label_x = obj_x + label_offset_x
            label_y = obj_y + label_offset_y

            # Keep label on screen
            label_x = max(10, min(self.width - 200, label_x))
            label_y = max(10, min(self.height - 50, label_y))

            # Draw label background (rounded rectangle)
            label_text = obj.label
            bbox = draw.textbbox((label_x, label_y), label_text, font=font)
            padding = 8

            # Semi-transparent background
            bg_coords = [
                bbox[0] - padding,
                bbox[1] - padding,
                bbox[2] + padding,
                bbox[3] + padding
            ]

            # Draw background rectangle
            draw.rectangle(bg_coords, fill=(40, 40, 50, 220), outline=(100, 100, 120))

            # Draw text
            text_color = tuple(int(c * 255) for c in obj.color)
            draw.text((label_x, label_y), label_text, font=font, fill=text_color)

            # Draw arrow from label to object
            arrow_start = (
                (bg_coords[0] + bg_coords[2]) // 2,
                bg_coords[3]
            )
            arrow_end = (obj_x, obj_y)

            # Draw arrow line
            draw.line([arrow_start, arrow_end], fill=(150, 150, 170), width=2)

            # Draw arrowhead
            self._draw_arrowhead(draw, arrow_start, arrow_end, (150, 150, 170))

        return np.array(pil_image)

    def _draw_arrowhead(self, draw, start, end, color, size=10):
        """Draw an arrowhead at the end of a line."""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = np.sqrt(dx*dx + dy*dy)
        if length < 1:
            return

        # Normalize
        dx /= length
        dy /= length

        # Arrowhead points
        p1 = (
            int(end[0] - size * dx + size/2 * dy),
            int(end[1] - size * dy - size/2 * dx)
        )
        p2 = (
            int(end[0] - size * dx - size/2 * dy),
            int(end[1] - size * dy + size/2 * dx)
        )

        draw.polygon([end, p1, p2], fill=color)

    def generate_static_image(self,
                              frame_idx: int = 0,
                              output_name: str = "semantic_lidar_scene.png") -> str:
        """
        Generate a high-resolution static image.

        Args:
            frame_idx: Frame to render
            output_name: Output filename

        Returns:
            Path to saved image
        """
        print(f"\nGenerating static image from frame {frame_idx}...")

        image = self.render_frame_to_image(frame_idx, camera_offset=0.0)

        if image is None:
            print("Failed to render frame")
            return None

        output_path = os.path.join(self.output_dir, output_name)
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        print(f"  Saved: {output_path}")
        print(f"  Resolution: {self.width}x{self.height}")

        return output_path

    def generate_video(self,
                       num_frames: int = 150,
                       fps: int = 30,
                       output_name: str = "semantic_lidar_video.mp4") -> str:
        """
        Generate a smooth video with camera motion and moving objects.

        Args:
            num_frames: Number of video frames (10-15 seconds at 30fps = 300-450 frames)
            fps: Frames per second
            output_name: Output video filename

        Returns:
            Path to saved video
        """
        print(f"\nGenerating {num_frames} frames for video with object motion...")

        # Reset tracker
        KalmanObjectTracker._next_id = 0
        self.tracker = MultiObjectTracker(max_age=5, min_hits=2)

        frames_dir = os.path.join(self.output_dir, "frames")

        # Use all available data frames to show real object motion
        data_frames = len(self.csv_files)
        print(f"  Using {data_frames} LiDAR data frames for object motion")

        # Calculate how many times to repeat data if we need more video frames
        # Each data frame shows slightly different object positions
        frames_per_data = max(1, num_frames // data_frames)

        frame_count = 0
        for data_idx in range(min(data_frames, num_frames)):
            # Render each data frame (objects will be in different positions)
            for repeat in range(frames_per_data):
                if frame_count >= num_frames:
                    break

                # Camera offset for smooth motion
                camera_offset = frame_count * 0.2

                # Render frame with actual data (objects move between data frames)
                image = self.render_frame_to_image(data_idx, camera_offset)

                if image is not None:
                    frame_path = os.path.join(frames_dir, f"frame_{frame_count:04d}.png")
                    cv2.imwrite(frame_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

                frame_count += 1

            if (data_idx + 1) % 10 == 0:
                print(f"  Processed data frame {data_idx + 1}/{min(data_frames, num_frames)}, video frame {frame_count}/{num_frames}")

        print(f"  Total video frames generated: {frame_count}")

        # Create video with ffmpeg
        output_path = os.path.join(self.output_dir, output_name)
        frame_pattern = os.path.join(frames_dir, "frame_%04d.png")

        print(f"\nCreating video: {output_path}")

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-vf", "format=yuv420p",
            output_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  Video saved: {output_path}")
                duration = num_frames / fps
                print(f"  Duration: {duration:.1f} seconds")
                return output_path
            else:
                print(f"  FFmpeg error: {result.stderr[:500]}")
                return None
        except FileNotFoundError:
            print("  FFmpeg not found. Frames saved in output directory.")
            return None


# ============================================================================
# MATPLOTLIB FALLBACK (if Open3D not available)
# ============================================================================

class MatplotlibSemanticVisualizer:
    """
    Matplotlib-based fallback visualizer.
    """

    def __init__(self, data_dir: str, output_dir: str = "semantic_output"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        print(f"Found {len(self.csv_files)} CSV files (using matplotlib)")

        # Matplotlib style
        plt.style.use('dark_background')

    def load_and_process_frame(self, frame_idx: int):
        """Load and process a frame."""
        if frame_idx >= len(self.csv_files):
            return None, None, None, None

        csv_path = self.csv_files[frame_idx]
        df = pd.read_csv(csv_path, sep=';')
        points = df[['X', 'Y', 'Z']].values.astype(np.float32)

        if 'INTENSITY' in df.columns:
            intensity = df['INTENSITY'].values.astype(np.float32)
            points = np.column_stack([points, intensity / 255.0])
        else:
            points = np.column_stack([points, np.ones(len(points)) * 0.5])

        processed = preprocess_point_cloud(points)
        labels, num_clusters = cluster_point_cloud(processed)
        features_list, classifications = extract_and_classify(processed, labels)

        return processed, labels, features_list, classifications

    def generate_static_image(self, frame_idx: int = 0) -> str:
        """Generate static image using matplotlib."""
        print(f"\nGenerating matplotlib visualization...")

        points, labels, features, classifications = self.load_and_process_frame(frame_idx)
        if points is None:
            return None

        # Create figure
        fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

        # Set dark background
        ax.set_facecolor('#0a0a10')
        fig.patch.set_facecolor('#0a0a10')

        # Downsample for visualization
        step = max(1, len(points) // 20000)
        pts = points[::step]

        # Height-based coloring
        colors = colorize_by_height_vectorized(pts)

        # Plot point cloud
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=colors, s=0.5, alpha=0.8)

        # Add grid on ground plane
        grid_size = 50
        grid_spacing = 10
        for i in range(-grid_size, grid_size + 1, grid_spacing):
            ax.plot([i, i], [-grid_size, grid_size], [-1.5, -1.5],
                    color='#333340', linewidth=0.5, alpha=0.5)
            ax.plot([-grid_size, grid_size], [i, i], [-1.5, -1.5],
                    color='#333340', linewidth=0.5, alpha=0.5)

        # Highlight objects and add labels
        for feat in features:
            classification = classifications.get(feat.cluster_id, 'UNKNOWN')
            if classification in ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']:
                cluster_mask = labels == feat.cluster_id
                cluster_pts = points[cluster_mask]

                if len(cluster_pts) < 10:
                    continue

                # Bounding box
                bbox_min = np.min(cluster_pts[:, :3], axis=0)
                bbox_max = np.max(cluster_pts[:, :3], axis=0)
                center = (bbox_min + bbox_max) / 2

                color = SEMANTIC_COLORS.get(classification, (0.5, 0.5, 0.5))
                label = SEMANTIC_LABELS.get(classification, classification)

                # Draw bounding box edges
                self._draw_3d_bbox(ax, bbox_min, bbox_max, color)

                # Add label
                ax.text(center[0], center[1], bbox_max[2] + 1,
                        label, fontsize=10, color=color,
                        ha='center', va='bottom',
                        bbox=dict(boxstyle='round', facecolor='#1a1a24',
                                  edgecolor=color, alpha=0.8))

        # Set view
        ax.view_init(elev=25, azim=-60)
        ax.set_xlim(-30, 30)
        ax.set_ylim(-10, 50)
        ax.set_zlim(-3, 10)

        # Hide axes
        ax.set_axis_off()

        # Save
        output_path = os.path.join(self.output_dir, "semantic_lidar_scene.png")
        plt.savefig(output_path, dpi=100, bbox_inches='tight',
                    facecolor='#0a0a10', edgecolor='none')
        plt.close()

        print(f"  Saved: {output_path}")
        return output_path

    def _draw_3d_bbox(self, ax, bbox_min, bbox_max, color):
        """Draw 3D bounding box."""
        x = [bbox_min[0], bbox_max[0]]
        y = [bbox_min[1], bbox_max[1]]
        z = [bbox_min[2], bbox_max[2]]

        # Bottom face
        ax.plot([x[0], x[1]], [y[0], y[0]], [z[0], z[0]], color=color, linewidth=1.5)
        ax.plot([x[1], x[1]], [y[0], y[1]], [z[0], z[0]], color=color, linewidth=1.5)
        ax.plot([x[1], x[0]], [y[1], y[1]], [z[0], z[0]], color=color, linewidth=1.5)
        ax.plot([x[0], x[0]], [y[1], y[0]], [z[0], z[0]], color=color, linewidth=1.5)

        # Top face
        ax.plot([x[0], x[1]], [y[0], y[0]], [z[1], z[1]], color=color, linewidth=1.5)
        ax.plot([x[1], x[1]], [y[0], y[1]], [z[1], z[1]], color=color, linewidth=1.5)
        ax.plot([x[1], x[0]], [y[1], y[1]], [z[1], z[1]], color=color, linewidth=1.5)
        ax.plot([x[0], x[0]], [y[1], y[0]], [z[1], z[1]], color=color, linewidth=1.5)

        # Vertical edges
        ax.plot([x[0], x[0]], [y[0], y[0]], [z[0], z[1]], color=color, linewidth=1.5)
        ax.plot([x[1], x[1]], [y[0], y[0]], [z[0], z[1]], color=color, linewidth=1.5)
        ax.plot([x[1], x[1]], [y[1], y[1]], [z[0], z[1]], color=color, linewidth=1.5)
        ax.plot([x[0], x[0]], [y[1], y[1]], [z[0], z[1]], color=color, linewidth=1.5)


# ============================================================================
# MAIN ENTRY POINTS
# ============================================================================

def generate_semantic_visualization(data_dir: str,
                                    output_dir: str = "semantic_output",
                                    generate_image: bool = True,
                                    generate_video: bool = True,
                                    video_duration: float = 12.0,
                                    fps: int = 30):
    """
    Main function to generate semantic LiDAR visualization.

    Args:
        data_dir: Directory containing LiDAR CSV files
        output_dir: Output directory
        generate_image: Whether to generate static image
        generate_video: Whether to generate video
        video_duration: Video duration in seconds
        fps: Video frame rate
    """
    print("="*70)
    print("SEMANTIC 3D LIDAR PERCEPTION VISUALIZATION")
    print("="*70)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    if HAS_OPEN3D:
        viz = SemanticLiDARVisualizer(
            data_dir=data_dir,
            output_dir=output_dir,
            width=1920,
            height=1080
        )
    else:
        viz = MatplotlibSemanticVisualizer(
            data_dir=data_dir,
            output_dir=output_dir
        )

    outputs = {}

    if generate_image:
        print("\n" + "-"*70)
        print("GENERATING STATIC IMAGE")
        print("-"*70)
        image_path = viz.generate_static_image()
        outputs['image'] = image_path

    if generate_video and HAS_OPEN3D:
        print("\n" + "-"*70)
        print("GENERATING VIDEO")
        print("-"*70)
        num_frames = int(video_duration * fps)
        video_path = viz.generate_video(num_frames=num_frames, fps=fps)
        outputs['video'] = video_path
    elif generate_video:
        print("\nNote: Video generation requires Open3D. Skipping video.")

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)

    for key, path in outputs.items():
        if path:
            print(f"  {key}: {path}")

    return outputs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Semantic 3D LiDAR Visualization")
    parser.add_argument("--data", "-d", type=str, required=True,
                        help="Directory containing LiDAR CSV files")
    parser.add_argument("--output", "-o", type=str, default="semantic_output",
                        help="Output directory")
    parser.add_argument("--image-only", action="store_true",
                        help="Generate only static image")
    parser.add_argument("--video-only", action="store_true",
                        help="Generate only video")
    parser.add_argument("--duration", "-t", type=float, default=12.0,
                        help="Video duration in seconds")
    parser.add_argument("--fps", type=int, default=30,
                        help="Video frame rate")

    args = parser.parse_args()

    generate_image = not args.video_only
    generate_video = not args.image_only

    generate_semantic_visualization(
        data_dir=args.data,
        output_dir=args.output,
        generate_image=generate_image,
        generate_video=generate_video,
        video_duration=args.duration,
        fps=args.fps
    )
