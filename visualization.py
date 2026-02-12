"""
LiDAR Visualization Module
==========================
Autonomous Driving Perception Pipeline - Visualization

This module provides visualization functions for:
1. 3D point cloud visualization (Open3D)
2. Bird's Eye View (BEV) top-down projection
3. Bounding boxes and object IDs
4. Track trajectories

Author: Perception Engineer
Course: Localization, Motion Planning and Sensor Fusion
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow, Circle
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from typing import List, Dict, Optional, Tuple
import open3d as o3d

from classification import ObjectFeatures
from tracking import TrackState


# Color schemes
CLASS_COLORS = {
    'VEHICLE': [1.0, 0.0, 0.0],     # Red
    'PEDESTRIAN': [0.0, 1.0, 0.0],  # Green
    'UNKNOWN': [1.0, 1.0, 0.0],     # Yellow
}

CLASS_COLORS_MPL = {
    'VEHICLE': 'red',
    'PEDESTRIAN': 'green',
    'UNKNOWN': 'orange',
}


def generate_distinct_colors(n: int) -> np.ndarray:
    """
    Generate n visually distinct colors.

    Uses HSV color space with evenly spaced hues.

    Args:
        n: Number of colors to generate

    Returns:
        Array of RGB colors (n, 3) in range [0, 1]
    """
    colors = []
    for i in range(n):
        hue = i / max(n, 1)
        # Convert HSV to RGB (saturation=1, value=1)
        import colorsys
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        colors.append(rgb)
    return np.array(colors)


class PointCloudVisualizer:
    """
    3D point cloud visualization using Open3D.
    """

    def __init__(self, window_width: int = 1280, window_height: int = 720):
        """
        Initialize the visualizer.

        Args:
            window_width: Visualization window width
            window_height: Visualization window height
        """
        self.window_width = window_width
        self.window_height = window_height

    def visualize_point_cloud(self, points: np.ndarray,
                             title: str = "LiDAR Point Cloud",
                             color_by_height: bool = True) -> None:
        """
        Visualize a point cloud in 3D.

        Args:
            points: Point cloud (N, 4) [X, Y, Z, INTENSITY]
            title: Window title
            color_by_height: Color points by Z coordinate
        """
        print(f"\nLaunching 3D visualization: '{title}'")
        print("  Controls: Left mouse=Rotate, Right mouse=Pan, Scroll=Zoom, Q=Close")

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])

        if color_by_height:
            # Color by height (Z coordinate)
            z = points[:, 2]
            z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)

            colors = np.zeros((len(points), 3))
            colors[:, 0] = z_norm           # Red increases with height
            colors[:, 1] = 1 - abs(z_norm - 0.5) * 2  # Green peaks at middle
            colors[:, 2] = 1 - z_norm       # Blue decreases with height
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            # Use intensity for coloring
            intensity = points[:, 3] if points.shape[1] >= 4 else np.ones(len(points)) * 0.5
            colors = np.repeat(intensity.reshape(-1, 1), 3, axis=1)
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # Create coordinate frame for reference
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=5.0, origin=[0, 0, 0]
        )

        o3d.visualization.draw_geometries(
            [pcd, coord_frame],
            window_name=title,
            width=self.window_width,
            height=self.window_height
        )

    def visualize_clusters(self, points: np.ndarray,
                          labels: np.ndarray,
                          title: str = "Clustered Point Cloud",
                          show_noise: bool = False) -> None:
        """
        Visualize clustered point cloud with distinct colors per cluster.

        Args:
            points: Point cloud (N, 4)
            labels: Cluster labels for each point (-1 = noise)
            title: Window title
            show_noise: Whether to show noise points (in black)
        """
        print(f"\nLaunching 3D visualization: '{title}'")

        # Get unique cluster labels
        unique_labels = np.unique(labels)
        cluster_labels = unique_labels[unique_labels != -1]
        num_clusters = len(cluster_labels)

        # Generate colors for each cluster
        cluster_colors = generate_distinct_colors(num_clusters)

        # Assign colors to points
        colors = np.zeros((len(points), 3))
        for i, label in enumerate(cluster_labels):
            mask = labels == label
            colors[mask] = cluster_colors[i]

        # Handle noise points
        noise_mask = labels == -1
        if show_noise:
            colors[noise_mask] = [0.3, 0.3, 0.3]  # Gray for noise
            points_to_show = points
            colors_to_show = colors
        else:
            points_to_show = points[~noise_mask]
            colors_to_show = colors[~noise_mask]

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_to_show[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(colors_to_show)

        # Coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=5.0, origin=[0, 0, 0]
        )

        o3d.visualization.draw_geometries(
            [pcd, coord_frame],
            window_name=title,
            width=self.window_width,
            height=self.window_height
        )

    def visualize_detections(self, points: np.ndarray,
                            labels: np.ndarray,
                            features_list: List[ObjectFeatures],
                            title: str = "Object Detections") -> None:
        """
        Visualize detections with bounding boxes.

        Args:
            points: Point cloud (N, 4)
            labels: Cluster labels
            features_list: List of ObjectFeatures with classifications
            title: Window title
        """
        print(f"\nLaunching 3D visualization: '{title}'")

        geometries = []

        # Create colored point cloud
        unique_labels = np.unique(labels)
        cluster_labels = unique_labels[unique_labels != -1]

        # Points colored by cluster
        non_noise_mask = labels != -1
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[non_noise_mask, :3])

        # Color by classification
        colors = np.zeros((np.sum(non_noise_mask), 3))
        for features in features_list:
            mask_local = labels[non_noise_mask] == features.cluster_id
            color = CLASS_COLORS.get(features.classification, [0.5, 0.5, 0.5])
            colors[mask_local] = color

        pcd.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(pcd)

        # Add bounding boxes
        for features in features_list:
            bbox = features.bounding_box
            color = CLASS_COLORS.get(features.classification, [0.5, 0.5, 0.5])

            # Create line set for bounding box edges
            corners = self._get_bbox_corners(bbox)
            lines = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
                [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
                [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
            ]

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(corners)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))

            geometries.append(line_set)

        # Coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=5.0, origin=[0, 0, 0]
        )
        geometries.append(coord_frame)

        # Sensor position indicator
        sensor = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
        sensor.translate([0, 0, 1.5])
        sensor.paint_uniform_color([0, 0, 1])  # Blue
        geometries.append(sensor)

        o3d.visualization.draw_geometries(
            geometries,
            window_name=title,
            width=self.window_width,
            height=self.window_height
        )

    def _get_bbox_corners(self, bbox) -> np.ndarray:
        """Get 8 corners of a bounding box."""
        min_pt = bbox.min_point
        max_pt = bbox.max_point

        corners = np.array([
            [min_pt[0], min_pt[1], min_pt[2]],
            [max_pt[0], min_pt[1], min_pt[2]],
            [max_pt[0], max_pt[1], min_pt[2]],
            [min_pt[0], max_pt[1], min_pt[2]],
            [min_pt[0], min_pt[1], max_pt[2]],
            [max_pt[0], min_pt[1], max_pt[2]],
            [max_pt[0], max_pt[1], max_pt[2]],
            [min_pt[0], max_pt[1], max_pt[2]],
        ])
        return corners


class BEVVisualizer:
    """
    Bird's Eye View (BEV) visualization using Matplotlib.

    Shows top-down view of the scene with:
    - Point cloud projection
    - Bounding boxes
    - Track IDs and classifications
    - Velocity vectors
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 12)):
        """
        Initialize BEV visualizer.

        Args:
            figsize: Figure size (width, height) in inches
        """
        self.figsize = figsize

    def visualize_clusters(self, points: np.ndarray,
                          labels: np.ndarray,
                          title: str = "Bird's Eye View - Clusters",
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize clusters in Bird's Eye View.

        Args:
            points: Point cloud (N, 4)
            labels: Cluster labels
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Get unique clusters
        unique_labels = np.unique(labels)
        cluster_labels = unique_labels[unique_labels != -1]
        num_clusters = len(cluster_labels)

        # Generate colors
        colors = generate_distinct_colors(num_clusters)

        # Plot clusters
        for i, label in enumerate(cluster_labels):
            mask = labels == label
            cluster_points = points[mask]
            ax.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                c=[colors[i]],
                s=10,
                alpha=0.6,
                label=f'Cluster {label}'
            )

        # Plot noise in gray
        noise_mask = labels == -1
        if np.any(noise_mask):
            noise_points = points[noise_mask]
            ax.scatter(
                noise_points[:, 0],
                noise_points[:, 1],
                c='gray',
                s=2,
                alpha=0.2,
                label='Noise'
            )

        # Plot ego vehicle (sensor)
        ax.plot(0, 0, 'b*', markersize=15, label='Sensor')

        self._format_bev_plot(ax, title)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved to: {save_path}")

        return fig

    def visualize_detections(self, points: np.ndarray,
                            labels: np.ndarray,
                            features_list: List[ObjectFeatures],
                            title: str = "Bird's Eye View - Detections",
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize detections with bounding boxes in BEV.

        Args:
            points: Point cloud (N, 4)
            labels: Cluster labels
            features_list: List of ObjectFeatures
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot each detection
        for features in features_list:
            cluster_id = features.cluster_id
            classification = features.classification or 'UNKNOWN'
            color = CLASS_COLORS_MPL.get(classification, 'gray')

            # Get cluster points
            mask = labels == cluster_id
            cluster_points = points[mask]

            # Plot points
            ax.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                c=color,
                s=15,
                alpha=0.6
            )

            # Draw bounding box (2D projection)
            bbox = features.bounding_box
            rect = Rectangle(
                (bbox.min_point[0], bbox.min_point[1]),
                bbox.length,
                bbox.width,
                linewidth=2,
                edgecolor=color,
                facecolor='none',
                linestyle='--'
            )
            ax.add_patch(rect)

            # Add label
            center = features.center
            label_text = f"{classification}\nID:{cluster_id}"
            ax.text(
                center[0], center[1],
                label_text,
                fontsize=8,
                ha='center',
                va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
            )

        # Plot sensor
        ax.plot(0, 0, 'b*', markersize=15, label='Sensor')

        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                      markersize=10, label='Vehicle'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                      markersize=10, label='Pedestrian'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
                      markersize=10, label='Unknown'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        self._format_bev_plot(ax, title)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved to: {save_path}")

        return fig

    def visualize_tracking(self, tracks: List[TrackState],
                          frame_idx: int = 0,
                          title: str = "Bird's Eye View - Tracking",
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize tracking results with trajectories.

        Args:
            tracks: List of TrackState objects
            frame_idx: Current frame number
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        for track in tracks:
            color = CLASS_COLORS_MPL.get(track.classification, 'gray')
            pos = track.position[:2]
            vel = track.velocity
            dims = track.dimensions  # [L, W, H]

            # Draw bounding box
            rect = Rectangle(
                (pos[0] - dims[0]/2, pos[1] - dims[1]/2),
                dims[0], dims[1],
                linewidth=2.5,
                edgecolor=color,
                facecolor=color,
                alpha=0.3
            )
            ax.add_patch(rect)

            # Draw track ID and info
            label_text = f"ID:{track.track_id}\n{track.classification}"
            ax.text(
                pos[0], pos[1],
                label_text,
                fontsize=9,
                ha='center',
                va='center',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9)
            )

            # Draw velocity vector
            speed = np.sqrt(vel[0]**2 + vel[1]**2)
            if speed > 0.1:
                scale = 2.0  # Scale factor for visibility
                arrow = FancyArrow(
                    pos[0], pos[1],
                    vel[0] * scale, vel[1] * scale,
                    width=0.3,
                    head_width=0.8,
                    head_length=0.4,
                    fc=color,
                    ec='black',
                    linewidth=1,
                    alpha=0.7
                )
                ax.add_patch(arrow)

            # Draw track history (trajectory)
            if len(track.history) > 1:
                history = np.array(track.history)
                ax.plot(
                    history[:, 0], history[:, 1],
                    color=color,
                    linewidth=2,
                    linestyle='--',
                    alpha=0.5
                )

        # Plot sensor
        ax.plot(0, 0, 'b*', markersize=20, label='Sensor', zorder=10)

        # Add title with frame info
        full_title = f"{title} (Frame {frame_idx}, {len(tracks)} tracks)"
        self._format_bev_plot(ax, full_title)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved to: {save_path}")

        return fig

    def _format_bev_plot(self, ax: plt.Axes, title: str) -> None:
        """
        Apply standard formatting to BEV plot.

        Args:
            ax: Matplotlib axes
            title: Plot title
        """
        ax.set_xlabel('X (Forward) [m]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (Lateral) [m]', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axis('equal')

        # Add direction indicator
        ax.annotate('', xy=(0, 15), xytext=(0, 5),
                   arrowprops=dict(arrowstyle='->', color='blue', lw=2))
        ax.text(0, 17, 'Forward', ha='center', fontsize=10,
               color='blue', fontweight='bold')

        plt.tight_layout()


def show_pipeline_results(points: np.ndarray,
                         labels: np.ndarray,
                         features_list: List[ObjectFeatures],
                         tracks: Optional[List[TrackState]] = None,
                         frame_idx: int = 0,
                         show_3d: bool = True,
                         show_bev: bool = True,
                         save_dir: Optional[str] = None) -> None:
    """
    Convenience function to show all pipeline visualization results.

    Args:
        points: Preprocessed point cloud
        labels: Cluster labels
        features_list: Object features with classifications
        tracks: Optional tracking results
        frame_idx: Current frame number
        show_3d: Show 3D Open3D visualization
        show_bev: Show Bird's Eye View matplotlib plots
        save_dir: Optional directory to save figures
    """
    # 3D visualization
    if show_3d:
        viz3d = PointCloudVisualizer()
        viz3d.visualize_detections(
            points, labels, features_list,
            title=f"Frame {frame_idx} - Object Detections"
        )

    # BEV visualization
    if show_bev:
        viz_bev = BEVVisualizer()

        # Detections view
        save_path = f"{save_dir}/frame_{frame_idx:04d}_detections.png" if save_dir else None
        fig = viz_bev.visualize_detections(
            points, labels, features_list,
            title=f"Frame {frame_idx} - Detections",
            save_path=save_path
        )
        plt.show()
        plt.close(fig)

        # Tracking view (if tracks available)
        if tracks:
            save_path = f"{save_dir}/frame_{frame_idx:04d}_tracking.png" if save_dir else None
            fig = viz_bev.visualize_tracking(
                tracks, frame_idx,
                title=f"Frame {frame_idx} - Tracking",
                save_path=save_path
            )
            plt.show()
            plt.close(fig)


# Example usage
if __name__ == "__main__":
    from data_loader import generate_simulated_frame
    from preprocessing import preprocess_point_cloud
    from clustering import cluster_point_cloud
    from classification import extract_and_classify

    print("="*70)
    print("VISUALIZATION MODULE - TEST")
    print("="*70)

    # Generate test data
    print("\n[Test] Generating and processing test data...")
    raw_points = generate_simulated_frame(num_points=15000, seed=42)
    processed_points = preprocess_point_cloud(raw_points, verbose=False)
    labels, num_clusters = cluster_point_cloud(processed_points, verbose=False)
    features_list, classifications = extract_and_classify(processed_points, labels, verbose=False)
    print(f"  Processed {len(processed_points):,} points, {num_clusters} clusters")

    # Test visualizations
    print("\n[Test 1] BEV Cluster Visualization...")
    viz_bev = BEVVisualizer()
    fig = viz_bev.visualize_clusters(processed_points, labels)
    plt.show()
    plt.close(fig)

    print("\n[Test 2] BEV Detection Visualization...")
    fig = viz_bev.visualize_detections(processed_points, labels, features_list)
    plt.show()
    plt.close(fig)

    print("\n[Test 3] 3D Visualization (close window to continue)...")
    viz3d = PointCloudVisualizer()
    viz3d.visualize_detections(processed_points, labels, features_list)

    print("\n" + "="*70)
    print("Visualization module ready for use.")
    print("="*70)
