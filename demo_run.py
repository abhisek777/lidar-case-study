"""
LiDAR Object Detection and Tracking - Demo Script
===================================================
This script demonstrates the complete perception pipeline on
Blickfeld Cube 1 LiDAR CSV data.

It produces:
  1. Bounding box plots for detected objects (BEV + 3D)
  2. Object trajectory visualization across frames
  3. Summary statistics and performance report

Usage:
  python demo_run.py                          # runs on lidar_data/ folder
  python demo_run.py --data path/to/csv_dir   # custom data path
  python demo_run.py --frames 10              # process 10 frames

Output is saved to demo_output/ directory.

Author: Abhisek Maddi
Course: Localization, Motion Planning and Sensor Fusion
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
from matplotlib.lines import Line2D

# Pipeline modules
from data_loader import BlickfeldDataLoader
from preprocessing import LiDARPreprocessor
from clustering import PointCloudClusterer
from classification import FeatureExtractor, RuleBasedClassifier
from tracking import MultiObjectTracker, TrackState

# Colors for object classes
CLASS_COLORS = {
    'VEHICLE': 'red',
    'PEDESTRIAN': 'green',
    'UNKNOWN': 'orange',
}


def run_demo(data_dir, output_dir, num_frames):
    """
    Run the full LiDAR perception pipeline and generate visualizations.

    Args:
        data_dir:   Path to directory containing Blickfeld CSV files
        output_dir: Path to save output plots
        num_frames: Number of frames to process
    """
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("=" * 70)
    print("LiDAR OBJECT DETECTION & TRACKING DEMO")
    print("=" * 70)
    print(f"\nData directory : {data_dir}")
    print(f"Output directory: {output_dir}")

    loader = BlickfeldDataLoader(data_dir)
    total_frames = loader.get_num_frames()
    if total_frames == 0:
        print("ERROR: No CSV files found in", data_dir)
        sys.exit(1)

    frames_to_process = min(num_frames, total_frames)
    print(f"CSV files found : {total_frames}")
    print(f"Frames to process: {frames_to_process}")

    # ------------------------------------------------------------------
    # 2. Initialise pipeline components
    # ------------------------------------------------------------------
    preprocessor = LiDARPreprocessor(
        min_range=5.0, max_range=250.0,
        voxel_size=0.1, ground_threshold=0.2
    )
    clusterer = PointCloudClusterer(
        eps=0.5, min_samples=10, min_cluster_size=10
    )
    extractor = FeatureExtractor()
    classifier = RuleBasedClassifier()
    tracker = MultiObjectTracker(
        max_age=5, min_hits=3, association_threshold=5.0, dt=0.1
    )

    # Storage for trajectory and per-frame results
    all_tracks_history = {}   # track_id -> list of (frame, x, y, cls)
    frame_results = []        # per-frame metadata

    # ------------------------------------------------------------------
    # 3. Process each frame
    # ------------------------------------------------------------------
    print("\n--- Processing frames ---\n")
    for fi in range(frames_to_process):
        t0 = time.time()
        raw = loader.load_frame(fi)
        if raw is None or len(raw) == 0:
            continue

        # Preprocess
        pts = preprocessor.preprocess(raw, verbose=False)

        # Cluster
        labels = clusterer.cluster(pts, verbose=False)
        n_clusters = clusterer.get_num_clusters()

        # Extract features & classify
        features = extractor.extract_features(pts, labels, verbose=False)
        classifications = classifier.classify(features, verbose=False)

        # Track
        tracks = tracker.update(features, verbose=False)

        elapsed = (time.time() - t0) * 1000
        print(f"  Frame {fi:3d}/{frames_to_process-1}  |  "
              f"points {raw.shape[0]:>6,} -> {pts.shape[0]:>5,}  |  "
              f"clusters {n_clusters:2d}  |  "
              f"tracks {len(tracks):2d}  |  "
              f"{elapsed:6.1f} ms")

        # Store per-frame data
        frame_results.append({
            'frame': fi,
            'points': pts,
            'labels': labels,
            'features': features,
            'tracks': tracks,
            'n_clusters': n_clusters,
            'time_ms': elapsed,
        })

        # Accumulate trajectory history
        for t in tracks:
            tid = t.track_id
            if tid not in all_tracks_history:
                all_tracks_history[tid] = []
            all_tracks_history[tid].append(
                (fi, t.position[0], t.position[1], t.classification)
            )

        # --- Save BEV bounding-box plot for selected frames ---
        if fi % max(1, frames_to_process // 6) == 0 or fi == frames_to_process - 1:
            save_bounding_box_plot(
                pts, labels, features, tracks, fi,
                os.path.join(output_dir, f"detections_frame_{fi:03d}.png")
            )

    # ------------------------------------------------------------------
    # 4. Generate trajectory plot (all frames combined)
    # ------------------------------------------------------------------
    print("\n--- Generating trajectory plot ---")
    save_trajectory_plot(
        all_tracks_history, frames_to_process,
        os.path.join(output_dir, "object_trajectories.png")
    )

    # ------------------------------------------------------------------
    # 5. Generate summary statistics plot
    # ------------------------------------------------------------------
    print("--- Generating summary statistics ---")
    save_summary_plot(
        frame_results, all_tracks_history,
        os.path.join(output_dir, "pipeline_summary.png")
    )

    # ------------------------------------------------------------------
    # 6. Print final report
    # ------------------------------------------------------------------
    avg_ms = np.mean([r['time_ms'] for r in frame_results])
    total_vehicles = sum(
        1 for r in frame_results
        for f in r['features'] if f.classification == 'VEHICLE'
    )
    total_peds = sum(
        1 for r in frame_results
        for f in r['features'] if f.classification == 'PEDESTRIAN'
    )

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Frames processed      : {len(frame_results)}")
    print(f"  Avg processing time   : {avg_ms:.1f} ms/frame")
    print(f"  Estimated FPS         : {1000/avg_ms:.1f}")
    print(f"  Total vehicle detections   : {total_vehicles}")
    print(f"  Total pedestrian detections: {total_peds}")
    print(f"  Unique tracked objects     : {len(all_tracks_history)}")
    print(f"\n  Output saved to: {os.path.abspath(output_dir)}/")
    print("=" * 70)


# ======================================================================
# Plotting helpers
# ======================================================================

def save_bounding_box_plot(points, labels, features, tracks, frame_idx, path):
    """
    Save a Bird's Eye View plot showing detected objects with bounding boxes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    # --- Left panel: detection bounding boxes ---
    ax = axes[0]
    # Background points (noise) in light gray
    noise_mask = labels == -1
    if np.any(noise_mask):
        ax.scatter(points[noise_mask, 0], points[noise_mask, 1],
                   c='lightgray', s=1, alpha=0.15, rasterized=True)

    for feat in features:
        cls = feat.classification or 'UNKNOWN'
        color = CLASS_COLORS.get(cls, 'gray')
        mask = labels == feat.cluster_id
        cluster_pts = points[mask]

        # Cluster points
        ax.scatter(cluster_pts[:, 0], cluster_pts[:, 1],
                   c=color, s=8, alpha=0.6, rasterized=True)

        # Bounding box
        bb = feat.bounding_box
        rect = Rectangle(
            (bb.min_point[0], bb.min_point[1]),
            bb.length, bb.width,
            linewidth=2, edgecolor=color,
            facecolor='none', linestyle='--'
        )
        ax.add_patch(rect)

        # Label
        ax.text(feat.center[0], feat.center[1],
                f"{cls}\n{feat.length:.1f}x{feat.width:.1f}m",
                fontsize=7, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='white', alpha=0.85))

    ax.plot(0, 0, 'b*', markersize=14)
    ax.set_title(f"Detected Objects — Frame {frame_idx}", fontweight='bold')
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axis('equal')

    # Legend
    legend_elems = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red',
               markersize=10, label='Vehicle'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='green',
               markersize=10, label='Pedestrian'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='orange',
               markersize=10, label='Unknown'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='blue',
               markersize=12, label='Sensor'),
    ]
    ax.legend(handles=legend_elems, loc='upper right', fontsize=9)

    # --- Right panel: tracked objects with bounding boxes + velocity ---
    ax2 = axes[1]
    for track in tracks:
        cls = track.classification
        color = CLASS_COLORS.get(cls, 'gray')
        pos = track.position[:2]
        dims = track.dimensions  # [L, W, H]
        vel = track.velocity

        # Bounding box centred on position
        rect = Rectangle(
            (pos[0] - dims[0]/2, pos[1] - dims[1]/2),
            dims[0], dims[1],
            linewidth=2.5, edgecolor=color,
            facecolor=color, alpha=0.25
        )
        ax2.add_patch(rect)

        # Track ID label
        ax2.text(pos[0], pos[1],
                 f"ID:{track.track_id}\n{cls}",
                 fontsize=8, ha='center', va='center', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2',
                           facecolor='white', alpha=0.9))

        # Velocity arrow
        speed = np.sqrt(vel[0]**2 + vel[1]**2)
        if speed > 0.1:
            ax2.annotate('', xy=(pos[0]+vel[0]*2, pos[1]+vel[1]*2),
                         xytext=(pos[0], pos[1]),
                         arrowprops=dict(arrowstyle='->', color=color, lw=2))

        # Trajectory tail
        if len(track.history) > 1:
            hist = np.array(track.history)
            ax2.plot(hist[:, 0], hist[:, 1], color=color,
                     lw=1.5, ls='--', alpha=0.5)

    ax2.plot(0, 0, 'b*', markersize=14)
    ax2.set_title(f"Tracked Objects — Frame {frame_idx}  "
                  f"({len(tracks)} active tracks)", fontweight='bold')
    ax2.set_xlabel("X [m]")
    ax2.set_ylabel("Y [m]")
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.axis('equal')
    ax2.legend(handles=legend_elems, loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {path}")


def save_trajectory_plot(tracks_history, num_frames, path):
    """
    Save a plot showing the full trajectories of all tracked objects.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    for tid, entries in tracks_history.items():
        if len(entries) < 2:
            continue
        xs = [e[1] for e in entries]
        ys = [e[2] for e in entries]
        cls = entries[-1][3]
        color = CLASS_COLORS.get(cls, 'gray')

        ax.plot(xs, ys, color=color, lw=2, alpha=0.7)
        # Start marker
        ax.plot(xs[0], ys[0], 'o', color=color, markersize=6)
        # End marker (arrow-head direction)
        ax.plot(xs[-1], ys[-1], 's', color=color, markersize=7)
        # Label at end
        ax.text(xs[-1], ys[-1], f" ID:{tid}",
                fontsize=7, color=color, fontweight='bold')

    ax.plot(0, 0, 'b*', markersize=16, zorder=10)
    ax.set_title(f"Object Trajectories ({num_frames} frames, "
                 f"{len(tracks_history)} unique objects)",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("X [m]", fontsize=12)
    ax.set_ylabel("Y [m]", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axis('equal')

    legend_elems = [
        Line2D([0], [0], color='red', lw=2, label='Vehicle trajectory'),
        Line2D([0], [0], color='green', lw=2, label='Pedestrian trajectory'),
        Line2D([0], [0], color='orange', lw=2, label='Unknown trajectory'),
        Line2D([0], [0], marker='o', color='gray', lw=0,
               markersize=6, label='Start'),
        Line2D([0], [0], marker='s', color='gray', lw=0,
               markersize=7, label='End'),
        Line2D([0], [0], marker='*', color='blue', lw=0,
               markersize=12, label='Sensor'),
    ]
    ax.legend(handles=legend_elems, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {path}")


def save_summary_plot(frame_results, tracks_history, path):
    """
    Save a 2x2 summary figure with pipeline statistics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    frames = [r['frame'] for r in frame_results]

    # (a) Detections per frame by class
    ax = axes[0, 0]
    vehicles = [sum(1 for f in r['features'] if f.classification == 'VEHICLE')
                for r in frame_results]
    peds = [sum(1 for f in r['features'] if f.classification == 'PEDESTRIAN')
            for r in frame_results]
    unknown = [sum(1 for f in r['features'] if f.classification == 'UNKNOWN')
               for r in frame_results]
    ax.bar(frames, vehicles, color='red', alpha=0.7, label='Vehicle')
    ax.bar(frames, peds, bottom=vehicles, color='green', alpha=0.7,
           label='Pedestrian')
    bottoms = [v + p for v, p in zip(vehicles, peds)]
    ax.bar(frames, unknown, bottom=bottoms, color='orange', alpha=0.7,
           label='Unknown')
    ax.set_xlabel("Frame")
    ax.set_ylabel("Detections")
    ax.set_title("(a) Detections per Frame", fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # (b) Active tracks over time
    ax = axes[0, 1]
    n_tracks = [len(r['tracks']) for r in frame_results]
    ax.plot(frames, n_tracks, 'b-o', markersize=4, lw=2)
    ax.fill_between(frames, n_tracks, alpha=0.15, color='blue')
    ax.set_xlabel("Frame")
    ax.set_ylabel("Active Tracks")
    ax.set_title("(b) Active Tracks over Time", fontweight='bold')
    ax.grid(True, alpha=0.3)

    # (c) Processing time per frame
    ax = axes[1, 0]
    times = [r['time_ms'] for r in frame_results]
    ax.bar(frames, times, color='steelblue', alpha=0.8)
    ax.axhline(np.mean(times), color='red', ls='--', lw=1.5,
               label=f'Mean: {np.mean(times):.1f} ms')
    ax.set_xlabel("Frame")
    ax.set_ylabel("Time [ms]")
    ax.set_title("(c) Processing Time per Frame", fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # (d) Track duration histogram
    ax = axes[1, 1]
    durations = [len(v) for v in tracks_history.values()]
    if durations:
        ax.hist(durations, bins=max(5, len(set(durations))),
                color='purple', alpha=0.7, edgecolor='black')
    ax.set_xlabel("Track Length (frames)")
    ax.set_ylabel("Count")
    ax.set_title("(d) Track Duration Distribution", fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle("Pipeline Performance Summary", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {path}")


# ======================================================================
# Entry point
# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LiDAR Detection & Tracking Demo — "
                    "generates bounding-box and trajectory visualizations"
    )
    parser.add_argument(
        '--data', '-d', type=str, default='lidar_data',
        help='Path to directory with Blickfeld CSV files (default: lidar_data)'
    )
    parser.add_argument(
        '--output', '-o', type=str, default='demo_output',
        help='Directory to save output plots (default: demo_output)'
    )
    parser.add_argument(
        '--frames', '-n', type=int, default=51,
        help='Number of frames to process (default: all 51)'
    )
    args = parser.parse_args()

    run_demo(args.data, args.output, args.frames)
