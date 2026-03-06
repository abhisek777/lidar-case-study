"""
LiDAR Object Detection and Tracking Pipeline — Main Script
===========================================================
DLMDSEAAD02 -- Localization, Motion Planning and Sensor Fusion

Processes ALL available dataset frames (four ZIP parts combined) through:
  1. Data loading   -- Blickfeld Cube 1 CSV (X;Y;Z;DISTANCE;INTENSITY;TIMESTAMP)
  2. Preprocessing  -- range filter, voxel downsample, RANSAC ground, SOR noise
  3. Clustering     -- DBSCAN (eps=0.8 m, min_samples=8)
  4. Classification -- Rule-based (VEHICLE / PEDESTRIAN / STATIC_STRUCTURE / UNKNOWN)
  5. Tracking       -- Kalman filter MOT with static reclassification
  6. Verification   -- Honest proxy metrics (no ground truth)

Parameter note (professor feedback addressed):
  The text and code now use IDENTICAL values:
    max_range = 100.0 m   (sensor spec is 250 m; effective range in this
                           stationary-sensor dataset is < 100 m)
    eps       = 0.8 m     (DBSCAN neighbourhood radius)
  These are the parameters validated to work on this dataset.  The 250 m sensor
  maximum range is a hardware specification; it does not imply that objects
  appear at that distance in this specific recording.

Author: Kalpana Abhiseka Maddi
"""

import os
import sys
import glob
import time
import argparse
import numpy as np
import pandas as pd
from typing import List, Optional

from data_loader     import BlickfeldDataLoader, print_frame_statistics
from preprocessing   import LiDARPreprocessor
from clustering      import PointCloudClusterer
from classification  import FeatureExtractor, RuleBasedClassifier, ObjectFeatures
from tracking        import MultiObjectTracker, KalmanObjectTracker, TrackState
from performance_analysis import VerificationAnalyzer


# ── Pipeline parameters (documented for report) ───────────────────────────────
PIPELINE_PARAMS = {
    # Preprocessing
    'min_range':          2.0,    # m  -- removes near-field sensor artefacts
    'max_range':        100.0,    # m  -- effective range in this dataset
    'voxel_size':         0.15,   # m  -- uniform spatial resolution
    'ground_threshold':   0.25,   # m  -- RANSAC inlier tolerance

    # Clustering (DBSCAN)
    'eps':                0.8,    # m  -- neighbourhood radius
    'min_samples':        8,      # pts-- minimum dense-region size
    'min_cluster_size':   8,      # pts-- minimum valid cluster size

    # Tracking
    'max_age':            8,      # frames before track deletion
    'min_hits':           2,      # frames before track confirmation
    'assoc_threshold':    4.0,    # m  -- max centroid distance for matching
    'sensor_fps':        10.0,    # Hz -- used for time-step dt
}


# ── Dataset helpers ───────────────────────────────────────────────────────────

def collect_all_csv_files(lider_dir: str) -> List[str]:
    """
    Return sorted list of all CSV files across the four dataset parts.

    The dataset is split into four ZIP/directory parts:
      192.168.26.26_2020-11-25_20-01-45_frame-1899_part_1   (52 frames)
      192.168.26.26_2020-11-25_20-01-45_frame-2155_part_2  (256 frames)
      192.168.26.26_2020-11-25_20-01-45_frame-2414_part_3  (259 frames)
      192.168.26.26_2020-11-25_20-01-45_frame-2566_part_4  (152 frames)
    Total: ~719 frames ≈ 71.9 s at 10 fps

    Args:
        lider_dir: Path to the 'Lider datasets/' parent directory

    Returns:
        Sorted list of CSV file paths
    """
    parts     = sorted(glob.glob(os.path.join(lider_dir, '*_part_*')))
    csv_files: List[str] = []
    for part in parts:
        if os.path.isdir(part):
            files = sorted(glob.glob(os.path.join(part, '*.csv')))
            csv_files.extend(files)
    return csv_files


def load_frame_from_csv(csv_path: str) -> Optional[np.ndarray]:
    """Load one LiDAR frame from a Blickfeld CSV file."""
    try:
        df = pd.read_csv(csv_path, sep=';')
        df.columns = df.columns.str.upper().str.strip()
        if not all(c in df.columns for c in ['X', 'Y', 'Z']):
            return None
        intensity = (df['INTENSITY'].values if 'INTENSITY' in df.columns
                     else np.ones(len(df)) * 0.5)
        pts = np.column_stack([df['X'].values, df['Y'].values,
                               df['Z'].values, intensity]).astype(np.float32)
        return pts[~np.isnan(pts).any(axis=1)]
    except Exception as exc:
        print(f"  Warning: could not load {os.path.basename(csv_path)}: {exc}")
        return None


# ── Main pipeline class ───────────────────────────────────────────────────────

class PerceptionPipeline:
    """
    Complete LiDAR perception pipeline for V&V (Verification & Validation).

    Integrates preprocessing → clustering → classification → tracking.
    Collects verification metrics (no ground truth required).
    """

    def __init__(self, params: dict = None):
        p = params or PIPELINE_PARAMS

        self.preprocessor = LiDARPreprocessor(
            min_range        = p['min_range'],
            max_range        = p['max_range'],
            voxel_size       = p['voxel_size'],
            ground_threshold = p['ground_threshold'],
        )
        self.clusterer = PointCloudClusterer(
            eps              = p['eps'],
            min_samples      = p['min_samples'],
            min_cluster_size = p['min_cluster_size'],
        )
        self.extractor  = FeatureExtractor()
        self.classifier = RuleBasedClassifier()
        self.tracker    = MultiObjectTracker(
            max_age               = p['max_age'],
            min_hits              = p['min_hits'],
            association_threshold = p['assoc_threshold'],
            dt                    = 1.0 / p['sensor_fps'],
        )
        self.analyzer    = VerificationAnalyzer(sensor_fps=p['sensor_fps'])
        self.frame_count = 0
        KalmanObjectTracker._next_id = 0

    # ------------------------------------------------------------------
    def process_frame(self, points: np.ndarray,
                      verbose: bool = False) -> dict:
        """
        Process one LiDAR frame through the full pipeline.

        Returns:
            dict with keys:
              preprocessed_points, cluster_labels, features,
              classifications, tracks, processing_ms
        """
        t0 = time.time()
        self.frame_count += 1

        processed       = self.preprocessor.preprocess(points, verbose=verbose)
        labels          = self.clusterer.cluster(processed, verbose=verbose)
        features        = self.extractor.extract_features(processed, labels, verbose=verbose)
        classifications = self.classifier.classify(features, verbose=verbose)
        tracks          = self.tracker.update(features, verbose=verbose)

        proc_ms  = (time.time() - t0) * 1000
        n_clust  = int(np.sum(np.unique(labels) != -1))
        self.analyzer.record(n_clust, len(processed), tracks, proc_ms)

        return {
            'preprocessed_points': processed,
            'cluster_labels':      labels,
            'features':            features,
            'classifications':     classifications,
            'tracks':              tracks,
            'processing_ms':       proc_ms,
        }

    # ------------------------------------------------------------------
    def print_verification_report(self) -> None:
        """Print verification report after all frames have been processed."""
        report = self.analyzer.generate_report()
        self.analyzer.print_report(report)


# ── Entry point ───────────────────────────────────────────────────────────────

def run_pipeline(lider_dir:    str,
                 num_frames:   Optional[int] = None,
                 save_frames:  bool = False,
                 output_dir:   str  = 'pipeline_output',
                 verbose_each: bool = False) -> PerceptionPipeline:
    """
    Run the full pipeline on all dataset frames.

    Args:
        lider_dir:   Path to 'Lider datasets/' directory
        num_frames:  Process only this many frames (None = all)
        save_frames: Save per-frame BEV plots to output_dir
        output_dir:  Where to save plots (created if needed)
        verbose_each: Print detailed output for every frame

    Returns:
        Configured PerceptionPipeline instance
    """
    print("=" * 70)
    print("LIDAR PERCEPTION PIPELINE — DLMDSEAAD02")
    print("=" * 70)
    print("\nPipeline parameters:")
    for k, v in PIPELINE_PARAMS.items():
        print(f"  {k:<22}: {v}")

    # Collect all frames
    csv_files = collect_all_csv_files(lider_dir)
    if not csv_files:
        print(f"\nNo CSV files found in: {lider_dir}")
        print("Please check the dataset directory structure.")
        sys.exit(1)

    if num_frames:
        csv_files = csv_files[:num_frames]

    fps     = PIPELINE_PARAMS['sensor_fps']
    dur_s   = len(csv_files) / fps
    print(f"\nDataset: {len(csv_files)} frames ({dur_s:.1f} s at {fps} fps)")
    print(f"Source : {lider_dir}\n")

    if save_frames:
        os.makedirs(output_dir, exist_ok=True)

    # Run pipeline
    pipeline = PerceptionPipeline()

    for idx, csv_path in enumerate(csv_files):
        points = load_frame_from_csv(csv_path)
        if points is None or len(points) < 50:
            continue

        results = pipeline.process_frame(points, verbose=verbose_each)

        if verbose_each or (idx + 1) % 100 == 0 or idx == len(csv_files) - 1:
            n_t = len(results['tracks'])
            n_v = sum(1 for t in results['tracks'] if t.classification == 'VEHICLE')
            n_p = sum(1 for t in results['tracks'] if t.classification == 'PEDESTRIAN')
            n_s = sum(1 for t in results['tracks'] if t.classification == 'STATIC_STRUCTURE')
            print(f"  [{idx+1:4d}/{len(csv_files)}]  "
                  f"pts={len(results['preprocessed_points']):5d}  "
                  f"clusters={int(np.sum(np.unique(results['cluster_labels']) != -1)):3d}  "
                  f"tracks={n_t:3d} (V={n_v} P={n_p} S={n_s})  "
                  f"t={results['processing_ms']:.0f}ms")

        if save_frames:
            _save_bev_frame(results, idx, output_dir)

    print(f"\nPipeline complete: {pipeline.frame_count} frames processed")
    pipeline.print_verification_report()
    return pipeline


# ── BEV frame saver (lightweight, no open3d) ─────────────────────────────────

def _save_bev_frame(results: dict, idx: int, output_dir: str) -> None:
    """Save a minimal Bird's Eye View figure for one frame."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        pts    = results['preprocessed_points']
        tracks = results['tracks']
        feats  = results['features']

        COLORS = {
            'VEHICLE':          '#FF4444',
            'PEDESTRIAN':       '#44FF44',
            'STATIC_STRUCTURE': '#FF8800',
            'UNKNOWN':          '#AAAAAA',
        }

        fig, ax = plt.subplots(figsize=(10, 10), facecolor='#0A0A1A')
        ax.set_facecolor('#0A0A1A')

        # Point cloud
        if len(pts):
            z = pts[:, 2]
            zn = (z - z.min()) / max(z.max() - z.min(), 0.1)
            ax.scatter(pts[:, 0], pts[:, 1], c=zn, cmap='RdYlGn_r',
                       s=1.0, alpha=0.5, rasterized=True)

        # Detection boxes
        for f in feats:
            cls   = f.classification or 'UNKNOWN'
            color = COLORS.get(cls, '#888888')
            bb    = f.bounding_box
            rect  = patches.Rectangle(
                (bb.min_point[0], bb.min_point[1]), bb.length, bb.width,
                lw=1.5, edgecolor=color, facecolor=color, alpha=0.15, ls='--')
            ax.add_patch(rect)

        # Confirmed tracks
        for t in tracks:
            cls   = t.classification
            color = COLORS.get(cls, '#888888')
            pos   = t.position[:2]
            dims  = t.dimensions
            rect  = patches.Rectangle(
                (pos[0] - dims[0]/2, pos[1] - dims[1]/2), dims[0], dims[1],
                lw=2, edgecolor=color, facecolor=color, alpha=0.25)
            ax.add_patch(rect)
            if len(t.history) > 1:
                h = np.array(t.history)
                ax.plot(h[:, 0], h[:, 1], color=color, lw=1.5, alpha=0.5)
            ax.text(pos[0], pos[1],
                    f"ID:{t.track_id}\n{cls[:3]}",
                    color='white', fontsize=7, ha='center', va='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))

        ax.plot(0, 0, 'o', color='#4488FF', ms=10, zorder=10)
        ax.set_xlim(-50, 50);  ax.set_ylim(-50, 50)
        ax.set_xlabel('X [m]', color='white');  ax.set_ylabel('Y [m]', color='white')
        ax.set_title(f'Frame {idx:04d}  — tracks={len(tracks)}',
                     color='white', fontsize=12, fontweight='bold')
        ax.tick_params(colors='white')
        ax.grid(True, color='#333344', lw=0.4, alpha=0.5)
        for sp in ax.spines.values():
            sp.set_color('#444444')

        plt.tight_layout()
        path = os.path.join(output_dir, f'frame_{idx:04d}.png')
        plt.savefig(path, dpi=100, bbox_inches='tight', facecolor='#0A0A1A')
        plt.close(fig)
    except Exception:
        pass   # Don't let visualisation errors stop the pipeline


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LiDAR Perception Pipeline')
    parser.add_argument('--data', '-d', type=str,
                        default='/Users/abhisekmaddi/Documents/assignment/'
                                'AI Autonomus/new model/Lider datasets',
                        help='Path to Lider datasets/ directory')
    parser.add_argument('--frames', '-n', type=int, default=None,
                        help='Max frames to process (default: all)')
    parser.add_argument('--save-frames', action='store_true',
                        help='Save per-frame BEV PNG images')
    parser.add_argument('--output', '-o', type=str, default='pipeline_output',
                        help='Output directory for saved frames')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detail for every frame')

    args = parser.parse_args()
    run_pipeline(
        lider_dir    = args.data,
        num_frames   = args.frames,
        save_frames  = args.save_frames,
        output_dir   = args.output,
        verbose_each = args.verbose,
    )
