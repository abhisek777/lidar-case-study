"""
Performance Analysis — Honest Verification Metrics (No Ground Truth)
=====================================================================
DLMDSEAAD02 -- Localization, Motion Planning and Sensor Fusion

Since the dataset contains NO ground-truth annotations, classical supervised
metrics (precision, recall, F1, MOTA, MOTP, classification accuracy) CANNOT
be computed.  Any numeric claim for those metrics would require comparing
predictions against known labels -- which do not exist.

This module therefore computes ONLY verification-oriented proxy metrics that
are derivable from the pipeline output itself:

  1. Detection stability
       mean ± std of cluster count per frame
       low std → preprocessing / clustering behaves consistently

  2. Temporal track stability index (TSI)
       TSI = std(active_tracks_per_frame) / mean(active_tracks_per_frame)
       TSI ≈ 0 is ideal; TSI > 0.5 indicates unstable detection

  3. Track length distribution
       How many frames does the average track persist?
       Long tracks indicate reliable data association.
       Note: track length is in frames; multiply by dt for seconds.

  4. Classification distribution
       Fraction of detections assigned each label.
       High UNKNOWN fraction indicates conservative (safe) behaviour.
       This is NOT accuracy -- it is a distribution statistic.

  5. Classification consistency per track
       Over the lifetime of each confirmed track, what percentage of frames
       keep the same class label?  High consistency indicates stable features.

What is explicitly NOT reported:
  - MOTA / MOTP  (require ground truth)
  - Precision / Recall / F1  (require ground truth)
  - Classification accuracy as a percentage  (requires ground truth)
  - False alarm rate per hour  (requires ground truth)

Reference for verification-oriented evaluation without ground truth:
  Pendleton et al. (2017). Perception, planning, control, and coordination
  for autonomous vehicles. Machines, 5(1), 6.

Author: Kalpana Abhiseka Maddi
"""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class DetectionStability:
    """Detection count statistics across all frames."""
    frames_processed:     int
    mean_clusters:        float
    std_clusters:         float
    min_clusters:         int
    max_clusters:         int
    mean_points_per_frame: float


@dataclass
class TrackingMetrics:
    """Track persistence statistics (no ground truth required)."""
    total_tracks_created: int
    mean_track_length_frames: float
    std_track_length_frames:  float
    max_track_length_frames:  int
    min_track_length_frames:  int
    temporal_stability_index: float   # std / mean of active-tracks-per-frame
    mean_active_tracks:       float
    std_active_tracks:        float


@dataclass
class ClassificationStats:
    """Classification distribution (NOT accuracy -- no ground truth)."""
    total_detections:     int
    class_counts:         Dict[str, int]
    class_fractions:      Dict[str, float]
    # Per-track classification consistency (0–1; 1 = same class all frames)
    mean_class_consistency:   float
    std_class_consistency:    float


@dataclass
class VerificationReport:
    """Full verification report without ground truth."""
    detection:       DetectionStability
    tracking:        TrackingMetrics
    classification:  ClassificationStats
    processing_fps:  float
    sensor_fps:      float
    total_frames:    int
    total_duration_s: float


# ── Analyzer ──────────────────────────────────────────────────────────────────

class VerificationAnalyzer:
    """
    Collects per-frame pipeline outputs and computes verification metrics.

    Usage:
        analyzer = VerificationAnalyzer(sensor_fps=10.0)
        for frame in frames:
            analyzer.record(clusters, tracks, processing_time_ms)
        report = analyzer.generate_report()
        analyzer.print_report(report)
    """

    def __init__(self, sensor_fps: float = 10.0):
        self.sensor_fps = sensor_fps
        self._cluster_counts:    List[int]   = []
        self._point_counts:      List[int]   = []
        self._active_track_counts: List[int] = []
        self._processing_times:  List[float] = []
        self._track_lengths:     Dict[int, int] = defaultdict(int)
        self._track_classes:     Dict[int, List[str]] = defaultdict(list)
        self.frame_count = 0

    # ------------------------------------------------------------------
    def record(self,
               num_clusters:     int,
               num_points:       int,
               tracks:           list,
               processing_ms:    float) -> None:
        """
        Record output of one frame.

        Args:
            num_clusters:  Number of DBSCAN clusters found this frame
            num_points:    Number of preprocessed points this frame
            tracks:        Confirmed TrackState objects this frame
            processing_ms: Wall-clock time for this frame in milliseconds
        """
        self.frame_count += 1
        self._cluster_counts.append(num_clusters)
        self._point_counts.append(num_points)
        self._active_track_counts.append(len(tracks))
        self._processing_times.append(processing_ms)

        for t in tracks:
            tid = t.track_id
            self._track_lengths[tid] += 1
            self._track_classes[tid].append(t.classification)

    # ------------------------------------------------------------------
    def generate_report(self) -> VerificationReport:
        """Compute and return the full verification report."""
        if self.frame_count == 0:
            raise RuntimeError("No frames recorded.")

        # ── Detection stability ───────────────────────────────────────────────
        cc = np.array(self._cluster_counts)
        det = DetectionStability(
            frames_processed      = self.frame_count,
            mean_clusters         = float(np.mean(cc)),
            std_clusters          = float(np.std(cc)),
            min_clusters          = int(cc.min()),
            max_clusters          = int(cc.max()),
            mean_points_per_frame = float(np.mean(self._point_counts)),
        )

        # ── Tracking metrics ──────────────────────────────────────────────────
        lengths = np.array(list(self._track_lengths.values()), dtype=float)
        atc     = np.array(self._active_track_counts, dtype=float)
        tsi     = float(np.std(atc) / max(np.mean(atc), 0.1))

        trk = TrackingMetrics(
            total_tracks_created      = len(self._track_lengths),
            mean_track_length_frames  = float(np.mean(lengths)) if len(lengths) else 0.0,
            std_track_length_frames   = float(np.std(lengths))  if len(lengths) else 0.0,
            max_track_length_frames   = int(lengths.max())       if len(lengths) else 0,
            min_track_length_frames   = int(lengths.min())       if len(lengths) else 0,
            temporal_stability_index  = tsi,
            mean_active_tracks        = float(np.mean(atc)),
            std_active_tracks         = float(np.std(atc)),
        )

        # ── Classification distribution ───────────────────────────────────────
        # Aggregate all classification labels from all confirmed tracks
        all_labels: List[str] = []
        for labels in self._track_classes.values():
            all_labels.extend(labels)

        class_counts: Dict[str, int] = defaultdict(int)
        for lbl in all_labels:
            class_counts[lbl] += 1
        total = max(len(all_labels), 1)

        class_fractions = {k: v / total for k, v in class_counts.items()}

        # Per-track consistency
        consistencies = []
        for labels in self._track_classes.values():
            if len(labels) < 2:
                continue
            modal = max(set(labels), key=labels.count)
            consistencies.append(labels.count(modal) / len(labels))

        cls_stats = ClassificationStats(
            total_detections       = len(all_labels),
            class_counts           = dict(class_counts),
            class_fractions        = class_fractions,
            mean_class_consistency = float(np.mean(consistencies)) if consistencies else 1.0,
            std_class_consistency  = float(np.std(consistencies))  if consistencies else 0.0,
        )

        # ── Processing performance ────────────────────────────────────────────
        avg_ms  = float(np.mean(self._processing_times)) if self._processing_times else 0.0
        proc_fps = 1000.0 / max(avg_ms, 0.1)

        duration_s = self.frame_count / self.sensor_fps

        return VerificationReport(
            detection        = det,
            tracking         = trk,
            classification   = cls_stats,
            processing_fps   = proc_fps,
            sensor_fps       = self.sensor_fps,
            total_frames     = self.frame_count,
            total_duration_s = duration_s,
        )

    # ------------------------------------------------------------------
    def print_report(self, report: Optional[VerificationReport] = None) -> None:
        """Print the verification report to stdout."""
        if report is None:
            report = self.generate_report()

        sep = "=" * 70

        print(f"\n{sep}")
        print("VERIFICATION REPORT  (Ground-truth-free proxy metrics)")
        print(f"{sep}")
        print("NOTE: No ground-truth labels are available.  Metrics below are")
        print("      verification proxies only -- NOT precision / recall / accuracy.")
        print(f"{sep}")

        r = report
        fps_sensor  = r.sensor_fps
        dt          = 1.0 / fps_sensor

        print(f"\n  Dataset            : {r.total_frames} frames  "
              f"({r.total_duration_s:.1f} s at {fps_sensor} fps)")
        print(f"  Processing speed   : {r.processing_fps:.1f} fps  "
              f"(avg {1000/max(r.processing_fps,0.1):.0f} ms/frame)")

        print(f"\n{'─'*70}")
        print("  1. Detection Stability  (std/mean of cluster count per frame)")
        print(f"{'─'*70}")
        d = r.detection
        cv = d.std_clusters / max(d.mean_clusters, 0.1)
        print(f"  Frames processed         : {d.frames_processed}")
        print(f"  Mean clusters / frame    : {d.mean_clusters:.1f}")
        print(f"  Std  clusters / frame    : {d.std_clusters:.1f}  "
              f"(CV = {cv:.2f}  -- lower is more stable)")
        print(f"  Range                    : [{d.min_clusters}, {d.max_clusters}]")
        print(f"  Mean preprocessed points : {d.mean_points_per_frame:.0f}")

        print(f"\n{'─'*70}")
        print("  2. Temporal Stability Index  (TSI = std/mean active tracks)")
        print(f"{'─'*70}")
        t = r.tracking
        print(f"  Active tracks — mean ± std : {t.mean_active_tracks:.1f} ± {t.std_active_tracks:.1f}")
        print(f"  Temporal Stability Index   : {t.temporal_stability_index:.3f}  "
              f"(0 = perfectly stable; <0.3 = good)")

        print(f"\n{'─'*70}")
        print("  3. Track Length Distribution")
        print(f"{'─'*70}")
        mean_s = t.mean_track_length_frames * dt
        max_s  = t.max_track_length_frames  * dt
        print(f"  Total unique tracks        : {t.total_tracks_created}")
        print(f"  Mean track length          : {t.mean_track_length_frames:.1f} frames  "
              f"= {mean_s:.2f} s  (at {fps_sensor} fps)")
        print(f"  Std  track length          : {t.std_track_length_frames:.1f} frames")
        print(f"  Max  track length          : {t.max_track_length_frames} frames  "
              f"= {max_s:.2f} s")
        print(f"  Min  track length          : {t.min_track_length_frames} frames")

        print(f"\n{'─'*70}")
        print("  4. Classification Distribution  (NOT accuracy)")
        print(f"{'─'*70}")
        c = r.classification
        print(f"  Total detections (all frames) : {c.total_detections}")
        for lbl in ['VEHICLE', 'PEDESTRIAN', 'STATIC_STRUCTURE', 'UNKNOWN']:
            cnt  = c.class_counts.get(lbl, 0)
            frac = c.class_fractions.get(lbl, 0.0)
            print(f"    {lbl:<20}: {cnt:5d}  ({frac:.1%})")
        print(f"\n  NOTE: UNKNOWN + STATIC_STRUCTURE fraction reflects the")
        print(f"        conservative design -- unclassified objects are never")
        print(f"        forced into a wrong class.")

        print(f"\n{'─'*70}")
        print("  5. Per-Track Classification Consistency")
        print(f"{'─'*70}")
        print(f"  Mean consistency (modal class fraction per track) :")
        print(f"    {c.mean_class_consistency:.1%} ± {c.std_class_consistency:.1%}")
        print(f"  (1.0 = track never changes its label; robust association)")

        print(f"\n{sep}")
        print("  METRICS NOT REPORTED (require ground truth not available):")
        print("    MOTA / MOTP, Precision, Recall, F1,")
        print("    Classification accuracy (%), False Alarm Rate/hour")
        print(f"{sep}\n")


# ── Convenience runner ────────────────────────────────────────────────────────

def run_verification(data_dir: str,
                     num_frames: Optional[int] = None,
                     sensor_fps: float = 10.0,
                     verbose:    bool = True) -> VerificationReport:
    """
    Run complete verification analysis on the dataset.

    Loads ALL available frames from the four dataset parts, runs the pipeline,
    and returns the VerificationReport.

    Args:
        data_dir:   Path to 'Lider datasets/' directory
        num_frames: Cap on frames to process (None = all)
        sensor_fps: Sensor update rate for time conversion (default 10 Hz)
        verbose:    Print frame-level progress

    Returns:
        VerificationReport
    """
    import glob, os, pandas as pd
    from preprocessing   import LiDARPreprocessor
    from clustering      import PointCloudClusterer
    from classification  import FeatureExtractor, RuleBasedClassifier
    from tracking        import MultiObjectTracker, KalmanObjectTracker

    # Collect all CSV files from all parts
    parts    = sorted(glob.glob(os.path.join(data_dir, '*_part_*')))
    csv_files: List[str] = []
    for part in parts:
        if os.path.isdir(part):
            csv_files.extend(sorted(glob.glob(os.path.join(part, '*.csv'))))
    if num_frames:
        csv_files = csv_files[:num_frames]

    print(f"\nVerification: {len(csv_files)} frames  "
          f"({len(csv_files)/sensor_fps:.1f} s at {sensor_fps} fps)\n")

    # Pipeline components
    preprocessor = LiDARPreprocessor(min_range=2.0, max_range=100.0,
                                     voxel_size=0.15, ground_threshold=0.25)
    clusterer    = PointCloudClusterer(eps=0.8, min_samples=8, min_cluster_size=8)
    extractor    = FeatureExtractor()
    classifier   = RuleBasedClassifier()
    tracker      = MultiObjectTracker(max_age=8, min_hits=2,
                                      association_threshold=4.0, dt=1.0/sensor_fps)
    KalmanObjectTracker._next_id = 0

    analyzer = VerificationAnalyzer(sensor_fps=sensor_fps)

    for idx, csv_path in enumerate(csv_files):
        t0 = time.time()
        try:
            df = pd.read_csv(csv_path, sep=';')
            df.columns = df.columns.str.upper().str.strip()
            if not all(c in df.columns for c in ['X', 'Y', 'Z']):
                continue
            intensity = (df['INTENSITY'].values if 'INTENSITY' in df.columns
                         else np.ones(len(df)) * 0.5)
            pts = np.column_stack([df['X'].values, df['Y'].values,
                                   df['Z'].values, intensity]).astype(np.float32)
            pts = pts[~np.isnan(pts).any(axis=1)]
            if len(pts) < 50:
                continue

            processed  = preprocessor.preprocess(pts, verbose=False)
            labels     = clusterer.cluster(processed, verbose=False)
            features   = extractor.extract_features(processed, labels, verbose=False)
            classifier.classify(features, verbose=False)
            tracks     = tracker.update(features, verbose=False)

            n_clusters = int(np.sum(np.unique(labels) != -1))
            proc_ms    = (time.time() - t0) * 1000
            analyzer.record(n_clusters, len(processed), tracks, proc_ms)

        except Exception as exc:
            if verbose:
                print(f"  Warning: frame {idx} failed: {exc}")
            continue

        if verbose and ((idx + 1) % 100 == 0 or idx == len(csv_files) - 1):
            print(f"  [{idx+1}/{len(csv_files)}] clusters={n_clusters}  "
                  f"active_tracks={len(tracks)}")

    report = analyzer.generate_report()
    analyzer.print_report(report)
    return report


if __name__ == '__main__':
    BASE = '/Users/abhisekmaddi/Documents/assignment/AI Autonomus/new model/Lider datasets'
    run_verification(data_dir=BASE, verbose=True)
