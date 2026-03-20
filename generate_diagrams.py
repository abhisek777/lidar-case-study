"""
Generate all 7 report diagrams from real LiDAR pipeline data.
=============================================================
DLMDSEAAD02 — Localization, Motion Planning and Sensor Fusion

Diagrams produced:
  fig1_point_cloud_3d.png          — 3-D LiDAR point cloud visualization
  fig2_bounding_boxes.png          — Detected clusters with bounding boxes (BEV)
  fig3_trajectories.png            — Object trajectories from the tracking algorithm
  fig4_classification_dist.png     — Distribution of object classifications
  fig5_active_tracks_over_time.png — Number of active tracks over time
  fig6_track_length_dist.png       — Distribution of track lengths
  fig7_detections_per_frame.png    — Number of detected objects per frame

Author: Kalpana Abhiseka Maddi
"""

import os
import sys
import glob
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from collections import defaultdict
from typing import List, Dict, Tuple

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, 'Lider datasets')
OUT_DIR    = os.path.join(BASE_DIR, 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, BASE_DIR)

from preprocessing   import LiDARPreprocessor
from clustering      import PointCloudClusterer
from classification  import FeatureExtractor, RuleBasedClassifier
from tracking        import MultiObjectTracker, KalmanObjectTracker

# ── Pipeline parameters (must match main_pipeline.py exactly) ─────────────────
P = dict(
    min_range=2.0, max_range=100.0, voxel_size=0.15, ground_threshold=0.25,
    eps=0.8, min_samples=8, min_cluster_size=8,
    max_age=8, min_hits=2, assoc_threshold=4.0, sensor_fps=10.0,
)

# ── Colour palette ────────────────────────────────────────────────────────────
CLASS_COLORS = {
    'VEHICLE':          '#E74C3C',
    'PEDESTRIAN':       '#2ECC71',
    'STATIC_STRUCTURE': '#F39C12',
    'UNKNOWN':          '#95A5A6',
}
DARK_BG   = '#0D1117'
GRID_COL  = '#21262D'
TEXT_COL  = '#E6EDF3'
ACCENT    = '#58A6FF'


# ── Helpers ───────────────────────────────────────────────────────────────────

def collect_csv_files(data_dir: str) -> List[str]:
    parts = sorted(glob.glob(os.path.join(data_dir, '*_part_*')))
    files = []
    for p in parts:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, '*.csv'))))
    return files


def load_csv(path: str):
    try:
        df = pd.read_csv(path, sep=';')
        df.columns = df.columns.str.upper().str.strip()
        if not all(c in df.columns for c in ['X', 'Y', 'Z']):
            return None
        intensity = df['INTENSITY'].values if 'INTENSITY' in df.columns else np.ones(len(df)) * 0.5
        pts = np.column_stack([df['X'].values, df['Y'].values, df['Z'].values, intensity]).astype(np.float32)
        return pts[~np.isnan(pts).any(axis=1)]
    except Exception:
        return None


def styled_fig(figsize=(12, 7)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT_COL, which='both')
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.title.set_color(TEXT_COL)
    for sp in ax.spines.values():
        sp.set_color(GRID_COL)
    ax.grid(True, color=GRID_COL, linewidth=0.6, alpha=0.8)
    return fig, ax


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f'  Saved: {name}')


# ── Run full pipeline, collect everything needed ──────────────────────────────

def run_and_collect(csv_files: List[str]):
    preprocessor = LiDARPreprocessor(
        min_range=P['min_range'], max_range=P['max_range'],
        voxel_size=P['voxel_size'], ground_threshold=P['ground_threshold'])
    clusterer = PointCloudClusterer(
        eps=P['eps'], min_samples=P['min_samples'], min_cluster_size=P['min_cluster_size'])
    extractor  = FeatureExtractor()
    classifier = RuleBasedClassifier()
    tracker    = MultiObjectTracker(
        max_age=P['max_age'], min_hits=P['min_hits'],
        association_threshold=P['assoc_threshold'], dt=1.0/P['sensor_fps'])
    KalmanObjectTracker._next_id = 0

    # Storage
    detections_per_frame   = []   # int: cluster count per frame
    active_tracks_per_frame = []  # int: confirmed track count per frame
    track_lengths: Dict[int, int] = defaultdict(int)
    track_classes: Dict[int, List[str]] = defaultdict(list)
    track_histories: Dict[int, List[np.ndarray]] = defaultdict(list)

    # Pick a representative frame for static diagrams (around frame 360 = midpoint)
    snap_idx  = len(csv_files) // 2
    snap_data = None   # will hold the frame snapshot

    print(f'  Running pipeline on {len(csv_files)} frames...')
    for idx, path in enumerate(csv_files):
        pts = load_csv(path)
        if pts is None or len(pts) < 50:
            continue

        processed  = preprocessor.preprocess(pts, verbose=False)
        labels     = clusterer.cluster(processed, verbose=False)
        features   = extractor.extract_features(processed, labels, verbose=False)
        classifier.classify(features, verbose=False)
        tracks     = tracker.update(features, verbose=False)

        n_clusters = int(np.sum(np.unique(labels) != -1))
        detections_per_frame.append(n_clusters)
        active_tracks_per_frame.append(len(tracks))

        for t in tracks:
            track_lengths[t.track_id] += 1
            track_classes[t.track_id].append(t.classification)
            track_histories[t.track_id].extend([np.array(h) for h in t.history])

        # Capture snapshot for static diagrams
        if idx == snap_idx:
            snap_data = dict(
                processed=processed.copy(),
                labels=labels.copy(),
                features=list(features),
                tracks=list(tracks),
            )

        if (idx + 1) % 100 == 0 or idx == len(csv_files) - 1:
            print(f'    [{idx+1}/{len(csv_files)}]  clusters={n_clusters}  tracks={len(tracks)}')

    return dict(
        detections_per_frame=detections_per_frame,
        active_tracks_per_frame=active_tracks_per_frame,
        track_lengths=dict(track_lengths),
        track_classes=dict(track_classes),
        track_histories=dict(track_histories),
        snap=snap_data,
    )


# ── Figure 1: 3-D point cloud ─────────────────────────────────────────────────

def fig1_point_cloud_3d(snap):
    pts = snap['processed']
    fig = plt.figure(figsize=(13, 9), facecolor=DARK_BG)
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(DARK_BG)

    # Colour by distance from sensor
    dist = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2 + pts[:, 2]**2)
    norm = (dist - dist.min()) / max(dist.max() - dist.min(), 0.1)
    sc   = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                      c=norm, cmap='plasma', s=0.5, alpha=0.7, rasterized=True)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label('Normalised Distance', color=TEXT_COL)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COL)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COL)

    ax.set_xlabel('X [m]', color=TEXT_COL, labelpad=6)
    ax.set_ylabel('Y [m]', color=TEXT_COL, labelpad=6)
    ax.set_zlabel('Z [m]', color=TEXT_COL, labelpad=6)
    ax.tick_params(colors=TEXT_COL)
    ax.set_title('Three-Dimensional LiDAR Point Cloud — Recorded Traffic Scene',
                 color=TEXT_COL, fontsize=12, fontweight='bold', pad=12)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(GRID_COL)
    ax.yaxis.pane.set_edgecolor(GRID_COL)
    ax.zaxis.pane.set_edgecolor(GRID_COL)
    ax.grid(True, color=GRID_COL, linewidth=0.4)

    # Mark sensor origin
    ax.scatter([0], [0], [0], c=ACCENT, s=60, zorder=10, label='Sensor origin')
    ax.legend(facecolor=DARK_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL, fontsize=9)

    plt.tight_layout()
    save(fig, 'fig1_point_cloud_3d.png')


# ── Figure 2: Bounding boxes (BEV) ────────────────────────────────────────────

def fig2_bounding_boxes(snap):
    pts     = snap['processed']
    feats   = snap['features']
    tracks  = snap['tracks']

    fig, ax = plt.subplots(figsize=(11, 11), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    # Point cloud
    dist = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    norm = (dist - dist.min()) / max(dist.max() - dist.min(), 0.1)
    ax.scatter(pts[:, 0], pts[:, 1], c=norm, cmap='viridis',
               s=1.0, alpha=0.6, rasterized=True)

    # Raw cluster bounding boxes (dashed, faint)
    for f in feats:
        bb  = f.bounding_box
        cls = f.classification or 'UNKNOWN'
        col = CLASS_COLORS.get(cls, '#AAAAAA')
        rect = patches.Rectangle(
            (bb.min_point[0], bb.min_point[1]), bb.length, bb.width,
            linewidth=1.0, edgecolor=col, facecolor=col,
            alpha=0.12, linestyle='--')
        ax.add_patch(rect)

    # Confirmed track bounding boxes (solid, bright)
    for t in tracks:
        col  = CLASS_COLORS.get(t.classification, '#AAAAAA')
        pos  = t.position[:2]
        dims = t.dimensions
        rect = patches.Rectangle(
            (pos[0] - dims[0]/2, pos[1] - dims[1]/2), dims[0], dims[1],
            linewidth=2.0, edgecolor=col, facecolor=col, alpha=0.25)
        ax.add_patch(rect)
        label_short = {'VEHICLE': 'VEH', 'PEDESTRIAN': 'PED',
                       'STATIC_STRUCTURE': 'STR', 'UNKNOWN': 'UNK'}.get(t.classification, '?')
        ax.text(pos[0], pos[1], f'ID:{t.track_id}\n{label_short}',
                color='white', fontsize=6.5, ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=col, alpha=0.75))

    # Legend
    legend_handles = [patches.Patch(facecolor=v, label=k)
                      for k, v in CLASS_COLORS.items()]
    ax.legend(handles=legend_handles, loc='upper right',
              facecolor=DARK_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL, fontsize=9)

    # Sensor
    ax.plot(0, 0, 'o', color=ACCENT, ms=10, zorder=10, label='Sensor')

    ax.set_xlim(-50, 50);  ax.set_ylim(-10, 80)
    ax.set_xlabel('Lateral X [m]', color=TEXT_COL)
    ax.set_ylabel('Forward Y [m]', color=TEXT_COL)
    ax.set_title('Detected Object Clusters — Bounding Box Overlay (Bird\'s-Eye View)',
                 color=TEXT_COL, fontsize=12, fontweight='bold')
    ax.tick_params(colors=TEXT_COL)
    ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.8)
    for sp in ax.spines.values():
        sp.set_color(GRID_COL)

    plt.tight_layout()
    save(fig, 'fig2_bounding_boxes.png')


# ── Figure 3: Trajectories ────────────────────────────────────────────────────

def fig3_trajectories(track_histories, track_classes):
    fig, ax = plt.subplots(figsize=(12, 10), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    # Only tracks with ≥ 5 unique positions and dominant class != UNKNOWN
    plotted = 0
    for tid, hist in track_histories.items():
        if len(hist) < 5:
            continue
        labels = track_classes.get(tid, ['UNKNOWN'])
        modal  = max(set(labels), key=labels.count)
        color  = CLASS_COLORS.get(modal, '#AAAAAA')
        pts    = np.array(hist)

        # Remove duplicate consecutive positions
        unique_mask = np.ones(len(pts), dtype=bool)
        for i in range(1, len(pts)):
            if np.linalg.norm(pts[i] - pts[i-1]) < 0.05:
                unique_mask[i] = False
        pts = pts[unique_mask]
        if len(pts) < 3:
            continue

        ax.plot(pts[:, 0], pts[:, 1], '-', color=color,
                linewidth=1.2, alpha=0.7)
        ax.scatter(pts[0, 0], pts[0, 1], color=color, s=25, zorder=5, marker='o')
        ax.scatter(pts[-1, 0], pts[-1, 1], color=color, s=50, zorder=6, marker='>')
        plotted += 1

    # Legend
    legend_handles = [patches.Patch(facecolor=v, label=k)
                      for k, v in CLASS_COLORS.items()]
    ax.legend(handles=legend_handles, loc='upper right',
              facecolor=DARK_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL, fontsize=9,
              title='Object Class', title_fontsize=9)

    # Sensor origin
    ax.plot(0, 0, '*', color=ACCENT, ms=14, zorder=10, label='Sensor origin')

    ax.set_xlim(-60, 60);  ax.set_ylim(-20, 90)
    ax.set_xlabel('Lateral X [m]', color=TEXT_COL)
    ax.set_ylabel('Forward Y [m]', color=TEXT_COL)
    ax.set_title(f'Example Object Trajectories Generated by the Tracking Algorithm\n'
                 f'({plotted} tracks shown  ·  circle = start, arrow = end)',
                 color=TEXT_COL, fontsize=12, fontweight='bold')
    ax.tick_params(colors=TEXT_COL)
    ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.8)
    for sp in ax.spines.values():
        sp.set_color(GRID_COL)

    plt.tight_layout()
    save(fig, 'fig3_trajectories.png')


# ── Figure 4: Classification distribution ────────────────────────────────────

def fig4_classification_dist(track_classes):
    # Count across all confirmed track-frame observations
    counts: Dict[str, int] = defaultdict(int)
    for labels in track_classes.values():
        for lbl in labels:
            counts[lbl] += 1

    order   = ['VEHICLE', 'PEDESTRIAN', 'STATIC_STRUCTURE', 'UNKNOWN']
    vals    = [counts.get(k, 0) for k in order]
    colors  = [CLASS_COLORS[k] for k in order]
    total   = sum(vals)
    labels  = [f'{k}\n{counts.get(k,0):,}  ({counts.get(k,0)/max(total,1):.1%})'
               for k in order]

    fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK_BG)

    # Bar chart
    ax_bar.set_facecolor(DARK_BG)
    bars = ax_bar.bar(order, vals, color=colors, edgecolor=DARK_BG, linewidth=1.5, width=0.6)
    for bar, val in zip(bars, vals):
        ax_bar.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + total * 0.005,
                    f'{val:,}\n({val/max(total,1):.1%})',
                    ha='center', va='bottom', color=TEXT_COL, fontsize=9, fontweight='bold')
    ax_bar.set_xlabel('Object Class', color=TEXT_COL)
    ax_bar.set_ylabel('Track-Frame Observations', color=TEXT_COL)
    ax_bar.set_title('Classification Count per Class', color=TEXT_COL, fontweight='bold')
    ax_bar.tick_params(colors=TEXT_COL)
    ax_bar.set_ylim(0, max(vals) * 1.18)
    ax_bar.grid(True, axis='y', color=GRID_COL, linewidth=0.5)
    for sp in ax_bar.spines.values():
        sp.set_color(GRID_COL)

    # Pie chart
    ax_pie.set_facecolor(DARK_BG)
    wedge_props = dict(linewidth=1.5, edgecolor=DARK_BG)
    ax_pie.pie(vals, labels=labels, colors=colors,
               autopct='%1.1f%%', startangle=140,
               wedgeprops=wedge_props, textprops={'color': TEXT_COL, 'fontsize': 9})
    ax_pie.set_title('Classification Fraction', color=TEXT_COL, fontweight='bold')

    fig.suptitle('Distribution of Object Classifications Across All Confirmed Track Observations',
                 color=TEXT_COL, fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    save(fig, 'fig4_classification_dist.png')


# ── Figure 5: Active tracks over time ────────────────────────────────────────

def fig5_active_tracks_over_time(active_tracks_per_frame, sensor_fps=10.0):
    time_axis = np.arange(len(active_tracks_per_frame)) / sensor_fps
    arr = np.array(active_tracks_per_frame, dtype=float)

    fig, ax = styled_fig(figsize=(14, 5))

    # Rolling mean (window = 10 frames = 1 s)
    window = 10
    rolling = np.convolve(arr, np.ones(window)/window, mode='same')

    ax.fill_between(time_axis, arr, alpha=0.2, color=ACCENT)
    ax.plot(time_axis, arr, color=ACCENT, linewidth=0.8, alpha=0.5, label='Per-frame count')
    ax.plot(time_axis, rolling, color='#FF7B54', linewidth=1.8, label=f'{window}-frame rolling mean')

    mean_val = arr.mean()
    ax.axhline(mean_val, color='#FFD700', linewidth=1.2, linestyle='--',
               label=f'Mean = {mean_val:.1f}')

    ax.set_xlabel('Time [s]', color=TEXT_COL)
    ax.set_ylabel('Confirmed Active Tracks', color=TEXT_COL)
    ax.set_title('Number of Active Object Tracks Over Time  (718 frames · 71.8 s)',
                 color=TEXT_COL, fontsize=12, fontweight='bold')
    ax.set_xlim(0, time_axis[-1])
    ax.set_ylim(0, arr.max() * 1.15)
    ax.legend(facecolor=DARK_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL, fontsize=9)

    plt.tight_layout()
    save(fig, 'fig5_active_tracks_over_time.png')


# ── Figure 6: Track length distribution ──────────────────────────────────────

def fig6_track_length_dist(track_lengths, sensor_fps=10.0):
    lengths = np.array(list(track_lengths.values()), dtype=float)

    fig, ax = styled_fig(figsize=(12, 6))

    bins = np.arange(1, lengths.max() + 2) - 0.5
    n, bins_out, patches_list = ax.hist(lengths, bins=bins, color=ACCENT,
                                         edgecolor=DARK_BG, linewidth=0.6)

    # Colour bars by length category
    for patch, left in zip(patches_list, bins_out[:-1]):
        frames = left + 0.5
        secs   = frames / sensor_fps
        if secs < 0.5:
            patch.set_facecolor('#95A5A6')   # short (< 0.5 s)
        elif secs < 2.0:
            patch.set_facecolor(ACCENT)       # medium
        else:
            patch.set_facecolor('#2ECC71')    # long (> 2 s)

    mean_f = lengths.mean()
    ax.axvline(mean_f, color='#FFD700', linewidth=1.8, linestyle='--',
               label=f'Mean = {mean_f:.1f} frames ({mean_f/sensor_fps:.2f} s)')
    ax.axvline(np.median(lengths), color='#FF7B54', linewidth=1.5, linestyle=':',
               label=f'Median = {np.median(lengths):.0f} frames ({np.median(lengths)/sensor_fps:.2f} s)')

    # Custom legend for bar colours
    short_p  = patches.Patch(facecolor='#95A5A6', label='< 0.5 s  (boundary objects)')
    medium_p = patches.Patch(facecolor=ACCENT,    label='0.5 – 2.0 s  (transient)')
    long_p   = patches.Patch(facecolor='#2ECC71', label='> 2.0 s  (persistent)')
    handles  = [short_p, medium_p, long_p,
                plt.Line2D([0], [0], color='#FFD700', lw=1.8, ls='--', label=f'Mean = {mean_f:.1f} f'),
                plt.Line2D([0], [0], color='#FF7B54', lw=1.5, ls=':', label=f'Median = {np.median(lengths):.0f} f')]
    ax.legend(handles=handles, facecolor=DARK_BG, edgecolor=GRID_COL,
              labelcolor=TEXT_COL, fontsize=9)

    ax.set_xlabel('Track Length [frames]  (1 frame = 0.10 s at 10 Hz)', color=TEXT_COL)
    ax.set_ylabel('Number of Tracks', color=TEXT_COL)
    ax.set_title(f'Distribution of Track Lengths Across the Dataset  '
                 f'({len(lengths)} confirmed tracks total)',
                 color=TEXT_COL, fontsize=12, fontweight='bold')
    ax.set_xlim(0, min(lengths.max() + 2, 100))

    plt.tight_layout()
    save(fig, 'fig6_track_length_dist.png')


# ── Figure 7: Detections per frame ────────────────────────────────────────────

def fig7_detections_per_frame(detections_per_frame, sensor_fps=10.0):
    time_axis = np.arange(len(detections_per_frame)) / sensor_fps
    arr = np.array(detections_per_frame, dtype=float)
    window = 15
    rolling = np.convolve(arr, np.ones(window)/window, mode='same')

    fig, ax = styled_fig(figsize=(14, 5))

    ax.fill_between(time_axis, arr, alpha=0.15, color='#C678DD')
    ax.plot(time_axis, arr, color='#C678DD', linewidth=0.7, alpha=0.5, label='Per-frame count')
    ax.plot(time_axis, rolling, color='#E5C07B', linewidth=1.8,
            label=f'{window}-frame rolling mean ({window/sensor_fps:.1f} s)')

    mean_val = arr.mean()
    ax.axhline(mean_val, color='#56B6C2', linewidth=1.2, linestyle='--',
               label=f'Mean = {mean_val:.1f} clusters/frame')

    ax.set_xlabel('Time [s]', color=TEXT_COL)
    ax.set_ylabel('DBSCAN Clusters Detected', color=TEXT_COL)
    ax.set_title('Number of Detected Objects per Frame  (718 frames · 71.8 s)',
                 color=TEXT_COL, fontsize=12, fontweight='bold')
    ax.set_xlim(0, time_axis[-1])
    ax.set_ylim(0, arr.max() * 1.15)
    ax.legend(facecolor=DARK_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL, fontsize=9)

    plt.tight_layout()
    save(fig, 'fig7_detections_per_frame.png')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('LiDAR Report Diagram Generator — DLMDSEAAD02')
    print('=' * 60)

    csv_files = collect_csv_files(DATA_DIR)
    if not csv_files:
        print(f'ERROR: No CSV files found in {DATA_DIR}')
        sys.exit(1)
    print(f'Found {len(csv_files)} frames')

    data = run_and_collect(csv_files)

    snap = data['snap']
    if snap is None:
        print('ERROR: Snapshot frame not captured.')
        sys.exit(1)

    print('\nGenerating figures...')
    fig1_point_cloud_3d(snap)
    fig2_bounding_boxes(snap)
    fig3_trajectories(data['track_histories'], data['track_classes'])
    fig4_classification_dist(data['track_classes'])
    fig5_active_tracks_over_time(data['active_tracks_per_frame'])
    fig6_track_length_dist(data['track_lengths'])
    fig7_detections_per_frame(data['detections_per_frame'])

    print('\nAll 7 figures saved:')
    for i, name in enumerate([
        'fig1_point_cloud_3d.png',
        'fig2_bounding_boxes.png',
        'fig3_trajectories.png',
        'fig4_classification_dist.png',
        'fig5_active_tracks_over_time.png',
        'fig6_track_length_dist.png',
        'fig7_detections_per_frame.png',
    ], 1):
        path = os.path.join(OUT_DIR, name)
        exists = '✓' if os.path.exists(path) else '✗'
        print(f'  {exists}  Fig {i}: {name}')


if __name__ == '__main__':
    main()
