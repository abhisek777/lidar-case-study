"""
3D LiDAR Video Generator — Blickfeld Viewer Style
===================================================
DLMDSEAAD02 -- Localization, Motion Planning and Sensor Fusion

Replicates the visual style of the Blickfeld Cube 1 viewer:
  - Pure black background
  - White grid on the ground plane extending to the horizon
  - Low camera angle (~15 deg elevation) — "standing at ground level"
  - Jet colormap by point height: blue=low, green/yellow=mid, red=high
  - Distance colorbar on the left
  - Dense small points, wide field of view
  - 3D bounding boxes + track labels overlaid on the raw point cloud

Author: Kalpana Abhiseka Maddi
"""

import os
import sys
import glob
import shutil
import argparse
import time
from typing import List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import cv2

from preprocessing  import LiDARPreprocessor
from clustering     import PointCloudClusterer
from classification import FeatureExtractor, RuleBasedClassifier, ObjectFeatures
from tracking       import MultiObjectTracker, KalmanObjectTracker, TrackState


# ── Visual constants — Blickfeld viewer style ─────────────────────────────────
FIG_W    = 19.2        # inches  →  1920 px at 100 dpi
FIG_H    = 10.8        # inches  →  1080 px at 100 dpi
DPI      = 100

# Camera — ground-level view: camera almost at street level looking forward
CAM_ELEV = 5           # degrees elevation (ground level — like standing on the street)
CAM_AZIM = -50         # degrees azimuth

SCENE_R   = 60.0       # metres each side of sensor shown
SCENE_Z   = 20.0       # max Z shown
GRID_STEP = 5          # metres between grid lines

MAX_PTS  = 6000        # max points per frame (performance)

CLASS_COLOR = {
    'VEHICLE':          '#FF3333',
    'PEDESTRIAN':       '#33FF88',
    'STATIC_STRUCTURE': '#FF9900',
    'UNKNOWN':          '#AAAAAA',
}


# ── Ground grid ───────────────────────────────────────────────────────────────

def _draw_ground_grid(ax) -> None:
    """White-ish grid on the z=0 plane, extending to SCENE_R."""
    ticks = np.arange(-SCENE_R, SCENE_R + GRID_STEP, GRID_STEP)
    segs = []
    for v in ticks:
        segs.append([(v, -SCENE_R, 0), (v,  SCENE_R, 0)])   # x-lines
        segs.append([(-SCENE_R, v, 0), ( SCENE_R, v, 0)])   # y-lines
    col = Line3DCollection(segs, colors='#484848', linewidths=0.5,
                           alpha=0.9, zorder=1)
    ax.add_collection3d(col)


# ── 3D bounding box ───────────────────────────────────────────────────────────

def _box_edges(cx, cy, cz_base, L, W, H):
    """Return list of (p1, p2) edge pairs for a 3-D bounding box."""
    dx, dy = L / 2, W / 2
    corners = [
        (cx-dx, cy-dy, cz_base),   (cx+dx, cy-dy, cz_base),
        (cx+dx, cy+dy, cz_base),   (cx-dx, cy+dy, cz_base),
        (cx-dx, cy-dy, cz_base+H), (cx+dx, cy-dy, cz_base+H),
        (cx+dx, cy+dy, cz_base+H), (cx-dx, cy+dy, cz_base+H),
    ]
    idx = [(0,1),(1,2),(2,3),(3,0),
           (4,5),(5,6),(6,7),(7,4),
           (0,4),(1,5),(2,6),(3,7)]
    return [(corners[a], corners[b]) for a, b in idx]


def _draw_box3d(ax, cx, cy, L, W, H, color, lw=2.0) -> None:
    segs = _box_edges(cx, cy, 0.0, L, W, H)
    col  = Line3DCollection(segs, colors=color, linewidths=lw, alpha=0.95, zorder=6)
    ax.add_collection3d(col)


# ── Render single frame ───────────────────────────────────────────────────────

def render_frame(points:     np.ndarray,
                 features:   List[ObjectFeatures],
                 tracks:     List[TrackState],
                 frame_idx:  int,
                 sensor_fps: float,
                 save_path:  str) -> None:

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_facecolor('black')

    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')

    # Hide ALL default matplotlib 3-D decorations (panes, spines, ticks, labels)
    ax.set_axis_off()
    ax.xaxis.pane.fill     = False
    ax.yaxis.pane.fill     = False
    ax.zaxis.pane.fill     = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    ax.grid(False)

    # ── Ground grid ───────────────────────────────────────────────────────────
    _draw_ground_grid(ax)

    # ── Point cloud ───────────────────────────────────────────────────────────
    if len(points) > 0:
        pts = points
        if len(pts) > MAX_PTS:
            idx = np.random.choice(len(pts), MAX_PTS, replace=False)
            pts = pts[idx]

        z     = pts[:, 2]
        z_min = float(z.min())
        z_max = float(max(z.max(), z_min + 0.1))
        z_n   = (z - z_min) / (z_max - z_min)

        sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                        c=z_n, cmap='jet',
                        s=1.0, alpha=0.85,
                        depthshade=True, rasterized=True, zorder=2)
    else:
        z_min, z_max = 0.0, 10.0
        sc = None

    # ── Camera & limits ───────────────────────────────────────────────────────
    ax.set_xlim(-SCENE_R, SCENE_R)
    ax.set_ylim(-SCENE_R, SCENE_R)
    ax.set_zlim(0.0, SCENE_Z)
    ax.view_init(elev=CAM_ELEV, azim=CAM_AZIM)

    # ── 3D bounding boxes + labels ────────────────────────────────────────────
    for t in tracks:
        cls   = t.classification
        color = CLASS_COLOR.get(cls, '#AAAAAA')
        pos   = t.position
        dims  = t.dimensions      # [L, W, H]
        L, W, H = float(dims[0]), float(dims[1]), float(dims[2])

        _draw_box3d(ax, float(pos[0]), float(pos[1]), L, W, H, color)

        # Trajectory on ground plane
        if len(t.history) > 1:
            h = np.array(t.history)
            ax.plot3D(h[:, 0], h[:, 1], np.zeros(len(h)),
                      color=color, lw=1.2, alpha=0.6, zorder=5)

        # Label above box
        short = {'VEHICLE': 'V', 'PEDESTRIAN': 'P',
                 'STATIC_STRUCTURE': 'S', 'UNKNOWN': '?'}.get(cls, '?')
        ax.text(float(pos[0]), float(pos[1]), H + 0.5,
                f'{short}{t.track_id}',
                color='white', fontsize=7, fontweight='bold',
                ha='center', va='bottom', zorder=10,
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor=color, alpha=0.80, edgecolor='none'))

    # ── Sensor dot at origin ──────────────────────────────────────────────────
    ax.scatter([0], [0], [0.05], color='white', s=30, zorder=10)

    # Tight layout before adding manual colorbar text
    plt.tight_layout(pad=0)

    # ── Left-side colorbar (manual axes, Blickfeld style) ─────────────────────
    cbar_ax = fig.add_axes([0.02, 0.15, 0.018, 0.60])   # [left, bottom, w, h]
    norm    = mcolors.Normalize(vmin=z_min, vmax=z_max)
    cb      = fig.colorbar(cm.ScalarMappable(norm=norm, cmap='jet'),
                           cax=cbar_ax)
    cb.set_label('Distance [m]', color='white', fontsize=8, labelpad=4)
    cb.ax.yaxis.set_tick_params(color='#aaaaaa', labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#aaaaaa')
    cb.outline.set_edgecolor('#555555')

    # ── Top status bar (Blickfeld viewer style) ───────────────────────────────
    t_sec = frame_idx / sensor_fps
    n_v = sum(1 for t in tracks if t.classification == 'VEHICLE')
    n_p = sum(1 for t in tracks if t.classification == 'PEDESTRIAN')
    n_s = sum(1 for t in tracks if t.classification == 'STATIC_STRUCTURE')
    status = (f'Playing: 192.168.26.26_2020-11-25_20-01-45   '
              f'Frame: {frame_idx:04d}   t = {t_sec:.1f} s   '
              f'Tracks: {len(tracks)}   V={n_v}  P={n_p}  S={n_s}')
    fig.text(0.5, 0.975, status,
             ha='center', va='top', color='#cccccc', fontsize=9,
             fontfamily='monospace',
             bbox=dict(facecolor='black', alpha=0.0, edgecolor='none'))

    # ── Scale marker bottom-right (Blickfeld style) ───────────────────────────
    fig.text(0.97, 0.04, '1m', ha='right', va='bottom',
             color='white', fontsize=8, fontfamily='monospace')

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=CLASS_COLOR['VEHICLE'],          label='Vehicle'),
        mpatches.Patch(color=CLASS_COLOR['PEDESTRIAN'],       label='Pedestrian'),
        mpatches.Patch(color=CLASS_COLOR['STATIC_STRUCTURE'], label='Static Structure'),
        mpatches.Patch(color=CLASS_COLOR['UNKNOWN'],          label='Unknown'),
    ]
    fig.legend(handles=legend_handles, loc='lower right',
               facecolor='#111111', edgecolor='#444444',
               labelcolor='white', fontsize=8, framealpha=0.85,
               bbox_to_anchor=(0.99, 0.04))

    plt.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='black',
                pad_inches=0.05)
    plt.close(fig)


# ── Compile PNGs to MP4 ───────────────────────────────────────────────────────

def compile_to_mp4(frame_dir: str, output_path: str, fps: float) -> None:
    pngs = sorted(glob.glob(os.path.join(frame_dir, 'frame_*.png')))
    if not pngs:
        print('  No frames found.')
        return
    sample = cv2.imread(pngs[0])
    if sample is None:
        return
    h, w   = sample.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    print(f'  Compiling {len(pngs)} frames -> {output_path}  ({w}x{h} @ {fps} fps) ...')
    for p in pngs:
        img = cv2.imread(p)
        if img is not None:
            writer.write(img)
    writer.release()
    size_mb = os.path.getsize(output_path) / 1e6
    print(f'  Done: {output_path}  ({size_mb:.1f} MB)')


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_video(lider_dir:   str,
                   output_path: str           = 'lidar_3d_tracking.mp4',
                   sensor_fps:  float         = 10.0,
                   max_frames:  Optional[int] = None,
                   keep_frames: bool          = False) -> None:

    print('=' * 70)
    print('LIDAR 3D VIDEO  —  Blickfeld viewer style')
    print('=' * 70)

    parts     = sorted(glob.glob(os.path.join(lider_dir, '*_part_*')))
    csv_files: List[str] = []
    for part in parts:
        if os.path.isdir(part):
            csv_files.extend(sorted(glob.glob(os.path.join(part, '*.csv'))))
    if max_frames:
        csv_files = csv_files[:max_frames]
    if not csv_files:
        print(f'No CSV files found in {lider_dir}'); sys.exit(1)

    print(f'Frames  : {len(csv_files)}  ({len(csv_files)/sensor_fps:.1f} s)')
    print(f'Output  : {output_path}')
    print(f'Res     : {int(FIG_W*DPI)}x{int(FIG_H*DPI)}')

    # Pipeline
    preprocessor = LiDARPreprocessor(min_range=2.0, max_range=100.0,
                                     voxel_size=0.15, ground_threshold=0.25)
    clusterer    = PointCloudClusterer(eps=0.8, min_samples=8, min_cluster_size=8)
    extractor    = FeatureExtractor()
    classifier   = RuleBasedClassifier()
    tracker      = MultiObjectTracker(max_age=8, min_hits=2,
                                      association_threshold=4.0,
                                      dt=1.0 / sensor_fps)
    KalmanObjectTracker._next_id = 0

    out_dir    = os.path.dirname(os.path.abspath(output_path))
    frames_tmp = os.path.join(out_dir, '_video_frames_tmp')
    os.makedirs(frames_tmp, exist_ok=True)

    rendered = 0
    t_start  = time.time()
    print('\nRendering ...')

    for idx, csv_path in enumerate(csv_files):
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

            processed = preprocessor.preprocess(pts,            verbose=False)
            labels    = clusterer.cluster(processed,             verbose=False)
            features  = extractor.extract_features(processed, labels, verbose=False)
            classifier.classify(features,                        verbose=False)
            tracks    = tracker.update(features,                 verbose=False)

            save_path = os.path.join(frames_tmp, f'frame_{idx:05d}.png')
            render_frame(processed, features, tracks, idx, sensor_fps, save_path)
            rendered += 1

        except Exception as exc:
            print(f'  Warning: frame {idx} skipped — {exc}')
            continue

        if (idx + 1) % 50 == 0 or idx == len(csv_files) - 1:
            elapsed = time.time() - t_start
            eta     = elapsed / max(idx + 1, 1) * (len(csv_files) - idx - 1)
            n_v = sum(1 for t in tracks if t.classification == 'VEHICLE')
            n_p = sum(1 for t in tracks if t.classification == 'PEDESTRIAN')
            print(f'  [{idx+1:4d}/{len(csv_files)}]  rendered={rendered}  '
                  f'tracks={len(tracks)} (V={n_v} P={n_p})  '
                  f'elapsed={elapsed:.0f}s  eta={eta:.0f}s')

    print(f'\n{rendered} frames rendered in {time.time()-t_start:.0f} s')
    compile_to_mp4(frames_tmp, output_path, sensor_fps)

    if not keep_frames:
        shutil.rmtree(frames_tmp, ignore_errors=True)
    else:
        print(f'  Frames kept in: {frames_tmp}')


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Lider datasets')
    OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lidar_3d_tracking.mp4')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data',   '-d', type=str,   default=BASE)
    parser.add_argument('--output', '-o', type=str,   default=OUT)
    parser.add_argument('--fps',          type=float, default=10.0)
    parser.add_argument('--frames', '-n', type=int,   default=None)
    parser.add_argument('--keep-frames',  action='store_true')
    args = parser.parse_args()

    generate_video(lider_dir   = args.data,
                   output_path = args.output,
                   sensor_fps  = args.fps,
                   max_frames  = args.frames,
                   keep_frames = args.keep_frames)
