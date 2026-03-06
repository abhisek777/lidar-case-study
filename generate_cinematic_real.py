"""
Cinematic Real-Data LiDAR Visualization
=========================================
Reads the actual Blickfeld Cube 1 dataset (718 CSV frames) through the full
perception pipeline (preprocessing → DBSCAN → classification → Kalman tracking)
and renders each frame with the cinematic cv2 style:

  * Pure black background
  * White/grey ground grid
  * Jet colormap by distance (blue=close, red=far)
  * Neon-glow 3-D bounding boxes for tracked objects
  * VEHICLE / PEDESTRIAN labels above boxes
  * Distance colorbar + spinning LiDAR indicator

Output: lidar_cinematic_real.mp4

Coordinate transform applied on load:
  Blickfeld CSV  →  Cinematic (looking-forward-X)
      old Y      →  new X  (forward / depth)
      old X      →  new Y  (lateral)
      old Z      →  new Z  (vertical)

Author: Kalpana Abhiseka Maddi
Course: DLMDSEAAD02
"""

import os, sys, glob, time, shutil
import numpy as np
import cv2
import pandas as pd
from sklearn.cluster import DBSCAN

# ── Make sure project modules are importable ───────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from classification import FeatureExtractor, RuleBasedClassifier
from tracking import MultiObjectTracker, KalmanObjectTracker

# ── Output config ──────────────────────────────────────────────────────────────
W, H        = 1920, 1080
FPS         = 10               # real data captured at ~10 Hz
MAX_RANGE   = 80.0             # colour scale (m)
FOV_H_DEG   = 65               # horizontal field of view shown (narrower = closer/front view)
GRID_STEP   = 5.0              # metres between grid lines
SENSOR_Z    = 2.0              # sensor height above ground (approx)
GROUND_Z    = -SENSOR_Z        # ground elevation in sensor frame

OUTPUT_MP4  = 'lidar_cinematic_real.mp4'
TMP_DIR     = '_cinematic_real_frames'

DATASET_ROOT = os.path.join(os.path.dirname(__file__), 'Lider datasets')

# ── Object colours (BGR) ──────────────────────────────────────────────────────
OBJ_COLOR = {
    'VEHICLE':    (0,  80, 255),   # bright red-orange
    'PEDESTRIAN': (0, 255,  80),   # bright green
}


# ── Jet colormap ──────────────────────────────────────────────────────────────

def jet_bgr(v: np.ndarray) -> np.ndarray:
    """v: 1-D float32 in [0,1] → (N,3) uint8 BGR."""
    v_u8 = (np.clip(v, 0, 1) * 255).astype(np.uint8).reshape(-1, 1)
    return cv2.applyColorMap(v_u8, cv2.COLORMAP_JET).reshape(-1, 3)

def jet_single(v: float):
    c = jet_bgr(np.array([v], np.float32))[0]
    return (int(c[0]), int(c[1]), int(c[2]))


# ── CSV loading ────────────────────────────────────────────────────────────────

def collect_csv_files(root: str):
    """Return sorted list of all CSV files in the dataset root."""
    pattern = os.path.join(root, '**', '*.csv')
    files   = sorted(glob.glob(pattern, recursive=True))
    return files


def load_frame_csv(path: str) -> np.ndarray:
    """
    Load one CSV frame and transform to cinematic coordinate frame.

    Blickfeld Cube 1 CSV columns: X Y Z DISTANCE INTENSITY …
    We transform:
        new_X = old_Y  (forward / depth axis)
        new_Y = old_X  (lateral)
        new_Z = old_Z  (vertical, unchanged)

    Returns (N, 4) float32: [new_X, new_Y, new_Z, INTENSITY]
    """
    try:
        df = pd.read_csv(path, sep=';', usecols=['X', 'Y', 'Z', 'INTENSITY'])
        x  = df['Y'].values.astype(np.float32)   # forward
        y  = df['X'].values.astype(np.float32)   # lateral
        z  = df['Z'].values.astype(np.float32)   # vertical (unchanged)
        intens = df['INTENSITY'].values.astype(np.float32)
        pts = np.column_stack([x, y, z, intens])
        # Remove NaN
        valid = ~np.isnan(pts).any(axis=1)
        return pts[valid]
    except Exception as e:
        print(f'  [load] Error {path}: {e}')
        return np.zeros((0, 4), np.float32)


# ── Lightweight preprocessing (no open3d) ────────────────────────────────────

def preprocess(pts: np.ndarray) -> np.ndarray:
    """
    Lightweight preprocessing for speed:
      1. Range filter: 1.0 m – MAX_RANGE
      2. Ground removal: Z < (GROUND_Z + 0.4)  keeps objects on/above ground
      3. Height cap:   Z > GROUND_Z + 6.0  removes distant building tops
    Returns (M, 4) non-ground object points.
    """
    if len(pts) == 0:
        return pts

    dist  = np.linalg.norm(pts[:, :3], axis=1)
    mask  = (dist >= 1.0) & (dist <= MAX_RANGE)
    mask &= pts[:, 2] > (GROUND_Z + 0.4)     # above ground
    mask &= pts[:, 2] < (GROUND_Z + 8.0)     # below tall buildings
    return pts[mask]


def get_raw_display_pts(pts: np.ndarray) -> np.ndarray:
    """
    Points to display on screen (ground + objects, range-limited).
    Returns ALL points within MAX_RANGE for the dot cloud.
    """
    if len(pts) == 0:
        return pts
    dist = np.linalg.norm(pts[:, :3], axis=1)
    mask = (dist >= 0.5) & (dist <= MAX_RANGE)
    return pts[mask]


# ── DBSCAN clustering ──────────────────────────────────────────────────────────

def cluster_points(pts: np.ndarray):
    """DBSCAN on XYZ, returns label array (-1 = noise)."""
    if len(pts) < 10:
        return np.full(len(pts), -1, dtype=int)
    db = DBSCAN(eps=0.8, min_samples=8, n_jobs=1).fit(pts[:, :3])
    return db.labels_


# ── Perspective projection ────────────────────────────────────────────────────

def project(world_pts: np.ndarray, sensor_pos: np.ndarray):
    """
    Camera looks along +X (new_X = forward).
    right  = +Y,  up = +Z.
    sensor_pos = [0, 0, SENSOR_Z]
    Returns (u, v, dist, valid_mask).
    """
    rel  = world_pts - sensor_pos
    cx   =  rel[:, 1]           # screen right
    cy   = -rel[:, 2]           # screen down (−Z up)
    cz   =  rel[:, 0]           # depth
    dist = np.sqrt((rel**2).sum(axis=1))

    f     = (W / 2) / np.tan(np.radians(FOV_H_DEG / 2))
    valid = (cz > 0.5) & (dist < MAX_RANGE)

    u = np.full(len(world_pts), -1, np.int32)
    v = np.full(len(world_pts), -1, np.int32)

    if valid.sum() > 0:
        idx   = np.where(valid)[0]
        u_f   = W / 2 + f * cx[valid] / cz[valid]
        v_f   = H / 2 + f * cy[valid] / cz[valid]
        in_sc = (u_f >= 0) & (u_f < W) & (v_f >= 0) & (v_f < H)
        good  = idx[in_sc]
        u[good] = u_f[in_sc].astype(int)
        v[good] = v_f[in_sc].astype(int)
        valid[idx[~in_sc]] = False

    return u, v, dist, valid


# ── Ground grid ───────────────────────────────────────────────────────────────

def draw_grid(frame: np.ndarray, sensor_pos: np.ndarray) -> None:
    sx, sy, sz = float(sensor_pos[0]), float(sensor_pos[1]), float(sensor_pos[2])
    f  = (W / 2) / np.tan(np.radians(FOV_H_DEG / 2))
    gc = (48, 48, 48)   # dark grey grid

    def gp(wx, wy):
        cz = wx - sx
        if cz < 0.5:
            return None
        u = int(W / 2 + f * (wy - sy) / cz)
        # ground Z = GROUND_Z  →  rel_z = GROUND_Z - sz  →  cy = -(GROUND_Z - sz) = sz - GROUND_Z
        cy_ground = sz - GROUND_Z
        v = int(H / 2 + f * cy_ground / cz)
        if 0 <= u < W and 0 <= v < H:
            return (u, v)
        return None

    # Forward lines (constant X range, varying Y)
    for off in np.arange(GRID_STEP, MAX_RANGE + GRID_STEP, GRID_STEP):
        gx = sx + off
        p1 = gp(gx, sy - MAX_RANGE * 0.6)
        p2 = gp(gx, sy + MAX_RANGE * 0.6)
        if p1 and p2:
            cv2.line(frame, p1, p2, gc, 1, cv2.LINE_AA)

    # Lateral lines (constant Y, forward sweep)
    for gy in np.arange(sy - MAX_RANGE * 0.6, sy + MAX_RANGE * 0.6 + GRID_STEP, GRID_STEP):
        p1 = gp(sx + 1.0,       gy)
        p2 = gp(sx + MAX_RANGE, gy)
        if p1 and p2:
            cv2.line(frame, p1, p2, gc, 1, cv2.LINE_AA)


# ── 3-D bounding box ──────────────────────────────────────────────────────────

def draw_object_box(frame, glow_layer, label, cx, cy, cz_base, L, bW, bH, sensor_pos):
    color  = OBJ_COLOR.get(label, (200, 200, 200))
    FW, FH = W, H
    sx, sy, sz = float(sensor_pos[0]), float(sensor_pos[1]), float(sensor_pos[2])
    f = (FW / 2) / np.tan(np.radians(FOV_H_DEG / 2))

    def proj(wx, wy, wz):
        cz_c = wx - sx
        if cz_c < 0.5:
            return None
        u = int(FW / 2 + f * (wy - sy) / cz_c)
        v = int(FH / 2 + f * (-(wz - sz)) / cz_c)
        return (np.clip(u, 0, FW-1), np.clip(v, 0, FH-1))

    dx, dy = L / 2, bW / 2
    corners = [
        (cx-dx, cy-dy, cz_base),      (cx+dx, cy-dy, cz_base),
        (cx+dx, cy+dy, cz_base),      (cx-dx, cy+dy, cz_base),
        (cx-dx, cy-dy, cz_base+bH),   (cx+dx, cy-dy, cz_base+bH),
        (cx+dx, cy+dy, cz_base+bH),   (cx-dx, cy+dy, cz_base+bH),
    ]
    c2d = [proj(*c) for c in corners]

    edges = [(0,1),(1,2),(2,3),(3,0),
             (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]

    for a, b in edges:
        pa, pb = c2d[a], c2d[b]
        if pa and pb:
            cv2.line(glow_layer, pa, pb, color,  5, cv2.LINE_AA)
            cv2.line(frame,      pa, pb, color,  1, cv2.LINE_AA)

    # Label tag above box
    top = proj(cx, cy, cz_base + bH + 0.3)
    if top:
        short = 'VEH' if label == 'VEHICLE' else 'PED'
        tx, ty = np.clip(top[0], 5, FW-80), np.clip(top[1], 16, FH-5)
        (tw, th), _ = cv2.getTextSize(short, cv2.FONT_HERSHEY_SIMPLEX, 0.80, 2)
        cv2.rectangle(glow_layer, (tx-8, ty-th-8), (tx+tw+8, ty+8), color, -1)
        cv2.rectangle(frame,      (tx-5, ty-th-5), (tx+tw+5, ty+5), color, -1)
        cv2.putText(frame, short, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 0, 0), 2, cv2.LINE_AA)


# ── Colorbar ──────────────────────────────────────────────────────────────────

def draw_colorbar(frame: np.ndarray) -> None:
    bx, by, bw, bh = 38, 150, 22, 600
    for i in range(bh):
        v_norm = 1.0 - i / bh
        c = jet_single(v_norm)
        cv2.rectangle(frame, (bx, by+i), (bx+bw, by+i+1), c, -1)
    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (100, 100, 100), 1)
    for k in range(6):
        frac = k / 5
        yt   = by + int(frac * bh)
        dl   = MAX_RANGE * (1 - frac)
        cv2.line(frame, (bx+bw, yt), (bx+bw+5, yt), (150, 150, 150), 1)
        cv2.putText(frame, f'{dl:.0f}', (bx+bw+8, yt+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, 'Distance', (bx-5, by-22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(frame, '[m]', (bx+3, by-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1, cv2.LINE_AA)


# ── Scan indicator ────────────────────────────────────────────────────────────

def draw_scan_indicator(frame: np.ndarray, angle_deg: float) -> None:
    cx_, cy_, r = W - 85, 80, 42
    cv2.circle(frame, (cx_, cy_), r, (38, 38, 38), -1)
    cv2.circle(frame, (cx_, cy_), r, (75, 75, 75),  1)
    rad = np.radians(-angle_deg)
    x2  = int(cx_ + r * np.cos(rad))
    y2  = int(cy_ + r * np.sin(rad))
    cv2.line(frame,   (cx_, cy_), (x2, y2), (0, 220, 220), 2, cv2.LINE_AA)
    cv2.circle(frame, (cx_, cy_), 3, (0, 220, 220), -1)
    cv2.putText(frame, 'LiDAR', (cx_-18, cy_+r+14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 130, 130), 1)


# ── Frame renderer ────────────────────────────────────────────────────────────

def render_frame(raw_pts: np.ndarray,
                 tracks:   list,
                 sensor_pos: np.ndarray,
                 frame_idx: int,
                 scan_angle: float,
                 n_frames:  int,
                 csv_name:  str) -> np.ndarray:

    frame = np.zeros((H, W, 3), dtype=np.uint8)

    # Ground grid
    draw_grid(frame, sensor_pos)

    # Project all display points
    if len(raw_pts) > 0:
        u_arr, v_arr, dist_arr, valid = project(raw_pts[:, :3], sensor_pos)

        if valid.sum() > 0:
            uv = u_arr[valid];  vv = v_arr[valid];  dv = dist_arr[valid]
            # Sort far→near (near overwrites far)
            order = np.argsort(-dv)
            uv, vv, dv = uv[order], vv[order], dv[order]
            colors = jet_bgr(dv / MAX_RANGE)
            frame[vv, uv] = colors
            # Larger blobs for close points
            close = dv < 15.0
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                vr = np.clip(vv[close]+dr, 0, H-1)
                uc = np.clip(uv[close]+dc, 0, W-1)
                frame[vr, uc] = colors[close]

    # Bounding boxes with neon glow
    glow_layer = np.zeros_like(frame)
    n_veh = n_ped = 0
    for ts in tracks:
        lbl = ts.classification
        if lbl not in OBJ_COLOR:
            continue
        if lbl == 'VEHICLE':
            n_veh += 1
        elif lbl == 'PEDESTRIAN':
            n_ped += 1
        cx_, cy_ = float(ts.position[0]), float(ts.position[1])
        L_,  W_  = float(ts.dimensions[0]), float(ts.dimensions[1])
        H_       = float(ts.dimensions[2])
        cz_base  = GROUND_Z + 0.05   # box starts just above ground
        draw_object_box(frame, glow_layer, lbl,
                        cx_, cy_, cz_base, L_, W_, H_, sensor_pos)

    if tracks:
        glow_blur = cv2.GaussianBlur(glow_layer, (25, 25), 0)
        frame[:] = cv2.addWeighted(frame, 1.0, glow_blur, 0.60, 0)

    # Overlays
    draw_scan_indicator(frame, scan_angle)
    draw_colorbar(frame)

    # HUD — top bar
    n_pts = len(raw_pts)
    cv2.putText(frame,
                f'Playing: {csv_name}   Frame: {frame_idx+1:04d}/{n_frames}',
                (82, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180,180,180), 1, cv2.LINE_AA)
    cv2.putText(frame,
                f'Points: {n_pts:,}   Sensor: Blickfeld Cube 1   t = {frame_idx / FPS:.1f} s',
                (82, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (110,110,110), 1, cv2.LINE_AA)

    # Active object counters — bottom-right, bold coloured boxes
    box_x, box_y = W - 280, H - 90
    # Vehicles counter (red-orange)
    cv2.rectangle(frame, (box_x, box_y), (box_x+120, box_y+36), (0, 60, 180), -1)
    cv2.rectangle(frame, (box_x, box_y), (box_x+120, box_y+36), (0, 80, 255),  1)
    cv2.putText(frame, f'VEH: {n_veh:2d}', (box_x+8, box_y+25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 80, 255), 2, cv2.LINE_AA)
    # Pedestrians counter (green)
    cv2.rectangle(frame, (box_x+130, box_y), (box_x+260, box_y+36), (0, 100, 0), -1)
    cv2.rectangle(frame, (box_x+130, box_y), (box_x+260, box_y+36), (0, 255, 80), 1)
    cv2.putText(frame, f'PED: {n_ped:2d}', (box_x+138, box_y+25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 80), 2, cv2.LINE_AA)
    # Label
    cv2.putText(frame, 'ACTIVE OBJECTS', (box_x, box_y-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1, cv2.LINE_AA)

    return frame


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Reset track IDs across runs
    KalmanObjectTracker._next_id = 0

    print('=' * 64)
    print('CINEMATIC LIDAR VIDEO — REAL BLICKFELD CUBE 1 DATASET')
    print('=' * 64)

    # Collect CSV frames
    csv_files = collect_csv_files(DATASET_ROOT)
    if not csv_files:
        print(f'ERROR: No CSV files found in {DATASET_ROOT}')
        return
    n_frames = len(csv_files)
    print(f'  Found {n_frames} CSV frames')
    print(f'  Resolution: {W}x{H}   FPS: {FPS}')
    print(f'  Output: {OUTPUT_MP4}')

    # Initialise trackers
    feature_extractor = FeatureExtractor()
    classifier        = RuleBasedClassifier()
    tracker           = MultiObjectTracker(max_age=5, min_hits=3,
                                           association_threshold=5.0, dt=1.0/FPS)

    os.makedirs(TMP_DIR, exist_ok=True)
    t0 = time.time()

    print(f'\nProcessing {n_frames} frames...')
    for fi, csv_path in enumerate(csv_files):
        # ── Load ──────────────────────────────────────────────────────────────
        raw_all = load_frame_csv(csv_path)

        # Points for rendering (all valid distances — includes ground)
        disp_pts = get_raw_display_pts(raw_all)

        # Points for detection (ground removed)
        det_pts = preprocess(raw_all)

        # ── Detect → cluster → classify → track ───────────────────────────
        tracks = []
        if len(det_pts) >= 20:
            labels     = cluster_points(det_pts)
            features   = feature_extractor.extract_features(det_pts, labels, verbose=False)
            cls_map    = classifier.classify(features, verbose=False)
            for feat in features:
                feat.classification = cls_map.get(feat.cluster_id, 'UNKNOWN')
            tracks = tracker.update(features, verbose=False)

        # ── Render ────────────────────────────────────────────────────────────
        sensor_pos = np.array([0.0, 0.0, SENSOR_Z], np.float32)
        scan_angle = (fi * (360.0 * 10 / FPS)) % 360.0
        csv_name   = os.path.basename(csv_path)

        frame = render_frame(disp_pts, tracks, sensor_pos,
                             fi, scan_angle, n_frames, csv_name)
        cv2.imwrite(os.path.join(TMP_DIR, f'frame_{fi:05d}.png'), frame)

        if (fi + 1) % 50 == 0 or fi == n_frames - 1:
            el  = time.time() - t0
            eta = el / (fi+1) * (n_frames - fi - 1)
            n_veh = sum(1 for t in tracks if t.classification == 'VEHICLE')
            n_ped = sum(1 for t in tracks if t.classification == 'PEDESTRIAN')
            print(f'  [{fi+1:4d}/{n_frames}]  {el:.0f}s elapsed  '
                  f'eta={eta:.0f}s  ({(fi+1)/el:.1f} fps render)  '
                  f'veh={n_veh} ped={n_ped}')

    print(f'\nDone in {time.time()-t0:.0f}s. Compiling video...')

    # ── Compile MP4 ───────────────────────────────────────────────────────────
    pngs   = sorted(glob.glob(os.path.join(TMP_DIR, 'frame_*.png')))
    if not pngs:
        print('ERROR: no frames rendered')
        return

    sample = cv2.imread(pngs[0])
    hh, ww = sample.shape[:2]
    writer  = cv2.VideoWriter(OUTPUT_MP4,
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               FPS, (ww, hh))
    for p in pngs:
        img = cv2.imread(p)
        if img is not None:
            writer.write(img)
    writer.release()
    shutil.rmtree(TMP_DIR, ignore_errors=True)

    size_mb = os.path.getsize(OUTPUT_MP4) / 1e6
    dur_s   = n_frames / FPS
    print(f'\nVideo saved: {OUTPUT_MP4}')
    print(f'  Size: {size_mb:.1f} MB   Duration: {dur_s:.1f}s ({n_frames} frames @ {FPS} fps)')


if __name__ == '__main__':
    main()
