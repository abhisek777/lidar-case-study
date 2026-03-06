"""
Cinematic Synthetic LiDAR Visualization
========================================
Creates a 1920x1080 MP4 of a synthetic urban scene scanned by a vehicle-mounted LiDAR.
Visual style: Blickfeld viewer — pure black background, white grid floor,
jet colormap (blue=close, red=far), distance colorbar on left.

Usage:
    python generate_cinematic_video.py
    python generate_cinematic_video.py --output my_video.mp4 --duration 15

Author: Kalpana Abhiseka Maddi
"""

import os, shutil, glob, time, argparse
import numpy as np
import cv2

# ── Config ────────────────────────────────────────────────────────────────────
W, H        = 1920, 1080
FPS         = 20
SENSOR_H    = 1.8          # m — roof-mounted LiDAR height
MAX_RANGE   = 60.0         # m — color scale + visibility cutoff
SPEED       = 3.5          # m/s — vehicle speed (~12.6 km/h)
FOV_H_DEG   = 90           # horizontal field of view
GRID_STEP   = 5.0          # metres between grid lines
OUTPUT_MP4  = 'lidar_cinematic.mp4'
TMP_DIR     = '_cinematic_frames'

np.random.seed(42)


# ── Jet colormap (OpenCV built-in, correct) ───────────────────────────────────

def jet_bgr(v: np.ndarray) -> np.ndarray:
    """v: 1-D float32 array 0-1  →  (N, 3) uint8 BGR."""
    v_u8 = (np.clip(v, 0, 1) * 255).astype(np.uint8).reshape(-1, 1)
    return cv2.applyColorMap(v_u8, cv2.COLORMAP_JET).reshape(-1, 3)

def jet_single(v: float):
    c = jet_bgr(np.array([v], np.float32))[0]
    return (int(c[0]), int(c[1]), int(c[2]))


# ── Scene generation ──────────────────────────────────────────────────────────

def make_scene() -> np.ndarray:
    """Generate static urban scene as (N, 3) float32 point cloud."""
    parts = []

    # ── Road (z≈0, x=-20..200, y=-4..4) ──────────────────────────────────────
    n = 35000
    xs = np.random.uniform(-20, 200, n)
    ys = np.random.uniform(-4.0,  4.0, n)
    zs = np.random.normal(0, 0.015, n)
    parts.append(np.column_stack([xs, ys, zs]))

    # ── Sidewalks (raised 12 cm) ───────────────────────────────────────────────
    for y0, y1 in [(4.0, 6.8), (-6.8, -4.0)]:
        n = 10000
        xs = np.random.uniform(-15, 200, n)
        ys = np.random.uniform(y0, y1, n)
        zs = np.ones(n) * 0.12 + np.random.normal(0, 0.02, n)
        parts.append(np.column_stack([xs, ys, zs]))

    # ── Buildings RIGHT side (facade at y=7) ──────────────────────────────────
    for x0, x1, h in [(8,55,13), (58,108,18), (112,158,11), (162,210,15)]:
        # Road-facing facade
        n = int((x1-x0) * h * 10)
        xs = np.random.uniform(x0, x1, n)
        ys = np.ones(n) * 7.0 + np.random.normal(0, 0.04, n)
        zs = np.random.uniform(0, h, n)
        parts.append(np.column_stack([xs, ys, zs]))
        # Near-end facade
        n2 = int(10 * h * 6)
        xs2 = np.ones(n2) * x0 + np.random.normal(0, 0.04, n2)
        ys2 = np.random.uniform(7.0, 17.0, n2)
        zs2 = np.random.uniform(0, h, n2)
        parts.append(np.column_stack([xs2, ys2, zs2]))
        # Rooftop
        n3 = int((x1-x0) * 4)
        xs3 = np.random.uniform(x0, x1, n3)
        ys3 = np.random.uniform(7.0, 17.0, n3)
        zs3 = np.ones(n3) * h + np.random.normal(0, 0.02, n3)
        parts.append(np.column_stack([xs3, ys3, zs3]))

    # ── Buildings LEFT side (facade at y=-7) ──────────────────────────────────
    for x0, x1, h in [(5,52,12), (55,106,16), (110,155,13), (158,205,14)]:
        n = int((x1-x0) * h * 10)
        xs = np.random.uniform(x0, x1, n)
        ys = np.ones(n) * (-7.0) + np.random.normal(0, 0.04, n)
        zs = np.random.uniform(0, h, n)
        parts.append(np.column_stack([xs, ys, zs]))
        n2 = int(10 * h * 6)
        xs2 = np.ones(n2) * x0 + np.random.normal(0, 0.04, n2)
        ys2 = np.random.uniform(-17.0, -7.0, n2)
        zs2 = np.random.uniform(0, h, n2)
        parts.append(np.column_stack([xs2, ys2, zs2]))
        n3 = int((x1-x0) * 4)
        xs3 = np.random.uniform(x0, x1, n3)
        ys3 = np.random.uniform(-17.0, -7.0, n3)
        zs3 = np.ones(n3) * h + np.random.normal(0, 0.02, n3)
        parts.append(np.column_stack([xs3, ys3, zs3]))

    # ── Parked cars RIGHT (road face at y=5.0) ────────────────────────────────
    for cx in [18, 32, 46, 68, 84, 100, 118, 138, 155, 172]:
        # Side face (road-facing)
        xs = np.random.uniform(cx-2.2, cx+2.2, 120)
        ys = np.ones(120) * 5.0 + np.random.normal(0, 0.03, 120)
        zs = np.random.uniform(0, 1.5, 120)
        parts.append(np.column_stack([xs, ys, zs]))
        # Roof
        xs = np.random.uniform(cx-1.4, cx+1.4, 50)
        ys = np.random.uniform(5.1, 6.5, 50)
        zs = np.ones(50) * 1.5 + np.random.normal(0, 0.02, 50)
        parts.append(np.column_stack([xs, ys, zs]))
        # Front/rear faces
        for sign in [-1, 1]:
            xs = np.ones(35) * (cx + sign*2.2) + np.random.normal(0, 0.02, 35)
            ys = np.random.uniform(5.0, 6.5, 35)
            zs = np.random.uniform(0, 1.5, 35)
            parts.append(np.column_stack([xs, ys, zs]))

    # ── Parked cars LEFT (road face at y=-5.0) ────────────────────────────────
    for cx in [12, 28, 44, 65, 88, 105, 125, 145, 162, 180]:
        xs = np.random.uniform(cx-2.2, cx+2.2, 120)
        ys = np.ones(120) * (-5.0) + np.random.normal(0, 0.03, 120)
        zs = np.random.uniform(0, 1.5, 120)
        parts.append(np.column_stack([xs, ys, zs]))
        xs = np.random.uniform(cx-1.4, cx+1.4, 50)
        ys = np.random.uniform(-6.5, -5.1, 50)
        zs = np.ones(50) * 1.5 + np.random.normal(0, 0.02, 50)
        parts.append(np.column_stack([xs, ys, zs]))
        for sign in [-1, 1]:
            xs = np.ones(35) * (cx + sign*2.2) + np.random.normal(0, 0.02, 35)
            ys = np.random.uniform(-6.5, -5.0, 35)
            zs = np.random.uniform(0, 1.5, 35)
            parts.append(np.column_stack([xs, ys, zs]))

    # ── Extra road / buildings to cover 60-second drive (280 m) ──────────────
    xs = np.random.uniform(200, 280, 12000)
    ys = np.random.uniform(-4.0, 4.0, 12000)
    zs = np.random.normal(0, 0.015, 12000)
    parts.append(np.column_stack([xs, ys, zs]))
    for y0, y1 in [(4.0, 6.8), (-6.8, -4.0)]:
        xs = np.random.uniform(200, 280, 4000)
        ys = np.random.uniform(y0, y1, 4000)
        zs = np.ones(4000)*0.12 + np.random.normal(0, 0.02, 4000)
        parts.append(np.column_stack([xs, ys, zs]))
    for x0, x1, h in [(210,260,14),(262,280,10)]:
        n = int((x1-x0)*h*8)
        xs = np.random.uniform(x0, x1, n)
        ys = np.ones(n)*7.0 + np.random.normal(0, 0.04, n)
        zs = np.random.uniform(0, h, n)
        parts.append(np.column_stack([xs, ys, zs]))
        xs = np.random.uniform(x0, x1, n)
        ys = np.ones(n)*(-7.0) + np.random.normal(0, 0.04, n)
        parts.append(np.column_stack([xs, ys, zs]))

    # ── Street lights ─────────────────────────────────────────────────────────
    for lx in range(0, 280, 20):
        for ly in [5.5, -5.5]:
            xs = np.ones(25) * lx + np.random.normal(0, 0.03, 25)
            ys = np.ones(25) * ly + np.random.normal(0, 0.03, 25)
            zs = np.random.uniform(0, 5.5, 25)
            parts.append(np.column_stack([xs, ys, zs]))

    return np.vstack(parts).astype(np.float32)


# ── Moving objects ────────────────────────────────────────────────────────────

# Object descriptor: (label, cx, cy, L, W, H, box_color_bgr)
OBJ_COLOR = {
    'PED':  (0,   255,  80),   # bright green
    'CAR':  (0,    80, 255),   # bright red-orange
    'CYC':  (0,   220, 255),   # bright yellow
}

def moving_objects(t: float):
    """
    Returns:
        pts   : (N, 3) float32 point cloud of all moving objects
        objs  : list of (label, cx, cy, cz_base, L, W, H)
    """
    pts  = []
    objs = []
    sensor_x = SPEED * t

    def ped_pts(px, py, n=400):
        xs = np.random.normal(px, 0.13, n)
        ys = np.random.normal(py, 0.13, n)
        zs = np.random.uniform(0, 1.75, n)
        return np.column_stack([xs, ys, zs])

    def car_pts(cx, cy):
        p = []
        # Side facing road (dense)
        xs = np.random.uniform(cx-2.2, cx+2.2, 500)
        ys = np.ones(500)*cy + np.random.normal(0, 0.04, 500)
        zs = np.random.uniform(0, 1.5, 500)
        p.append(np.column_stack([xs, ys, zs]))
        # Roof
        xs = np.random.uniform(cx-1.4, cx+1.4, 250)
        ys = np.random.uniform(cy-0.9, cy+0.9, 250)
        zs = np.ones(250)*1.5 + np.random.normal(0, 0.02, 250)
        p.append(np.column_stack([xs, ys, zs]))
        # Front face
        xs = np.ones(250)*(cx-2.2) + np.random.normal(0, 0.02, 250)
        ys = np.random.uniform(cy-0.9, cy+0.9, 250)
        zs = np.random.uniform(0, 1.5, 250)
        p.append(np.column_stack([xs, ys, zs]))
        return np.vstack(p)

    # ── Pedestrians along right sidewalk (walk TOWARD vehicle) ────────────────
    for wx in [20, 60, 100, 140, 180, 220]:
        px = wx - t * 1.2
        if sensor_x - 5 < px < sensor_x + MAX_RANGE:
            pts.append(ped_pts(px, 5.4))
            objs.append(('PED', px, 5.4, 0.0, 0.65, 0.65, 1.75))

    # ── Pedestrians along left sidewalk (walk SAME direction as vehicle) ──────
    for wx in [15, 55, 95, 135, 175, 215]:
        px = wx + t * 1.0
        if sensor_x - 5 < px < sensor_x + MAX_RANGE:
            pts.append(ped_pts(px, -5.4))
            objs.append(('PED', px, -5.4, 0.0, 0.65, 0.65, 1.75))

    # ── Crossing pedestrians (appear at intervals, cross road) ────────────────
    for wx, t_start in [(40, 0), (90, 14), (140, 28), (190, 42)]:
        dt  = t - t_start
        p3y = 4.5 - dt * 0.7
        p3x = wx + dt * 0.2
        if 0 <= dt <= 13 and sensor_x - 5 < p3x < sensor_x + MAX_RANGE:
            pts.append(ped_pts(p3x, p3y))
            objs.append(('PED', p3x, p3y, 0.0, 0.65, 0.65, 1.75))

    # ── Cyclists (appear every 15 s) ──────────────────────────────────────────
    for t_start in [0, 15, 30, 45]:
        dt  = t - t_start
        cx  = (sensor_x + 25) + dt * 4.5   # 25m ahead at start, moves forward
        cy  = -1.5
        if 0 <= dt <= 12 and sensor_x - 5 < cx < sensor_x + MAX_RANGE:
            xs = np.random.uniform(cx-0.95, cx+0.95, 400)
            ys = np.random.normal(cy, 0.20, 400)
            zs = np.random.uniform(0, 1.15, 400)
            pts.append(np.column_stack([xs, ys, zs]))
            objs.append(('CYC', cx, cy, 0.0, 1.9, 0.65, 1.15))

    # ── Oncoming cars (one every ~10 s) ───────────────────────────────────────
    # car starts 60m ahead of current sensor position every 10 s
    for t_start in [0, 10, 20, 30, 40, 50]:
        dt      = t - t_start
        car_wx  = (sensor_x + 60) - dt * 8.5   # world x
        if sensor_x - 20 < car_wx < sensor_x + MAX_RANGE:
            pts.append(car_pts(car_wx, -1.85))
            objs.append(('CAR', car_wx, -1.85, 0.0, 4.5, 1.9, 1.5))

    if pts:
        return np.vstack(pts).astype(np.float32), objs
    return np.zeros((0, 3), np.float32), objs


# ── 3-D bounding box overlay ─────────────────────────────────────────────────

def draw_object_box(frame, glow_layer, label, cx, cy, cz_base, L, W, H, sensor_pos):
    """
    Draw bold 3-D bounding box with neon glow effect.
    - glow_layer: separate canvas that gets blurred and blended for glow
    - frame: final output frame
    """
    color = OBJ_COLOR.get(label, (255, 255, 255))
    FW, FH = globals()['W'], globals()['H']
    sx, sy, sz = float(sensor_pos[0]), float(sensor_pos[1]), float(sensor_pos[2])
    f = (FW / 2) / np.tan(np.radians(FOV_H_DEG / 2))

    def proj(wx, wy, wz):
        cz_c = wx - sx
        if cz_c < 0.2:
            return None
        u = int(FW / 2 + f * (wy - sy) / cz_c)
        v = int(FH / 2 + f * (-(wz - sz)) / cz_c)
        return (np.clip(u, 0, FW-1), np.clip(v, 0, FH-1))

    dx, dy = L / 2, W / 2
    corners = [
        (cx-dx, cy-dy, cz_base),   (cx+dx, cy-dy, cz_base),
        (cx+dx, cy+dy, cz_base),   (cx-dx, cy+dy, cz_base),
        (cx-dx, cy-dy, cz_base+H), (cx+dx, cy-dy, cz_base+H),
        (cx+dx, cy+dy, cz_base+H), (cx-dx, cy+dy, cz_base+H),
    ]
    c2d = [proj(*c) for c in corners]

    edges = [(0,1),(1,2),(2,3),(3,0),
             (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]

    for a, b in edges:
        pa, pb = c2d[a], c2d[b]
        if pa and pb:
            # Thick glow line on glow layer
            cv2.line(glow_layer, pa, pb, color, 10, cv2.LINE_AA)
            # Bold crisp line on frame
            cv2.line(frame,      pa, pb, color,  3, cv2.LINE_AA)

    # Bold label tag above box
    top = proj(cx, cy, cz_base + H + 0.4)
    if top:
        tx, ty = np.clip(top[0], 5, FW-80), np.clip(top[1], 14, FH-5)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        # Glow halo on label background
        cv2.rectangle(glow_layer, (tx-6, ty-th-6), (tx+tw+6, ty+6), color, -1)
        # Solid label box on frame
        cv2.rectangle(frame,      (tx-4, ty-th-4), (tx+tw+4, ty+4), color, -1)
        cv2.putText(frame, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)


# ── Perspective projection ────────────────────────────────────────────────────

def project(world_pts: np.ndarray, sensor_pos: np.ndarray):
    """
    Camera looks along +X world axis.
    right = +Y world,  up = +Z world.
    Returns (u, v, dist, valid_mask).
    """
    rel  = world_pts - sensor_pos          # (N, 3)
    cx   =  rel[:, 1]                      # screen right
    cy   = -rel[:, 2]                      # screen down (−Z)
    cz   =  rel[:, 0]                      # depth (forward)
    dist = np.sqrt((rel**2).sum(axis=1))

    f     = (W / 2) / np.tan(np.radians(FOV_H_DEG / 2))
    valid = (cz > 0.2) & (dist < MAX_RANGE)

    u = np.full(len(world_pts), -1, np.int32)
    v = np.full(len(world_pts), -1, np.int32)

    if valid.sum() > 0:
        idx   = np.where(valid)[0]
        u_f   = W/2 + f * cx[valid] / cz[valid]
        v_f   = H/2 + f * cy[valid] / cz[valid]
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
    gc = (52, 52, 52)   # dark grey grid

    def gp(wx, wy):
        cz = wx - sx
        if cz < 0.3:
            return None
        u = int(W/2 + f * (wy - sy) / cz)
        v = int(H/2 + f * sz / cz)          # ground z=0, cy = -(0-sz) = sz
        return (np.clip(u, 0, W-1), np.clip(v, 0, H-1))

    # Lines running left-right (constant x, varies y)
    for off in np.arange(GRID_STEP, MAX_RANGE + GRID_STEP, GRID_STEP):
        gx = sx + off
        p1 = gp(gx, sy - MAX_RANGE)
        p2 = gp(gx, sy + MAX_RANGE)
        if p1 and p2:
            cv2.line(frame, p1, p2, gc, 1, cv2.LINE_AA)

    # Lines running forward-back (constant y, varies x)
    for gy in np.arange(sy - MAX_RANGE, sy + MAX_RANGE + GRID_STEP, GRID_STEP):
        p1 = gp(sx + 1.0,       gy)
        p2 = gp(sx + MAX_RANGE, gy)
        if p1 and p2:
            cv2.line(frame, p1, p2, gc, 1, cv2.LINE_AA)


# ── Colorbar ──────────────────────────────────────────────────────────────────

def draw_colorbar(frame: np.ndarray, max_range: float) -> None:
    bx, by, bw, bh = 38, 150, 22, 600
    for i in range(bh):
        v_norm = 1.0 - i / bh          # top=red(far), bottom=blue(close)
        c = jet_single(v_norm)
        cv2.rectangle(frame, (bx, by+i), (bx+bw, by+i+1), c, -1)
    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (100, 100, 100), 1)

    for k in range(6):
        frac    = k / 5
        yt      = by + int(frac * bh)
        d_label = max_range * (1 - frac)
        cv2.line(frame, (bx+bw, yt), (bx+bw+5, yt), (150, 150, 150), 1)
        cv2.putText(frame, f'{d_label:.0f}', (bx+bw+8, yt+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

    cv2.putText(frame, 'Distance', (bx-5, by-22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(frame, '[m]', (bx+3,  by-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1, cv2.LINE_AA)


# ── Scan indicator ────────────────────────────────────────────────────────────

def draw_scan_indicator(frame: np.ndarray, angle_deg: float) -> None:
    """Spinning LiDAR radar dial in top-right corner."""
    cx, cy, r = W - 85, 80, 42
    cv2.circle(frame, (cx, cy), r,   (38, 38, 38), -1)
    cv2.circle(frame, (cx, cy), r,   (75, 75, 75),  1)
    rad = np.radians(-angle_deg)
    x2  = int(cx + r * np.cos(rad))
    y2  = int(cy + r * np.sin(rad))
    cv2.line(frame,   (cx, cy), (x2, y2), (0, 220, 220), 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 3, (0, 220, 220), -1)
    cv2.putText(frame, 'LiDAR', (cx-18, cy+r+14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 130, 130), 1)


# ── Frame renderer ────────────────────────────────────────────────────────────

def render_frame(scene:      np.ndarray,
                 moving:     np.ndarray,
                 objs:       list,
                 sensor_pos: np.ndarray,
                 frame_idx:  int,
                 scan_angle: float,
                 n_frames:   int) -> np.ndarray:

    frame = np.zeros((H, W, 3), dtype=np.uint8)

    # Grid (behind points)
    draw_grid(frame, sensor_pos)

    # Combine static + moving
    all_pts = np.vstack([scene, moving]) if len(moving) > 0 else scene

    # Project
    u_arr, v_arr, dist_arr, valid = project(all_pts, sensor_pos)

    if valid.sum() > 0:
        uv = u_arr[valid]
        vv = v_arr[valid]
        dv = dist_arr[valid]

        # Sort far→near so near points overwrite far
        order = np.argsort(-dv)
        uv, vv, dv = uv[order], vv[order], dv[order]

        # Color by distance
        colors = jet_bgr(dv / MAX_RANGE)

        # Render
        frame[vv, uv] = colors

        # Slightly larger blobs for close points (< 15 m)
        close = dv < 15.0
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            vr = np.clip(vv[close] + dr, 0, H-1)
            uc = np.clip(uv[close] + dc, 0, W-1)
            frame[vr, uc] = colors[close]

    # Draw 3-D bounding boxes with neon glow
    glow_layer = np.zeros_like(frame)
    for (label, cx, cy, cz_base, L, bw, bh) in objs:
        draw_object_box(frame, glow_layer, label, cx, cy, cz_base, L, bw, bh, sensor_pos)
    # Blur glow layer and blend onto frame
    if objs:
        glow_blur = cv2.GaussianBlur(glow_layer, (21, 21), 0)
        frame[:] = cv2.addWeighted(frame, 1.0, glow_blur, 0.55, 0)

    # Overlays
    draw_scan_indicator(frame, scan_angle)
    draw_colorbar(frame, MAX_RANGE)

    # HUD
    t_sec  = frame_idx / FPS
    sp_kmh = SPEED * 3.6
    n_pts  = int(valid.sum())
    cv2.putText(frame,
                f'Playing: urban_lidar_simulation.bfpc   '
                f'Frame: {frame_idx:04d}   t = {t_sec:.1f} s',
                (82, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (180,180,180), 1, cv2.LINE_AA)
    cv2.putText(frame,
                f'Speed: {sp_kmh:.1f} km/h   '
                f'Points: {n_pts:,}   Sensor: Blickfeld Cube 1',
                (82, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (110,110,110), 1, cv2.LINE_AA)
    cv2.putText(frame, '5m', (W-65, H-18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (150,150,150), 1, cv2.LINE_AA)

    return frame


# ── Main ──────────────────────────────────────────────────────────────────────

def main(output_mp4: str, duration_s: float) -> None:
    n_frames = int(FPS * duration_s)
    os.makedirs(TMP_DIR, exist_ok=True)

    print('=' * 60)
    print('CINEMATIC SYNTHETIC LIDAR VIDEO')
    print('=' * 60)
    print(f'  Duration  : {duration_s:.0f} s  ({n_frames} frames @ {FPS} fps)')
    print(f'  Resolution: {W}x{H}')
    print(f'  Output    : {output_mp4}')

    print('\nGenerating urban scene...')
    scene = make_scene()
    print(f'  {len(scene):,} static points')

    print(f'\nRendering {n_frames} frames...')
    t0 = time.time()

    for fi in range(n_frames):
        t          = fi / FPS
        sensor_pos = np.array([SPEED * t, 0.0, SENSOR_H], np.float32)
        scan_angle = (fi * (360.0 * 10 / FPS)) % 360.0   # 10 Hz spin

        mov, objs = moving_objects(t)
        frame = render_frame(scene, mov, objs, sensor_pos, fi, scan_angle, n_frames)
        cv2.imwrite(os.path.join(TMP_DIR, f'frame_{fi:05d}.png'), frame)

        if (fi + 1) % 40 == 0 or fi == n_frames - 1:
            el  = time.time() - t0
            eta = el / (fi+1) * (n_frames - fi - 1)
            print(f'  [{fi+1:4d}/{n_frames}]  '
                  f'{el:.0f}s elapsed  eta={eta:.0f}s  '
                  f'({(fi+1)/el:.1f} fps render)')

    print(f'\nDone in {time.time()-t0:.0f}s. Compiling to {output_mp4}...')

    pngs   = sorted(glob.glob(os.path.join(TMP_DIR, 'frame_*.png')))
    sample = cv2.imread(pngs[0])
    hh, ww = sample.shape[:2]
    writer  = cv2.VideoWriter(output_mp4,
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               FPS, (ww, hh))
    for p in pngs:
        img = cv2.imread(p)
        if img is not None:
            writer.write(img)
    writer.release()
    shutil.rmtree(TMP_DIR, ignore_errors=True)

    size_mb = os.path.getsize(output_mp4) / 1e6
    print(f'Video saved: {output_mp4}  ({size_mb:.1f} MB, {duration_s:.0f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',   '-o', default=OUTPUT_MP4)
    parser.add_argument('--duration', '-d', type=float, default=60.0,
                        help='Video duration in seconds (default: 60)')
    args = parser.parse_args()
    main(args.output, args.duration)
