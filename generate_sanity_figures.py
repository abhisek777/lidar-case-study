"""
Generate 4 Sanity Check figures from real LiDAR data.
=====================================================
DLMDSEAAD02 — Section 5: Sanity Check and Data Validation

  fig_s1_range_distribution.png     — Distribution of detection distances (5.1)
  fig_s2_density_vs_distance.png    — Point density vs distance (5.2)
  fig_s3_height_distribution.png    — Vertical (Z) distribution of points (5.3)
  fig_s4_intensity_distribution.png — Return intensity distribution (5.4)

Author: Kalpana Abhiseka Maddi
"""

import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'Lider datasets')
OUT_DIR  = BASE_DIR

DARK_BG  = '#0D1117'
GRID_COL = '#21262D'
TEXT_COL = '#E6EDF3'
ACCENT   = '#58A6FF'
ACCENT2  = '#2ECC71'
ACCENT3  = '#F39C12'
ACCENT4  = '#E74C3C'


# ── Load a sample of frames for statistics ────────────────────────────────────

def collect_csv_files(data_dir):
    parts = sorted(glob.glob(os.path.join(data_dir, '*_part_*')))
    files = []
    for p in parts:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, '*.csv'))))
    return files


def load_points_sample(csv_files, every_n=5):
    """Load every N-th frame and stack all points together."""
    all_pts = []
    sampled = csv_files[::every_n]
    print(f'  Loading {len(sampled)} frames (every {every_n}th of {len(csv_files)})...')
    for path in sampled:
        try:
            df = pd.read_csv(path, sep=';')
            df.columns = df.columns.str.upper().str.strip()
            if not all(c in df.columns for c in ['X', 'Y', 'Z']):
                continue
            intensity = df['INTENSITY'].values if 'INTENSITY' in df.columns else np.ones(len(df)) * 0.5
            pts = np.column_stack([df['X'].values, df['Y'].values,
                                   df['Z'].values, intensity]).astype(np.float32)
            pts = pts[~np.isnan(pts).any(axis=1)]
            all_pts.append(pts)
        except Exception:
            continue
    return np.vstack(all_pts) if all_pts else np.zeros((0, 4))


def styled_ax(ax):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT_COL, which='both')
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.title.set_color(TEXT_COL)
    for sp in ax.spines.values():
        sp.set_color(GRID_COL)
    ax.grid(True, color=GRID_COL, linewidth=0.6, alpha=0.8)


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f'  Saved: {name}')


# ── Figure S1: Range distribution ─────────────────────────────────────────────

def fig_s1_range_distribution(pts):
    dist = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2 + pts[:, 2]**2)
    # Clip to sensor spec max for clean display
    dist = dist[dist <= 250]

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=DARK_BG)
    styled_ax(ax)

    counts, bin_edges, patches = ax.hist(dist, bins=100, color=ACCENT,
                                          edgecolor=DARK_BG, linewidth=0.4)

    # Colour bars: near (< 20 m) = green, mid (20–100 m) = blue, far (> 100 m) = orange
    for patch, left in zip(patches, bin_edges[:-1]):
        if left < 20:
            patch.set_facecolor(ACCENT2)
        elif left < 100:
            patch.set_facecolor(ACCENT)
        else:
            patch.set_facecolor(ACCENT3)

    mean_d = float(np.mean(dist))
    median_d = float(np.median(dist))
    ax.axvline(mean_d,   color='#FFD700', lw=1.8, ls='--',
               label=f'Mean = {mean_d:.1f} m')
    ax.axvline(median_d, color='#FF7B54', lw=1.5, ls=':',
               label=f'Median = {median_d:.1f} m')
    ax.axvline(100, color='#FFFFFF', lw=1.2, ls='-.',
               label='Effective max range (100 m)', alpha=0.5)

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=ACCENT2, label='Near field (< 20 m)'),
        Patch(facecolor=ACCENT,  label='Mid range (20–100 m)'),
        Patch(facecolor=ACCENT3, label='Far field (> 100 m)'),
        plt.Line2D([0],[0], color='#FFD700', lw=1.8, ls='--', label=f'Mean = {mean_d:.1f} m'),
        plt.Line2D([0],[0], color='#FF7B54', lw=1.5, ls=':', label=f'Median = {median_d:.1f} m'),
        plt.Line2D([0],[0], color='#FFFFFF', lw=1.2, ls='-.', alpha=0.5, label='Effective max (100 m)'),
    ]
    ax.legend(handles=handles, facecolor=DARK_BG, edgecolor=GRID_COL,
              labelcolor=TEXT_COL, fontsize=9)

    ax.set_xlabel('Distance from Sensor [m]', color=TEXT_COL)
    ax.set_ylabel('Number of LiDAR Points', color=TEXT_COL)
    ax.set_title('Distribution of LiDAR Detection Distances\n'
                 f'(sampled from all 718 frames  ·  {len(dist):,} points)',
                 color=TEXT_COL, fontsize=12, fontweight='bold')
    ax.set_xlim(0, 130)

    plt.tight_layout()
    save(fig, 'fig_s1_range_distribution.png')


# ── Figure S2: Point density vs distance ──────────────────────────────────────

def fig_s2_density_vs_distance(pts):
    dist = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2 + pts[:, 2]**2)

    # Compute density per 2 m range bin (points / bin_width)
    bin_width = 2.0
    bin_edges = np.arange(0, 105, bin_width)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    counts, _ = np.histogram(dist, bins=bin_edges)

    # Theoretical inverse-square reference (normalised)
    ref_d = bin_centres[bin_centres > 0]
    ref_y = counts[bin_centres > 0].max() / (ref_d / ref_d[ref_d > 0][0]) ** 2

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=DARK_BG)
    styled_ax(ax)

    ax.bar(bin_centres, counts, width=bin_width * 0.85,
           color=ACCENT, edgecolor=DARK_BG, linewidth=0.4,
           alpha=0.8, label='Observed point count')
    ax.plot(ref_d, ref_y, color='#FFD700', lw=2.0, ls='--',
            label='Theoretical inverse-square falloff')

    ax.set_xlabel('Distance from Sensor [m]', color=TEXT_COL)
    ax.set_ylabel('LiDAR Point Count per 2 m Bin', color=TEXT_COL)
    ax.set_title('Point Density vs Distance from Sensor\n'
                 'Observed density decreases with distance — consistent with angular LiDAR geometry',
                 color=TEXT_COL, fontsize=12, fontweight='bold')
    ax.set_xlim(0, 105)
    ax.legend(facecolor=DARK_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL, fontsize=9)

    # Annotate zones
    ax.axvspan(0,   20,  alpha=0.06, color=ACCENT2, label='High density zone')
    ax.axvspan(20, 100,  alpha=0.04, color=ACCENT,  label='Medium density zone')
    ax.text(8,   counts.max() * 0.92, 'High\ndensity',  color=ACCENT2,
            fontsize=8, ha='center', fontweight='bold')
    ax.text(55,  counts.max() * 0.20, 'Medium density', color=ACCENT,
            fontsize=8, ha='center', fontweight='bold')

    plt.tight_layout()
    save(fig, 'fig_s2_density_vs_distance.png')


# ── Figure S3: Height distribution ────────────────────────────────────────────

def fig_s3_height_distribution(pts):
    z = pts[:, 2]
    z_clipped = z[(z >= -2) & (z <= 6)]

    fig, (ax_hist, ax_kde) = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK_BG)

    # Histogram
    styled_ax(ax_hist)
    counts, bin_edges, patches = ax_hist.hist(z_clipped, bins=120,
                                               color=ACCENT3, edgecolor=DARK_BG,
                                               linewidth=0.3, density=True)

    # Colour by height zone
    for patch, left in zip(patches, bin_edges[:-1]):
        if left < 0.15:
            patch.set_facecolor('#555566')   # ground
        elif left < 0.5:
            patch.set_facecolor(ACCENT2)      # low objects (kerbs, feet)
        elif left < 2.2:
            patch.set_facecolor(ACCENT)       # vehicles & pedestrians
        else:
            patch.set_facecolor(ACCENT3)      # tall structures

    ax_hist.axvline(0.0,  color='#FFFFFF', lw=1.2, ls=':',  alpha=0.5, label='Ground (Z=0)')
    ax_hist.axvline(0.25, color='#FFD700', lw=1.2, ls='--', alpha=0.7, label='Ground threshold (0.25 m)')
    ax_hist.axvline(2.2,  color=ACCENT4,   lw=1.2, ls='-.', alpha=0.7, label='Pedestrian max height (2.2 m)')

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor='#555566', label='Ground plane (Z < 0.15 m)'),
        Patch(facecolor=ACCENT2,   label='Low objects (0.15–0.5 m)'),
        Patch(facecolor=ACCENT,    label='Vehicles & pedestrians (0.5–2.2 m)'),
        Patch(facecolor=ACCENT3,   label='Tall structures (> 2.2 m)'),
        plt.Line2D([0],[0], color='#FFD700', lw=1.2, ls='--', label='Ground threshold (0.25 m)'),
        plt.Line2D([0],[0], color=ACCENT4,   lw=1.2, ls='-.', label='Pedestrian max height (2.2 m)'),
    ]
    ax_hist.legend(handles=handles, facecolor=DARK_BG, edgecolor=GRID_COL,
                   labelcolor=TEXT_COL, fontsize=8)
    ax_hist.set_xlabel('Height Z [m]', color=TEXT_COL)
    ax_hist.set_ylabel('Normalised Density', color=TEXT_COL)
    ax_hist.set_title('Height Distribution (Histogram)', color=TEXT_COL, fontweight='bold')
    ax_hist.set_xlim(-1.5, 5.5)

    # Cumulative distribution (right panel)
    styled_ax(ax_kde)
    z_sorted = np.sort(z_clipped)
    cdf = np.arange(1, len(z_sorted) + 1) / len(z_sorted)
    ax_kde.plot(z_sorted, cdf, color=ACCENT, lw=2.0, label='Empirical CDF')
    ax_kde.axvline(0.25, color='#FFD700', lw=1.4, ls='--', label='Ground threshold (0.25 m)')
    ax_kde.axhline(0.5,  color=GRID_COL,  lw=0.8, ls=':', alpha=0.6)

    ground_frac = float(np.mean(z_clipped < 0.25))
    ax_kde.text(0.30, ground_frac - 0.04,
                f'{ground_frac:.0%} of points\nbelow ground threshold',
                color='#FFD700', fontsize=8)

    ax_kde.set_xlabel('Height Z [m]', color=TEXT_COL)
    ax_kde.set_ylabel('Cumulative Fraction', color=TEXT_COL)
    ax_kde.set_title('Height Cumulative Distribution', color=TEXT_COL, fontweight='bold')
    ax_kde.set_xlim(-1.5, 5.5)
    ax_kde.set_ylim(0, 1.05)
    ax_kde.legend(facecolor=DARK_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL, fontsize=9)

    fig.suptitle('Vertical Distribution of LiDAR Points — Height Sanity Check',
                 color=TEXT_COL, fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    save(fig, 'fig_s3_height_distribution.png')


# ── Figure S4: Intensity distribution ─────────────────────────────────────────

def fig_s4_intensity_distribution(pts):
    intensity = pts[:, 3]
    # Remove obvious fill values
    intensity = intensity[(intensity > 0) & (intensity < 1e6)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK_BG)

    # Panel 1 — histogram
    styled_ax(ax1)
    counts, bin_edges, patches = ax1.hist(intensity, bins=100,
                                           color=ACCENT4, edgecolor=DARK_BG,
                                           linewidth=0.3, density=True)

    mean_i   = float(np.mean(intensity))
    median_i = float(np.median(intensity))
    std_i    = float(np.std(intensity))

    ax1.axvline(mean_i,   color='#FFD700', lw=1.8, ls='--',
                label=f'Mean = {mean_i:.1f}')
    ax1.axvline(median_i, color='#FF7B54', lw=1.5, ls=':',
                label=f'Median = {median_i:.1f}')
    ax1.axvspan(mean_i - 2*std_i, mean_i + 2*std_i,
                alpha=0.08, color='#FFD700', label='Mean ± 2σ range')

    ax1.set_xlabel('Return Intensity [a.u.]', color=TEXT_COL)
    ax1.set_ylabel('Normalised Density', color=TEXT_COL)
    ax1.set_title('Intensity Value Distribution', color=TEXT_COL, fontweight='bold')
    ax1.legend(facecolor=DARK_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL, fontsize=9)

    # Panel 2 — intensity vs distance scatter (sampled)
    styled_ax(ax2)
    dist = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2 + pts[:, 2]**2)
    mask = (dist > 0) & (dist < 100) & (intensity > 0) & (intensity < 1e6)
    d_s  = dist[mask]
    i_s  = intensity[mask]

    # Bin-mean plot instead of scatter (too many points)
    bin_edges = np.linspace(0, 100, 51)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_means, bin_stds = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask_bin = (d_s >= lo) & (d_s < hi)
        if mask_bin.sum() > 10:
            bin_means.append(np.mean(i_s[mask_bin]))
            bin_stds.append(np.std(i_s[mask_bin]))
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)

    bm = np.array(bin_means)
    bs = np.array(bin_stds)
    valid = ~np.isnan(bm)

    ax2.fill_between(bin_centres[valid], bm[valid] - bs[valid], bm[valid] + bs[valid],
                     alpha=0.2, color=ACCENT4)
    ax2.plot(bin_centres[valid], bm[valid], color=ACCENT4, lw=2.0,
             label='Mean intensity per 2 m bin')

    ax2.set_xlabel('Distance from Sensor [m]', color=TEXT_COL)
    ax2.set_ylabel('Return Intensity [a.u.]', color=TEXT_COL)
    ax2.set_title('Mean Intensity vs Distance\n(shaded band = ±1σ)',
                  color=TEXT_COL, fontweight='bold')
    ax2.set_xlim(0, 100)
    ax2.legend(facecolor=DARK_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL, fontsize=9)

    fig.suptitle('Distribution of LiDAR Return Intensity Values — Sensor Consistency Check',
                 color=TEXT_COL, fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    save(fig, 'fig_s4_intensity_distribution.png')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('Sanity Check Figure Generator — DLMDSEAAD02')
    print('=' * 60)

    csv_files = collect_csv_files(DATA_DIR)
    if not csv_files:
        print(f'ERROR: No CSV files found in {DATA_DIR}')
        sys.exit(1)
    print(f'Found {len(csv_files)} frames')

    pts = load_points_sample(csv_files, every_n=5)
    print(f'  Total points loaded: {len(pts):,}')

    print('\nGenerating sanity check figures...')
    fig_s1_range_distribution(pts)
    fig_s2_density_vs_distance(pts)
    fig_s3_height_distribution(pts)
    fig_s4_intensity_distribution(pts)

    print('\nAll 4 sanity check figures saved:')
    for name in ['fig_s1_range_distribution.png',
                 'fig_s2_density_vs_distance.png',
                 'fig_s3_height_distribution.png',
                 'fig_s4_intensity_distribution.png']:
        exists = '✓' if os.path.exists(os.path.join(OUT_DIR, name)) else '✗'
        print(f'  {exists}  {name}')


if __name__ == '__main__':
    main()
