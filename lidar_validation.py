"""
LiDAR Data Validation and Sanity Check Module
==============================================
Validates sensor data against Blickfeld Cube 1 specifications.

Validates:
1. Range verification (5-250m specification)
2. Point density analysis vs distance
3. Noise and outlier detection
4. Data quality metrics

Course: Localization, Motion Planning and Sensor Fusion
Assignment: LiDAR-based Object Detection and Tracking Pipeline
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import os
import glob


@dataclass
class LiDARValidationReport:
    """Complete validation report for LiDAR data."""
    # Basic statistics
    total_points: int
    total_frames: int
    points_per_frame_mean: float
    points_per_frame_std: float

    # Range analysis
    min_range: float
    max_range: float
    mean_range: float
    range_within_spec: float  # Percentage within 5-250m

    # Point density
    density_near: float   # Points/m^2 at 0-20m
    density_mid: float    # Points/m^2 at 20-50m
    density_far: float    # Points/m^2 at 50-100m
    density_very_far: float  # Points/m^2 at 100-250m

    # Noise analysis
    outlier_percentage: float
    ground_plane_fit_error: float
    intensity_mean: float
    intensity_std: float

    # Quality metrics
    data_quality_score: float  # 0-100
    specification_compliance: bool


class LiDARValidator:
    """
    Validates LiDAR data against sensor specifications.

    Blickfeld Cube 1 Specifications:
    - Range: 5m to 250m
    - Field of View: 70° x 30°
    - Points per second: Up to 100,000
    - Angular resolution: 0.18°
    """

    # Blickfeld Cube 1 specifications
    SPEC_MIN_RANGE = 5.0    # meters
    SPEC_MAX_RANGE = 250.0  # meters
    SPEC_FOV_H = 70.0       # degrees horizontal
    SPEC_FOV_V = 30.0       # degrees vertical

    def __init__(self):
        """Initialize validator."""
        self.validation_results = {}

    def validate_frame(self, points: np.ndarray,
                      verbose: bool = True) -> Dict:
        """
        Validate a single LiDAR frame.

        Args:
            points: Point cloud (N, 4) [X, Y, Z, INTENSITY]
            verbose: Print progress

        Returns:
            Dictionary with validation metrics
        """
        if verbose:
            print("\n" + "="*60)
            print("LIDAR DATA VALIDATION - SINGLE FRAME")
            print("="*60)

        results = {}

        # 1. Basic statistics
        results['num_points'] = len(points)
        results['has_intensity'] = points.shape[1] >= 4

        # 2. Range analysis
        distances = np.linalg.norm(points[:, :3], axis=1)
        results['range_min'] = float(np.min(distances))
        results['range_max'] = float(np.max(distances))
        results['range_mean'] = float(np.mean(distances))
        results['range_std'] = float(np.std(distances))

        # Range specification compliance
        in_spec = (distances >= self.SPEC_MIN_RANGE) & (distances <= self.SPEC_MAX_RANGE)
        results['range_compliance'] = float(np.sum(in_spec) / len(distances) * 100)

        below_min = np.sum(distances < self.SPEC_MIN_RANGE)
        above_max = np.sum(distances > self.SPEC_MAX_RANGE)
        results['points_below_min_range'] = int(below_min)
        results['points_above_max_range'] = int(above_max)

        # 3. Point density vs distance
        density = self._compute_density_profile(points, distances)
        results['density_profile'] = density

        # 4. Intensity analysis
        if points.shape[1] >= 4:
            intensity = points[:, 3]
            results['intensity_min'] = float(np.min(intensity))
            results['intensity_max'] = float(np.max(intensity))
            results['intensity_mean'] = float(np.mean(intensity))
            results['intensity_std'] = float(np.std(intensity))
        else:
            results['intensity_min'] = 0.0
            results['intensity_max'] = 0.0
            results['intensity_mean'] = 0.0
            results['intensity_std'] = 0.0

        # 5. Noise estimation
        results['noise_estimate'] = self._estimate_noise(points)

        # 6. Height distribution
        z_vals = points[:, 2]
        results['z_min'] = float(np.min(z_vals))
        results['z_max'] = float(np.max(z_vals))
        results['z_mean'] = float(np.mean(z_vals))

        # 7. Quality score
        results['quality_score'] = self._compute_quality_score(results)

        if verbose:
            self._print_validation_results(results)

        return results

    def validate_dataset(self, data_dir: str,
                        max_frames: int = 100,
                        verbose: bool = True) -> LiDARValidationReport:
        """
        Validate entire LiDAR dataset.

        Args:
            data_dir: Directory containing CSV files
            max_frames: Maximum frames to analyze
            verbose: Print progress

        Returns:
            Complete validation report
        """
        if verbose:
            print("\n" + "="*60)
            print("LIDAR DATASET VALIDATION")
            print("="*60)
            print(f"\nAnalyzing data from: {data_dir}")

        # Find CSV files
        csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        num_frames = min(len(csv_files), max_frames)

        if verbose:
            print(f"Found {len(csv_files)} CSV files, analyzing {num_frames} frames")

        if num_frames == 0:
            raise ValueError(f"No CSV files found in {data_dir}. Please provide a valid data directory containing LiDAR point cloud CSV files.")

        # Collect statistics across frames
        all_ranges = []
        all_intensities = []
        all_z_values = []
        points_per_frame = []
        density_profiles = []

        for i, csv_path in enumerate(csv_files[:num_frames]):
            try:
                df = pd.read_csv(csv_path, sep=';')
                points = df[['X', 'Y', 'Z']].values.astype(np.float32)

                if 'INTENSITY' in df.columns:
                    intensity = df['INTENSITY'].values.astype(np.float32)
                    points = np.column_stack([points, intensity / 255.0])
                else:
                    points = np.column_stack([points, np.ones(len(points)) * 0.5])

                # Collect stats
                distances = np.linalg.norm(points[:, :3], axis=1)
                all_ranges.extend(distances.tolist())
                all_intensities.extend(points[:, 3].tolist())
                all_z_values.extend(points[:, 2].tolist())
                points_per_frame.append(len(points))

                density = self._compute_density_profile(points, distances)
                density_profiles.append(density)

                if verbose and (i + 1) % 20 == 0:
                    print(f"  Processed {i + 1}/{num_frames} frames...")

            except Exception as e:
                if verbose:
                    print(f"  Error loading {csv_path}: {e}")
                continue

        # Check if any data was loaded
        if len(all_ranges) == 0 or len(density_profiles) == 0:
            raise ValueError(f"No valid point cloud data could be loaded from {data_dir}. Please check that the CSV files contain valid X, Y, Z columns.")

        # Aggregate statistics
        all_ranges = np.array(all_ranges)
        all_intensities = np.array(all_intensities)
        all_z_values = np.array(all_z_values)
        points_per_frame = np.array(points_per_frame)

        # Range compliance
        in_spec = (all_ranges >= self.SPEC_MIN_RANGE) & \
                  (all_ranges <= self.SPEC_MAX_RANGE)
        range_compliance = np.sum(in_spec) / len(all_ranges) * 100

        # Average density profile
        avg_density = {}
        for key in density_profiles[0].keys():
            avg_density[key] = np.mean([d[key] for d in density_profiles])

        # Noise estimation
        outlier_pct = self._estimate_dataset_noise(all_ranges, all_z_values)

        # Quality score
        quality_score = self._compute_dataset_quality(
            range_compliance, avg_density, outlier_pct, points_per_frame
        )

        # Create report
        report = LiDARValidationReport(
            total_points=len(all_ranges),
            total_frames=num_frames,
            points_per_frame_mean=float(np.mean(points_per_frame)),
            points_per_frame_std=float(np.std(points_per_frame)),
            min_range=float(np.min(all_ranges)),
            max_range=float(np.max(all_ranges)),
            mean_range=float(np.mean(all_ranges)),
            range_within_spec=float(range_compliance),
            density_near=avg_density['0-20m'],
            density_mid=avg_density['20-50m'],
            density_far=avg_density['50-100m'],
            density_very_far=avg_density['100-250m'],
            outlier_percentage=outlier_pct,
            ground_plane_fit_error=0.1,  # Placeholder
            intensity_mean=float(np.mean(all_intensities)),
            intensity_std=float(np.std(all_intensities)),
            data_quality_score=quality_score,
            specification_compliance=range_compliance > 95.0
        )

        if verbose:
            self._print_dataset_report(report)

        return report

    def _compute_density_profile(self, points: np.ndarray,
                                 distances: np.ndarray) -> Dict[str, float]:
        """Compute point density at different distance ranges."""
        ranges = [
            ('0-20m', 0, 20),
            ('20-50m', 20, 50),
            ('50-100m', 50, 100),
            ('100-250m', 100, 250)
        ]

        density = {}
        for name, r_min, r_max in ranges:
            mask = (distances >= r_min) & (distances < r_max)
            count = np.sum(mask)

            # Approximate area of spherical shell section
            # Area = 2 * pi * r^2 * (1 - cos(theta/2)) for FOV angle theta
            theta_h = np.radians(self.SPEC_FOV_H / 2)
            theta_v = np.radians(self.SPEC_FOV_V / 2)
            r_mid = (r_min + r_max) / 2

            # Simplified area calculation
            area = theta_h * theta_v * r_mid**2
            density[name] = count / max(area, 1.0)

        return density

    def _estimate_noise(self, points: np.ndarray) -> Dict:
        """Estimate noise level in point cloud."""
        # Use local variance as noise estimate
        from scipy.spatial import cKDTree

        if len(points) < 20:
            return {'local_variance': 0.0, 'outlier_fraction': 0.0}

        tree = cKDTree(points[:, :3])

        # Sample points for noise estimation
        sample_size = min(1000, len(points))
        sample_idx = np.random.choice(len(points), sample_size, replace=False)

        variances = []
        for idx in sample_idx:
            dists, _ = tree.query(points[idx, :3], k=10)
            variances.append(np.var(dists[1:]))  # Exclude self

        mean_variance = np.mean(variances)

        # Outlier detection using distance statistics
        all_distances = np.linalg.norm(points[:, :3], axis=1)
        z_scores = np.abs((all_distances - np.mean(all_distances)) / np.std(all_distances))
        outlier_fraction = np.sum(z_scores > 3) / len(points)

        return {
            'local_variance': float(mean_variance),
            'outlier_fraction': float(outlier_fraction)
        }

    def _estimate_dataset_noise(self, ranges: np.ndarray,
                                z_values: np.ndarray) -> float:
        """Estimate outlier percentage in dataset."""
        # Points with unusual z-values
        z_mean, z_std = np.mean(z_values), np.std(z_values)
        z_outliers = np.abs(z_values - z_mean) > 3 * z_std

        # Points with unusual ranges
        r_mean, r_std = np.mean(ranges), np.std(ranges)
        r_outliers = np.abs(ranges - r_mean) > 3 * r_std

        outlier_pct = (np.sum(z_outliers) + np.sum(r_outliers)) / (2 * len(ranges)) * 100

        return float(outlier_pct)

    def _compute_quality_score(self, results: Dict) -> float:
        """Compute quality score for single frame."""
        score = 100.0

        # Penalize for out-of-spec points
        out_of_spec = 100 - results['range_compliance']
        score -= out_of_spec * 0.5

        # Penalize for low point count
        if results['num_points'] < 5000:
            score -= 10

        # Penalize for high noise
        noise = results.get('noise_estimate', {})
        if noise.get('outlier_fraction', 0) > 0.05:
            score -= 10

        return max(0, min(100, score))

    def _compute_dataset_quality(self, range_compliance: float,
                                 density: Dict, outlier_pct: float,
                                 points_per_frame: np.ndarray) -> float:
        """Compute overall dataset quality score."""
        score = 100.0

        # Range compliance (weight: 30%)
        score -= (100 - range_compliance) * 0.3

        # Point density consistency (weight: 20%)
        density_score = min(density['0-20m'] / 100, 1.0) * 20
        score = score - 20 + density_score

        # Outlier percentage (weight: 20%)
        score -= outlier_pct * 2

        # Point count consistency (weight: 30%)
        cv = np.std(points_per_frame) / np.mean(points_per_frame)
        if cv > 0.3:
            score -= 15

        return max(0, min(100, score))

    def _print_validation_results(self, results: Dict):
        """Print single frame validation results."""
        print(f"\n--- Basic Statistics ---")
        print(f"  Points: {results['num_points']:,}")
        print(f"  Has intensity: {results['has_intensity']}")

        print(f"\n--- Range Analysis ---")
        print(f"  Specification: {self.SPEC_MIN_RANGE}m - {self.SPEC_MAX_RANGE}m")
        print(f"  Measured range: {results['range_min']:.2f}m - {results['range_max']:.2f}m")
        print(f"  Mean range: {results['range_mean']:.2f}m (+/- {results['range_std']:.2f}m)")
        print(f"  Within specification: {results['range_compliance']:.1f}%")
        print(f"  Points below min: {results['points_below_min_range']}")
        print(f"  Points above max: {results['points_above_max_range']}")

        print(f"\n--- Point Density Profile ---")
        for key, value in results['density_profile'].items():
            print(f"  {key}: {value:.2f} pts/m^2")

        print(f"\n--- Intensity Analysis ---")
        print(f"  Range: {results['intensity_min']:.3f} - {results['intensity_max']:.3f}")
        print(f"  Mean: {results['intensity_mean']:.3f} (+/- {results['intensity_std']:.3f})")

        print(f"\n--- Quality Score ---")
        print(f"  Score: {results['quality_score']:.1f}/100")

        print("="*60)

    def _print_dataset_report(self, report: LiDARValidationReport):
        """Print dataset validation report."""
        print(f"\n{'='*60}")
        print("DATASET VALIDATION REPORT")
        print(f"{'='*60}")

        print(f"\n--- Dataset Overview ---")
        print(f"  Total frames: {report.total_frames}")
        print(f"  Total points: {report.total_points:,}")
        print(f"  Points/frame: {report.points_per_frame_mean:.0f} (+/- {report.points_per_frame_std:.0f})")

        print(f"\n--- Range Validation ---")
        print(f"  Specification: {self.SPEC_MIN_RANGE}m - {self.SPEC_MAX_RANGE}m")
        print(f"  Measured: {report.min_range:.2f}m - {report.max_range:.2f}m")
        print(f"  Mean range: {report.mean_range:.2f}m")
        print(f"  Compliance: {report.range_within_spec:.1f}%")
        status = "PASS" if report.range_within_spec > 95 else "PARTIAL"
        print(f"  Status: {status}")

        print(f"\n--- Point Density vs Distance ---")
        print(f"  0-20m (near):     {report.density_near:.2f} pts/m^2")
        print(f"  20-50m (mid):     {report.density_mid:.2f} pts/m^2")
        print(f"  50-100m (far):    {report.density_far:.2f} pts/m^2")
        print(f"  100-250m (v.far): {report.density_very_far:.2f} pts/m^2")
        print(f"  Note: Density decreases with r^2 (inverse square law)")

        print(f"\n--- Noise Analysis ---")
        print(f"  Estimated outliers: {report.outlier_percentage:.2f}%")
        print(f"  Intensity mean: {report.intensity_mean:.3f}")
        print(f"  Intensity std: {report.intensity_std:.3f}")

        print(f"\n--- Quality Assessment ---")
        print(f"  Quality score: {report.data_quality_score:.1f}/100")
        print(f"  Specification compliance: {'PASS' if report.specification_compliance else 'FAIL'}")

        print(f"\n{'='*60}")

    def generate_validation_plots(self, data_dir: str,
                                  output_dir: str = "validation_plots",
                                  max_frames: int = 50) -> List[str]:
        """
        Generate validation plots for the report.

        Args:
            data_dir: Directory with CSV files
            output_dir: Output directory for plots
            max_frames: Maximum frames to analyze

        Returns:
            List of saved plot paths
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_plots = []

        print("\nGenerating validation plots...")

        # Collect data
        csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        num_frames = min(len(csv_files), max_frames)

        all_ranges = []
        all_intensities = []
        all_z = []

        for csv_path in csv_files[:num_frames]:
            try:
                df = pd.read_csv(csv_path, sep=';')
                points = df[['X', 'Y', 'Z']].values
                distances = np.linalg.norm(points, axis=1)
                all_ranges.extend(distances.tolist())
                all_z.extend(points[:, 2].tolist())

                if 'INTENSITY' in df.columns:
                    all_intensities.extend(df['INTENSITY'].values.tolist())
            except:
                continue

        all_ranges = np.array(all_ranges)
        all_z = np.array(all_z)
        all_intensities = np.array(all_intensities) if all_intensities else None

        # Plot 1: Range Distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(all_ranges, bins=100, color='steelblue', alpha=0.7, edgecolor='white')
        ax.axvline(x=5.0, color='red', linestyle='--', linewidth=2, label='Min spec (5m)')
        ax.axvline(x=250.0, color='red', linestyle='--', linewidth=2, label='Max spec (250m)')
        ax.set_xlabel('Distance from Sensor (m)', fontsize=12)
        ax.set_ylabel('Point Count', fontsize=12)
        ax.set_title('LiDAR Range Distribution vs Specification', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        path = os.path.join(output_dir, 'range_distribution.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        saved_plots.append(path)
        print(f"  Saved: {path}")

        # Plot 2: Point Density vs Distance
        fig, ax = plt.subplots(figsize=(10, 6))
        bins = np.arange(0, 260, 10)
        counts, _ = np.histogram(all_ranges, bins=bins)

        # Normalize by shell area
        bin_centers = (bins[:-1] + bins[1:]) / 2
        shell_areas = 4 * np.pi * bin_centers**2 * 10 / 360  # Approximate FOV fraction
        density = counts / np.maximum(shell_areas, 1)

        ax.bar(bin_centers, density, width=8, color='steelblue', alpha=0.7, edgecolor='white')
        ax.set_xlabel('Distance (m)', fontsize=12)
        ax.set_ylabel('Relative Point Density', fontsize=12)
        ax.set_title('Point Density vs Distance (Expected: 1/r² decay)', fontsize=14)

        # Overlay theoretical 1/r² curve
        r_theory = np.linspace(10, 250, 100)
        theory = density[2] * (bin_centers[2] / r_theory)**2  # Normalize to observed
        ax.plot(r_theory, theory, 'r--', linewidth=2, label='Theoretical 1/r² decay')
        ax.legend()
        ax.grid(True, alpha=0.3)

        path = os.path.join(output_dir, 'density_vs_distance.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        saved_plots.append(path)
        print(f"  Saved: {path}")

        # Plot 3: Height Distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(all_z, bins=100, color='forestgreen', alpha=0.7, edgecolor='white')
        ax.axvline(x=0, color='brown', linestyle='--', linewidth=2, label='Ground level')
        ax.set_xlabel('Height Z (m)', fontsize=12)
        ax.set_ylabel('Point Count', fontsize=12)
        ax.set_title('Height Distribution (Ground Plane Detection)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        path = os.path.join(output_dir, 'height_distribution.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        saved_plots.append(path)
        print(f"  Saved: {path}")

        # Plot 4: Intensity Distribution (if available)
        if all_intensities is not None and len(all_intensities) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(all_intensities, bins=100, color='orange', alpha=0.7, edgecolor='white')
            ax.set_xlabel('Intensity', fontsize=12)
            ax.set_ylabel('Point Count', fontsize=12)
            ax.set_title('LiDAR Intensity Distribution', fontsize=14)
            ax.grid(True, alpha=0.3)

            path = os.path.join(output_dir, 'intensity_distribution.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            saved_plots.append(path)
            print(f"  Saved: {path}")

        print(f"\nGenerated {len(saved_plots)} validation plots in {output_dir}/")

        return saved_plots


# Test
if __name__ == "__main__":
    print("="*70)
    print("LIDAR VALIDATION MODULE - TEST")
    print("="*70)

    from data_loader import generate_simulated_frame

    # Test with simulated data
    validator = LiDARValidator()

    points = generate_simulated_frame(15000, seed=42)
    results = validator.validate_frame(points, verbose=True)

    print("\nValidation module ready for use.")
