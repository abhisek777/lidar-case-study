"""
LiDAR Feature Extraction and Classification Module
==================================================
Autonomous Driving Perception Pipeline - Object Classification

Implements rule-based geometric classification into:
  VEHICLE         -- cars / vans / trucks
  PEDESTRIAN      -- people on foot
  STATIC_STRUCTURE -- walls, buildings, infrastructure (too large/flat to be dynamic)
  UNKNOWN         -- objects that cannot be confidently assigned

Classification thresholds (documented explicitly for the report):
  Vehicle:
    length  : 2.0 -- 8.0  m
    width   : 1.3 -- 3.0  m
    height  : 1.0 -- 3.5  m
    footprint (L*W) <= 18 m²  (filter out walls that fit L/W ranges)
    height  <= 4.0 m          (buildings are taller)
    aspect  L/W   <= 5.0      (walls have very high L/W)

  Pedestrian:
    length  : 0.2 -- 1.2  m
    width   : 0.2 -- 1.2  m
    height  : 1.2 -- 2.2  m
    H/W     >= 1.5           (taller than wide)

  Static Structure (explicitly labelled, not silently dropped):
    footprint > 18 m²  OR
    height    > 4.0  m  OR
    L/W       > 6.0  (wall-like elongated shape)

Course: DLMDSEAAD02 -- Localization, Motion Planning and Sensor Fusion
Author: Kalpana Abhiseka Maddi
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class BoundingBox3D:
    """3-D axis-aligned bounding box (AABB)."""
    min_point: np.ndarray   # [x_min, y_min, z_min]
    max_point: np.ndarray   # [x_max, y_max, z_max]
    center:    np.ndarray   # [x_c,   y_c,   z_c  ]
    length:    float        # X extent (forward / back)
    width:     float        # Y extent (left  / right)
    height:    float        # Z extent (up    / down)
    volume:    float        # L * W * H


@dataclass
class ObjectFeatures:
    """All geometric and statistical features for one detected cluster."""
    cluster_id:   int
    num_points:   int

    bounding_box: BoundingBox3D
    center:       np.ndarray
    length:       float
    width:        float
    height:       float
    volume:       float

    distance_from_sensor: float
    point_density:        float   # pts / m³

    aspect_ratio_lw:  float   # length / width
    aspect_ratio_lh:  float   # length / height
    aspect_ratio_hw:  float   # height / width

    mean_intensity:   float
    std_intensity:    float
    min_z:            float
    max_z:            float

    classification: Optional[str] = None
    confidence:     float          = 0.0


# ── Feature extractor ─────────────────────────────────────────────────────────

class FeatureExtractor:
    """Extracts 3-D geometric and statistical features from point cloud clusters."""

    def extract_features(self,
                         points:         np.ndarray,
                         cluster_labels: np.ndarray,
                         verbose:        bool = True) -> List[ObjectFeatures]:
        """
        Extract features from all clusters.

        Args:
            points:         Pre-processed cloud (N, 4) [X, Y, Z, INTENSITY]
            cluster_labels: Per-point cluster id; -1 == noise
            verbose:        Print summary table

        Returns:
            List of ObjectFeatures, one per cluster
        """
        unique = np.unique(cluster_labels)
        cluster_ids = unique[unique != -1]

        if verbose:
            print(f"\n[Feature Extraction] {len(cluster_ids)} clusters found")

        features_list = []
        for cid in cluster_ids:
            mask    = cluster_labels == cid
            pts_c   = points[mask]
            feat    = self._extract_single(cid, pts_c)
            features_list.append(feat)

        return features_list

    # ------------------------------------------------------------------
    def _extract_single(self, cluster_id: int,
                        points: np.ndarray) -> ObjectFeatures:
        xyz       = points[:, :3]
        intensity = points[:, 3] if points.shape[1] >= 4 else np.ones(len(points)) * 0.5

        min_pt  = np.min(xyz, axis=0)
        max_pt  = np.max(xyz, axis=0)
        center  = (min_pt + max_pt) / 2.0
        dims    = max_pt - min_pt
        length  = max(float(dims[0]), 0.01)
        width   = max(float(dims[1]), 0.01)
        height  = max(float(dims[2]), 0.01)
        volume  = length * width * height

        bbox = BoundingBox3D(min_point=min_pt, max_point=max_pt, center=center,
                             length=length, width=width, height=height, volume=volume)

        return ObjectFeatures(
            cluster_id=cluster_id,
            num_points=len(points),
            bounding_box=bbox,
            center=center,
            length=length,
            width=width,
            height=height,
            volume=volume,
            distance_from_sensor=float(np.linalg.norm(center)),
            point_density=len(points) / max(volume, 0.001),
            aspect_ratio_lw=length / max(width,  0.01),
            aspect_ratio_lh=length / max(height, 0.01),
            aspect_ratio_hw=height / max(width,  0.01),
            mean_intensity=float(np.mean(intensity)),
            std_intensity=float(np.std(intensity)),
            min_z=float(min_pt[2]),
            max_z=float(max_pt[2]),
        )


# ── Rule-based classifier ─────────────────────────────────────────────────────

class RuleBasedClassifier:
    """
    Conservative rule-based classifier using 3-D bounding box geometry.

    Threshold values are documented explicitly so they can be discussed
    and justified in the written report.
    """

    # ── Vehicle thresholds ───────────────────────────────────────────────────
    VEH_MIN_L      = 2.0    # m
    VEH_MAX_L      = 8.0    # m
    VEH_MIN_W      = 1.3    # m
    VEH_MAX_W      = 3.0    # m
    VEH_MIN_H      = 1.0    # m
    VEH_MAX_H      = 3.5    # m
    VEH_MIN_VOL    = 3.0    # m³
    VEH_MAX_FOOTPRINT = 18.0   # m²  (L*W > 18 is too large for a single vehicle)
    VEH_MAX_ASPECT_LW = 5.0    # L/W > 5 => wall-like, not a car
    VEH_MAX_H_HARD    = 4.0    # m  any object taller than 4 m is infrastructure

    # ── Pedestrian thresholds ────────────────────────────────────────────────
    PED_MIN_L  = 0.2
    PED_MAX_L  = 1.2
    PED_MIN_W  = 0.2
    PED_MAX_W  = 1.2
    PED_MIN_H  = 1.2
    PED_MAX_H  = 2.2
    PED_MAX_VOL = 2.0

    # ── Static structure triggers ────────────────────────────────────────────
    # If ANY of these conditions holds, classify as STATIC_STRUCTURE
    STATIC_MIN_FOOTPRINT = 18.0   # m²
    STATIC_MIN_HEIGHT    = 4.0    # m
    STATIC_MIN_LW_RATIO  = 6.0    # very elongated = wall / fence

    def classify(self,
                 features_list: List[ObjectFeatures],
                 verbose:       bool = True) -> Dict[int, str]:
        """
        Classify all detected objects.

        Returns:
            Dict mapping cluster_id -> class label
        """
        if verbose:
            print(f"\n[Classification] {len(features_list)} objects")
            print(f"  Thresholds: Vehicle L=[{self.VEH_MIN_L},{self.VEH_MAX_L}]m "
                  f"W=[{self.VEH_MIN_W},{self.VEH_MAX_W}]m H=[{self.VEH_MIN_H},{self.VEH_MAX_H}]m "
                  f"footprint<={self.VEH_MAX_FOOTPRINT}m² L/W<={self.VEH_MAX_ASPECT_LW}")
            print(f"  Thresholds: Pedestrian L=[{self.PED_MIN_L},{self.PED_MAX_L}]m "
                  f"W=[{self.PED_MIN_W},{self.PED_MAX_W}]m H=[{self.PED_MIN_H},{self.PED_MAX_H}]m")

        classifications = {}
        counts = {'VEHICLE': 0, 'PEDESTRIAN': 0, 'STATIC_STRUCTURE': 0, 'UNKNOWN': 0}

        for feat in features_list:
            label, conf = self._classify_single(feat)
            feat.classification = label
            feat.confidence     = conf
            classifications[feat.cluster_id] = label
            counts[label] += 1

        if verbose:
            total = len(features_list)
            print(f"  Vehicle={counts['VEHICLE']}  Pedestrian={counts['PEDESTRIAN']}  "
                  f"Static={counts['STATIC_STRUCTURE']}  Unknown={counts['UNKNOWN']}  "
                  f"(total={total})")
            labeled = counts['VEHICLE'] + counts['PEDESTRIAN']
            rate = labeled / max(total, 1)
            print(f"  Labeled rate (V+P / total): {rate:.1%}  "
                  f"[conservative design -- static/unknown suppresses false positives]")

        return classifications

    # ------------------------------------------------------------------
    def _classify_single(self, f: ObjectFeatures) -> Tuple[str, float]:
        """Classify one object. Returns (label, confidence)."""
        L, W, H     = f.length, f.width, f.height
        footprint   = L * W
        lw_ratio    = L / max(W, 0.01)

        # ── Step 1: Static structure pre-filter ──────────────────────────────
        # These are definitive disqualifiers for dynamic objects.
        if (footprint   >= self.STATIC_MIN_FOOTPRINT or
                H       >= self.STATIC_MIN_HEIGHT    or
                lw_ratio >= self.STATIC_MIN_LW_RATIO):
            return 'STATIC_STRUCTURE', 0.85

        # ── Step 2: Vehicle check ────────────────────────────────────────────
        if (self.VEH_MIN_L <= L <= self.VEH_MAX_L and
                self.VEH_MIN_W <= W <= self.VEH_MAX_W and
                self.VEH_MIN_H <= H <= self.VEH_MAX_H):
            conf = self._vehicle_confidence(f)
            return 'VEHICLE', conf

        # Volume-based vehicle (e.g., partial occlusion hides one dimension)
        if f.volume >= self.VEH_MIN_VOL and H >= 1.0 and footprint <= self.VEH_MAX_FOOTPRINT:
            conf = min(0.75, f.volume / 20.0)
            return 'VEHICLE', conf

        # ── Step 3: Pedestrian check ─────────────────────────────────────────
        if (self.PED_MIN_L <= L <= self.PED_MAX_L and
                self.PED_MIN_W <= W <= self.PED_MAX_W and
                self.PED_MIN_H <= H <= self.PED_MAX_H):
            hw = H / max(W, 0.01)
            conf = min(0.90, 0.65 + hw * 0.05) if hw >= 1.5 else 0.60
            return 'PEDESTRIAN', conf

        # Aspect-ratio-based pedestrian
        if f.aspect_ratio_hw >= 2.0 and H >= 1.2 and f.volume <= self.PED_MAX_VOL:
            conf = min(0.80, 0.50 + f.aspect_ratio_hw * 0.08)
            return 'PEDESTRIAN', conf

        # ── Step 4: Catch-all unknown ─────────────────────────────────────────
        return 'UNKNOWN', 0.40

    # ------------------------------------------------------------------
    def _vehicle_confidence(self, f: ObjectFeatures) -> float:
        """Confidence based on closeness to typical car (4.5 × 1.8 × 1.5 m)."""
        ls = max(0.0, 1.0 - abs(f.length - 4.5) / 4.5)
        ws = max(0.0, 1.0 - abs(f.width  - 1.8) / 1.8)
        hs = max(0.0, 1.0 - abs(f.height - 1.5) / 1.5)
        conf = 0.4 * ls + 0.3 * ws + 0.3 * hs
        if f.volume > 10:
            conf = min(0.92, conf + 0.10)
        return max(0.50, min(0.92, conf))


# ── Convenience wrapper ────────────────────────────────────────────────────────

def extract_and_classify(points:         np.ndarray,
                         cluster_labels: np.ndarray,
                         verbose:        bool = True) -> Tuple[List[ObjectFeatures],
                                                               Dict[int, str]]:
    """Extract features and classify in one call."""
    extractor = FeatureExtractor()
    features  = extractor.extract_features(points, cluster_labels, verbose)
    classifier = RuleBasedClassifier()
    labels    = classifier.classify(features, verbose)
    return features, labels
