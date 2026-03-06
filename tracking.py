"""
Multi-Object Tracking Module — Kalman Filter
=============================================
DLMDSEAAD02 -- Localization, Motion Planning and Sensor Fusion

Kalman-filter-based multi-object tracking (MOT) with constant-velocity model.

Additional feature (professor feedback fix):
  After a track has been confirmed for STATIC_RECLASS_FRAMES or more frames,
  its estimated speed is tested.  If speed < STATIC_SPEED_THRESHOLD (m/s) the
  track is reclassified to STATIC_STRUCTURE.  This catches walls, fences, and
  buildings that may have passed geometric classification but do not move.

State vector : [x, y, vx, vy, width, length]
Measurement  : [x, y, width, length]

Author: Kalpana Abhiseka Maddi
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Optional
from dataclasses import dataclass
from collections import deque

from classification import ObjectFeatures


# ── Constants ─────────────────────────────────────────────────────────────────
# Tracks active for this many frames with speed below threshold → STATIC_STRUCTURE
STATIC_RECLASS_FRAMES    = 8      # frames
STATIC_SPEED_THRESHOLD   = 0.40   # m/s   (sensor fps * threshold = real-world speed)


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class TrackState:
    """Complete state of a tracked object, used downstream for visualisation."""
    track_id:          int
    position:          np.ndarray    # [x, y, z]
    velocity:          np.ndarray    # [vx, vy]
    dimensions:        np.ndarray    # [length, width, height]
    classification:    str
    confidence:        float
    age:               int
    hits:              int
    time_since_update: int
    history:           List[np.ndarray]


# ── Single-object Kalman tracker ──────────────────────────────────────────────

class KalmanObjectTracker:
    """Kalman Filter tracker for a single object (constant-velocity model)."""

    _next_id: int = 0

    def __init__(self,
                 initial_position: np.ndarray,
                 initial_size:     np.ndarray,
                 classification:   str   = 'UNKNOWN',
                 dt:               float = 0.1):
        self.track_id      = KalmanObjectTracker._next_id
        KalmanObjectTracker._next_id += 1

        self.classification = classification
        self.dt             = dt
        self.age            = 0
        self.hits           = 1
        self.time_since_update = 0
        self.confidence     = 0.5
        self.history        = deque(maxlen=50)
        self.history.append(initial_position.copy())

        # Speed history for static reclassification
        self._speed_history: deque = deque(maxlen=STATIC_RECLASS_FRAMES)

        # 6-D Kalman Filter  state = [x, y, vx, vy, w, l]
        self.kf = KalmanFilter(dim_x=6, dim_z=4)

        self.kf.x = np.array([
            initial_position[0], initial_position[1],
            0.0, 0.0,
            initial_size[0], initial_size[1]
        ]).reshape(-1, 1)

        # State-transition matrix F (constant velocity)
        self.kf.F = np.array([
            [1, 0, dt, 0,  0, 0],
            [0, 1, 0,  dt, 0, 0],
            [0, 0, 1,  0,  0, 0],
            [0, 0, 0,  1,  0, 0],
            [0, 0, 0,  0,  1, 0],
            [0, 0, 0,  0,  0, 1],
        ])

        # Measurement matrix H  (we observe x, y, w, l)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])

        # Measurement noise R
        self.kf.R = np.diag([0.10, 0.10, 0.05, 0.05])

        # Process noise Q
        self.kf.Q = np.diag([0.01, 0.01, 0.10, 0.10, 0.001, 0.001])

        # Initial covariance P  (high uncertainty on velocity)
        self.kf.P = np.diag([1.0, 1.0, 10.0, 10.0, 0.5, 0.5])

    # ------------------------------------------------------------------
    def predict(self) -> np.ndarray:
        self.kf.predict()
        self.age               += 1
        self.time_since_update += 1
        return self.kf.x.flatten()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        self.kf.update(measurement.reshape(-1, 1))
        self.hits              += 1
        self.time_since_update  = 0
        pos = self.get_position()
        self.history.append(pos.copy())

        # Record speed for static reclassification
        speed = self.get_speed()
        self._speed_history.append(speed)

        # Reclassify static objects ------------------------------------------
        if (len(self._speed_history) >= STATIC_RECLASS_FRAMES and
                self.hits >= STATIC_RECLASS_FRAMES):
            avg_speed = float(np.mean(list(self._speed_history)))
            if (avg_speed < STATIC_SPEED_THRESHOLD and
                    self.classification not in ('STATIC_STRUCTURE', 'PEDESTRIAN')):
                self.classification = 'STATIC_STRUCTURE'
                self.confidence     = 0.80

        return self.kf.x.flatten()

    # ------------------------------------------------------------------
    def get_position(self)   -> np.ndarray: return np.array([self.kf.x[0, 0], self.kf.x[1, 0]])
    def get_velocity(self)   -> np.ndarray: return np.array([self.kf.x[2, 0], self.kf.x[3, 0]])
    def get_speed(self)      -> float:
        v = self.get_velocity();  return float(np.sqrt(v[0]**2 + v[1]**2))
    def get_dimensions(self) -> np.ndarray: return np.array([self.kf.x[4, 0], self.kf.x[5, 0]])

    def get_track_state(self, height: float = 1.5) -> TrackState:
        pos  = self.get_position()
        vel  = self.get_velocity()
        dims = self.get_dimensions()
        return TrackState(
            track_id          = self.track_id,
            position          = np.array([pos[0], pos[1], height / 2]),
            velocity          = vel,
            dimensions        = np.array([dims[1], dims[0], height]),  # [L, W, H]
            classification    = self.classification,
            confidence        = self.confidence,
            age               = self.age,
            hits              = self.hits,
            time_since_update = self.time_since_update,
            history           = [np.array(h) for h in self.history],
        )

    def is_confirmed(self) -> bool:
        return self.hits >= 3 and self.time_since_update <= 1

    def is_lost(self) -> bool:
        if self.time_since_update > 5:
            return True
        if self.age < 3 and self.time_since_update > 2:
            return True
        return False


# ── Multi-object tracker ──────────────────────────────────────────────────────

class MultiObjectTracker:
    """
    Manages a pool of KalmanObjectTrackers.

    Pipeline per frame:
      1. Predict  -- advance all tracks
      2. Associate -- Hungarian matching on Euclidean centroid distance
      3. Update   -- update matched tracks, reclassify if static
      4. Create   -- initialise new tracks for unmatched detections
      5. Delete   -- remove lost tracks
    """

    def __init__(self,
                 max_age:               int   = 5,
                 min_hits:              int   = 3,
                 association_threshold: float = 5.0,
                 dt:                   float = 0.1):
        self.max_age               = max_age
        self.min_hits              = min_hits
        self.association_threshold = association_threshold
        self.dt                    = dt
        self.trackers:  List[KalmanObjectTracker] = []
        self.frame_count = 0

    # ------------------------------------------------------------------
    def update(self,
               features_list: List[ObjectFeatures],
               verbose:       bool = False) -> List[TrackState]:
        """
        Update tracker with detections from the current frame.

        Args:
            features_list: Detections from classification step
            verbose:       Print per-frame tracking summary

        Returns:
            Confirmed TrackState objects
        """
        self.frame_count += 1

        # 1. Predict
        for t in self.trackers:
            t.predict()

        # 2. Associate
        matched, unmatched_dets, _ = self._associate(features_list)

        # 3. Update matched
        for det_idx, trk_idx in matched:
            f = features_list[det_idx]
            t = self.trackers[trk_idx]
            meas = np.array([f.center[0], f.center[1], f.width, f.length])
            t.update(meas)
            # Only update class if new detection is more specific
            if f.classification not in ('UNKNOWN', 'STATIC_STRUCTURE'):
                t.classification = f.classification
                t.confidence     = f.confidence

        # 4. New tracks for unmatched detections
        for det_idx in unmatched_dets:
            f   = features_list[det_idx]
            new = KalmanObjectTracker(
                initial_position = f.center[:2],
                initial_size     = np.array([f.width, f.length]),
                classification   = f.classification,
                dt               = self.dt,
            )
            new.confidence = f.confidence
            self.trackers.append(new)

        # 5. Remove lost tracks
        self.trackers = [t for t in self.trackers if not t.is_lost()]

        # 6. Return confirmed tracks
        confirmed = []
        for t in self.trackers:
            if t.is_confirmed():
                h = 1.7 if t.classification == 'PEDESTRIAN' else 1.5
                confirmed.append(t.get_track_state(h))

        if verbose:
            print(f"  Frame {self.frame_count}: "
                  f"dets={len(features_list)} active={len(self.trackers)} "
                  f"confirmed={len(confirmed)}")

        return confirmed

    # ------------------------------------------------------------------
    def _associate(self, features_list: List[ObjectFeatures]) -> Tuple[
            List[Tuple[int, int]], List[int], List[int]]:

        if not self.trackers:
            return [], list(range(len(features_list))), []
        if not features_list:
            return [], [], list(range(len(self.trackers)))

        # Euclidean centroid distance cost matrix
        cost = np.zeros((len(features_list), len(self.trackers)))
        for di, f in enumerate(features_list):
            for ti, t in enumerate(self.trackers):
                cost[di, ti] = np.linalg.norm(f.center[:2] - t.get_position())

        det_idx, trk_idx = linear_sum_assignment(cost)

        matched, unmatched_dets = [], list(range(len(features_list)))
        unmatched_trks          = list(range(len(self.trackers)))

        for di, ti in zip(det_idx, trk_idx):
            if cost[di, ti] < self.association_threshold:
                matched.append((di, ti))
                unmatched_dets.remove(di)
                unmatched_trks.remove(ti)

        return matched, unmatched_dets, unmatched_trks

    # ------------------------------------------------------------------
    def get_confirmed_tracks(self) -> List[TrackState]:
        tracks = []
        for t in self.trackers:
            if t.is_confirmed():
                h = 1.7 if t.classification == 'PEDESTRIAN' else 1.5
                tracks.append(t.get_track_state(h))
        return tracks

    def reset(self) -> None:
        self.trackers    = []
        self.frame_count = 0
        KalmanObjectTracker._next_id = 0
