"""
Enhanced Multi-Object Tracking Module
=====================================
Extended tracking with velocity estimation, motion vectors, and trajectories.

Features:
1. Kalman Filter-based tracking with constant velocity model
2. Velocity estimation from position measurements
3. Motion vector visualization
4. Trajectory history for each track
5. Track state management (tentative, confirmed, lost)

Course: Localization, Motion Planning and Sensor Fusion
Assignment: LiDAR-based Object Detection and Tracking Pipeline
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import time

from classification import ObjectFeatures


@dataclass
class EnhancedTrackState:
    """
    Enhanced state of a tracked object with velocity and trajectory.
    """
    track_id: int
    position: np.ndarray          # [x, y, z]
    velocity: np.ndarray          # [vx, vy] in m/s
    speed: float                  # Magnitude of velocity in m/s
    heading: float                # Direction of motion in radians
    dimensions: np.ndarray        # [length, width, height]
    classification: str
    confidence: float
    age: int                      # Frames since track creation
    hits: int                     # Number of successful detections
    time_since_update: int        # Frames since last measurement
    trajectory: List[np.ndarray]  # Position history
    velocity_history: List[np.ndarray]  # Velocity history
    is_confirmed: bool
    is_moving: bool               # True if speed > threshold


class EnhancedKalmanTracker:
    """
    Enhanced Kalman Filter tracker with velocity estimation.

    State Vector: [x, y, vx, vy, ax, ay, width, length]
    - Position (x, y)
    - Velocity (vx, vy) - estimated from measurements
    - Acceleration (ax, ay) - for smoother predictions
    - Dimensions (width, length)

    Measurement Vector: [x, y, width, length]
    """

    _next_id = 0

    def __init__(self,
                 initial_position: np.ndarray,
                 initial_size: np.ndarray,
                 classification: str = 'UNKNOWN',
                 dt: float = 0.1):
        """
        Initialize enhanced Kalman tracker.

        Args:
            initial_position: Initial [x, y] position
            initial_size: Initial [width, length]
            classification: Object class
            dt: Time step between frames (seconds)
        """
        self.track_id = EnhancedKalmanTracker._next_id
        EnhancedKalmanTracker._next_id += 1

        self.classification = classification
        self.dt = dt
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.confidence = 0.5
        self.height = 1.5  # Default height

        # Position and velocity history
        self.position_history = deque(maxlen=50)
        self.velocity_history = deque(maxlen=50)
        self.position_history.append(initial_position.copy())
        self.velocity_history.append(np.array([0.0, 0.0]))

        # Timestamps for velocity calculation
        self.last_update_time = time.time()

        # Initialize 8D Kalman Filter
        # State: [x, y, vx, vy, ax, ay, width, length]
        self.kf = KalmanFilter(dim_x=8, dim_z=4)

        # Initial state
        self.kf.x = np.array([
            initial_position[0],  # x
            initial_position[1],  # y
            0.0,                  # vx
            0.0,                  # vy
            0.0,                  # ax
            0.0,                  # ay
            initial_size[0],      # width
            initial_size[1]       # length
        ]).reshape(-1, 1)

        # State transition matrix F (Constant Acceleration model)
        dt2 = 0.5 * dt * dt
        self.kf.F = np.array([
            [1, 0, dt, 0,  dt2, 0,   0, 0],  # x
            [0, 1, 0,  dt, 0,   dt2, 0, 0],  # y
            [0, 0, 1,  0,  dt,  0,   0, 0],  # vx
            [0, 0, 0,  1,  0,   dt,  0, 0],  # vy
            [0, 0, 0,  0,  1,   0,   0, 0],  # ax
            [0, 0, 0,  0,  0,   1,   0, 0],  # ay
            [0, 0, 0,  0,  0,   0,   1, 0],  # width
            [0, 0, 0,  0,  0,   0,   0, 1],  # length
        ])

        # Measurement matrix H (observe position and size)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],  # x
            [0, 1, 0, 0, 0, 0, 0, 0],  # y
            [0, 0, 0, 0, 0, 0, 1, 0],  # width
            [0, 0, 0, 0, 0, 0, 0, 1],  # length
        ])

        # Measurement noise covariance R
        self.kf.R = np.diag([0.1, 0.1, 0.05, 0.05])

        # Process noise covariance Q
        self.kf.Q = np.diag([
            0.01,   # x
            0.01,   # y
            0.1,    # vx
            0.1,    # vy
            0.5,    # ax
            0.5,    # ay
            0.001,  # width
            0.001,  # length
        ])

        # Initial state covariance P
        self.kf.P = np.diag([
            1.0,    # x
            1.0,    # y
            10.0,   # vx
            10.0,   # vy
            50.0,   # ax
            50.0,   # ay
            0.5,    # width
            0.5,    # length
        ])

    def predict(self) -> np.ndarray:
        """Predict next state."""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.kf.x.flatten()

    def update(self, measurement: np.ndarray, height: float = None) -> np.ndarray:
        """
        Update state with new measurement.

        Args:
            measurement: Observed [x, y, width, length]
            height: Object height (if available)
        """
        self.kf.update(measurement.reshape(-1, 1))
        self.hits += 1
        self.time_since_update = 0

        if height is not None:
            self.height = height

        # Update histories
        position = self.get_position()
        velocity = self.get_velocity()
        self.position_history.append(position.copy())
        self.velocity_history.append(velocity.copy())

        return self.kf.x.flatten()

    def get_position(self) -> np.ndarray:
        """Get current position [x, y]."""
        return np.array([self.kf.x[0, 0], self.kf.x[1, 0]])

    def get_velocity(self) -> np.ndarray:
        """Get current velocity [vx, vy] in m/s."""
        return np.array([self.kf.x[2, 0], self.kf.x[3, 0]])

    def get_acceleration(self) -> np.ndarray:
        """Get current acceleration [ax, ay] in m/s^2."""
        return np.array([self.kf.x[4, 0], self.kf.x[5, 0]])

    def get_speed(self) -> float:
        """Get current speed (magnitude of velocity) in m/s."""
        vel = self.get_velocity()
        return np.sqrt(vel[0]**2 + vel[1]**2)

    def get_heading(self) -> float:
        """Get heading angle in radians (-pi to pi)."""
        vel = self.get_velocity()
        return np.arctan2(vel[1], vel[0])

    def get_dimensions(self) -> np.ndarray:
        """Get dimensions [width, length]."""
        return np.array([self.kf.x[6, 0], self.kf.x[7, 0]])

    def is_confirmed(self) -> bool:
        """Check if track is confirmed (reliable)."""
        return self.hits >= 3 and self.time_since_update <= 1

    def is_moving(self, threshold: float = 0.5) -> bool:
        """Check if object is moving (speed > threshold)."""
        return self.get_speed() > threshold

    def is_lost(self, max_age: int = 5) -> bool:
        """Check if track should be deleted."""
        if self.time_since_update > max_age:
            return True
        if self.age < 3 and self.time_since_update > 2:
            return True
        return False

    def get_motion_vector(self, scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get motion vector for visualization.

        Args:
            scale: Scale factor for vector length

        Returns:
            Tuple of (start_point, end_point) for the motion arrow
        """
        pos = self.get_position()
        vel = self.get_velocity()

        start = np.array([pos[0], pos[1], self.height / 2])
        end = np.array([pos[0] + vel[0] * scale,
                       pos[1] + vel[1] * scale,
                       self.height / 2])

        return start, end

    def get_trajectory(self) -> List[np.ndarray]:
        """Get position trajectory history."""
        return [np.array([p[0], p[1], self.height / 2])
                for p in self.position_history]

    def get_track_state(self) -> EnhancedTrackState:
        """Get complete track state for output."""
        pos = self.get_position()
        vel = self.get_velocity()
        dims = self.get_dimensions()

        return EnhancedTrackState(
            track_id=self.track_id,
            position=np.array([pos[0], pos[1], self.height / 2]),
            velocity=vel,
            speed=self.get_speed(),
            heading=self.get_heading(),
            dimensions=np.array([dims[1], dims[0], self.height]),
            classification=self.classification,
            confidence=self.confidence,
            age=self.age,
            hits=self.hits,
            time_since_update=self.time_since_update,
            trajectory=self.get_trajectory(),
            velocity_history=[np.array(v) for v in self.velocity_history],
            is_confirmed=self.is_confirmed(),
            is_moving=self.is_moving()
        )


class EnhancedMultiObjectTracker:
    """
    Enhanced Multi-Object Tracker with velocity estimation and trajectories.

    Features:
    - Kalman Filter tracking with velocity estimation
    - Hungarian algorithm for data association
    - Motion vector computation
    - Trajectory history for each track
    - Track lifecycle management
    """

    def __init__(self,
                 max_age: int = 5,
                 min_hits: int = 3,
                 association_threshold: float = 3.0,
                 dt: float = 0.1,
                 velocity_weight: float = 0.3):
        """
        Initialize enhanced multi-object tracker.

        Args:
            max_age: Maximum frames to keep unmatched track
            min_hits: Minimum hits to confirm track
            association_threshold: Max distance for matching
            dt: Time step between frames
            velocity_weight: Weight for velocity in cost calculation
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.association_threshold = association_threshold
        self.dt = dt
        self.velocity_weight = velocity_weight

        self.trackers: List[EnhancedKalmanTracker] = []
        self.frame_count = 0

        # Statistics
        self.total_tracks_created = 0
        self.total_tracks_deleted = 0

    def update(self, features_list: List[ObjectFeatures],
               verbose: bool = False) -> List[EnhancedTrackState]:
        """
        Update tracker with new detections.

        Args:
            features_list: Detected objects from current frame
            verbose: Print progress

        Returns:
            List of confirmed track states with velocities
        """
        self.frame_count += 1

        if verbose:
            print(f"\n--- Frame {self.frame_count} Enhanced Tracking ---")
            print(f"  Detections: {len(features_list)}")
            print(f"  Active tracks: {len(self.trackers)}")

        # Step 1: Predict all tracks
        predictions = []
        for tracker in self.trackers:
            tracker.predict()
            predictions.append(tracker.get_position())

        # Step 2: Data association
        matched, unmatched_dets, unmatched_tracks = self._associate(
            features_list, predictions
        )

        if verbose:
            print(f"  Matched: {len(matched)}")
            print(f"  Unmatched detections: {len(unmatched_dets)}")
            print(f"  Unmatched tracks: {len(unmatched_tracks)}")

        # Step 3: Update matched tracks
        for det_idx, track_idx in matched:
            features = features_list[det_idx]
            tracker = self.trackers[track_idx]

            measurement = np.array([
                features.center[0],
                features.center[1],
                features.width,
                features.length
            ])

            tracker.update(measurement, height=features.height)
            tracker.classification = features.classification
            tracker.confidence = features.confidence

        # Step 4: Create new tracks
        for det_idx in unmatched_dets:
            features = features_list[det_idx]

            new_tracker = EnhancedKalmanTracker(
                initial_position=features.center[:2],
                initial_size=np.array([features.width, features.length]),
                classification=features.classification,
                dt=self.dt
            )
            new_tracker.height = features.height
            new_tracker.confidence = features.confidence

            self.trackers.append(new_tracker)
            self.total_tracks_created += 1

            if verbose:
                print(f"  Created track ID: {new_tracker.track_id}")

        # Step 5: Delete lost tracks
        active_trackers = []
        for tracker in self.trackers:
            if not tracker.is_lost(self.max_age):
                active_trackers.append(tracker)
            else:
                self.total_tracks_deleted += 1
                if verbose:
                    print(f"  Deleted track ID: {tracker.track_id}")

        self.trackers = active_trackers

        # Return confirmed tracks
        confirmed_tracks = []
        for tracker in self.trackers:
            if tracker.is_confirmed():
                confirmed_tracks.append(tracker.get_track_state())

        if verbose:
            print(f"  Confirmed tracks: {len(confirmed_tracks)}")
            for track in confirmed_tracks:
                print(f"    ID {track.track_id}: {track.classification}, "
                      f"pos=({track.position[0]:.1f}, {track.position[1]:.1f}), "
                      f"vel=({track.velocity[0]:.2f}, {track.velocity[1]:.2f}) m/s, "
                      f"speed={track.speed:.2f} m/s")

        return confirmed_tracks

    def _associate(self, features_list: List[ObjectFeatures],
                   predictions: List[np.ndarray]) -> Tuple[
                       List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections to tracks using Hungarian algorithm.

        Cost combines position distance and velocity consistency.
        """
        if len(self.trackers) == 0:
            return [], list(range(len(features_list))), []

        if len(features_list) == 0:
            return [], [], list(range(len(self.trackers)))

        # Build cost matrix
        cost_matrix = np.zeros((len(features_list), len(self.trackers)))

        for d_idx, features in enumerate(features_list):
            det_pos = features.center[:2]

            for t_idx, tracker in enumerate(self.trackers):
                # Position distance
                track_pos = tracker.get_position()
                pos_dist = np.linalg.norm(det_pos - track_pos)

                # Velocity-based prediction distance
                vel = tracker.get_velocity()
                predicted_pos = track_pos + vel * self.dt
                vel_dist = np.linalg.norm(det_pos - predicted_pos)

                # Combined cost
                cost = (1 - self.velocity_weight) * pos_dist + \
                       self.velocity_weight * vel_dist

                cost_matrix[d_idx, t_idx] = cost

        # Solve assignment
        det_indices, track_indices = linear_sum_assignment(cost_matrix)

        # Filter by threshold
        matched = []
        unmatched_dets = list(range(len(features_list)))
        unmatched_tracks = list(range(len(self.trackers)))

        for d_idx, t_idx in zip(det_indices, track_indices):
            if cost_matrix[d_idx, t_idx] < self.association_threshold:
                matched.append((d_idx, t_idx))
                unmatched_dets.remove(d_idx)
                unmatched_tracks.remove(t_idx)

        return matched, unmatched_dets, unmatched_tracks

    def get_all_tracks(self) -> List[EnhancedTrackState]:
        """Get all track states including tentative."""
        return [t.get_track_state() for t in self.trackers]

    def get_confirmed_tracks(self) -> List[EnhancedTrackState]:
        """Get only confirmed track states."""
        return [t.get_track_state() for t in self.trackers if t.is_confirmed()]

    def get_statistics(self) -> Dict:
        """Get tracker statistics."""
        confirmed = [t for t in self.trackers if t.is_confirmed()]
        moving = [t for t in confirmed if t.is_moving()]

        return {
            'frame_count': self.frame_count,
            'active_tracks': len(self.trackers),
            'confirmed_tracks': len(confirmed),
            'moving_tracks': len(moving),
            'total_created': self.total_tracks_created,
            'total_deleted': self.total_tracks_deleted
        }

    def reset(self):
        """Reset tracker state."""
        self.trackers = []
        self.frame_count = 0
        self.total_tracks_created = 0
        self.total_tracks_deleted = 0
        EnhancedKalmanTracker._next_id = 0


# Test
if __name__ == "__main__":
    print("="*70)
    print("ENHANCED TRACKING MODULE - TEST")
    print("="*70)

    from data_loader import generate_simulated_frame
    from preprocessing import preprocess_point_cloud
    from clustering import cluster_point_cloud
    from classification import extract_and_classify

    # Reset ID counter
    EnhancedKalmanTracker._next_id = 0

    # Initialize tracker
    tracker = EnhancedMultiObjectTracker(
        max_age=5,
        min_hits=2,
        association_threshold=5.0,
        dt=0.1
    )

    print("\nProcessing 10-frame sequence...")

    for frame_idx in range(10):
        points = generate_simulated_frame(15000, seed=42, frame_index=frame_idx)
        processed = preprocess_point_cloud(points, verbose=False)
        labels, _ = cluster_point_cloud(processed, verbose=False)
        features, _ = extract_and_classify(processed, labels, verbose=False)

        tracks = tracker.update(features, verbose=True)

        for track in tracks:
            print(f"    Track {track.track_id}: speed={track.speed:.2f} m/s, "
                  f"moving={track.is_moving}, trajectory_len={len(track.trajectory)}")

    stats = tracker.get_statistics()
    print(f"\nStatistics: {stats}")
