"""
Multi-Object Tracking Module with Kalman Filter
===============================================
Autonomous Driving Perception Pipeline - Object Tracking

This module implements Kalman Filter-based multi-object tracking (MOT)
to maintain consistent object IDs across frames and predict trajectories.

Key Components:
1. KalmanTracker: Single object tracker using Kalman Filter
2. HungarianMatcher: Data association using Hungarian algorithm
3. MultiObjectTracker: Manages multiple tracks across frames

Motion Model: Constant Velocity (CV)
State Vector: [x, y, vx, vy, width, length]

Author: Perception Engineer
Course: Localization, Motion Planning and Sensor Fusion
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque

from classification import ObjectFeatures


@dataclass
class TrackState:
    """
    State of a tracked object.

    Contains position, velocity, dimensions, and metadata.
    """
    track_id: int
    position: np.ndarray      # [x, y, z]
    velocity: np.ndarray      # [vx, vy]
    dimensions: np.ndarray    # [length, width, height]
    classification: str
    confidence: float
    age: int                  # Frames since track creation
    hits: int                 # Number of successful detections
    time_since_update: int    # Frames since last measurement
    history: List[np.ndarray] # Position history


class KalmanObjectTracker:
    """
    Kalman Filter-based tracker for a single object.

    Uses Constant Velocity motion model:
    - State: [x, y, vx, vy, width, length]
    - Measurements: [x, y, width, length]

    The Kalman Filter provides:
    1. Smooth state estimation from noisy measurements
    2. Velocity estimation from position-only measurements
    3. Prediction during temporary occlusions
    """

    # Class variable for unique ID generation
    _next_id = 0

    def __init__(self,
                 initial_position: np.ndarray,
                 initial_size: np.ndarray,
                 classification: str = 'UNKNOWN',
                 dt: float = 0.1):
        """
        Initialize Kalman tracker for a single object.

        Args:
            initial_position: Initial [x, y] position
            initial_size: Initial [width, length]
            classification: Object class (VEHICLE, PEDESTRIAN, UNKNOWN)
            dt: Time step between frames (seconds)
        """
        # Assign unique track ID
        self.track_id = KalmanObjectTracker._next_id
        KalmanObjectTracker._next_id += 1

        # Metadata
        self.classification = classification
        self.dt = dt
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.confidence = 0.5

        # Position history for trajectory visualization
        self.history = deque(maxlen=30)  # Store last 30 positions
        self.history.append(initial_position.copy())

        # Initialize 6D Kalman Filter
        # State: [x, y, vx, vy, width, length]
        self.kf = KalmanFilter(dim_x=6, dim_z=4)

        # Initial state
        self.kf.x = np.array([
            initial_position[0],  # x
            initial_position[1],  # y
            0.0,                   # vx (initially stationary)
            0.0,                   # vy (initially stationary)
            initial_size[0],       # width
            initial_size[1]        # length
        ]).reshape(-1, 1)

        # State transition matrix F (Constant Velocity model)
        # x(k+1) = x(k) + vx(k) * dt
        # y(k+1) = y(k) + vy(k) * dt
        # vx(k+1) = vx(k)
        # vy(k+1) = vy(k)
        # width(k+1) = width(k)
        # length(k+1) = length(k)
        self.kf.F = np.array([
            [1, 0, dt, 0,  0, 0],  # x
            [0, 1, 0,  dt, 0, 0],  # y
            [0, 0, 1,  0,  0, 0],  # vx
            [0, 0, 0,  1,  0, 0],  # vy
            [0, 0, 0,  0,  1, 0],  # width
            [0, 0, 0,  0,  0, 1],  # length
        ])

        # Measurement matrix H
        # We measure: [x, y, width, length]
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],  # x
            [0, 1, 0, 0, 0, 0],  # y
            [0, 0, 0, 0, 1, 0],  # width
            [0, 0, 0, 0, 0, 1],  # length
        ])

        # Measurement noise covariance R
        # How noisy are the measurements from detection?
        self.kf.R = np.diag([
            0.1,   # x position noise (meters)
            0.1,   # y position noise (meters)
            0.05,  # width noise (meters)
            0.05,  # length noise (meters)
        ])

        # Process noise covariance Q
        # How much does the true state deviate from our model?
        q_pos = 0.01   # Position process noise
        q_vel = 0.1    # Velocity process noise (higher - velocity changes)
        q_dim = 0.001  # Dimension process noise (very low - size is constant)

        self.kf.Q = np.diag([
            q_pos,  # x
            q_pos,  # y
            q_vel,  # vx
            q_vel,  # vy
            q_dim,  # width
            q_dim,  # length
        ])

        # Initial state covariance P
        # High uncertainty for velocity, lower for position/size
        self.kf.P = np.diag([
            1.0,    # x
            1.0,    # y
            10.0,   # vx (very uncertain)
            10.0,   # vy (very uncertain)
            0.5,    # width
            0.5,    # length
        ])

    def predict(self) -> np.ndarray:
        """
        Predict next state (time update).

        Called every frame, even without a matching detection.
        Allows tracking to continue during occlusions.

        Returns:
            Predicted state [x, y, vx, vy, width, length]
        """
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.kf.x.flatten()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Update state with new measurement (measurement update).

        Args:
            measurement: Observed [x, y, width, length]

        Returns:
            Updated state [x, y, vx, vy, width, length]
        """
        self.kf.update(measurement.reshape(-1, 1))
        self.hits += 1
        self.time_since_update = 0

        # Add to position history
        position = self.get_position()
        self.history.append(position.copy())

        return self.kf.x.flatten()

    def get_state(self) -> np.ndarray:
        """Get current state estimate."""
        return self.kf.x.flatten()

    def get_position(self) -> np.ndarray:
        """Get current position [x, y]."""
        return np.array([self.kf.x[0, 0], self.kf.x[1, 0]])

    def get_velocity(self) -> np.ndarray:
        """Get current velocity [vx, vy]."""
        return np.array([self.kf.x[2, 0], self.kf.x[3, 0]])

    def get_speed(self) -> float:
        """Get current speed (magnitude of velocity)."""
        vel = self.get_velocity()
        return np.sqrt(vel[0]**2 + vel[1]**2)

    def get_dimensions(self) -> np.ndarray:
        """Get current dimensions [width, length]."""
        return np.array([self.kf.x[4, 0], self.kf.x[5, 0]])

    def get_bbox(self) -> np.ndarray:
        """Get bounding box [x, y, width, length]."""
        pos = self.get_position()
        dims = self.get_dimensions()
        return np.array([pos[0], pos[1], dims[0], dims[1]])

    def get_track_state(self, height: float = 1.5) -> TrackState:
        """
        Get complete track state for visualization.

        Args:
            height: Object height (not tracked, use last known)

        Returns:
            TrackState dataclass
        """
        pos = self.get_position()
        vel = self.get_velocity()
        dims = self.get_dimensions()

        return TrackState(
            track_id=self.track_id,
            position=np.array([pos[0], pos[1], height/2]),
            velocity=vel,
            dimensions=np.array([dims[1], dims[0], height]),  # [L, W, H]
            classification=self.classification,
            confidence=self.confidence,
            age=self.age,
            hits=self.hits,
            time_since_update=self.time_since_update,
            history=[np.array(h) for h in self.history]
        )

    def is_confirmed(self) -> bool:
        """
        Check if track is confirmed (reliable).

        A track is confirmed if:
        - Matched to detections multiple times (not a false positive)
        - Recently updated (not lost)
        """
        return self.hits >= 3 and self.time_since_update <= 1

    def is_lost(self) -> bool:
        """
        Check if track should be deleted.

        Delete tracks that:
        - Haven't been updated for too long
        - Are young and haven't been matched (likely false positive)
        """
        # Lost if not updated for >5 frames
        if self.time_since_update > 5:
            return True

        # Delete young tracks that haven't been matched
        if self.age < 3 and self.time_since_update > 2:
            return True

        return False


class MultiObjectTracker:
    """
    Multi-Object Tracker (MOT) using Kalman Filters.

    Handles:
    1. Track initialization from new detections
    2. Data association (matching detections to tracks)
    3. Track state management (confirmed, tentative, lost)
    4. Track deletion for objects that left the scene
    """

    def __init__(self,
                 max_age: int = 5,
                 min_hits: int = 3,
                 association_threshold: float = 5.0,
                 dt: float = 0.1):
        """
        Initialize multi-object tracker.

        Args:
            max_age: Maximum frames to keep unmatched track alive
            min_hits: Minimum hits to confirm a track
            association_threshold: Maximum distance for detection-track matching
            dt: Time step between frames
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.association_threshold = association_threshold
        self.dt = dt

        self.trackers: List[KalmanObjectTracker] = []
        self.frame_count = 0

    def update(self, features_list: List[ObjectFeatures],
               verbose: bool = False) -> List[TrackState]:
        """
        Update tracker with new detections from current frame.

        Tracking Pipeline:
        1. Predict: Move all tracks forward in time
        2. Associate: Match detections to predicted tracks
        3. Update: Update matched tracks with measurements
        4. Create: Initialize new tracks for unmatched detections
        5. Delete: Remove lost tracks

        Args:
            features_list: List of ObjectFeatures from current frame
            verbose: Print progress information

        Returns:
            List of TrackState for confirmed tracks
        """
        self.frame_count += 1

        if verbose:
            print(f"\n--- Frame {self.frame_count} Tracking ---")
            print(f"  Detections: {len(features_list)}")
            print(f"  Active tracks: {len(self.trackers)}")

        # Step 1: Predict all existing tracks
        for tracker in self.trackers:
            tracker.predict()

        # Step 2: Data association (Hungarian algorithm)
        matched, unmatched_dets, unmatched_tracks = self._associate(features_list)

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

            tracker.update(measurement)
            tracker.classification = features.classification
            tracker.confidence = features.confidence

        # Step 4: Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            features = features_list[det_idx]

            new_tracker = KalmanObjectTracker(
                initial_position=features.center[:2],
                initial_size=np.array([features.width, features.length]),
                classification=features.classification,
                dt=self.dt
            )
            new_tracker.confidence = features.confidence

            self.trackers.append(new_tracker)

            if verbose:
                print(f"  Created new track ID: {new_tracker.track_id}")

        # Step 5: Delete lost tracks
        active_trackers = []
        for tracker in self.trackers:
            if not tracker.is_lost():
                active_trackers.append(tracker)
            elif verbose:
                print(f"  Deleted track ID: {tracker.track_id}")

        self.trackers = active_trackers

        # Return confirmed track states
        confirmed_tracks = []
        for tracker in self.trackers:
            if tracker.is_confirmed():
                # Estimate height from classification
                if tracker.classification == 'PEDESTRIAN':
                    height = 1.7
                elif tracker.classification == 'VEHICLE':
                    height = 1.5
                else:
                    height = 1.5

                confirmed_tracks.append(tracker.get_track_state(height))

        if verbose:
            print(f"  Confirmed tracks: {len(confirmed_tracks)}")

        return confirmed_tracks

    def _associate(self, features_list: List[ObjectFeatures]) -> Tuple[
            List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections to existing tracks using Hungarian algorithm.

        Uses Euclidean distance as cost metric.

        Args:
            features_list: List of detections

        Returns:
            Tuple of (matched_pairs, unmatched_detections, unmatched_tracks)
        """
        if len(self.trackers) == 0:
            return [], list(range(len(features_list))), []

        if len(features_list) == 0:
            return [], [], list(range(len(self.trackers)))

        # Build cost matrix: Euclidean distance
        cost_matrix = np.zeros((len(features_list), len(self.trackers)))

        for d_idx, features in enumerate(features_list):
            det_pos = features.center[:2]

            for t_idx, tracker in enumerate(self.trackers):
                track_pos = tracker.get_position()
                distance = np.linalg.norm(det_pos - track_pos)
                cost_matrix[d_idx, t_idx] = distance

        # Solve assignment problem using Hungarian algorithm
        det_indices, track_indices = linear_sum_assignment(cost_matrix)

        # Filter matches by threshold
        matched = []
        unmatched_dets = list(range(len(features_list)))
        unmatched_tracks = list(range(len(self.trackers)))

        for d_idx, t_idx in zip(det_indices, track_indices):
            if cost_matrix[d_idx, t_idx] < self.association_threshold:
                matched.append((d_idx, t_idx))
                unmatched_dets.remove(d_idx)
                unmatched_tracks.remove(t_idx)
            else:
                # Distance too large - not a valid match
                pass

        return matched, unmatched_dets, unmatched_tracks

    def get_all_tracks(self) -> List[TrackState]:
        """Get all active track states (including tentative)."""
        tracks = []
        for tracker in self.trackers:
            height = 1.7 if tracker.classification == 'PEDESTRIAN' else 1.5
            tracks.append(tracker.get_track_state(height))
        return tracks

    def get_confirmed_tracks(self) -> List[TrackState]:
        """Get only confirmed track states."""
        tracks = []
        for tracker in self.trackers:
            if tracker.is_confirmed():
                height = 1.7 if tracker.classification == 'PEDESTRIAN' else 1.5
                tracks.append(tracker.get_track_state(height))
        return tracks

    def reset(self) -> None:
        """Reset the tracker, clearing all tracks."""
        self.trackers = []
        self.frame_count = 0
        KalmanObjectTracker._next_id = 0


# Example usage and testing
if __name__ == "__main__":
    from data_loader import generate_simulated_frame
    from preprocessing import preprocess_point_cloud
    from clustering import cluster_point_cloud
    from classification import extract_and_classify

    print("="*70)
    print("MULTI-OBJECT TRACKING MODULE - TEST")
    print("="*70)

    # Reset tracker ID counter
    KalmanObjectTracker._next_id = 0

    # Initialize tracker
    tracker = MultiObjectTracker(
        max_age=5,
        min_hits=3,
        association_threshold=5.0,
        dt=0.1
    )

    print("\n[Test] Processing simulated sequence (10 frames)...")

    for frame_idx in range(10):
        # Generate frame with motion
        raw_points = generate_simulated_frame(
            num_points=15000,
            seed=42,
            frame_index=frame_idx
        )

        # Preprocess
        processed_points = preprocess_point_cloud(raw_points, verbose=False)

        # Cluster
        labels, num_clusters = cluster_point_cloud(processed_points, verbose=False)

        # Extract features and classify
        features_list, classifications = extract_and_classify(
            processed_points, labels, verbose=False
        )

        # Update tracker
        confirmed_tracks = tracker.update(features_list, verbose=True)

        # Print track info
        for track in confirmed_tracks:
            pos = track.position
            vel = track.velocity
            speed = np.sqrt(vel[0]**2 + vel[1]**2)
            print(f"    Track {track.track_id}: {track.classification}, "
                  f"pos=({pos[0]:.1f}, {pos[1]:.1f}), "
                  f"speed={speed:.2f}m/s, age={track.age}")

    print("\n" + "="*70)
    print("Tracking module ready for use.")
    print("="*70)
