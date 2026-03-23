"""
Action space converter for translating NoMaD waypoints to Habitat discrete actions.

NoMaD outputs continuous waypoints (δx, δy) in meters, while Habitat uses discrete
actions (MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, STOP). This module bridges the gap
using a proportional controller strategy.
"""

import numpy as np
from typing import Optional
from enum import IntEnum


class HabitatActions(IntEnum):
    """Habitat simulator discrete action space."""
    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3


class WaypointToHabitatConverter:
    """
    Converts NoMaD's continuous waypoints to Habitat's discrete actions.

    Strategy:
    - Angle-first policy: Turn towards waypoint if not aligned
    - Then move forward when aligned
    - Handles both coarse and fine-grained turning
    """

    def __init__(
        self,
        forward_step: float = 0.25,
        turn_angle: float = 10.0,
        distance_threshold: float = 0.15,
        angle_threshold: float = 5.0,
    ):
        """
        Initialize the action converter.

        Args:
            forward_step: Distance moved by MOVE_FORWARD action (meters)
            turn_angle: Angle turned by TURN_LEFT/RIGHT actions (degrees)
            distance_threshold: Minimum distance to waypoint to move forward (meters)
            angle_threshold: Minimum angle difference to trigger fine turning (degrees)
        """
        self.forward_step = forward_step
        self.turn_angle_deg = turn_angle
        self.turn_angle_rad = np.radians(turn_angle)
        self.distance_threshold = distance_threshold
        self.angle_threshold_rad = np.radians(angle_threshold)

        # Thresholds for decision making
        self.coarse_turn_threshold = np.radians(20.0)  # Large angle → need to turn

    def waypoint_to_action(self, waypoint: np.ndarray) -> int:
        """
        Convert a single waypoint to a Habitat action.

        Waypoints are in the robot's egocentric frame:
        - x: forward (positive = ahead)
        - y: lateral (positive = left)

        Decision logic:
        1. If angle to waypoint > 20°: Turn towards it (coarse)
        2. Elif distance > threshold and aligned: Move forward
        3. Elif angle > 5°: Fine-tune turn
        4. Else: Move forward (default)

        Args:
            waypoint: Array of shape (2,) with (δx, δy) in meters

        Returns:
            action: Habitat action ID (0-3)
        """
        dx, dy = waypoint

        # Compute distance and angle to waypoint
        distance = np.sqrt(dx**2 + dy**2)
        angle_to_waypoint = np.arctan2(dy, dx)

        # Decision logic
        # Case 1: Large angle deviation - need significant turn
        if abs(angle_to_waypoint) > self.coarse_turn_threshold:
            if angle_to_waypoint > 0:
                return HabitatActions.TURN_LEFT
            else:
                return HabitatActions.TURN_RIGHT

        # Case 2: Aligned enough and has distance to cover
        elif distance > self.distance_threshold:
            return HabitatActions.MOVE_FORWARD

        # Case 3: Small angle adjustment needed
        elif abs(angle_to_waypoint) > self.angle_threshold_rad:
            if angle_to_waypoint > 0:
                return HabitatActions.TURN_LEFT
            else:
                return HabitatActions.TURN_RIGHT

        # Case 4: Very close to waypoint or well-aligned - move forward
        else:
            return HabitatActions.MOVE_FORWARD

    def waypoints_to_action(
        self,
        waypoints: np.ndarray,
        strategy: str = 'index',
        waypoint_index: int = 2
    ) -> int:
        """
        Convert trajectory of waypoints to a single action.

        Args:
            waypoints: Array of shape (num_samples, len_traj_pred, 2) or (len_traj_pred, 2)
            strategy: Selection strategy ('index' or 'closest')
            waypoint_index: Index to select if using 'index' strategy (default: 2)

        Returns:
            action: Habitat action ID (0-3)
        """
        # Handle different input shapes
        if waypoints.ndim == 3:
            # (num_samples, len_traj_pred, 2) - take first sample
            waypoints = waypoints[0]
        elif waypoints.ndim != 2:
            raise ValueError(f"Expected waypoints with 2 or 3 dims, got {waypoints.ndim}")

        # Select which waypoint to execute
        if strategy == 'index':
            # Use specified index (matches deployment behavior)
            selected_waypoint = waypoints[waypoint_index]
        elif strategy == 'closest':
            # Select closest non-trivial waypoint
            distances = np.linalg.norm(waypoints, axis=1)
            valid_idx = np.where(distances > 0.05)[0]
            if len(valid_idx) > 0:
                selected_waypoint = waypoints[valid_idx[0]]
            else:
                selected_waypoint = waypoints[0]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Convert to action
        return self.waypoint_to_action(selected_waypoint)

    def get_action_name(self, action: int) -> str:
        """Get human-readable action name."""
        return HabitatActions(action).name
