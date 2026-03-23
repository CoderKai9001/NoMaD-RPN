"""
Episode generation utilities for Habitat ImageNav integration.

Provides functions for generating random navigation episodes with start/goal pairs
and capturing goal images.
"""

import numpy as np
from typing import Tuple, Optional, List, TYPE_CHECKING
from PIL import Image

# Type checking imports
if TYPE_CHECKING:
    import habitat
    from habitat.core.simulator import Observations

# Runtime import
try:
    import habitat
    HABITAT_AVAILABLE = True
except ImportError:
    HABITAT_AVAILABLE = False
    habitat = None


def generate_random_navigable_positions(
    sim,  # habitat.Simulator
    min_geodesic_distance: float = 5.0,
    max_geodesic_distance: float = 15.0,
    max_attempts: int = 100
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Generate random start and goal positions that are navigable.

    Args:
        sim: Habitat simulator instance
        min_geodesic_distance: Minimum geodesic distance between start and goal (meters)
        max_geodesic_distance: Maximum geodesic distance between start and goal (meters)
        max_attempts: Maximum number of sampling attempts

    Returns:
        Tuple of (start_position, goal_position, geodesic_distance) or None if failed
    """
    pathfinder = sim.pathfinder

    for attempt in range(max_attempts):
        # Sample random navigable positions
        start_position = pathfinder.get_random_navigable_point()
        goal_position = pathfinder.get_random_navigable_point()

        # Check if path exists and get geodesic distance
        import habitat_sim
        path = habitat_sim.ShortestPath()
        path.requested_start = start_position
        path.requested_end = goal_position

        found_path = pathfinder.find_path(path)

        if found_path:
            geodesic_distance = path.geodesic_distance

            # Check distance constraints
            if min_geodesic_distance <= geodesic_distance <= max_geodesic_distance:
                return start_position, goal_position, geodesic_distance

    # Failed to generate valid positions
    print(f"Warning: Could not generate valid episode after {max_attempts} attempts")
    return None


def capture_goal_image(
    env,  # habitat.Env
    goal_position: np.ndarray,
    goal_rotation: Optional[np.ndarray] = None
) -> Image.Image:
    """
    Capture RGB observation from goal position by teleporting agent.

    Args:
        env: Habitat environment
        goal_position: 3D position (x, y, z) to capture from
        goal_rotation: Optional quaternion rotation. If None, uses random rotation.

    Returns:
        goal_image: PIL Image captured from goal position
    """
    # Save current agent state
    current_state = env.sim.get_agent_state()

    # Teleport to goal position
    import habitat_sim
    goal_state = habitat_sim.AgentState()
    goal_state.position = goal_position

    if goal_rotation is not None:
        goal_state.rotation = goal_rotation
    else:
        # Random rotation (look in random direction)
        random_angle = np.random.uniform(0, 2 * np.pi)
        goal_state.rotation = quaternion_from_angle_axis(random_angle, np.array([0, 1, 0]))

    env.sim.set_agent_state(goal_state.position, goal_state.rotation)

    # Capture observation
    observations = env.sim.get_observations_at()
    goal_rgb = observations['rgb']
    if goal_rgb.shape[-1] == 4:
        goal_rgb = goal_rgb[..., :3]
    # Convert to PIL Image
    goal_image = Image.fromarray(goal_rgb)

    # Restore agent to original position
    env.sim.set_agent_state(current_state.position, current_state.rotation)

    return goal_image


def quaternion_from_angle_axis(angle: float, axis: np.ndarray) -> np.ndarray:
    """
    Create quaternion from angle-axis representation.

    Args:
        angle: Rotation angle in radians
        axis: Rotation axis (3D unit vector)

    Returns:
        quaternion: Array of shape (4,) as [x, y, z, w]
    """
    axis = axis / np.linalg.norm(axis)
    half_angle = angle / 2.0
    w = np.cos(half_angle)
    x, y, z = axis * np.sin(half_angle)
    return np.array([x, y, z, w])


def compute_geodesic_distance(
    sim,  # habitat.Simulator
    start_position: np.ndarray,
    goal_position: np.ndarray
) -> float:
    """
    Compute geodesic (shortest path) distance between two positions.

    Args:
        sim: Habitat simulator
        start_position: Start 3D position
        goal_position: Goal 3D position

    Returns:
        geodesic_distance: Distance in meters, or float('inf') if no path exists
    """
    path = habitat.ShortestPath()
    path.requested_start = start_position
    path.requested_end = goal_position

    found_path = sim.pathfinder.find_path(path)

    if found_path:
        return path.geodesic_distance
    else:
        return float('inf')


def compute_path_length(trajectory: List[np.ndarray]) -> float:
    """
    Compute the total path length from a trajectory.

    Args:
        trajectory: List of 3D positions

    Returns:
        path_length: Total distance traveled in meters
    """
    if len(trajectory) < 2:
        return 0.0

    distances = []
    for i in range(1, len(trajectory)):
        dist = np.linalg.norm(trajectory[i] - trajectory[i - 1])
        distances.append(dist)

    return sum(distances)


def compute_euclidean_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two positions.

    Args:
        pos1: First position (3D)
        pos2: Second position (3D)

    Returns:
        distance: Euclidean distance in meters
    """
    return np.linalg.norm(pos1 - pos2)
