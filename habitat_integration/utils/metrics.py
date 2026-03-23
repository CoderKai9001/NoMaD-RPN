"""
Evaluation metrics for Habitat ImageNav with NoMaD.

Tracks standard navigation metrics (success rate, SPL) and provides
utilities for computing and logging results.
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from datetime import datetime


def compute_spl(success: bool, path_length: float, geodesic_distance: float) -> float:
    """
    Compute Success weighted by Path Length (SPL).

    SPL = Success × (geodesic_distance / max(path_length, geodesic_distance))

    Args:
        success: Whether the episode was successful
        path_length: Actual path length traveled (meters)
        geodesic_distance: Shortest possible path length (meters)

    Returns:
        spl: SPL score between 0 and 1
    """
    if not success:
        return 0.0

    if path_length < 1e-6:
        return 0.0

    return geodesic_distance / max(path_length, geodesic_distance)


class NavigationMetrics:
    """
    Track and compute navigation metrics across multiple episodes.
    """

    def __init__(self):
        """Initialize empty metrics tracker."""
        self.episodes = []

    def add_episode(self, episode_info: Dict[str, Any]):
        """
        Record results from a single episode.

        Args:
            episode_info: Dictionary with keys:
                - scene_id: str - HM3D scene identifier
                - episode_id: int - Episode number
                - success: bool - Whether goal was reached
                - spl: float - Success weighted by Path Length
                - distance_to_goal_start: float - Initial distance to goal (m)
                - distance_to_goal_end: float - Final distance to goal (m)
                - path_length: float - Total distance traveled (m)
                - geodesic_distance: float - Shortest path distance (m)
                - steps_taken: int - Number of actions taken
                - stop_reason: str - Why episode ended (SUCCESS/TIMEOUT/STUCK/etc)
                - trajectory: List[np.ndarray] - Agent positions (optional)
                - predicted_distances: List[float] - Model distance predictions (optional)
        """
        self.episodes.append(episode_info)

    def compute_aggregate_metrics(self) -> Dict[str, Any]:
        """
        Compute aggregate statistics across all recorded episodes.

        Returns:
            metrics: Dictionary with aggregate metrics
        """
        if not self.episodes:
            return {}

        # Basic counts
        num_episodes = len(self.episodes)
        successes = [e['success'] for e in self.episodes]
        num_successes = sum(successes)

        # Core metrics
        success_rate = np.mean(successes)
        spl_values = [e['spl'] for e in self.episodes]
        mean_spl = np.mean(spl_values)

        # Path metrics (all episodes)
        path_lengths = [e['path_length'] for e in self.episodes]
        geodesic_distances = [e['geodesic_distance'] for e in self.episodes]
        steps_taken = [e['steps_taken'] for e in self.episodes]

        # Navigation error
        navigation_errors = [
            e['path_length'] - e['geodesic_distance']
            for e in self.episodes
        ]

        # Success-only metrics
        successful_episodes = [e for e in self.episodes if e['success']]
        if successful_episodes:
            success_path_length = np.mean([e['path_length'] for e in successful_episodes])
            success_steps = np.mean([e['steps_taken'] for e in successful_episodes])
            success_spl = np.mean([e['spl'] for e in successful_episodes])
        else:
            success_path_length = 0.0
            success_steps = 0.0
            success_spl = 0.0

        # Final distances
        final_distances = [e['distance_to_goal_end'] for e in self.episodes]

        # Stop reasons distribution
        stop_reasons = [e['stop_reason'] for e in self.episodes]
        stop_reason_counts = Counter(stop_reasons)

        # Per-scene breakdown
        scenes = list(set(e['scene_id'] for e in self.episodes))
        per_scene_metrics = {}
        for scene_id in scenes:
            scene_episodes = [e for e in self.episodes if e['scene_id'] == scene_id]
            scene_successes = [e['success'] for e in scene_episodes]
            per_scene_metrics[scene_id] = {
                'num_episodes': len(scene_episodes),
                'success_rate': np.mean(scene_successes),
                'spl': np.mean([e['spl'] for e in scene_episodes])
            }

        # Compile all metrics
        metrics = {
            'num_episodes': num_episodes,
            'num_successes': num_successes,
            'success_rate': float(success_rate),
            'spl': float(mean_spl),
            'avg_path_length': float(np.mean(path_lengths)),
            'avg_geodesic_distance': float(np.mean(geodesic_distances)),
            'avg_navigation_error': float(np.mean(navigation_errors)),
            'avg_steps': float(np.mean(steps_taken)),
            'avg_final_distance_to_goal': float(np.mean(final_distances)),
            'success_only': {
                'avg_path_length': float(success_path_length),
                'avg_steps': float(success_steps),
                'avg_spl': float(success_spl),
            },
            'stop_reason_distribution': dict(stop_reason_counts),
            'per_scene': per_scene_metrics
        }

        return metrics

    def print_summary(self):
        """Print a formatted summary of the metrics."""
        metrics = self.compute_aggregate_metrics()

        if not metrics:
            print("No episodes recorded yet.")
            return

        print("\n" + "="*60)
        print(f"{'NAVIGATION EVALUATION SUMMARY':^60}")
        print("="*60)
        print(f"\nTotal Episodes: {metrics['num_episodes']}")
        print(f"Successes: {metrics['num_successes']}")
        print(f"\n{'CORE METRICS':^60}")
        print("-"*60)
        print(f"Success Rate:        {metrics['success_rate']:.1%}")
        print(f"SPL:                 {metrics['spl']:.3f}")
        print(f"\n{'PATH METRICS (ALL EPISODES)':^60}")
        print("-"*60)
        print(f"Avg Path Length:     {metrics['avg_path_length']:.2f} m")
        print(f"Avg Geodesic Dist:   {metrics['avg_geodesic_distance']:.2f} m")
        print(f"Avg Nav Error:       {metrics['avg_navigation_error']:.2f} m")
        print(f"Avg Steps:           {metrics['avg_steps']:.1f}")
        print(f"Avg Final Dist:      {metrics['avg_final_distance_to_goal']:.2f} m")

        if metrics['num_successes'] > 0:
            print(f"\n{'SUCCESS-ONLY METRICS':^60}")
            print("-"*60)
            print(f"Avg Path Length:     {metrics['success_only']['avg_path_length']:.2f} m")
            print(f"Avg Steps:           {metrics['success_only']['avg_steps']:.1f}")
            print(f"Avg SPL:             {metrics['success_only']['avg_spl']:.3f}")

        print(f"\n{'STOP REASONS':^60}")
        print("-"*60)
        for reason, count in metrics['stop_reason_distribution'].items():
            percentage = (count / metrics['num_episodes']) * 100
            print(f"{reason:20s}: {count:3d} ({percentage:5.1f}%)")

        if metrics.get('per_scene'):
            print(f"\n{'PER-SCENE BREAKDOWN':^60}")
            print("-"*60)
            for scene_id, scene_metrics in metrics['per_scene'].items():
                scene_name = scene_id.split('/')[-1][:30]  # Truncate long names
                print(f"{scene_name:30s}: SR={scene_metrics['success_rate']:.1%}, SPL={scene_metrics['spl']:.3f}, N={scene_metrics['num_episodes']}")

        print("="*60 + "\n")

    def save_results(self, output_path: str):
        """
        Save detailed results to JSON file.

        Args:
            output_path: Path to save JSON results
        """
        results = {
            'aggregate_metrics': self.compute_aggregate_metrics(),
            'per_episode': self.episodes,
            'timestamp': datetime.now().isoformat(),
            'num_episodes': len(self.episodes)
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)  # default=str handles numpy types

        print(f"Results saved to {output_path}")


class StoppingCriterion:
    """
    Multi-signal stopping criterion for navigation episodes.
    """

    def __init__(
        self,
        success_distance: float = 0.2,
        temporal_distance_threshold: float = 3.0,
        max_steps: int = 500,
        stuck_threshold: int = 10,
        stuck_distance: float = 0.1
    ):
        """
        Initialize stopping criterion.

        Args:
            success_distance: Euclidean distance threshold for success (meters)
            temporal_distance_threshold: Model's temporal distance threshold
            max_steps: Maximum steps before timeout
            stuck_threshold: Number of steps to check for stuck detection
            stuck_distance: Maximum displacement to be considered stuck (meters)
        """
        self.success_distance = success_distance
        self.temporal_distance_threshold = temporal_distance_threshold
        self.max_steps = max_steps
        self.stuck_threshold = stuck_threshold
        self.stuck_distance = stuck_distance

    def should_stop(
        self,
        euclidean_distance_to_goal: float,
        predicted_temporal_distance: Optional[float],
        steps_taken: int,
        position_history: List[np.ndarray]
    ) -> Tuple[bool, str]:
        """
        Determine if episode should stop.

        Args:
            euclidean_distance_to_goal: Current distance to goal (meters)
            predicted_temporal_distance: Model's predicted distance (or None)
            steps_taken: Number of steps taken so far
            position_history: List of agent positions

        Returns:
            Tuple of (should_stop, stop_reason)
        """
        # Primary success condition
        if euclidean_distance_to_goal < self.success_distance:
            return True, "SUCCESS"

        # Secondary: Model confidence (only if reasonably close)
        if predicted_temporal_distance is not None:
            if predicted_temporal_distance < self.temporal_distance_threshold:
                if euclidean_distance_to_goal < 0.5:  # Within 50cm
                    return True, "MODEL_CONFIDENT"

        # Timeout
        if steps_taken >= self.max_steps:
            return True, "TIMEOUT"

        # Stuck detection
        if len(position_history) >= self.stuck_threshold:
            recent_positions = position_history[-self.stuck_threshold:]
            displacement = np.linalg.norm(
                recent_positions[-1] - recent_positions[0]
            )
            if displacement < self.stuck_distance:
                return True, "STUCK"

        return False, ""
