"""
Main evaluation script for running NoMaD in Habitat simulator on HM3D scenes.

This script orchestrates the full evaluation pipeline:
1. Load HM3D scenes in Habitat
2. Generate random ImageNav episodes
3. Run NoMaD model to predict actions
4. Track navigation metrics (success rate, SPL)
"""

import argparse
import os
import sys
import yaml
from typing import Dict, Any, List
import numpy as np
import torch
from PIL import Image
import habitat
from habitat.config.default import get_config as get_habitat_config
from habitat.core.simulator import Observations

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nomad_wrapper import NoMaDHabitatWrapper
from action_converter import WaypointToHabitatConverter, HabitatActions
from utils.episode_utils import (
    generate_random_navigable_positions,
    capture_goal_image,
    compute_path_length,
    compute_euclidean_distance
)
from utils.metrics import NavigationMetrics, StoppingCriterion, compute_spl


def create_habitat_env(config_path: str, scene_id: str):
    import habitat_sim
    import numpy as np
    
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_id
    backend_cfg.enable_physics = False
    
    # Sensor Configuration
    sensor_spec = habitat_sim.CameraSensorSpec()
    sensor_spec.uuid = "rgb"
    sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    sensor_spec.resolution = [120, 160] # height, width
    sensor_spec.position = [0.0, 0.88, 0.0]
    # Default FOV is 90 degrees
    
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_spec]
    agent_cfg.height = 0.88
    agent_cfg.radius = 0.18
    agent_cfg.action_space = dict(
        move_forward=habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        turn_left=habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
        turn_right=habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
    )
    
    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    
    class SimWrapper:
        def __init__(self, sim):
            self._sim = sim
            
        def __getattr__(self, name):
            return getattr(self._sim, name)
            
        def get_agent_state(self, agent_id=0):
            return self._sim.get_agent(agent_id).get_state()
            
        def set_agent_state(self, position, rotation, agent_id=0, reset_sensors=True):
            state = self._sim.get_agent(agent_id).get_state()
            state.position = position
            if rotation is not None:
                if isinstance(rotation, list) or isinstance(rotation, np.ndarray):
                    import quaternion
                    rotation = np.quaternion(rotation[3], rotation[0], rotation[1], rotation[2])
                state.rotation = rotation
            self._sim.get_agent(agent_id).set_state(state)
            
        def get_observations_at(self, position=None, rotation=None, keep_agent_at_new_pose=False):
            if position is None:
                return self._sim.get_sensor_observations()
            current_state = self.get_agent_state()
            self.set_agent_state(position, rotation)
            obs = self._sim.get_sensor_observations()
            if not keep_agent_at_new_pose:
                self.set_agent_state(current_state.position, current_state.rotation)
            return obs
            
        def geodesic_distance(self, position_a, position_b, episode=None):
            import habitat_sim
            path = habitat_sim.ShortestPath()
            path.requested_start = position_a
            path.requested_end = position_b
            self._sim.pathfinder.find_path(path)
            return path.geodesic_distance

    class DummyEnv:
        def __init__(self, sim):
            self.sim = SimWrapper(sim)
            
        def step(self, action):
            if isinstance(action, dict):
                action = action.get("action", action)
            
            if action == 1 or action == "MOVE_FORWARD":
                action_name = "move_forward"
            elif action == 2 or action == "TURN_LEFT":
                action_name = "turn_left"
            elif action == 3 or action == "TURN_RIGHT":
                action_name = "turn_right"
            else:
                action_name = action
                
            obs = self.sim._sim.step(action_name)
            return obs
            
        def close(self):
            if hasattr(self.sim._sim, 'close'):
                self.sim._sim.close()

    sim = habitat_sim.Simulator(cfg)
    return DummyEnv(sim)


def run_episode(
    env: habitat.Env,
    nomad_model: NoMaDHabitatWrapper,
    action_converter: WaypointToHabitatConverter,
    stopping_criterion: StoppingCriterion,
    start_position: np.ndarray,
    goal_position: np.ndarray,
    goal_image: Image.Image,
    episode_id: int,
    scene_id: str,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run a single ImageNav episode with NoMaD.

    Args:
        env: Habitat environment
        nomad_model: NoMaD model wrapper
        action_converter: Waypoint to action converter
        stopping_criterion: Stopping criterion checker
        start_position: Starting 3D position
        goal_position: Goal 3D position
        goal_image: Goal image (PIL Image)
        episode_id: Episode number for logging
        scene_id: Scene identifier
        verbose: Whether to print step-by-step info

    Returns:
        episode_info: Dictionary with episode results
    """
    # Reset model context
    nomad_model.reset_context()

    # Set agent to start position
    start_state = env.sim.get_agent_state()
    start_state.position = start_position
    # Keep current rotation (or set random)
    env.sim.set_agent_state(start_state.position, start_state.rotation)

    # Get initial observation
    observations = env.sim.get_observations_at()
    rgb_obs = observations['rgb']

    # Add initial observation to context (repeat 3 times for cold start)
    for _ in range(nomad_model.context_size):
        nomad_model.add_observation(rgb_obs)

    # Track episode state
    trajectory = [start_position.copy()]
    predicted_distances = []
    actions_taken = []
    steps = 0
    done = False

    # Compute initial distance
    geodesic_distance = env.sim.geodesic_distance(start_position, goal_position)
    distance_to_goal_start = compute_euclidean_distance(start_position, goal_position)

    if verbose:
        print(f"\n--- Episode {episode_id} ---")
        print(f"Scene: {scene_id.split('/')[-1]}")
        print(f"Start: {start_position}")
        print(f"Goal: {goal_position}")
        print(f"Geodesic distance: {geodesic_distance:.2f}m")
        print(f"Euclidean distance: {distance_to_goal_start:.2f}m")

    # Episode loop
    while not done:
        # Get current agent state
        agent_state = env.sim.get_agent_state()
        current_position = agent_state.position
        current_distance = compute_euclidean_distance(current_position, goal_position)

        # Predict waypoints from model
        waypoints, pred_distance = nomad_model.predict_waypoints(goal_image)

        if waypoints is not None:
            # Convert waypoints to Habitat action
            action = action_converter.waypoints_to_action(
                waypoints,
                strategy='index',
                waypoint_index=nomad_model.waypoint_index
            )
            predicted_distances.append(pred_distance)

            if verbose:
                selected_wp = waypoints[0, nomad_model.waypoint_index]
                print(f"Step {steps}: Waypoint=({selected_wp[0]:.2f}, {selected_wp[1]:.2f}), "
                      f"Action={action_converter.get_action_name(action)}, "
                      f"Dist={current_distance:.2f}m, PredDist={pred_distance:.1f}")
        else:
            # Not enough context yet - move forward
            action = HabitatActions.MOVE_FORWARD
            if verbose:
                print(f"Step {steps}: Warming up context... Action=MOVE_FORWARD")

        # Execute action
        observations = env.step(action)
        rgb_obs = observations['rgb']
        actions_taken.append(int(action))

        # Add new observation to context
        nomad_model.add_observation(rgb_obs)

        # Update trajectory
        new_position = env.sim.get_agent_state().position
        trajectory.append(new_position.copy())
        steps += 1

        # Check stopping condition
        done, stop_reason = stopping_criterion.should_stop(
            euclidean_distance_to_goal=current_distance,
            predicted_temporal_distance=pred_distance if waypoints is not None else None,
            steps_taken=steps,
            position_history=trajectory
        )

    # Compute episode metrics
    final_position = env.sim.get_agent_state().position
    distance_to_goal_end = compute_euclidean_distance(final_position, goal_position)
    path_length = compute_path_length(trajectory)
    success = distance_to_goal_end < stopping_criterion.success_distance
    spl = compute_spl(success, path_length, geodesic_distance)

    if verbose:
        print(f"\n{'EPISODE RESULT':^40}")
        print(f"Success: {success}")
        print(f"Stop Reason: {stop_reason}")
        print(f"Final Distance: {distance_to_goal_end:.2f}m")
        print(f"Path Length: {path_length:.2f}m")
        print(f"SPL: {spl:.3f}")
        print(f"Steps: {steps}")

    # Compile episode info
    episode_info = {
        'scene_id': scene_id,
        'episode_id': episode_id,
        'success': success,
        'spl': spl,
        'distance_to_goal_start': float(distance_to_goal_start),
        'distance_to_goal_end': float(distance_to_goal_end),
        'path_length': float(path_length),
        'geodesic_distance': float(geodesic_distance),
        'steps_taken': steps,
        'stop_reason': stop_reason,
        'trajectory': [pos.tolist() for pos in trajectory],
        'predicted_distances': predicted_distances,
        'actions_taken': actions_taken
    }

    return episode_info


def run_evaluation(
    habitat_config_path: str,
    nomad_config_path: str,
    eval_config: Dict[str, Any]
):
    """
    Run full evaluation across multiple HM3D scenes.

    Args:
        habitat_config_path: Path to Habitat configuration YAML
        nomad_config_path: Path to NoMaD configuration YAML
        eval_config: Evaluation configuration dictionary
    """
    # Load NoMaD configuration
    with open(nomad_config_path, 'r') as f:
        nomad_config = yaml.safe_load(f)

    # Initialize NoMaD model
    print("Initializing NoMaD model...")
    nomad_model = NoMaDHabitatWrapper(
        checkpoint_path=nomad_config['checkpoint_path'],
        model_config_path=nomad_config['model_config_path'],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_samples=nomad_config.get('num_samples', 8),
        waypoint_index=nomad_config.get('waypoint_index', 2)
    )

    # Initialize action converter
    action_converter = WaypointToHabitatConverter(
        forward_step=nomad_config.get('forward_step', 0.25),
        turn_angle=nomad_config.get('turn_angle', 10.0),
        distance_threshold=nomad_config.get('distance_threshold', 0.15),
        angle_threshold=nomad_config.get('angle_threshold', 5.0)
    )

    # Initialize stopping criterion
    stopping_criterion = StoppingCriterion(
        success_distance=eval_config.get('success_distance', 0.2),
        temporal_distance_threshold=eval_config.get('temporal_distance_threshold', 3.0),
        max_steps=eval_config.get('max_steps', 500),
        stuck_threshold=eval_config.get('stuck_threshold', 10),
        stuck_distance=eval_config.get('stuck_distance', 0.1)
    )

    # Initialize metrics tracker
    metrics = NavigationMetrics()

    # Get scene list
    scenes = eval_config['hm3d_scenes']
    num_episodes_per_scene = eval_config.get('num_episodes_per_scene', 20)
    log_frequency = eval_config.get('log_frequency', 5)

    print(f"\nStarting evaluation on {len(scenes)} scenes")
    print(f"Episodes per scene: {num_episodes_per_scene}")
    print(f"Total episodes: {len(scenes) * num_episodes_per_scene}\n")

    total_episode_count = 0

    # Iterate over scenes
    for scene_idx, scene_id in enumerate(scenes):
        print(f"\n{'='*60}")
        print(f"Scene {scene_idx + 1}/{len(scenes)}: {scene_id.split('/')[-1]}")
        print(f"{'='*60}")

        # Create environment for this scene
        try:
            env = create_habitat_env(habitat_config_path, scene_id)
        except Exception as e:
            print(f"Error loading scene {scene_id}: {e}")
            continue

        # Run episodes for this scene
        for episode_idx in range(num_episodes_per_scene):
            total_episode_count += 1

            # Generate random episode
            result = generate_random_navigable_positions(
                env.sim,
                min_geodesic_distance=eval_config.get('min_geodesic_distance', 5.0),
                max_geodesic_distance=eval_config.get('max_geodesic_distance', 15.0),
                max_attempts=100
            )

            if result is None:
                print(f"Warning: Could not generate episode {episode_idx} for scene {scene_id}")
                continue

            start_position, goal_position, geodesic_distance = result

            # Capture goal image
            goal_image = capture_goal_image(env, goal_position)

            # Run episode
            verbose = (total_episode_count % log_frequency == 0)
            episode_info = run_episode(
                env=env,
                nomad_model=nomad_model,
                action_converter=action_converter,
                stopping_criterion=stopping_criterion,
                start_position=start_position,
                goal_position=goal_position,
                goal_image=goal_image,
                episode_id=total_episode_count,
                scene_id=scene_id,
                verbose=verbose
            )

            # Record metrics
            metrics.add_episode(episode_info)

            # Periodic summary
            if total_episode_count % log_frequency == 0:
                current_metrics = metrics.compute_aggregate_metrics()
                print(f"\n[Progress] {total_episode_count} episodes completed")
                print(f"Current Success Rate: {current_metrics['success_rate']:.1%}")
                print(f"Current SPL: {current_metrics['spl']:.3f}")

        # Close environment for this scene
        env.close()

    # Final summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    metrics.print_summary()

    # Save results
    output_dir = eval_config.get('output_dir', 'results')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"{eval_config.get('experiment_name', 'nomad_hm3d')}_results.json"
    )
    metrics.save_results(output_path)

    return metrics


def main():
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate NoMaD on Habitat ImageNav with HM3D scenes"
    )
    parser.add_argument(
        '--habitat-config',
        type=str,
        default='config/habitat.yaml',
        help='Path to Habitat configuration file'
    )
    parser.add_argument(
        '--nomad-config',
        type=str,
        default='config/nomad.yaml',
        help='Path to NoMaD model configuration file'
    )
    parser.add_argument(
        '--eval-config',
        type=str,
        default='config/evaluation.yaml',
        help='Path to evaluation configuration file'
    )
    args = parser.parse_args()

    # Load evaluation configuration
    with open(args.eval_config, 'r') as f:
        eval_config = yaml.safe_load(f)

    # Run evaluation
    run_evaluation(
        habitat_config_path=args.habitat_config,
        nomad_config_path=args.nomad_config,
        eval_config=eval_config
    )


if __name__ == '__main__':
    main()
