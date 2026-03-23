import os
import sys
import yaml
import torch
import numpy as np
from PIL import Image
import habitat_sim

# Add parent directory to path to access local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from habitat_integration.nomad_wrapper import NoMaDHabitatWrapper
from habitat_integration.action_converter import WaypointToHabitatConverter, HabitatActions
from habitat_integration.utils.episode_utils import compute_euclidean_distance


def create_headless_env(scene_id: str):
    """
    Create a headless Habitat Simulator instance for NoMaD.
    """
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_id
    backend_cfg.enable_physics = False
    
    # Sensor Configuration (matching LoCoBot)
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
        move_backward=habitat_sim.agent.ActionSpec(
            "move_backward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        turn_left=habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
        turn_right=habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
    )
    
    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    return sim


def format_observation(obs):
    """Convert Habitat sensor observation to RGB PIL Image cleanly."""
    rgb = obs['rgb']
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]  # Drop alpha channel
    return Image.fromarray(rgb.astype(np.uint8))


def explore_environment(sim, num_steps=150, save_interval=10, frames_list=None):
    """
    Exploration Phase: Execute a random walk, avoiding obstacles where possible,
    and build a topological 'map' of images and poses.
    """
    print(f"\n--- EXPLORATION PHASE ({num_steps} steps) ---")
    topomap = []
    
    # Initialize position to a random navigable point
    start_pos = sim.pathfinder.get_random_navigable_point()
    agent = sim.get_agent(0)
    state = agent.get_state()
    state.position = start_pos
    agent.set_state(state)
    
    obs = sim.get_sensor_observations()
    if frames_list is not None:
        frames_list.append(np.array(format_observation(obs)))
    
    for step in range(num_steps):
        # Prefer moving forward, but turn if we hit an obstacle (naively detected by lack of movement)
        prev_pos = agent.get_state().position
        
        # Simple exploration heuristic: move forward mostly, turn occasionally
        if np.random.rand() < 0.2:
            action = np.random.choice(["turn_left", "turn_right"])
        else:
            action = "move_forward"
            
        obs = sim.step(action)
        curr_pos = agent.get_state().position
        
        # If we tried to move forward but didn't move much, we hit a wall -> Turn
        if action == "move_forward" and compute_euclidean_distance(prev_pos, curr_pos) < 0.05:
            obs = sim.step(np.random.choice(["turn_left", "turn_right"]))
            curr_pos = agent.get_state().position

        if frames_list is not None:
            frames_list.append(np.array(format_observation(obs)))
            
        # Save node to topological map
        if step % save_interval == 0:
            pil_img = format_observation(obs)
            topomap.append({
                "id": len(topomap),
                "image": pil_img,
                "position": curr_pos.copy(),
                "rotation": agent.get_state().rotation
            })
            print(f"[Explore] Saved Topomap Node {len(topomap)-1} at position {curr_pos}")

    print(f"Exploration complete. Topomap contains {len(topomap)} nodes.")
    return topomap

def load_topomap_from_data(data_dir):
    import quaternion
    print(f"\n--- LOADING TOPOMAP FROM DATA ({data_dir}) ---")
    topomap = []
    
    poses_file = os.path.join(data_dir, "poses", "poses_odom.txt")
    images_dir = os.path.join(data_dir, "images_fov90")
    
    with open(poses_file, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        if line.startswith("#"):
            continue
        parts = line.strip().split()
        if len(parts) >= 8:
            frame_idx = int(parts[0])
            position = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            
            # Habitat expects quaternion. quaternion uses w, x, y, z
            rot_quat = np.quaternion(float(parts[7]), float(parts[4]), float(parts[5]), float(parts[6]))
            
            img_path = os.path.join(images_dir, f"{frame_idx:05d}.jpg")
            if os.path.exists(img_path):
                pil_img = Image.open(img_path).convert("RGB")
                
                # Resize image to match the sensor expectation (160x120 matches the NoMaD config width x height)
                pil_img = pil_img.resize((160, 120))
                
                topomap.append({
                    "id": len(topomap),
                    "image": pil_img,
                    "position": position,
                    "rotation": rot_quat,
                    "frame_idx": frame_idx
                })
                print(f"[Data] Loaded Topomap Node {len(topomap)-1} from frame {frame_idx} at {position}")
                
    print(f"Loading complete. Topomap contains {len(topomap)} nodes.")
    return topomap

def navigate_environment(sim, nomad_model, topomap, start_frame=1, target_frame=20, frames_list=None, enable_recovery=True):
    """
    Navigation Phase: Route back to a specific target in the topological map 
    from a completely different starting location.
    """
    # Find start and target nodes by frame_idx, fallback to list index
    start_node = next((n for n in topomap if n.get('frame_idx') == start_frame), topomap[1] if len(topomap) > 1 else topomap[0])
    target_node = next((n for n in topomap if n.get('frame_idx') == target_frame), topomap[20] if len(topomap) > 20 else topomap[-1])

    print(f"\n--- NAVIGATION PHASE ---")
    print(f"Targeting Node {target_node['id']} (Frame {target_node.get('frame_idx')}) at {target_node['position']}")

    # Teleport to the start node to simulate a return trip
    agent = sim.get_agent(0)
    state = agent.get_state()
    state.position = start_node['position']
    state.rotation = start_node['rotation']
    agent.set_state(state)
    print(f"Teleported agent to start position (Node {start_node['id']}, Frame {start_node.get('frame_idx')}): {state.position}")

    # Setup Navigation
    action_converter = WaypointToHabitatConverter()
    nomad_model.reset_context()

    # Get initial context
    obs = sim.get_sensor_observations()
    pil_img = format_observation(obs)
    if frames_list is not None:
        frames_list.append(np.array(pil_img))
    
    # Warmup context queue
    for _ in range(nomad_model.context_size):
        nomad_model.add_observation(np.array(pil_img))

    steps = 0
    max_steps = 1000
    goal_reached = False

    while steps < max_steps:
        # Distance to goal
        curr_pos = agent.get_state().position
        dist_to_goal = compute_euclidean_distance(curr_pos, target_node['position'])
        
        if dist_to_goal < 0.05:
            print(f"Goal reached! Final distance: {dist_to_goal:.2f}m")
            goal_reached = True
            break

        # Predict waypoint
        try:
            waypoints, pred_distance = nomad_model.predict_waypoints(target_node['image'])
        except Exception as e:
            print(f"Prediction failed: {e}")
            break

        # Convert waypoint to action
        if nomad_model.has_sufficient_context():
            action = action_converter.waypoints_to_action(
                waypoints,
                strategy='index',
                waypoint_index=nomad_model.waypoint_index
            )
        else:
            action = HabitatActions.MOVE_FORWARD

        # Step Simulator
        action_name = action_converter.get_action_name(action).lower()
        prev_pos = agent.get_state().position
        
        obs = sim.step(action_name)
        curr_pos = agent.get_state().position
        
        # Obstacle Recovery
        # A move_forward action normally moves 0.25m. If we move less than 0.15m, we are sliding against geometry.
        if enable_recovery and action_name == "move_forward" and compute_euclidean_distance(prev_pos, curr_pos) < 0.15:
            print(f"[Recovery] Collision detected at step {steps} (Moved {compute_euclidean_distance(prev_pos, curr_pos):.2f}m). Executing reverse maneuver...")
            
            # Step backward to disengage from collision geometry
            obs = sim.step("move_backward")
            if frames_list is not None:
                frames_list.append(np.array(format_observation(obs)))
                
            # Turn slightly to look for a clear path
            recovery_turn = np.random.choice(["turn_left", "turn_right"])
            for _ in range(3): 
                obs = sim.step(recovery_turn)
                if frames_list is not None:
                    frames_list.append(np.array(format_observation(obs)))
        
        # Add new observation
        new_img = format_observation(obs)
        if frames_list is not None:
            frames_list.append(np.array(new_img))
        nomad_model.add_observation(np.array(new_img))
        
        steps += 1
        if steps % 10 == 0:
            print(f"Nav Step {steps}: Dist to goal = {dist_to_goal:.2f}m, Pred Dist = {pred_distance:.2f}, Action = {action_name}")

    if not goal_reached:
        print(f"Navigation timed out after {steps} steps. Final distance: {dist_to_goal:.2f}m")
        
    return target_node


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Navigate using NoMaD and a Topomap")
    parser.add_argument("--start-frame", type=int, default=1, help="Start frame ID")
    parser.add_argument("--target-frame", type=int, default=70, help="Target/Goal frame ID")
    parser.add_argument("--no-recovery", action="store_true", help="Disable collision recovery maneuver")
    parser.add_argument("--scene-id", type=str, default="/scratch/aditya.vadali/00011-1W61QJVDBqe/1W61QJVDBqe.basis.glb", help="Path to scene GLB")
    parser.add_argument("--data-dir", type=str, default="/scratch/aditya.vadali/data/1W61QJVDBqe", help="Path to topomap data")
    args = parser.parse_args()

    # Model Configuration
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nomad_config_path = os.path.join(base_dir, "habitat_integration/config/nomad.yaml")
    
    with open(nomad_config_path, 'r') as f:
        nomad_config = yaml.safe_load(f)
        
    model_path = os.path.join(base_dir, "deployment/model_weights/nomad.pth")
    model_config_path = os.path.join(base_dir, "train/config/nomad.yaml")
    
    scene_id = args.scene_id
    
    print("Loading Headless Habitat Environment...")
    sim = create_headless_env(scene_id)
    
    print(f"Loading NoMaD Model from {model_path}...")
    nomad_model = NoMaDHabitatWrapper(
        checkpoint_path=model_path,
        model_config_path=model_config_path,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_samples=nomad_config.get('num_samples', 8),
        waypoint_index=nomad_config.get('waypoint_index', 2)
    )
    
    video_frames = []
    
    # 1. Load Data
    data_dir = args.data_dir
    topomap = load_topomap_from_data(data_dir)
    
    # 2. Navigation
    target_node = navigate_environment(
        sim, 
        nomad_model, 
        topomap, 
        start_frame=args.start_frame, 
        target_frame=args.target_frame, 
        frames_list=video_frames,
        enable_recovery=not args.no_recovery
    )
    
    sim.close()
    
    # Save Video and Images
    import cv2
    out_dir = "/scratch/aditya.vadali/results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "nomad_trajectory.mp4")
    
    # Save target/goal image
    target_img_path = os.path.join(out_dir, "target_image.jpg")
    if target_node is not None:
        target_node['image'].save(target_img_path)
        print(f"Goal image saved to {target_img_path}")
    
    # Save start image
    start_img_path = os.path.join(out_dir, "start_image.jpg")
    if len(video_frames) > 0:
        # Save first frame of video sequence as start image
        start_img = Image.fromarray(video_frames[0])
        start_img.save(start_img_path)
        print(f"Start image saved to {start_img_path}")
        
        print(f"Saving video with {len(video_frames)} frames to {out_path}...")
        h, w = video_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, 10.0, (w, h))
        for frame in video_frames:
            # OpenCV expects BGR
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        print("Video saved successfully.")
    
    print("Done!")

if __name__ == "__main__":
    main()
