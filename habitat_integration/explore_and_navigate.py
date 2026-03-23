import os
import sys
import yaml
import torch
import numpy as np
from PIL import Image
import habitat_sim
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.10)
        ),
        move_backward=habitat_sim.agent.ActionSpec(
            "move_backward", habitat_sim.agent.ActuationSpec(amount=0.10)
        ),
        turn_left=habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=4.0)
        ),
        turn_right=habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=4.0)
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


def render_stitched_frame(rgb_img, waypoints=None, waypoint_index=2):
    """
    Renders the RGB image side-by-side with a top-down trajectory plot of the waypoints.
    """
    rgb_img_resized = cv2.resize(rgb_img, (320, 240))
    fig, ax = plt.subplots(figsize=(3.2, 2.4), dpi=100)
    ax.set_title("Predicted Waypoints")
    ax.grid(True)
    
    if waypoints is not None:
        for i in range(waypoints.shape[0]):
            ax.plot(-waypoints[i, :, 1], waypoints[i, :, 0], color='cyan', alpha=0.3)
        ax.plot(-waypoints[0, :, 1], waypoints[0, :, 0], color='blue', linewidth=2, label='Trajectory')
        ax.scatter(-waypoints[0, :, 1], waypoints[0, :, 0], color='blue', s=20, label='Waypoints', zorder=4)
        selected_wp = waypoints[0, waypoint_index]
        ax.scatter(-selected_wp[1], selected_wp[0], color='red', s=40, label='Target Waypoint', zorder=5)
        
    ax.scatter(0, 0, color='black', marker='^', s=80, label='Robot', zorder=10)
    
    # Base 2m x 2m window (-1m to 1m left/right, -0.2m to 1.8m forward)
    x_bound = 1.0
    y_bound_max = 1.8
    y_bound_min = -0.2
    
    # Auto-scale if waypoints go beyond the 2m x 2m box
    if waypoints is not None:
        max_x = np.max(np.abs(-waypoints[:, :, 1]))
        max_y = np.max(waypoints[:, :, 0])
        min_y = np.min(waypoints[:, :, 0])
        
        x_bound = max(x_bound, max_x + 0.2)
        y_bound_max = max(y_bound_max, max_y + 0.2)
        y_bound_min = min(y_bound_min, min_y - 0.2)

    ax.set_xlim(-x_bound, x_bound)
    ax.set_ylim(y_bound_min, y_bound_max)
    ax.set_xlabel("Left/Right (m)")
    ax.set_ylabel("Forward (m)")
    if waypoints is not None:
        ax.legend(loc='upper right', fontsize='x-small')
    
    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    plot_img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig)
    
    if plot_img.shape[0] != rgb_img_resized.shape[0]:
        plot_img = cv2.resize(plot_img, (int(plot_img.shape[1] * rgb_img_resized.shape[0] / plot_img.shape[0]), rgb_img_resized.shape[0]))
        
    return np.hstack((rgb_img_resized, plot_img))


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
        
    image_idx = 0
    for line in lines:
        if line.startswith("#"):
            continue
        parts = line.strip().split()
        if len(parts) >= 8:
            frame_idx = int(parts[0])
            position = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            
            # Habitat expects quaternion. quaternion uses w, x, y, z
            rot_quat = np.quaternion(float(parts[7]), float(parts[4]), float(parts[5]), float(parts[6]))
            
            # Map sequential rows directly to corresponding sequential image files (0.jpg, 1.jpg, etc.)
            img_path = os.path.join(images_dir, f"{image_idx:05d}.jpg")
            if os.path.exists(img_path):
                pil_img = Image.open(img_path).convert("RGB")
                
                # Resize image to match the sensor expectation (160x120 matches the NoMaD config width x height)
                pil_img = pil_img.resize((160, 120))
                
                topomap.append({
                    "id": len(topomap),
                    "image": pil_img,
                    "position": position,
                    "rotation": rot_quat,
                    "frame_idx": image_idx  # We use the logical image index to lookup start/target frames
                })
                print(f"[Data] Loaded Topomap Node {len(topomap)-1} (Image {image_idx}.jpg, Pose {frame_idx}) at {position}")
            
            image_idx += 1
                
    print(f"Loading complete. Topomap contains {len(topomap)} nodes.")
    return topomap

def navigate_environment(sim, nomad_model, topomap, start_frame=1, target_frame=20, frames_list=None, enable_recovery=True, step_size=0.10, diff_drive=True):
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
    
    # Warmup context queue
    for _ in range(nomad_model.context_size):
        nomad_model.add_observation(np.array(pil_img))

    steps = 0
    max_steps = 500
    goal_reached = False

    while steps < max_steps:
        # Distance to goal
        curr_pos = agent.get_state().position
        dist_to_goal = compute_euclidean_distance(curr_pos, target_node['position'])
        
        if dist_to_goal < 0.8:
            print(f"Goal reached! Final distance: {dist_to_goal:.2f}m")
            goal_reached = True
            break

        # Predict waypoint
        try:
            waypoints, pred_distance = nomad_model.predict_waypoints(target_node['image'])
        except Exception as e:
            print(f"Prediction failed: {e}")
            break

        # Save visualized frame using current context and predicted waypoints
        if frames_list is not None and waypoints is not None:
            current_obs_arr = np.array(nomad_model.context_queue[-1])
            frames_list.append(render_stitched_frame(current_obs_arr, waypoints, nomad_model.waypoint_index))

        if diff_drive:
            # Differential Drive execution
            # Waypoints are in egocentric frame: x is forward (m), y is left (m)
            if nomad_model.has_sufficient_context() and waypoints is not None:
                target_wp = waypoints[0, nomad_model.waypoint_index]
            else:
                target_wp = np.array([step_size, 0.0]) # just move forward slightly
                
            dx, dy = target_wp[0], target_wp[1]
            
            # Max forward step is step_size. Max turn is 15 degrees per step.
            v = min(abs(dx), step_size) * np.sign(dx)
            angle = np.arctan2(dy, dx)
            omega = np.clip(angle, -np.radians(15.0), np.radians(15.0))
            
            import habitat_sim.utils.common as sim_utils
            state = agent.get_state()
            prev_pos = state.position.copy()
            
            # 1. Turn
            rot_offset = sim_utils.quat_from_angle_axis(omega, np.array([0.0, 1.0, 0.0]))
            new_rotation = state.rotation * rot_offset
            
            # 2. Translate forward along new look axis. In Habitat, -Z is forward.
            forward_dir = sim_utils.quat_rotate_vector(new_rotation, np.array([0.0, 0.0, -1.0]))
            desired_pos = prev_pos + forward_dir * v
            
            # 3. Step utilizing Pathfinder to respect NavMesh obstacles
            new_pos = sim.pathfinder.try_step(prev_pos, desired_pos)
            
            # 4. Apply State
            state.position = new_pos
            state.rotation = new_rotation
            agent.set_state(state)
            
            # 5. Extract fresh observation from the world
            obs = sim.get_sensor_observations()
            curr_pos = new_pos
            action_name = f"diff_drive(v={v:.2f}, w={np.degrees(omega):.1f}°)"
            v_check = v
            
        else:
            # Plain Actions execution (discrete)
            if nomad_model.has_sufficient_context() and waypoints is not None:
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
            v_check = step_size if action_name == "move_forward" else 0.0
            
        # Obstacle Recovery
        # If we asked to move forward (v > 0.05m) but moved notably less than the intended step size, we are blocked.
        if enable_recovery and v_check > 0.05 and compute_euclidean_distance(prev_pos, curr_pos) < (v_check * 0.2):
            print(f"[Recovery] Collision detected at step {steps} (Moved {compute_euclidean_distance(prev_pos, curr_pos):.2f}m). Executing reverse maneuver...")
            
            # Step backward to disengage from collision geometry
            obs = sim.step("move_backward")
            if frames_list is not None:
                frames_list.append(render_stitched_frame(np.array(format_observation(obs)), None))
                
            # Turn slightly to look for a clear path
            recovery_turn = np.random.choice(["turn_left", "turn_right"])
            for _ in range(3): 
                obs = sim.step(recovery_turn)
                if frames_list is not None:
                    frames_list.append(render_stitched_frame(np.array(format_observation(obs)), None))
        
        # Add new observation
        new_img = format_observation(obs)
        nomad_model.add_observation(np.array(new_img))
        
        steps += 1
        if steps % 10 == 0:
            print(f"Nav Step {steps}: Dist to goal = {dist_to_goal:.2f}m, Pred Dist = {pred_distance:.2f}, Action = {action_name}")

    if not goal_reached:
        print(f"Navigation timed out after {steps} steps. Final distance: {dist_to_goal:.2f}m")
        
    if goal_reached and frames_list is not None:
        final_obs = sim.get_sensor_observations()
        frames_list.append(render_stitched_frame(np.array(format_observation(final_obs)), None))
        
    return target_node


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Navigate using NoMaD and a Topomap")
    parser.add_argument("--start-frame", type=int, default=1, help="Start frame ID")
    parser.add_argument("--target-frame", type=int, default=70, help="Target/Goal frame ID")
    parser.add_argument("--no-recovery", action="store_true", help="Disable collision recovery maneuver")
    parser.add_argument("--scene-id", type=str, default="/scratch/aditya.vadali/data/1W61QJVDBqe/1W61QJVDBqe.basis.glb", help="Path to scene GLB")
    parser.add_argument("--data-dir", type=str, default="/scratch/aditya.vadali/data/1W61QJVDBqe", help="Path to topomap data")
    parser.add_argument("--topomap-stride", type=int, default=1, help="Stride multiplier for subsampling topomap nodes")
    parser.add_argument("--step-size", type=float, default=0.25, help="Simulation translation target step size in meters")
    parser.add_argument("--plain-actions", action="store_true", help="Fallback to discrete habitat step actions")
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
        enable_recovery=not args.no_recovery,
        step_size=args.step_size,
        diff_drive=not args.plain_actions
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
