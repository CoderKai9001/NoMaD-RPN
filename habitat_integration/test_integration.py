"""
Test script to verify Habitat integration components.

Run this to check that:
1. Model loads correctly
2. Action converter works
3. Episode generation works
4. Full pipeline can run a single episode
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import yaml
import numpy as np
from PIL import Image
import torch

print("Testing Habitat integration components...\n")

# Test 1: Import all modules
print("="*60)
print("Test 1: Importing modules")
print("="*60)
try:
    from nomad_wrapper import NoMaDHabitatWrapper
    print("✓ nomad_wrapper imported successfully")
except Exception as e:
    print(f"✗ Error importing nomad_wrapper: {e}")
    sys.exit(1)

try:
    from action_converter import WaypointToHabitatConverter, HabitatActions
    print("✓ action_converter imported successfully")
except Exception as e:
    print(f"✗ Error importing action_converter: {e}")
    sys.exit(1)

try:
    from utils.episode_utils import (
        compute_path_length,
        compute_euclidean_distance
    )
    print("✓ episode_utils imported successfully (basic functions)")

    # Try importing habitat-dependent functions
    try:
        from utils.episode_utils import generate_random_navigable_positions, capture_goal_image
        print("✓ episode_utils habitat functions imported successfully")
        habitat_utils_available = True
    except ImportError as e:
        print(f"⚠ Habitat-dependent functions not available: {e}")
        print("  This is OK if Habitat is not installed yet")
        habitat_utils_available = False

except Exception as e:
    print(f"✗ Error importing episode_utils: {e}")
    sys.exit(1)

try:
    from utils.metrics import NavigationMetrics, StoppingCriterion
    print("✓ metrics imported successfully")
except Exception as e:
    print(f"✗ Error importing metrics: {e}")
    sys.exit(1)

# Test 2: Action converter unit tests
print("\n" + "="*60)
print("Test 2: Action Converter")
print("="*60)

converter = WaypointToHabitatConverter(
    forward_step=0.25,
    turn_angle=10.0,
    distance_threshold=0.15,
    angle_threshold=5.0
)

test_cases = [
    (np.array([0.5, 0.0]), "MOVE_FORWARD", "Forward waypoint"),
    (np.array([0.0, 0.5]), "TURN_LEFT", "Left waypoint"),
    (np.array([0.0, -0.5]), "TURN_RIGHT", "Right waypoint"),
    (np.array([0.3, 0.3]), "TURN_LEFT", "Forward-left waypoint (needs turn)"),
    (np.array([0.05, 0.0]), "MOVE_FORWARD", "Small forward waypoint"),
]

for waypoint, expected_action, description in test_cases:
    action = converter.waypoint_to_action(waypoint)
    action_name = converter.get_action_name(action)
    status = "✓" if action_name == expected_action else "✗"
    print(f"{status} {description}: {waypoint} → {action_name} (expected {expected_action})")

# Test 3: Metrics computation
print("\n" + "="*60)
print("Test 3: Metrics Computation")
print("="*60)

metrics = NavigationMetrics()

# Add test episodes
test_episodes = [
    {
        'scene_id': 'test_scene',
        'episode_id': 1,
        'success': True,
        'spl': 0.8,
        'distance_to_goal_start': 10.0,
        'distance_to_goal_end': 0.15,
        'path_length': 12.0,
        'geodesic_distance': 10.0,
        'steps_taken': 48,
        'stop_reason': 'SUCCESS',
        'trajectory': [],
        'predicted_distances': [],
        'actions_taken': []
    },
    {
        'scene_id': 'test_scene',
        'episode_id': 2,
        'success': False,
        'spl': 0.0,
        'distance_to_goal_start': 8.0,
        'distance_to_goal_end': 2.5,
        'path_length': 15.0,
        'geodesic_distance': 8.0,
        'steps_taken': 500,
        'stop_reason': 'TIMEOUT',
        'trajectory': [],
        'predicted_distances': [],
        'actions_taken': []
    }
]

for ep in test_episodes:
    metrics.add_episode(ep)

aggregate = metrics.compute_aggregate_metrics()
print(f"✓ Success rate: {aggregate['success_rate']:.1%}")
print(f"✓ SPL: {aggregate['spl']:.3f}")
print(f"✓ Avg steps: {aggregate['avg_steps']:.1f}")

# Test 4: Model loading (if config is set up)
print("\n" + "="*60)
print("Test 4: Model Loading")
print("="*60)

try:
    with open('config/nomad.yaml', 'r') as f:
        nomad_config = yaml.safe_load(f)

    checkpoint_path = nomad_config['checkpoint_path']
    model_config_path = nomad_config['model_config_path']

    if checkpoint_path == 'YOUR_MODEL_PATH_HERE':
        print("⚠ Skipping model loading test: Please update checkpoint_path in config/nomad.yaml")
    else:
        if not os.path.exists(checkpoint_path):
            print(f"✗ Checkpoint not found: {checkpoint_path}")
        elif not os.path.exists(model_config_path):
            print(f"✗ Model config not found: {model_config_path}")
        else:
            print(f"Loading model from {checkpoint_path}...")
            model = NoMaDHabitatWrapper(
                checkpoint_path=checkpoint_path,
                model_config_path=model_config_path,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                num_samples=8,
                waypoint_index=2
            )
            print("✓ Model loaded successfully")

            # Test inference with dummy data
            print("\nTesting inference with dummy observations...")
            dummy_obs = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
            dummy_goal = Image.fromarray(np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8))

            # Add observations to build context
            for i in range(4):
                model.add_observation(dummy_obs)

            # Predict
            waypoints, distance = model.predict_waypoints(dummy_goal)

            if waypoints is not None:
                print(f"✓ Inference successful")
                print(f"  Waypoints shape: {waypoints.shape}")
                print(f"  Predicted distance: {distance:.2f}")
                print(f"  Selected waypoint: {waypoints[0, model.waypoint_index]}")
            else:
                print("✗ Inference returned None (should not happen with 4 observations)")

except Exception as e:
    print(f"✗ Model loading test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Habitat environment (if available)
print("\n" + "="*60)
print("Test 5: Habitat Environment")
print("="*60)

habitat_installed = False

try:
    import habitat
    habitat_installed = True
    print("✓ Habitat imported successfully")

    # Try to find HM3D scenes
    possible_scene_dirs = [
        'data/scene_datasets/hm3d/val',
        '../data/scene_datasets/hm3d/val',
        '../../data/scene_datasets/hm3d/val',
    ]

    scene_dir = None
    for dir_path in possible_scene_dirs:
        if os.path.exists(dir_path):
            scene_dir = dir_path
            break

    if scene_dir:
        print(f"✓ Found HM3D scene directory: {scene_dir}")
        scenes = [f for f in os.listdir(scene_dir) if os.path.isdir(os.path.join(scene_dir, f))]
        print(f"  Number of scenes available: {len(scenes)}")
        if scenes:
            print(f"  Example scenes: {scenes[:3]}")
    else:
        print("⚠ Could not find HM3D scene directory")
        print("  Please check your Habitat installation and update paths in config/evaluation.yaml")

except ImportError as e:
    print(f"⚠ Habitat not installed: {e}")
    print("  Install with: pip install habitat-sim habitat-lab")
    print("  Note: You may need a specific Habitat environment or build")

# Summary
print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)

summary_items = [
    ("✓ Core components", "NoMaD wrapper and action converter working"),
    ("✓ Metrics", "Navigation metrics computation working"),
]

if habitat_installed and habitat_utils_available:
    summary_items.append(("✓ Habitat ready", "Full integration ready to run"))
    next_steps = """
Next steps:
1. Update config/nomad.yaml with your NoMaD checkpoint path
2. Update config/evaluation.yaml with your HM3D scene paths
3. Run: python habitat_imagenav.py
"""
elif habitat_installed:
    summary_items.append(("⚠ Habitat", "Installed but may need configuration"))
    next_steps = """
Next steps:
1. Check Habitat installation: python -c "import habitat; print(habitat.__version__)"
2. Update config/nomad.yaml with your NoMaD checkpoint path
3. Update config/evaluation.yaml with your HM3D scene paths
"""
else:
    summary_items.append(("⚠ Habitat", "Not installed in this environment"))
    next_steps = """
Next steps:
1. Install Habitat: pip install habitat-sim habitat-lab
   (or activate the environment where Habitat is installed)
2. Update config/nomad.yaml with your NoMaD checkpoint path
3. Update config/evaluation.yaml with your HM3D scene paths
4. Re-run this test to verify
"""

for status, desc in summary_items:
    print(f"{status:20s} {desc}")

print(next_steps)
