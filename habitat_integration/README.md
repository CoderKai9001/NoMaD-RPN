# Habitat Integration for NoMaD-RPN

Run NoMaD visual navigation models in Habitat simulator on HM3D scenes using the ImageNav task.

## Quick Start

```bash
# 1. Activate environment and install dependencies
conda activate nomad_train
pip install habitat-sim habitat-lab
pip install "huggingface_hub<0.19.0"  # Fix diffusers compatibility

# 2. Update configs (see Configuration section below)
# - config/nomad.yaml: Set checkpoint_path
# - config/evaluation.yaml: Set hm3d_scenes paths

# 3. Test integration
cd habitat_integration
python test_integration.py

# 4. Run evaluation
python habitat_imagenav.py
```

## Overview

This integration allows you to evaluate pre-trained NoMaD models in simulation:
- **Task:** ImageNav (navigate to a goal specified by an image)
- **Scenes:** HM3D (indoor scenes)
- **Model:** Pre-trained NoMaD with diffusion-based action generation
- **Action space:** Converts NoMaD's continuous waypoints to Habitat's discrete actions

## Setup

### Prerequisites

1. **Activate the NoMaD conda environment:**
   ```bash
   conda activate nomad_train
   ```

2. **Install/verify dependencies:**
   ```bash
   conda activate nomad_train
   pip install habitat-sim habitat-lab

   # Fix diffusers compatibility issue 
   pip install "huggingface_hub<0.19.0"

   # Or install all requirements at once
   pip install -r requirements.txt
   ```

   **Note:** The nomad_train environment has diffusers==0.11.1 which requires an older huggingface_hub version. If you see import errors related to `cached_download`, downgrade huggingface_hub.

   Look at `ENV_SETUP.md` for environment setup details.

3. **Habitat and HM3D installed** 

4. **Pre-trained NoMaD model checkpoint** 

### Configuration

Before running, update the following configuration files:

#### 1. `config/nomad.yaml`
```yaml
checkpoint_path: /path/to/your/nomad_checkpoint.pth  # UPDATE THIS
```

#### 2. `config/evaluation.yaml`
Update the HM3D scene paths to match your installation:
```yaml
hm3d_scenes:
  - /path/to/hm3d/val/scene1/scene1.basis.glb  # UPDATE THESE
  - /path/to/hm3d/val/scene2/scene2.basis.glb
  - /path/to/hm3d/val/scene3/scene3.basis.glb
```

To find your HM3D scenes:
```bash
# Example: Find all HM3D validation scenes
find /path/to/habitat/data/scene_datasets/hm3d/val -name "*.basis.glb"
```

## Usage

### Test the Integration First

Before running a full evaluation, verify everything works:

```bash
conda activate nomad_train
cd habitat_integration
python test_integration.py
```

This will check:
- All modules import correctly
- Action converter works
- Metrics computation works
- Model can be loaded (if checkpoint path is set)

### Basic Evaluation

Run evaluation on all configured scenes:

```bash
conda activate nomad_train
cd habitat_integration
python habitat_imagenav.py
```

### Visual Output & Topological Exploration

To run a specific sequence visually and generate `.mp4` tracking outputs from loaded topological maps, use the `explore_and_navigate.py` script:

```bash
python explore_and_navigate.py --start-frame 1 --target-frame 70
```

Options:
- `--start-frame`: ID of the frame to spawn the agent at (default: 1).
- `--target-frame`: ID of the goal image frame (default: 70).
- `--no-recovery`: Disable backward-step collision recovery maneuvers.
- `--scene-id`: Custom `.glb` scene path.
- `--data-dir`: Custom path containing `/poses/` and `/images_fov90/` structure.

### Custom Configuration

Override default config files:

```bash
python habitat_imagenav.py \
  --habitat-config config/habitat.yaml \
  --nomad-config config/nomad.yaml \
  --eval-config config/evaluation.yaml
```

### Quick Test

For a quick test on 1 scene with 3 episodes, edit `config/evaluation.yaml`:
```yaml
num_episodes_per_scene: 3
hm3d_scenes:
  - /path/to/single/scene.glb  # Just one scene
```

## Configuration Guide

### Action Converter Tuning

If the agent's behavior seems off, tune these parameters in `config/nomad.yaml`:

- `distance_threshold` (default: 0.15): Minimum distance to waypoint to move forward
  - Increase if agent moves forward too eagerly
  - Decrease if agent turns too much

- `angle_threshold` (default: 5°): Minimum angle to trigger fine turning
  - Increase for less turning, straighter paths
  - Decrease for more precise alignment

- `waypoint_index` (default: 2): Which waypoint to execute (0-7)
  - Lower values (0-1): More reactive, shorter lookahead
  - Higher values (3-4): More planning, longer lookahead
  - Default 2 matches robot deployment

### Episode Generation

Adjust difficulty in `config/evaluation.yaml`:

- `min_geodesic_distance`: Shorter = easier episodes
- `max_geodesic_distance`: Longer = harder episodes
- Default range (5-15m) provides moderate difficulty

### Stopping Criteria

Tune when episodes terminate in `config/evaluation.yaml`:

- `success_distance` (default: 0.2m): Standard ImageNav success threshold
- `temporal_distance_threshold` (default: 3): Model's predicted distance threshold
- `max_steps` (default: 500): Timeout
- `stuck_threshold` (default: 10): Steps to check for stuck detection

## Output

Results are saved to `results/<experiment_name>_results.json`:

```json
{
  "aggregate_metrics": {
    "num_episodes": 15,
    "success_rate": 0.467,
    "spl": 0.312,
    "avg_path_length": 8.45,
    "avg_steps": 67.2,
    ...
  },
  "per_episode": [
    {
      "scene_id": "...",
      "episode_id": 1,
      "success": true,
      "spl": 0.654,
      "path_length": 7.23,
      "steps_taken": 58,
      "stop_reason": "SUCCESS",
      "trajectory": [[x1,y1,z1], [x2,y2,z2], ...],
      ...
    },
    ...
  ]
}
```

**Key metrics:**
- **Success Rate:** % of episodes reaching within 0.2m of goal
- **SPL:** Success weighted by path efficiency (higher is better)
- **Path Length:** Average distance traveled
- **Steps:** Average number of actions taken

## Troubleshooting

### Import errors for deployment/train modules

The code adds paths automatically, but if you get import errors:
```python
export PYTHONPATH="${PYTHONPATH}:/home2/aditya.vadali/NoMaD-RPN/deployment/src"
export PYTHONPATH="${PYTHONPATH}:/home2/aditya.vadali/NoMaD-RPN/train"
```

### Model checkpoint not found

Verify your checkpoint path:
```bash
ls -lh /path/to/your/nomad_checkpoint.pth
```

### Scene files not found

Check your HM3D installation:
```bash
ls data/scene_datasets/hm3d/val/
```

Habitat expects scenes in: `data/scene_datasets/` relative to working directory, or absolute paths.

### Low success rate

Expected due to domain shift (NoMaD trained on outdoor robots, testing on indoor HM3D):
- Success rate > 5% is reasonable for zero-shot transfer
- Success rate 20-40% would be quite good
- Can improve by fine-tuning on Habitat data (not implemented yet)

### Agent gets stuck

Try tuning:
- Increase `distance_threshold` to move forward more aggressively
- Decrease `angle_threshold` for more precise turns
- Try different `waypoint_index` values (1 for more reactive, 3 for more planning)

## Architecture

**Data flow:**
1. Habitat provides RGB observation (160×120×3)
2. NoMaD wrapper maintains context queue (3 frames)
3. Model predicts 8 waypoint trajectories via diffusion
4. Select waypoint at index 2 from first sample
5. Action converter maps waypoint (δx, δy) → discrete action
6. Habitat executes action, returns new observation
7. Repeat until goal reached or timeout

**Key components:**
- `habitat_imagenav.py`: Main evaluation orchestrator
- `nomad_wrapper.py`: Model loading and inference
- `action_converter.py`: Waypoint → action conversion
- `utils/episode_utils.py`: Episode generation and goal image capture
- `utils/metrics.py`: Metric tracking and logging

## Next Steps

1. **Update config files** with your paths
2. **Test on single episode** to verify integration works
3. **Run small evaluation** (3 scenes × 5 episodes = 15 total)
4. **Analyze results** and tune parameters if needed
5. **Scale up** to more scenes and episodes

## Expected Performance

Given the domain gap (outdoor robots → indoor HM3D), expect:
- Success rate: 10-40% (zero-shot transfer)
- SPL: 0.1-0.3 (lower due to suboptimal paths)
- Common failure modes: Getting stuck, taking circuitous routes, overshooting

This establishes a baseline for the integration. Performance can be improved by:
- Fine-tuning on Habitat demonstrations
- Tuning action conversion parameters
- Adjusting waypoint selection strategy
