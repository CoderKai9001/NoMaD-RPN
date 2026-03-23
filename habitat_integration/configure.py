#!/usr/bin/env python
"""
Helper script to configure the integration after installing Habitat and downloading HM3D.

This script will:
1. Find HM3D scenes on your system
2. Prompt for NoMaD checkpoint path
3. Update configuration files automatically
"""

import os
import sys
import glob
import yaml


def find_hm3d_scenes(search_paths=None):
    """Find HM3D scene files on the system."""
    if search_paths is None:
        search_paths = [
            'data/scene_datasets/hm3d',
            '../data/scene_datasets/hm3d',
            '../../data/scene_datasets/hm3d',
            os.path.expanduser('~/habitat/data/scene_datasets/hm3d'),
            '/home2/aditya.vadali/habitat/data/scene_datasets/hm3d',
        ]

    found_scenes = []

    for base_path in search_paths:
        if not os.path.exists(base_path):
            continue

        # Look for .basis.glb files
        pattern = os.path.join(base_path, '**/*.basis.glb')
        scenes = glob.glob(pattern, recursive=True)
        found_scenes.extend(scenes)

    return sorted(set(found_scenes))


def update_nomad_config(checkpoint_path):
    """Update nomad.yaml with checkpoint path."""
    config_path = 'config/nomad.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['checkpoint_path'] = checkpoint_path

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Updated {config_path}")


def update_evaluation_config(scene_paths, num_scenes=3):
    """Update evaluation.yaml with scene paths."""
    config_path = 'config/evaluation.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['hm3d_scenes'] = scene_paths[:num_scenes]

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Updated {config_path} with {len(scene_paths[:num_scenes])} scenes")


def main():
    print("="*60)
    print("NoMaD-RPN Habitat Integration Configuration Helper")
    print("="*60)
    print()

    # Step 1: Find HM3D scenes
    print("Step 1: Searching for HM3D scenes...")
    scenes = find_hm3d_scenes()

    if not scenes:
        print("✗ No HM3D scenes found!")
        print("\nPlease download HM3D dataset first. See DATASET_SETUP.md for instructions.")
        print("\nQuick start:")
        print("  mkdir -p data/scene_datasets")
        print("  python -m habitat_sim.utils.datasets_download --uids hm3d_minival_v0.2 --data-path data/")
        sys.exit(1)

    print(f"✓ Found {len(scenes)} HM3D scenes")
    for i, scene in enumerate(scenes[:5]):
        scene_name = os.path.basename(scene).replace('.basis.glb', '')
        print(f"  {i+1}. {scene_name}")
    if len(scenes) > 5:
        print(f"  ... and {len(scenes) - 5} more")

    # Step 2: Get NoMaD checkpoint path
    print("\nStep 2: NoMaD model checkpoint")
    checkpoint_path = input("Enter path to your NoMaD checkpoint (.pth file): ").strip()

    if not checkpoint_path or checkpoint_path == "":
        print("✗ No checkpoint path provided. Please update config/nomad.yaml manually.")
        checkpoint_path = None
    elif not os.path.exists(checkpoint_path):
        print(f"⚠ Warning: Checkpoint file not found at {checkpoint_path}")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            checkpoint_path = None

    # Step 3: Select number of scenes
    print(f"\nStep 3: Scene selection")
    print(f"Found {len(scenes)} scenes. How many to use for initial testing?")
    num_scenes_input = input(f"Enter number (default: 3): ").strip()

    try:
        num_scenes = int(num_scenes_input) if num_scenes_input else 3
        num_scenes = min(num_scenes, len(scenes))
    except ValueError:
        num_scenes = 3

    print(f"Using {num_scenes} scenes for evaluation")

    # Step 4: Update configuration files
    print("\nStep 4: Updating configuration files...")

    if checkpoint_path:
        update_nomad_config(checkpoint_path)
    else:
        print("⚠ Skipping nomad.yaml update (no checkpoint path provided)")

    update_evaluation_config(scenes, num_scenes)

    # Step 5: Summary
    print("\n" + "="*60)
    print("CONFIGURATION COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Test integration: python test_integration.py")
    print("2. Run evaluation: python habitat_imagenav.py")
    print("\nConfiguration files updated:")
    if checkpoint_path:
        print(f"  - config/nomad.yaml (checkpoint: {os.path.basename(checkpoint_path)})")
    print(f"  - config/evaluation.yaml ({num_scenes} scenes)")


if __name__ == '__main__':
    main()
