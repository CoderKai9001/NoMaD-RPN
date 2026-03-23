"""
NoMaD model wrapper for Habitat integration.

This module provides a clean interface to the NoMaD model for use in Habitat simulator,
reusing the model loading and inference code from the deployment module.
"""

import sys
import os
from collections import deque
from typing import Tuple, Optional
import numpy as np
import torch
import yaml
from PIL import Image
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# Add paths to import from train modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../train'))

# Import from local model_utils (ROS-independent)
from model_utils import load_model, transform_images, to_numpy

# Import from train module
from vint_train.training.train_utils import get_action


class NoMaDHabitatWrapper:
    """
    Wrapper for NoMaD model to use in Habitat simulator.

    Handles model loading, temporal context management, and waypoint prediction
    using diffusion-based inference.
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_config_path: str,
        device: str = 'cuda',
        num_samples: int = 8,
        waypoint_index: int = 2,
    ):
        """
        Initialize the NoMaD model wrapper.

        Args:
            checkpoint_path: Path to pre-trained model weights (.pth file)
            model_config_path: Path to model configuration YAML (train/config/nomad.yaml)
            device: Device to run model on ('cuda' or 'cpu')
            num_samples: Number of trajectory samples to generate (default: 8)
            waypoint_index: Which waypoint to select from trajectory (default: 2)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"NoMaD model using device: {self.device}")

        # Load model configuration
        with open(model_config_path, 'r') as f:
            self.model_params = yaml.safe_load(f)

        # Verify model type
        if self.model_params['model_type'] != 'nomad':
            raise ValueError(f"Expected model_type 'nomad', got '{self.model_params['model_type']}'")

        # Store model parameters
        self.context_size = self.model_params['context_size']
        self.image_size = self.model_params['image_size']
        self.len_traj_pred = self.model_params['len_traj_pred']
        self.num_diffusion_iters = self.model_params['num_diffusion_iters']
        self.normalize = self.model_params.get('normalize', True)

        # Inference parameters
        self.num_samples = num_samples
        self.waypoint_index = waypoint_index

        # Load model
        print(f"Loading NoMaD model from {checkpoint_path}...")
        self.model = load_model(checkpoint_path, self.model_params, self.device)
        self.model.eval()
        print("Model loaded successfully")

        # Initialize diffusion scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

        # Initialize context queue (stores PIL Images)
        # Size is context_size + 1 to match deployment behavior
        self.context_queue = deque(maxlen=self.context_size + 1)

        # Normalization constants (from deployment config)
        # These are used to denormalize waypoints: waypoint * (MAX_V / RATE)
        self.MAX_V = 0.2  # m/s
        self.RATE = 4.0  # Hz

    def reset_context(self):
        """Clear the context queue at the start of a new episode."""
        self.context_queue.clear()

    def add_observation(self, rgb_obs: np.ndarray):
        """
        Add an observation to the context queue.

        Args:
            rgb_obs: RGB observation from Habitat (H, W, 3) or (H, W, 4)
        """
        # Drop alpha channel if present
        if rgb_obs.shape[-1] == 4:
            rgb_obs = rgb_obs[..., :3]

        # Convert numpy array to PIL Image
        if rgb_obs.dtype == np.uint8:
            pil_img = Image.fromarray(rgb_obs)
        else:
            # If float, convert to uint8
            rgb_obs = (rgb_obs * 255).astype(np.uint8) if rgb_obs.max() <= 1.0 else rgb_obs.astype(np.uint8)
            pil_img = Image.fromarray(rgb_obs)

        self.context_queue.append(pil_img)

    def has_sufficient_context(self) -> bool:
        """Check if we have enough observations for inference."""
        return len(self.context_queue) > self.context_size

    def predict_waypoints(
        self,
        goal_image: Image.Image,
    ) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Predict waypoints to reach the goal image using diffusion.

        This implements the same diffusion inference as navigate.py lines 134-194.

        Args:
            goal_image: Goal image as PIL Image

        Returns:
            waypoints: Array of shape (num_samples, len_traj_pred, 2) with (x, y) waypoints
                      in meters, or None if insufficient context
            distance: Predicted temporal distance to goal, or None if insufficient context
        """
        if not self.has_sufficient_context():
            return None, None

        with torch.no_grad():
            # Transform observation images (context_size + 1 frames)
            # This creates a tensor of shape (1, 3*(context_size+1), H, W)
            obs_images = transform_images(
                list(self.context_queue),
                self.image_size,
                center_crop=False
            )
            obs_images = torch.split(obs_images, 3, dim=1)
            obs_images = torch.cat(obs_images, dim=1)
            obs_images = obs_images.to(self.device)

            # Transform goal image
            goal_img_tensor = transform_images(
                goal_image,
                self.image_size,
                center_crop=False
            ).to(self.device)

            # Goal mask (0 = use goal, 1 = mask goal for exploration)
            # For navigation, always use goal (mask=0)
            mask = torch.zeros(1).long().to(self.device)

            # Vision encoder: create observation-goal embedding
            # Repeat obs for single goal image
            obsgoal_cond = self.model(
                'vision_encoder',
                obs_img=obs_images,
                goal_img=goal_img_tensor,
                input_goal_mask=mask
            )

            # Distance prediction
            dist_pred = self.model('dist_pred_net', obsgoal_cond=obsgoal_cond)
            dist_pred_value = to_numpy(dist_pred).item()

            # Prepare for diffusion
            # Repeat conditioning for num_samples
            if len(obsgoal_cond.shape) == 2:
                obs_cond = obsgoal_cond.repeat(self.num_samples, 1)
            else:
                obs_cond = obsgoal_cond.repeat(self.num_samples, 1, 1)

            # Initialize action from Gaussian noise
            # Shape: (num_samples, len_traj_pred, 2)
            noisy_action = torch.randn(
                (self.num_samples, self.len_traj_pred, 2),
                device=self.device
            )

            # Initialize noise scheduler
            self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

            # Diffusion denoising loop (DDPM)
            for k in self.noise_scheduler.timesteps:
                # Predict noise
                noise_pred = self.model(
                    'noise_pred_net',
                    sample=noisy_action,
                    timestep=k,
                    global_cond=obs_cond
                )

                # Inverse diffusion step (remove noise)
                noisy_action = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=noisy_action
                ).prev_sample

            # Get final action (converts deltas to cumulative waypoints)
            action = get_action(noisy_action)
            action = to_numpy(action)

            # Denormalize if the model was trained with normalization
            if self.normalize:
                action = action * (self.MAX_V / self.RATE)

            return action, dist_pred_value

    def get_current_waypoint(
        self,
        goal_image: Image.Image,
    ) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Get the current waypoint to execute (convenience method).

        Args:
            goal_image: Goal image as PIL Image

        Returns:
            waypoint: Single waypoint (x, y) in meters, or None if insufficient context
            distance: Predicted temporal distance to goal, or None if insufficient context
        """
        waypoints, distance = self.predict_waypoints(goal_image)

        if waypoints is None:
            return None, None

        # Select waypoint at specified index from first sample
        # waypoints shape: (num_samples, len_traj_pred, 2)
        current_waypoint = waypoints[0, self.waypoint_index]

        return current_waypoint, distance
