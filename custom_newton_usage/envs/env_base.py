from __future__ import annotations

import math
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import torch
import warp as wp
from torch.func import functional_call, vmap

import newton
import newton.examples
from newton.selection import ArticulationView
from newton.solvers import SolverNotifyFlags


torch.set_float32_matmul_precision("medium")

from .configs import EnvConfig


class NewtonEnvBase(ABC):
    """
    Abstract base class for vectorized Newton physics environments.

    This class provides common infrastructure for:
    - Warp/CUDA initialization
    - Simulation timing
    - CUDA graph capture
    - Generic step/render logic

    Subclasses must implement environment-specific methods:
    - obs_dim, action_dim properties
    - _build_world(), _simulate_internal()
    - _get_obs(), _compute_rewards()
    - reset(), _apply_actions()
    """

    def __init__(self, viewer: Any, config: EnvConfig):
        self.viewer = viewer
        self.config = config
        self.num_worlds = config.num_worlds

        # Initialize warp and device
        torch.manual_seed(config.seed)
        wp.init()
        self.device = "cuda" if wp.get_device().is_cuda else "cpu"

        # Timing
        self.fps = config.fps
        self.frame_dt = config.frame_dt
        self.sim_time = 0.0
        self.sim_substeps = config.sim_substeps
        self.sim_dt = config.sim_dt

        # To be set by subclass in _build_world()
        self.model: Any = None
        self.solver: Any = None
        self.state_0: Any = None
        self.state_1: Any = None
        self.control: Any = None
        self.contacts: Any = None
        self.graph: Any = None

        # Build environment
        self._build_world()
        self._setup_viewer()
        self._capture_graph()

    @property
    @abstractmethod
    def obs_dim(self) -> int:
        """Observation space dimension."""
        pass

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Action space dimension."""
        pass

    @abstractmethod
    def _build_world(self) -> None:
        """
        Build the Newton model, solver, and states.

        Must set:
        - self.model
        - self.solver
        - self.state_0, self.state_1
        - self.control
        - self.contacts (optional, for collision-enabled envs)
        """
        pass

    @abstractmethod
    def _simulate_internal(self) -> None:
        """Run simulation substeps. Called during CUDA graph capture."""
        pass

    @abstractmethod
    def _get_obs(self) -> torch.Tensor:
        """
        Get observations from current state.

        Returns:
            Tensor of shape (num_worlds, obs_dim)
        """
        pass

    @abstractmethod
    def _compute_rewards(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rewards and done flags.

        Args:
            obs: Current observations (num_worlds, obs_dim)

        Returns:
            rewards: Tensor of shape (num_worlds,)
            dones: Boolean tensor of shape (num_worlds,)
        """
        pass

    @abstractmethod
    def reset(self) -> torch.Tensor:
        """
        Reset environment and return initial observations.

        Returns:
            Initial observations (num_worlds, obs_dim)
        """
        pass

    @abstractmethod
    def _apply_actions(self, actions: torch.Tensor) -> None:
        """
        Apply actions to the environment.

        Args:
            actions: Tensor of shape (num_worlds,) or (num_worlds, action_dim)
        """
        pass

    def _setup_viewer(self) -> None:
        """Setup viewer with model. Override for custom viewer setup."""
        if self.viewer is not None and hasattr(self.viewer, "set_model"):
            self.viewer.set_model(self.model)

    def _capture_graph(self) -> None:
        """Capture CUDA graph for simulation."""
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self._simulate_internal()
            self.graph = capture.graph

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Step the environment.

        Args:
            actions: Tensor of shape (num_worlds,) or (num_worlds, action_dim)

        Returns:
            obs: Tensor of shape (num_worlds, obs_dim)
            rewards: Tensor of shape (num_worlds,)
            dones: Boolean tensor of shape (num_worlds,)
            info: Dictionary with additional info
        """
        # Apply actions
        self._apply_actions(actions)

        # Step simulation
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._simulate_internal()
        self.sim_time += self.frame_dt

        # Get state & rewards
        obs = self._get_obs()
        rewards, dones = self._compute_rewards(obs)

        return obs, rewards, dones, {}

    def render(self) -> None:
        """Render current state."""
        if self.viewer is None:
            return
        if hasattr(self.viewer, "begin_frame"):
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            if self.contacts is not None and hasattr(self.viewer, "log_contacts"):
                self.viewer.log_contacts(self.contacts, self.state_0)
            if hasattr(self.viewer, "end_frame"):
                self.viewer.end_frame()

    def close(self) -> None:
        """Clean up resources."""
        if self.viewer is not None and hasattr(self.viewer, "close"):
            self.viewer.close()