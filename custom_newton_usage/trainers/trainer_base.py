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

from .configs import TrainerConfig
from custom_newton_usage.envs import NewtonBaseEnv


class TrainerBase(ABC):
    """
    Abstract base class for trainers.

    Trainers operate on NewtonEnvBase instances and implement
    specific training algorithms (GA, PPO, SAC, etc.)
    """

    def __init__(self, env: NewtonBaseEnv, config: TrainerConfig):
        self.env = env
        self.config = config
        self.device = env.device

    @abstractmethod
    def train(self) -> None:
        """Run the training loop."""
        pass

    @abstractmethod
    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get action for given observation (inference mode).

        Args:
            obs: Observations (num_worlds, obs_dim)

        Returns:
            Actions (num_worlds,) or (num_worlds, action_dim)
        """
        pass