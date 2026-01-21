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

from .base_cfg import BaseEnvConfig

@dataclass
class CartpoleConfig(BaseEnvConfig):
    """Configuration for Cartpole environment."""

    control_hz: int = 60
    physics_hz: int = 120

    # World layout
    spacing: Tuple[float, float, float] = (0.8, 4.0, 0.0)

    # Physics
    density: float = 100.0
    armature: float = 0.1

    # Reset ranges
    cart_pos_range: float = 0.5
    pole_angle_range: float = 0.25

    # Reward coefficients
    angle_cost_weight: float = 2.0
    vel_cost_weight: float = 0.05
    pos_cost_weight: float = 0.1

    # Termination thresholds
    pole_angle_threshold: float = 0.8
    cart_pos_threshold: float = 2.4

    disable_contacts: bool = True

    # Actions
    action_scale: float = 20.0  # Scale actions
    action_limit: float = 20.0  # Clamp actions to [-limit, limit]