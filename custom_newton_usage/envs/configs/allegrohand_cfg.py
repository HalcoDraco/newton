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
class AllegroHandConfig(BaseEnvConfig):
    """Configuration for Allegro Hand environment."""

    # Timing (override defaults for hand)
    control_hz: int = 30
    physics_hz: int = 120

    # Physics
    shape_ke: float = 1.0e3  # Contact stiffness
    shape_kd: float = 1.0e2  # Contact damping
    joint_target_ke: float = 150.0  # Joint drive stiffness
    joint_target_kd: float = 5.0  # Joint drive damping

    # Actions
    action_range: tuple[float, float] = (-21.0, 21.0) # Torque in Nm

    # Solver settings (override base config)
    solver: str = "newton"
    integrator: str = "implicitfast"
    njmax: int = 200
    nconmax: int = 150
    impratio: float = 10.0
    cone: str = "elliptic"
    iterations: int = 100
    ls_iterations: int = 15

    # Task: cube reorientation
    cube_drop_threshold: float = 0.1  # Below this z-height, episode ends

    # Reward weights for cube reorientation
    orientation_reward_weight: float = 2.0  # Main task: match target orientation
    position_reward_weight: float = 0.5  # Keep cube near starting position
    angular_vel_penalty_weight: float = 0.01  # Penalize fast spinning
    action_penalty_weight: float = 0.01  # Penalize large actions
    success_bonus: float = 5.0  # Bonus for reaching target orientation

    # Orientation task thresholds
    orientation_tolerance: float = 0.1  # Quaternion distance for success
    position_tolerance: float = 0.15  # Max position deviation from target

    # Initial grasp position for joints
    initial_grasp_pos: float = 0.3

    # Initial cube position [x, y, z] - set to None to use USD default (above hand)
    # For in-hand start, use something like [0.0, -0.15, 1.0]
    initial_cube_position: tuple[float, float, float] | None = (0.0, -0.15, 1.08)
    desired_cube_position: tuple[float, float, float] = (0.0, -0.15, 1.08)