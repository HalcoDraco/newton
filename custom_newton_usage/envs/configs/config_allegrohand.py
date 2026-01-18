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

from .config_base import EnvConfig



@dataclass
class AllegroHandConfig(EnvConfig):
    """Configuration for Allegro Hand environment."""

    # Timing (override defaults for hand)
    fps: int = 60
    sim_substeps: int = 2

    # Physics
    shape_ke: float = 1.0e3  # Contact stiffness
    shape_kd: float = 1.0e2  # Contact damping
    joint_target_ke: float = 150.0  # Joint drive stiffness
    joint_target_kd: float = 5.0  # Joint drive damping

    # Solver settings
    solver_type: str = "newton"
    integrator: str = "implicit"
    njmax: int = 200
    nconmax: int = 150
    impratio: float = 10.0
    cone: str = "elliptic"
    iterations: int = 100
    ls_iterations: int = 15

    # Task: cube height target
    cube_target_height: float = 0.5  # Target height for cube
    cube_drop_threshold: float = 0.1  # Below this, episode ends

    # Reward weights
    height_reward_weight: float = 1.0
    action_penalty_weight: float = 0.01
    velocity_penalty_weight: float = 0.001