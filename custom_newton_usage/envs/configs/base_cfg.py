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


@dataclass
class BaseEnvConfig:
    """Base configuration for Newton environments."""

    seed: int = 42
    wp_dtype = wp.float32
    torch_dtype = torch.float32

    # Number of parallel environments
    num_worlds: int = 64

    # Timing
    control_hz: int = 20
    physics_hz: int = 100

    @property
    def sim_substeps(self) -> int:
        """Number of simulation substeps per control step."""
        return self.physics_hz // self.control_hz

    @property
    def sim_dt(self) -> float:
        """Simulation timestep."""
        return 1.0 / self.physics_hz

    @property
    def frame_dt(self) -> float:
        """Control timestep."""
        return 1.0 / self.control_hz

    solver_class: str = "SolverMuJoCo"

    # Solver settings
    mjw_model = None
    mjw_data = None
    separate_worlds: bool | None = None
    njmax: int | None = None
    nconmax: int | None = None
    iterations: int = 20
    ls_iterations: int = 10
    solver: int | str = "cg"
    integrator: int | str = "implicitfast"
    cone: int | str = "pyramidal"
    impratio: float = 1
    use_mujoco_cpu: bool = False
    disable_contacts: bool = False
    default_actuator_gear: float | None = None
    actuator_gears: dict[str, float] | None = None
    update_data_interval: int = 1
    save_to_mjcf: str | None = None
    ls_parallel: bool = False
    use_mujoco_contacts: bool = True
    tolerance: float = 0.000001
    ls_tolerance: float = 0.01
    include_sites: bool = True

    # Actions
    action_scale: float = 1.0  # Scale actions
    action_limit: float = 1.0  # Clamp actions to [-limit, limit]