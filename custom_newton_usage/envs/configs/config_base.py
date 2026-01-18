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
class EnvConfig:
    """Base configuration for Newton environments."""

    # Number of parallel environments
    num_worlds: int 

    # Maximum number of parallel environments. 
    # If num_worlds > max_num_worlds, environments will be stepped in batches
    # with size (num_worlds / ((num_worlds // max_num_worlds) + 1))
    max_num_worlds: int 

    # Timing
    control_hz: int = 20
    physics_hz: int = 100

    assert physics_hz % control_hz == 0, "physics_hz must be multiple of control_hz"

    sim_substeps: int = physics_hz // control_hz

    sim_dt: float = 1.0 / physics_hz
    frame_dt: float = 1.0 / control_hz

    seed: int = 0
