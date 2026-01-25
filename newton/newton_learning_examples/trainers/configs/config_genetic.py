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

from .config_base import TrainerConfig

@dataclass
class GeneticTrainerConfig(TrainerConfig):
    """Configuration for Genetic Algorithm trainer."""

    elite_frac: float = 0.2
    noise_std: float = 0.05
    hidden_size: int = 64
    action_scale: float = 10.0
    force_limit: float = 40.0