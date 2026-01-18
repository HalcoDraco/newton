"""
Modular RL training framework for Newton physics environments.

This module provides:
- NewtonEnvBase: Abstract base class for vectorized Newton environments
- CartpoleEnv: Cartpole balancing environment
- AllegroHandEnv: Allegro hand cube manipulation environment
- TrainerBase: Abstract base class for trainers
- GeneticTrainer: Genetic algorithm trainer
- RandomTrainer: Random action baseline trainer

Run AllegroHand example (default):
    uv run --extra torch-cu12 custom_newton_usage_modular.py --viewer gl

Run Cartpole example:
    # Change AllegroHandEnv to CartpoleEnv in main() and use:
    uv run --extra torch-cu12 custom_newton_usage_modular.py --viewer gl

Future work:
- Add Brax wrapper for NewtonEnvBase
- Add PPO/SAC trainers
"""

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

from custom_newton_usage.envs import CartpoleEnv, AllegroHandEnv
from custom_newton_usage.envs.configs import CartpoleConfig, AllegroHandConfig
from custom_newton_usage.trainers.configs import RandomTrainerConfig, GeneticTrainerConfig
from custom_newton_usage.trainers import RandomTrainer, GeneticTrainer

# =============================================================================
# CLI and Main
# =============================================================================


def make_parser():
    """Create argument parser with all options."""
    parser = newton.examples.create_parser()

    # Environment options
    parser.add_argument("--num-worlds", type=int, default=64, help="Number of parallel environments.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    # Trainer options
    parser.add_argument("--generations", type=int, default=50, help="Training generations.")
    parser.add_argument("--episode-steps", type=int, default=300, help="Steps per episode.")
    parser.add_argument("--elite-frac", type=float, default=0.2, help="Elite fraction for selection.")
    parser.add_argument("--noise-std", type=float, default=0.05, help="Mutation noise std.")
    parser.add_argument("--hidden-size", type=int, default=64, help="Policy hidden layer size.")
    parser.add_argument("--action-scale", type=float, default=10.0, help="Action scaling factor.")
    parser.add_argument("--force-limit", type=float, default=40.0, help="Maximum force magnitude.")
    parser.add_argument("--render-every", type=int, default=1, help="Render every N generations (0 disables).")

    return parser


def main():
    # parser = make_parser()
    # viewer, args = newton.examples.init(parser)

    # if args.viewer == "null":
    #     args.render_every = 0

    # # Create environment config
    # env_config = CartpoleConfig(
    #     num_worlds=args.num_worlds,
    #     seed=args.seed,
    # )

    # # Create trainer config
    # trainer_config = RandomTrainerConfig()

    # # Initialize environment and trainer
    # env = CartpoleEnv(viewer, env_config)
    # trainer = RandomTrainer(env, trainer_config)

    # # Train
    # trainer.train()

    # # Cleanup
    # env.close()


    parser = make_parser()
    viewer, args = newton.examples.init(parser)

    if args.viewer == "null":
        args.render_every = 0
        viewer = None

    # Create environment config (using AllegroHand)
    env_config = AllegroHandConfig(
        num_worlds=args.num_worlds,
        seed=args.seed,
    )

    # env_config = CartpoleConfig(
    #     num_worlds=args.num_worlds,
    #     seed=args.seed,
    # )

    # Create trainer config
    # Note: action_scale and force_limit are less relevant for position control
    trainer_config = RandomTrainerConfig()
    # trainer_config = GeneticTrainerConfig(
    #     generations=args.generations,
    #     episode_steps=args.episode_steps,
    #     elite_frac=args.elite_frac,
    #     noise_std=args.noise_std,
    #     hidden_size=args.hidden_size,
    #     action_scale=10.0,  # Actions are [-1, 1] for position targets
    #     force_limit=40.0,   # Clamp to [-1, 1]
    #     render_every=args.render_every,
    # )

    # Initialize environment and trainer
    env = AllegroHandEnv(viewer, env_config)
    # env = CartpoleEnv(viewer, env_config)
    # trainer = GeneticTrainer(env, trainer_config)
    trainer = RandomTrainer(env, trainer_config)

    # Train
    trainer.train()

    # Cleanup
    env.close()


if __name__ == "__main__":
    main()