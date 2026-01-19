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

from .configs import RandomTrainerConfig
from custom_newton_usage.envs import NewtonBaseEnv
from .trainer_base import TrainerBase



class RandomTrainer(TrainerBase):
    """
    Random Action trainer (baseline).

    Samples random actions uniformly from the action space.
    Useful as a baseline for comparison with learning algorithms.
    """

    def __init__(self, env: NewtonBaseEnv, config: RandomTrainerConfig):
        super().__init__(env, config)
        self._config = config

    @property
    def random_config(self) -> RandomTrainerConfig:
        return self._config

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Generate random actions."""
        cfg = self.random_config

        # Sample random actions uniformly from [-1, 1]
        if obs.dim() == 1:
            # Single observation
            actions = torch.rand(self.env.action_dim, device=self.device) * 2 - 1
        else:
            # Batch of observations
            batch_size = obs.shape[0]
            actions = torch.rand(batch_size, self.env.action_dim, device=self.device) * 2 - 1

        # Scale and clamp
        actions = actions * cfg.action_scale
        actions = torch.clamp(actions, -cfg.action_limit, cfg.action_limit)

        # Return single action if input was single obs, otherwise batch
        return actions.squeeze(0) if obs.dim() == 1 else actions

    def train(self) -> None:
        """Run random action rollouts (no actual training)."""
        cfg = self.random_config

        best_reward = -float("inf")
        total_reward_sum = 0.0
        total_episodes = 0

        for gen in range(cfg.generations):
            # Check if viewer is still running
            if self.env.viewer is not None and hasattr(self.env.viewer, "is_running") and not self.env.viewer.is_running():
                break

            gen_start = time.perf_counter()

            # Reset environment
            obs = self.env.reset()
            episode_returns = torch.zeros(self.env.num_worlds, device=self.device)
            alive_mask = torch.ones(self.env.num_worlds, device=self.device, dtype=torch.bool)

            # Rollout episode with random actions
            for step in range(cfg.episode_steps):
                # Generate random actions
                actions = self.get_action(obs)

                # Mask actions for dead agents
                if actions.dim() == 1:
                    masked_actions = actions * alive_mask
                else:
                    masked_actions = actions * alive_mask.unsqueeze(-1)

                # Step environment
                obs, reward, dones = self.env.step(masked_actions)

                # Update alive mask and accumulate rewards
                alive_mask = alive_mask & (~dones)
                reward = reward * alive_mask
                episode_returns += reward

                # Render
                if cfg.render_every > 0 and gen % cfg.render_every == 0:
                    self.env.render()

                if not alive_mask.any():
                    break  # All agents are done

            # Logging
            mean_reward = episode_returns.mean().item()
            top_reward = episode_returns.max().item()

            if top_reward > best_reward:
                best_reward = top_reward

            total_reward_sum += mean_reward
            total_episodes += 1

            # Logging
            elapsed = time.perf_counter() - gen_start
            control_steps = (step + 1) * self.env.num_worlds
            steps_per_s = control_steps / elapsed if elapsed > 0 else float("inf")
            physics_steps = control_steps * self.env.sim_substeps
            physics_steps_per_s = physics_steps / elapsed if elapsed > 0 else float("inf")
            print(
                f"[Gen {gen:03d}] mean_reward={mean_reward:.3f} "
                f"best_all={best_reward:.3f} "
                f"time={elapsed:.2f}s "
                f"fps={steps_per_s:.1f} "
                f"physics_fps={physics_steps_per_s:.1f}"
            )

        # Final statistics
        avg_reward = total_reward_sum / total_episodes if total_episodes > 0 else 0.0
        print(f"\nRandom baseline results: avg_reward={avg_reward:.3f} best={best_reward:.3f}")