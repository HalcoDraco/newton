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

from .configs import GeneticTrainerConfig
from custom_newton_usage.envs import NewtonEnvBase
from .trainer_base import TrainerBase



class Policy(torch.nn.Module):
    """Simple MLP policy network."""

    def __init__(self, obs_dim: int, hidden: int, action_dim: int = 1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GeneticTrainer(TrainerBase):
    """
    Genetic Algorithm trainer.

    Maintains a population of policies (one per world), evaluates them
    in parallel, and evolves via elitism + mutation.
    """

    def __init__(self, env: NewtonEnvBase, config: GeneticTrainerConfig):
        super().__init__(env, config)
        self._config = config

        self._build_policy()
        self._init_population()

    @property
    def ga_config(self) -> GeneticTrainerConfig:
        return self._config

    def _build_policy(self) -> None:
        """Build policy network and extract base parameters."""
        self.policy = Policy(
            self.env.obs_dim,
            self.ga_config.hidden_size,
            action_dim=self.env.action_dim,
        ).to(self.device)
        self.policy.eval()

        self.base_params = {k: v.detach().to(self.device) for k, v in self.policy.named_parameters()}

    def _init_population(self) -> None:
        """Initialize population with noisy copies of base policy."""
        cfg = self.ga_config

        self.population_params: Dict[str, torch.Tensor] = {}
        for name, param in self.base_params.items():
            noise = cfg.noise_std * torch.randn(
                (self.env.num_worlds,) + param.shape,
                device=self.device,
                dtype=param.dtype,
            )
            self.population_params[name] = param.unsqueeze(0).expand_as(noise) + noise

        # Create batched policy function
        params_in_dims = {k: 0 for k in self.population_params}

        def policy_apply(p, obs):
            return functional_call(self.policy, p, (obs,))

        self.batched_policy = vmap(policy_apply, in_dims=(params_in_dims, 0))

    def _compute_actions(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute actions for all policies in population."""
        cfg = self.ga_config

        actions = self.batched_policy(self.population_params, obs)
        # Apply tanh and scaling - keep shape (num_worlds, action_dim) or (num_worlds,)
        if actions.dim() == 3 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)  # (num_worlds, 1) -> (num_worlds,)
        actions = torch.tanh(actions) * cfg.action_scale
        actions = torch.clamp(actions, -cfg.force_limit, cfg.force_limit)
        return actions

    def _mutate(self, elite_idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Create new population from elite policies with mutation."""
        cfg = self.ga_config

        new_params: Dict[str, torch.Tensor] = {}
        repeats = math.ceil(self.env.num_worlds / elite_idx.numel())

        for name, tensor in self.population_params.items():
            elites = tensor[elite_idx]
            tiled = elites.repeat((repeats,) + (1,) * (elites.dim() - 1))
            trimmed = tiled[: self.env.num_worlds]
            noise = cfg.noise_std * torch.randn_like(trimmed)
            new_params[name] = trimmed + noise

        return new_params

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Get action using best policy (for inference)."""
        cfg = self.ga_config

        # Use first policy in population (should be best after training)
        params = {k: v[0:1] for k, v in self.population_params.items()}
        obs_batched = obs.unsqueeze(0) if obs.dim() == 2 else obs

        actions = self.batched_policy(params, obs_batched)
        if actions.dim() == 3 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)
        actions = torch.tanh(actions) * cfg.action_scale
        actions = torch.clamp(actions, -cfg.force_limit, cfg.force_limit)

        return actions.squeeze(0)

    def train(self) -> None:
        """Run genetic algorithm training loop."""
        cfg = self.ga_config

        best_reward = -float("inf")
        best_params = None

        for gen in range(cfg.generations):
            # Check if viewer is still running
            if hasattr(self.env.viewer, "is_running") and not self.env.viewer.is_running():
                break

            gen_start = time.perf_counter()

            # Reset environment
            obs = self.env.reset()
            episode_returns = torch.zeros(self.env.num_worlds, device=self.device)
            alive_mask = torch.ones(self.env.num_worlds, device=self.device, dtype=torch.bool)

            # Rollout episode
            for step in range(cfg.episode_steps):
                # Compute actions
                actions = self._compute_actions(obs)
                # Mask actions for dead agents - handle both 1D and 2D actions
                if actions.dim() == 1:
                    masked_actions = actions * alive_mask
                else:
                    masked_actions = actions * alive_mask.unsqueeze(-1)

                # Step environment
                obs, reward, dones, _ = self.env.step(masked_actions)

                # Update alive mask and accumulate rewards
                alive_mask = alive_mask & (~dones)
                reward = reward * alive_mask
                episode_returns += reward

                # Render
                if cfg.render_every > 0 and gen % cfg.render_every == 0:
                    self.env.render()

            # Evolution step
            mean_reward = episode_returns.mean().item()
            top_reward, top_idx = torch.max(episode_returns, dim=0)

            if top_reward.item() > best_reward:
                best_reward = top_reward.item()
                best_params = {k: v[top_idx].detach().clone() for k, v in self.population_params.items()}

            elite_count = max(1, int(cfg.elite_frac * self.env.num_worlds))
            elite_scores, elite_idx = torch.topk(episode_returns, k=elite_count)
            self.population_params = self._mutate(elite_idx)

            # Logging
            elapsed = time.perf_counter() - gen_start
            frames = cfg.episode_steps * self.env.num_worlds
            fps = frames / elapsed if elapsed > 0 else float("inf")
            print(
                f"[Gen {gen:03d}] mean_reward={mean_reward:.3f} "
                f"best_gen={elite_scores.max().item():.3f} "
                f"best_all={best_reward:.3f} time={elapsed:.2f}s fps={fps:.1f}"
            )

        # Set population to best params for inference
        if best_params is not None:
            self.population_params = {
                k: v.unsqueeze(0).expand(self.env.num_worlds, *v.shape) for k, v in best_params.items()
            }