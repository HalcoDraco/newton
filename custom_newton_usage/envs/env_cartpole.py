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

from .env_base import NewtonEnvBase
from .configs import CartpoleConfig

class CartpoleEnv(NewtonEnvBase):
    """
    Vectorized Cartpole environment.

    Observation: [cart_pos, pole_angle, cart_vel, pole_vel]
    Action: Force applied to cart (1D)
    """

    def __init__(self, viewer: Any, config: CartpoleConfig):
        self._config = config  # Store typed config before super().__init__
        super().__init__(viewer, config)

    @property
    def cartpole_config(self) -> CartpoleConfig:
        """Typed access to cartpole-specific config."""
        return self._config

    @property
    def obs_dim(self) -> int:
        return 2 * self.articulation.joint_coord_count

    @property
    def action_dim(self) -> int:
        return 1

    def _build_world(self) -> None:
        cfg = self.cartpole_config

        # Build single cartpole
        world = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(world)
        world.default_shape_cfg.density = cfg.density
        world.default_joint_cfg.armature = cfg.armature
        world.default_body_armature = cfg.armature

        world.add_usd(
            newton.examples.get_asset("cartpole_single_pendulum.usda"),
            collapse_fixed_joints=False,
            enable_self_collisions=False,
        )
        world.joint_q[-2:] = [0.0, 0.3]

        # Replicate across worlds
        scene = newton.ModelBuilder()
        scene.replicate(world, num_worlds=self.num_worlds, spacing=cfg.spacing)

        self.model = scene.finalize()
        self.solver = newton.solvers.SolverMuJoCo(self.model, disable_contacts=True)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None  # No contacts for cartpole

        # Articulation view for easy joint access
        self.articulation = ArticulationView(self.model, "/cartpole", verbose=False)

        # Forward kinematics
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Pre-allocate action template
        self._joint_f_template = torch.zeros(
            (self.num_worlds, self.articulation.joint_dof_count),
            device=self.device,
            dtype=torch.float32,
        )

    def _setup_viewer(self) -> None:
        super()._setup_viewer()
        if hasattr(self.viewer, "set_world_offsets"):
            self.viewer.set_world_offsets(self.cartpole_config.spacing)

    def _simulate_internal(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            if hasattr(self.viewer, "apply_forces"):
                self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _get_obs(self) -> torch.Tensor:
        joint_q = wp.to_torch(self.articulation.get_attribute("joint_q", self.state_0)).to(self.device)
        joint_qd = wp.to_torch(self.articulation.get_attribute("joint_qd", self.state_0)).to(self.device)
        return torch.cat([joint_q, joint_qd], dim=1)

    def _compute_rewards(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cartpole_config

        # Unpack observations: [cart_pos, pole_angle, cart_vel, pole_vel]
        cart_pos, pole_angle, cart_vel, pole_vel = obs.T

        # Reward shaping
        angle_cost = cfg.angle_cost_weight * pole_angle.abs()
        vel_cost = cfg.vel_cost_weight * (cart_vel.abs() + pole_vel.abs())
        pos_cost = cfg.pos_cost_weight * cart_pos.abs()

        reward = 1.0 - (angle_cost + vel_cost + pos_cost)

        # Termination conditions
        dones = (pole_angle.abs() > cfg.pole_angle_threshold) | (cart_pos.abs() > cfg.cart_pos_threshold)

        return reward, dones

    def reset(self) -> torch.Tensor:
        cfg = self.cartpole_config

        # Randomize initial state
        cart_positions = cfg.cart_pos_range * (torch.rand(self.num_worlds, device=self.device) - 0.5)
        pole_angles = cfg.pole_angle_range * (torch.rand(self.num_worlds, device=self.device) - 0.5)

        joint_q = torch.stack([cart_positions, pole_angles], dim=1)
        joint_qd = torch.zeros_like(joint_q)

        self.articulation.set_attribute("joint_q", self.state_0, joint_q)
        self.articulation.set_attribute("joint_qd", self.state_0, joint_qd)

        if not isinstance(self.solver, newton.solvers.SolverMuJoCo):
            self.articulation.eval_fk(self.state_0)

        return self._get_obs()

    def _apply_actions(self, actions: torch.Tensor) -> None:
        joint_f = self._joint_f_template.clone()
        # Actions are forces on the cart (first joint)
        joint_f[:, 0] = actions.squeeze(-1) if actions.dim() > 1 else actions
        self.articulation.set_attribute("joint_f", self.control, joint_f)