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
from .configs import AllegroHandConfig

class AllegroHandEnv(NewtonEnvBase):
    """
    Vectorized Allegro Hand environment with cube manipulation task.

    Observation: [finger_joint_q (20), finger_joint_qd (20), cube_pos (3), cube_vel (3)] = 46 dims
    Action: Joint target positions for finger joints (20D)
    Task: Keep the cube elevated above a target height
    """

    # Number of actuated finger joints (4 fingers x 4 joints + 4 tip joints = 20 controllable DOFs)
    # From USD: index(4) + middle(5) + ring(5) + thumb(5) = 19 finger joints, but 20 DOFs
    NUM_FINGER_DOFS: int = 20
    # Total joints per hand including root and mount
    JOINTS_PER_HAND: int = 22
    # DOFs per hand (root=6, mount=0, fingers=20, cube joints=0)
    DOFS_PER_HAND: int = 20

    def __init__(self, viewer: Any, config: AllegroHandConfig):
        self._config = config
        super().__init__(viewer, config)

    @property
    def hand_config(self) -> AllegroHandConfig:
        """Typed access to hand-specific config."""
        return self._config

    @property
    def obs_dim(self) -> int:
        # finger joints q + qd + cube_pos (3) + cube_vel (3)
        return self._dofs_per_hand * 2 + 6

    @property
    def action_dim(self) -> int:
        # Control finger joints
        return self._dofs_per_hand

    def _build_world(self) -> None:
        cfg = self.hand_config

        # Build single allegro hand with cube
        allegro_hand = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(allegro_hand)
        allegro_hand.default_shape_cfg.ke = cfg.shape_ke
        allegro_hand.default_shape_cfg.kd = cfg.shape_kd

        # Load allegro hand USD
        asset_path = newton.utils.download_asset("wonik_allegro")
        asset_file = str(asset_path / "usd" / "allegro_left_hand_with_cube.usda")
        allegro_hand.add_usd(
            asset_file,
            xform=wp.transform(wp.vec3(0, 0, 0.5)),
            enable_self_collisions=True,
            ignore_paths=[".*Dummy", ".*CollisionPlane", ".*goal", ".*DexCube/visuals"],
        )

        # Hide collision shapes for the hand links
        for i, key in enumerate(allegro_hand.shape_key):
            if re.match(".*Robot/.*?/collision", key):
                allegro_hand.shape_flags[i] &= ~newton.ShapeFlags.VISIBLE

        # Set joint drive gains (for all DOFs)
        for i in range(allegro_hand.joint_dof_count):
            allegro_hand.joint_target_ke[i] = cfg.joint_target_ke
            allegro_hand.joint_target_kd[i] = cfg.joint_target_kd
            allegro_hand.joint_target_pos[i] = 0.3  # Initial grasp position

        # Store counts before replication
        self._single_hand_body_count = allegro_hand.body_count
        self._single_hand_joint_count = allegro_hand.joint_count
        self._single_hand_dof_count = allegro_hand.joint_dof_count
        self._dofs_per_hand = self._single_hand_dof_count

        # The cube is body index 21 (DexCube), Dummy is 22
        # After ignoring Dummy, cube should be at index 21
        self._cube_body_offset = 21  # DexCube body index in single hand

        # Replicate across worlds
        builder = newton.ModelBuilder()
        builder.replicate(allegro_hand, self.num_worlds)

        # Add ground plane
        builder.default_shape_cfg.ke = cfg.shape_ke
        builder.default_shape_cfg.kd = cfg.shape_kd
        builder.add_ground_plane()

        self.model = builder.finalize()

        # Forward kinematics
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)

        # Store initial state for reset
        self._initial_joint_q = self.model.joint_q.numpy().copy()
        self._initial_joint_qd = self.model.joint_qd.numpy().copy()
        self._initial_body_q = self.model.body_q.numpy().copy()
        self._initial_body_qd = self.model.body_qd.numpy().copy()

        # Create solver with proper settings for contact
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            solver=cfg.solver_type,
            integrator=cfg.integrator,
            njmax=cfg.njmax,
            nconmax=cfg.nconmax,
            impratio=cfg.impratio,
            cone=cfg.cone,
            iterations=cfg.iterations,
            ls_iterations=cfg.ls_iterations,
            use_mujoco_cpu=False,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        # Pre-allocate tensors for direct array access
        self._joint_target_buffer = torch.zeros(
            self.model.joint_dof_count,
            device=self.device,
            dtype=torch.float32,
        )

        # Get joint limits for clamping actions
        self._joint_limit_lower = wp.to_torch(self.model.joint_limit_lower).to(self.device)
        self._joint_limit_upper = wp.to_torch(self.model.joint_limit_upper).to(self.device)

        # Create indices for extracting per-world joint data
        # Each world has _single_hand_dof_count DOFs
        self._world_dof_indices = torch.arange(
            self.num_worlds, device=self.device
        ).unsqueeze(1) * self._single_hand_dof_count + torch.arange(
            self._single_hand_dof_count, device=self.device
        ).unsqueeze(0)

        # Cube body indices per world
        self._cube_body_indices = (
            torch.arange(self.num_worlds, device=self.device) * self._single_hand_body_count
            + self._cube_body_offset
        )

    def _simulate_internal(self) -> None:
        # Update contacts
        self.contacts = self.model.collide(self.state_0)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # Apply forces from viewer (picking, etc.)
            if hasattr(self.viewer, "apply_forces"):
                self.viewer.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _get_obs(self) -> torch.Tensor:
        """Get observations: finger joints + cube state."""
        # Get all joint positions and velocities
        joint_q = wp.to_torch(self.state_0.joint_q).to(self.device)
        joint_qd = wp.to_torch(self.state_0.joint_qd).to(self.device)

        # Extract per-world joint data using gathered indices
        # Shape: (num_worlds, dofs_per_hand)
        world_joint_q = joint_q[self._world_dof_indices]
        world_joint_qd = joint_qd[self._world_dof_indices]

        # Get cube positions and velocities
        body_q = wp.to_torch(self.state_0.body_q).to(self.device)  # (num_bodies, 7)
        body_qd = wp.to_torch(self.state_0.body_qd).to(self.device)  # (num_bodies, 6)

        # Extract cube for each world
        cube_pos = body_q[self._cube_body_indices, :3]  # xyz position
        cube_vel = body_qd[self._cube_body_indices, :3]  # linear velocity

        # Concatenate all observations
        obs = torch.cat([world_joint_q, world_joint_qd, cube_pos, cube_vel], dim=1)
        return obs

    def _compute_rewards(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.hand_config

        # Extract cube height from observations
        # obs layout: [joint_q (dofs), joint_qd (dofs), cube_pos (3), cube_vel (3)]
        dofs = self._dofs_per_hand
        cube_z = obs[:, 2 * dofs + 2]  # z position
        cube_vel = obs[:, 2 * dofs + 3: 2 * dofs + 6]  # velocity

        # Height reward: closer to target is better
        height_diff = torch.abs(cube_z - cfg.cube_target_height)
        height_reward = cfg.height_reward_weight * (1.0 - torch.clamp(height_diff, 0, 1))

        # Bonus for keeping cube high
        height_bonus = torch.clamp(cube_z - cfg.cube_drop_threshold, 0, 1)

        # Velocity penalty (prefer stable cube)
        vel_penalty = cfg.velocity_penalty_weight * cube_vel.norm(dim=1)

        # Total reward
        reward = height_reward + height_bonus - vel_penalty

        # Episode terminates if cube drops below threshold
        dones = cube_z < cfg.cube_drop_threshold

        return reward, dones

    def reset(self) -> torch.Tensor:
        """Reset environment to initial state."""
        # Reset joint positions and velocities via state
        wp.copy(
            self.state_0.joint_q,
            wp.array(self._initial_joint_q, dtype=wp.float32, device=self.device),
        )
        wp.copy(
            self.state_0.joint_qd,
            wp.array(self._initial_joint_qd, dtype=wp.float32, device=self.device),
        )

        # Reset body poses
        wp.copy(
            self.state_0.body_q,
            wp.array(self._initial_body_q, dtype=wp.float32, device=self.device),
        )
        wp.copy(
            self.state_0.body_qd,
            wp.array(self._initial_body_qd, dtype=wp.float32, device=self.device),
        )

        # Reset control targets to initial grasp position
        initial_targets = torch.full(
            (self.model.joint_dof_count,),
            0.3,
            device=self.device,
            dtype=torch.float32,
        )
        wp.copy(
            self.control.joint_target_pos,
            wp.from_torch(initial_targets),
        )

        return self._get_obs()

    def _apply_actions(self, actions: torch.Tensor) -> None:
        """Apply actions as joint target positions."""
        # Actions shape: (num_worlds, action_dim) or (num_worlds,)
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)

        # Ensure actions are the right shape
        if actions.shape[1] != self._dofs_per_hand:
            # Pad or truncate
            if actions.shape[1] < self._dofs_per_hand:
                actions = torch.nn.functional.pad(
                    actions, (0, self._dofs_per_hand - actions.shape[1])
                )
            else:
                actions = actions[:, : self._dofs_per_hand]

        # Map actions from [-1, 1] to joint target range [0, 0.6]
        scaled_actions = (actions + 1.0) * 0.3  # Maps [-1,1] to [0, 0.6]

        # Flatten and scatter to the full joint target array
        joint_targets = self._joint_target_buffer.clone()

        # Scatter per-world actions to the flat array
        flat_indices = self._world_dof_indices.flatten()
        flat_actions = scaled_actions.flatten()
        joint_targets[flat_indices] = flat_actions

        # Copy to control
        wp.copy(
            self.control.joint_target_pos,
            wp.from_torch(joint_targets),
        )