from __future__ import annotations

import re
from typing import Any, Dict, Tuple

import torch
import warp as wp

import newton
import newton.examples
from newton.selection import ArticulationView

torch.set_float32_matmul_precision("medium")

from .env_base import NewtonEnvBase
from .configs import AllegroHandConfig


class AllegroHandEnv(NewtonEnvBase):
    """
    Vectorized Allegro Hand environment with cube manipulation task.

    Uses ArticulationView for efficient batch access to joint/body data.

    Observation: [finger_joint_q (20), finger_joint_qd (20), cube_pos (3), cube_vel (3)] = 46 dims
    Action: Joint target positions for finger joints (20D)
    Task: Keep the cube elevated above a target height

    Note on collision detection:
    - SolverMuJoCo handles collision detection internally
    - model.collide() is NOT needed and would be redundant (2x slowdown)
    - Pass contacts=None to solver.step() - MuJoCo computes contacts internally
    """

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
        return self.articulation.joint_dof_count * 2 + 6

    @property
    def action_dim(self) -> int:
        # Control finger joints (all DOFs in articulation)
        return self.articulation.joint_dof_count

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

        # Store body count before replication (for cube index calculation)
        self._single_hand_body_count = allegro_hand.body_count
        self._cube_body_offset = 21  # DexCube body index in single hand

        # Replicate across worlds
        builder = newton.ModelBuilder()
        builder.replicate(allegro_hand, self.num_worlds)

        # Add ground plane
        builder.default_shape_cfg.ke = cfg.shape_ke
        builder.default_shape_cfg.kd = cfg.shape_kd
        builder.add_ground_plane()

        self.model = builder.finalize()

        # Create ArticulationView for efficient batch access to hand joints
        # This provides optimized get/set methods for joint data across all worlds
        self.articulation = ArticulationView(self.model, "*Robot*", verbose=False)

        # Forward kinematics
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)

        # Store initial state as Warp arrays (avoid numpy copies in hot path)
        self._initial_joint_q = wp.clone(self.model.joint_q)
        self._initial_joint_qd = wp.clone(self.model.joint_qd)
        self._initial_body_q = wp.clone(self.model.body_q)
        self._initial_body_qd = wp.clone(self.model.body_qd)

        # Create solver - MuJoCo handles collision detection internally
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
        # No external collision detection needed - MuJoCo handles it internally
        self.contacts = None

        # Pre-compute cube body indices for each world (for observation extraction)
        self._cube_body_indices = torch.tensor(
            [i * self._single_hand_body_count + self._cube_body_offset for i in range(self.num_worlds)],
            device=self.device,
            dtype=torch.long,
        )

        # Pre-allocate action template for ArticulationView
        self._joint_target_template = torch.zeros(
            (self.num_worlds, self.articulation.joint_dof_count),
            device=self.device,
            dtype=torch.float32,
        )

    def _simulate_internal(self) -> None:
        # MuJoCo solver handles collision detection internally when contacts=None
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # Apply forces from viewer (picking, etc.) - only if viewer exists
            if self.viewer is not None and hasattr(self.viewer, "apply_forces"):
                self.viewer.apply_forces(self.state_0)

            # Pass contacts=None - MuJoCo computes collisions internally
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _get_obs(self) -> torch.Tensor:
        """Get observations using ArticulationView for efficient batch access."""
        # Get joint positions and velocities via ArticulationView (optimized batch access)
        joint_q = wp.to_torch(self.articulation.get_attribute("joint_q", self.state_0)).to(self.device)
        joint_qd = wp.to_torch(self.articulation.get_attribute("joint_qd", self.state_0)).to(self.device)

        # Get cube body positions and velocities
        body_q = wp.to_torch(self.state_0.body_q).to(self.device)  # (num_bodies, 7)
        body_qd = wp.to_torch(self.state_0.body_qd).to(self.device)  # (num_bodies, 6)

        # Extract cube for each world
        cube_pos = body_q[self._cube_body_indices, :3]  # xyz position
        cube_vel = body_qd[self._cube_body_indices, :3]  # linear velocity

        # Concatenate all observations
        obs = torch.cat([joint_q, joint_qd, cube_pos, cube_vel], dim=1)
        return obs

    def _compute_rewards(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.hand_config
        dofs = self.articulation.joint_dof_count

        # Extract cube height from observations
        # obs layout: [joint_q (dofs), joint_qd (dofs), cube_pos (3), cube_vel (3)]
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
        """Reset environment to initial state using efficient Warp array copies."""
        # Reset state arrays (Warp-to-Warp copy, stays on GPU)
        wp.copy(self.state_0.joint_q, self._initial_joint_q)
        wp.copy(self.state_0.joint_qd, self._initial_joint_qd)
        wp.copy(self.state_0.body_q, self._initial_body_q)
        wp.copy(self.state_0.body_qd, self._initial_body_qd)

        # Reset control targets to initial grasp position
        self._joint_target_template.fill_(0.3)
        self.articulation.set_attribute("joint_target_pos", self.control, self._joint_target_template)

        return self._get_obs()

    def _apply_actions(self, actions: torch.Tensor) -> None:
        """Apply actions as joint target positions using ArticulationView."""
        # Actions shape: (num_worlds, action_dim) or (num_worlds,)
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)

        # Ensure actions are the right shape
        expected_dim = self.articulation.joint_dof_count
        if actions.shape[1] != expected_dim:
            if actions.shape[1] < expected_dim:
                actions = torch.nn.functional.pad(actions, (0, expected_dim - actions.shape[1]))
            else:
                actions = actions[:, :expected_dim]

        # Scale actions from [-1, 1] to joint target range [0, 0.6]
        scaled_actions = (actions + 1.0) * 0.3

        # Apply via ArticulationView (handles index mapping efficiently)
        self.articulation.set_attribute("joint_target_pos", self.control, scaled_actions)