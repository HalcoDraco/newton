from __future__ import annotations

import re
from typing import Tuple

import torch
import warp as wp

import newton
from newton.selection import ArticulationView

from .base import NewtonBaseEnv, NewtonTorchEnv
from .configs import AllegroHandConfig


class AllegroHandEnv(NewtonBaseEnv):
    """
    Vectorized Allegro Hand environment with cube reorientation task.

    Uses ArticulationView for efficient batch access to joint/body data.

    Observation: [finger_joint_q (16), finger_joint_qd (16), cube_pos (3), cube_quat (4), cube_lin_vel (3), cube_ang_vel (3), target_quat (4)] = 49 dims
    Action: Joint target positions for finger joints (16D)
    Task: Reorient the cube to match a randomly sampled target orientation

    Note on collision detection:
    - SolverMuJoCo handles collision detection internally
    - model.collide() is NOT needed and would be redundant
    - Pass contacts=None to solver.step() - MuJoCo computes contacts internally
    """

    def __init__(self, viewer: newton.viewer.ViewerNull | newton.viewer.ViewerGL, config: AllegroHandConfig):
        self.config = config
        # Target orientation for each world (will be set in reset)
        self._target_quat: torch.Tensor | None = None
        super().__init__(viewer, config)

    @property
    def obs_dim(self) -> int:
        # finger joints q + qd + cube_pos (3) + cube_quat (4) + cube_lin_vel (3) + cube_ang_vel (3) + target_quat (4)
        return self.hand_articulation.joint_dof_count * 2 + 3 + 4 + 3 + 3 + 4

    @property
    def action_dim(self) -> int:
        # Control finger joints (all DOFs in articulation)
        return self.hand_articulation.joint_dof_count

    def _build_model(self, config: AllegroHandConfig) -> newton.Model:

        # Build single allegro hand with cube
        allegro_hand = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(allegro_hand)
        allegro_hand.default_shape_cfg.ke = config.shape_ke
        allegro_hand.default_shape_cfg.kd = config.shape_kd

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
            allegro_hand.joint_target_ke[i] = config.joint_target_ke
            allegro_hand.joint_target_kd[i] = config.joint_target_kd
            allegro_hand.joint_target_pos[i] = config.initial_grasp_pos

        # Modify cube's initial position if specified
        # The cube's floating joint DOFs store [x, y, z, qx, qy, qz, qw]
        # Find the cube's joint and modify its position
        if config.initial_cube_position is not None:
            # Find the cube joint (it's a floating joint for DexCube)
            for i, key in enumerate(allegro_hand.joint_key):
                if "joint_22" in key or "DexCube" in key.lower():
                    # Get the DOF start for this joint
                    dof_start = allegro_hand.joint_q_start[i]
                    # Set position [x, y, z]
                    allegro_hand.joint_q[dof_start + 0] = config.initial_cube_position[0]
                    allegro_hand.joint_q[dof_start + 1] = config.initial_cube_position[1]
                    allegro_hand.joint_q[dof_start + 2] = config.initial_cube_position[2]
                    break

        # Replicate across worlds
        builder = newton.ModelBuilder()
        builder.replicate(allegro_hand, self.num_worlds)

        # Add ground plane
        builder.default_shape_cfg.ke = config.shape_ke
        builder.default_shape_cfg.kd = config.shape_kd
        builder.add_ground_plane()

        model = builder.finalize()
        return model

    def _define_articulation_views(self) -> None:
        """Define articulation views for efficient batch access."""
        # ArticulationView for the hand (Robot articulation)
        # Pattern "*Robot" matches articulation keys like "/World/envs/env_0/Robot"
        self.hand_articulation = ArticulationView(self.model, "*Robot", verbose=False)

        # ArticulationView for the cube (DexCube articulation)
        # Pattern "*DexCube" matches articulation keys like "/World/envs/env_0/object/DexCube"
        self.cube_articulation = ArticulationView(self.model, "*DexCube", verbose=False)

        # NOTE: eval_fk was already called in base.__init__ after _build_model
        # Now model.body_q contains FK-computed transforms

    def _pre_allocate_buffers(self) -> None:
        """Pre-allocate any buffers needed for simulation."""
        # Store initial state as Warp arrays for efficient reset (GPU-to-GPU copy)
        self._initial_joint_q = wp.clone(self.model.joint_q)
        self._initial_joint_qd = wp.clone(self.model.joint_qd)
        self._initial_body_q = wp.clone(self.model.body_q)
        self._initial_body_qd = wp.clone(self.model.body_qd)

        # Pre-allocate templates for efficient batched operations
        self._joint_target_template = torch.zeros(
            (self.num_worlds, self.hand_articulation.joint_dof_count),
            device=self.device,
            dtype=self.torch_dtype,
        )

        # Initialize target orientation for each world
        self._target_quat = torch.zeros(
            (self.num_worlds, 4),
            device=self.device,
            dtype=self.torch_dtype,
        )
        # Identity quaternion (x, y, z, w) format - warp uses scalar-last
        self._target_quat[:, 3] = 1.0

        # Desired cube position for position reward (same for all worlds in local coords)
        # The position is in local world coordinates, which are the same for all replicated worlds
        self._desired_cube_pos = torch.tensor(
            self.config.desired_cube_position,
            device=self.device,
            dtype=self.torch_dtype,
        ).unsqueeze(0).expand(self.num_worlds, -1)

    def _apply_actions(self, actions: wp.array) -> None:
        """Apply actions as joint target positions using ArticulationView."""
        # Actions shape: (num_worlds, action_dim)
        # actions_torch = wp.to_torch(actions)
        self._joint_target_template = wp.to_torch(actions)

        # expected_dim = self.hand_articulation.joint_dof_count
        # if actions_torch.shape[1] != expected_dim:
        #     if actions_torch.shape[1] < expected_dim:
        #         actions_torch = torch.nn.functional.pad(actions_torch, (0, expected_dim - actions_torch.shape[1]))
        #     else:
        #         actions_torch = actions_torch[:, :expected_dim]

        # Set joint target positions via ArticulationView
        # self._joint_target_template[:] = actions_torch
        self.hand_articulation.set_attribute("joint_target_pos", self.control, self._joint_target_template)

    def _get_obs(self) -> torch.Tensor:
        """Get observations using ArticulationView for efficient batch access."""
        # Get hand joint positions and velocities via ArticulationView
        joint_q = wp.to_torch(self.hand_articulation.get_attribute("joint_q", self.state_0))
        joint_qd = wp.to_torch(self.hand_articulation.get_attribute("joint_qd", self.state_0))

        # Get cube body state via ArticulationView
        # body_q shape: (num_worlds, num_bodies, 7) - [px, py, pz, qw, qx, qy, qz]
        # The DexCube articulation has 2 bodies: DexCube (index 0) and Dummy (index 1)
        # We only need the DexCube body (first body)
        cube_body_q = wp.to_torch(self.cube_articulation.get_attribute("body_q", self.state_0))
        cube_body_qd = wp.to_torch(self.cube_articulation.get_attribute("body_qd", self.state_0))

        # Select the DexCube body (index 0) - squeeze out the body dimension
        cube_body_q = cube_body_q[:, 0, :]  # Shape: (num_worlds, 7)
        cube_body_qd = cube_body_qd[:, 0, :]  # Shape: (num_worlds, 6)

        cube_pos = cube_body_q[:, :3]  # xyz position
        cube_quat = cube_body_q[:, 3:]  # quaternion (w, x, y, z)
        cube_lin_vel = cube_body_qd[:, :3]  # linear velocity
        cube_ang_vel = cube_body_qd[:, 3:]  # angular velocity

        # Concatenate all observations including target orientation
        # obs has size (num_worlds, obs_dim)
        obs = torch.cat([
            joint_q,
            joint_qd,
            cube_pos,
            cube_quat,
            cube_lin_vel,
            cube_ang_vel,
            self._target_quat,
        ], dim=1)

        return obs

    def _compute_rewards(self, obs: torch.Tensor) -> Tuple[wp.array, wp.array]:
        """Compute rewards for cube reorientation task."""
        # obs has size (num_worlds, obs_dim)
        dofs = self.hand_articulation.joint_dof_count

        # Parse observation layout:
        # [joint_q (dofs), joint_qd (dofs), cube_pos (3), cube_quat (4), cube_lin_vel (3), cube_ang_vel (3), target_quat (4)]
        offset = 2 * dofs
        cube_pos = obs[:, offset:offset + 3]
        cube_quat = obs[:, offset + 3:offset + 7]
        target_quat = obs[:, offset + 7 + 6:offset + 7 + 10]

        cube_z = cube_pos[:, 2]

        # Orientation reward: quaternion distance to target
        # Using quaternion dot product - closer to 1 (or -1) means better alignment
        # |q1 · q2| gives similarity (handles quaternion double-cover)
        quat_dot = torch.abs(torch.sum(cube_quat * target_quat, dim=1))
        quat_dot = torch.clamp(quat_dot, 0.0, 1.0)
        orientation_reward = self.config.orientation_reward_weight * quat_dot

        # Position reward: penalize deviation from desired position
        pos_diff = torch.norm(cube_pos - self._desired_cube_pos, dim=1)
        position_reward = self.config.position_reward_weight * torch.exp(-pos_diff / self.config.position_tolerance)

        # Success bonus: reward reaching target orientation
        orientation_error = 1.0 - quat_dot
        success_mask = (orientation_error < self.config.orientation_tolerance).float()
        success_bonus = self.config.success_bonus * success_mask

        # Total reward
        reward = orientation_reward + position_reward + success_bonus

        # Episode terminates if cube drops below threshold
        dones = (cube_z < self.config.cube_drop_threshold).to(torch.uint8)

        reward_wp = wp.from_torch(reward, dtype=self.wp_dtype)
        dones_wp = wp.from_torch(dones, dtype=wp.uint8)

        return reward_wp, dones_wp

    def _reset_state(self) -> None:
        """Reset environment to initial state and sample new target orientations."""
        # Reset ALL state arrays using efficient Warp-to-Warp copy (stays on GPU)
        # This restores both hand joints AND cube body position/orientation
        wp.copy(self.state_0.joint_q, self._initial_joint_q)
        wp.copy(self.state_0.joint_qd, self._initial_joint_qd)
        wp.copy(self.state_0.body_q, self._initial_body_q)
        wp.copy(self.state_0.body_qd, self._initial_body_qd)

        # ALSO copy joint_q to model.joint_q so eval_fk in base.reset() uses correct values
        wp.copy(self.model.joint_q, self._initial_joint_q)
        wp.copy(self.model.joint_qd, self._initial_joint_qd)

        # Reset control targets to initial grasp position
        self._joint_target_template.fill_(self.config.initial_grasp_pos)
        self.hand_articulation.set_attribute("joint_target_pos", self.control, self._joint_target_template)

        # Sample random target orientations for each world
        self._sample_target_orientations()

    def _sample_target_orientations(self) -> None:
        """Sample random target quaternions for cube reorientation task."""
        # Sample random rotations using uniform quaternion sampling
        # Method: sample 4 Gaussian values and normalize
        # Format: [x, y, z, w] (warp scalar-last convention)
        random_quat = torch.randn(self.num_worlds, 4, device=self.device, dtype=self.torch_dtype)
        random_quat = random_quat / random_quat.norm(dim=1, keepdim=True)

        # Ensure positive w component (canonical form) - w is at index 3
        sign = torch.sign(random_quat[:, 3:4])
        sign[sign == 0] = 1.0
        random_quat = random_quat * sign

        self._target_quat[:] = random_quat


    def render(self) -> None:
        """Render with orientation visualization arrows."""
        if type(self.viewer) is newton.viewer.ViewerNull:
            return

        # Get current cube positions and orientations
        cube_body_q = wp.to_torch(self.cube_articulation.get_attribute("body_q", self.state_0))
        cube_pos = cube_body_q[:, 0, :3]  # (num_worlds, 3)
        cube_quat = cube_body_q[:, 0, 3:]  # (num_worlds, 4) - [x, y, z, w] (warp scalar-last)

        # Get world offsets from viewer for visual positioning
        if self.viewer.world_offsets is not None:
            world_offsets = wp.to_torch(self.viewer.world_offsets)[:self.num_worlds]
        else:
            world_offsets = torch.zeros((self.num_worlds, 3), device=self.device, dtype=self.torch_dtype)

        # Apply world offsets to cube positions for visualization
        cube_pos_vis = cube_pos + world_offsets

        # Arrow length for visualization
        arrow_length = 0.15

        # Compute arrow directions from quaternions
        # We visualize the "up" direction (z-axis) rotated by each quaternion
        # This shows which way the cube's local z-axis is pointing

        # For current orientation (blue arrows)
        current_dir = self._quat_rotate_vector(cube_quat, torch.tensor([0.0, 0.0, 1.0], device=self.device))

        # For target orientation (green arrows)
        target_dir = self._quat_rotate_vector(self._target_quat, torch.tensor([0.0, 0.0, 1.0], device=self.device))

        # Build arrow arrays: first num_worlds are target (green), next num_worlds are current (blue)
        arrow_starts_torch = torch.zeros((self.num_worlds * 2, 3), device=self.device, dtype=self.torch_dtype)
        arrow_ends_torch = torch.zeros((self.num_worlds * 2, 3), device=self.device, dtype=self.torch_dtype)
        arrow_colors_torch = torch.zeros((self.num_worlds * 2, 3), device=self.device, dtype=self.torch_dtype)

        # Target orientation arrows (green) - start at cube position (with world offset)
        arrow_starts_torch[:self.num_worlds] = cube_pos_vis
        arrow_ends_torch[:self.num_worlds] = cube_pos_vis + target_dir * arrow_length
        arrow_colors_torch[:self.num_worlds] = torch.tensor([0.0, 1.0, 0.0], device=self.device)  # Green

        # Current orientation arrows (blue) - start at cube position, offset slightly
        offset = torch.tensor([0.02, 0.0, 0.0], device=self.device)
        arrow_starts_torch[self.num_worlds:] = cube_pos_vis + offset
        arrow_ends_torch[self.num_worlds:] = cube_pos_vis + offset + current_dir * arrow_length
        arrow_colors_torch[self.num_worlds:] = torch.tensor([0.0, 0.5, 1.0], device=self.device)  # Blue

        # Convert to warp arrays
        arrow_starts = wp.from_torch(arrow_starts_torch.contiguous(), dtype=wp.vec3)
        arrow_ends = wp.from_torch(arrow_ends_torch.contiguous(), dtype=wp.vec3)
        arrow_colors = wp.from_torch(arrow_colors_torch.contiguous(), dtype=wp.vec3)

        # Call base render
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)

        # Log orientation arrows
        self.viewer.log_lines(
            "/orientation_arrows",
            arrow_starts,
            arrow_ends,
            arrow_colors,
            width=0.005,
        )

        self.viewer.end_frame()

    def _quat_rotate_vector(self, quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """
        Rotate a vector by quaternions (batched).

        Args:
            quat: Quaternions of shape (N, 4) in [x, y, z, w] format (warp scalar-last)
            vec: Vector of shape (3,) to rotate

        Returns:
            Rotated vectors of shape (N, 3)
        """
        # Expand vector to match batch size
        vec = vec.unsqueeze(0).expand(quat.shape[0], -1)  # (N, 3)

        # Extract quaternion components [x, y, z, w]
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        # Rodrigues' rotation formula via quaternion
        # v' = v + 2*w*(q_xyz x v) + 2*(q_xyz x (q_xyz x v))
        q_xyz = quat[:, :3]  # (N, 3)

        # Cross products
        uv = torch.cross(q_xyz, vec, dim=1)
        uuv = torch.cross(q_xyz, uv, dim=1)

        return vec + 2.0 * (w.unsqueeze(1) * uv + uuv)


class AllegroHandTorchEnv(NewtonTorchEnv, AllegroHandEnv):
    """Allegro Hand environment with PyTorch interface."""
    pass
