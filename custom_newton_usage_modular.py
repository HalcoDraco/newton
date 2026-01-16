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


# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass
class EnvConfig:
    """Base configuration for Newton environments."""

    num_worlds: int
    fps: int = 60
    sim_substeps: int = 10
    seed: int = 0

    @property
    def frame_dt(self) -> float:
        return 1.0 / self.fps

    @property
    def sim_dt(self) -> float:
        return self.frame_dt / self.sim_substeps


@dataclass
class CartpoleConfig(EnvConfig):
    """Configuration for Cartpole environment."""

    # World layout
    spacing: Tuple[float, float, float] = (0.8, 4.0, 0.0)

    # Physics
    density: float = 100.0
    armature: float = 0.1

    # Reset ranges
    cart_pos_range: float = 0.5
    pole_angle_range: float = 0.25

    # Reward coefficients
    angle_cost_weight: float = 2.0
    vel_cost_weight: float = 0.05
    pos_cost_weight: float = 0.1

    # Termination thresholds
    pole_angle_threshold: float = 0.8
    cart_pos_threshold: float = 2.4


@dataclass
class AllegroHandConfig(EnvConfig):
    """Configuration for Allegro Hand environment."""

    # Timing (override defaults for hand)
    fps: int = 50
    sim_substeps: int = 8

    # Physics
    shape_ke: float = 1.0e3  # Contact stiffness
    shape_kd: float = 1.0e2  # Contact damping
    joint_target_ke: float = 150.0  # Joint drive stiffness
    joint_target_kd: float = 5.0  # Joint drive damping

    # Solver settings
    solver_type: str = "newton"
    integrator: str = "implicitfast"
    njmax: int = 200
    nconmax: int = 150
    impratio: float = 10.0
    cone: str = "elliptic"
    iterations: int = 100
    ls_iterations: int = 50

    # Task: cube height target
    cube_target_height: float = 0.5  # Target height for cube
    cube_drop_threshold: float = 0.1  # Below this, episode ends

    # Reward weights
    height_reward_weight: float = 1.0
    action_penalty_weight: float = 0.01
    velocity_penalty_weight: float = 0.001


@dataclass
class TrainerConfig:
    """Base configuration for trainers."""

    generations: int = 50
    episode_steps: int = 300
    render_every: int = 1


@dataclass
class GeneticTrainerConfig(TrainerConfig):
    """Configuration for Genetic Algorithm trainer."""

    elite_frac: float = 0.2
    noise_std: float = 0.05
    hidden_size: int = 64
    action_scale: float = 10.0
    force_limit: float = 40.0


@dataclass
class RandomTrainerConfig(TrainerConfig):
    """Configuration for Random Action trainer (baseline)."""

    action_scale: float = 10.0  # Scale for random actions
    action_limit: float = 40.0  # Clamp actions to [-limit, limit]


# =============================================================================
# Policy Network
# =============================================================================


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


# =============================================================================
# Abstract Base Classes
# =============================================================================


class NewtonEnvBase(ABC):
    """
    Abstract base class for vectorized Newton physics environments.

    This class provides common infrastructure for:
    - Warp/CUDA initialization
    - Simulation timing
    - CUDA graph capture
    - Generic step/render logic

    Subclasses must implement environment-specific methods:
    - obs_dim, action_dim properties
    - _build_world(), _simulate_internal()
    - _get_obs(), _compute_rewards()
    - reset(), _apply_actions()
    """

    def __init__(self, viewer: Any, config: EnvConfig):
        self.viewer = viewer
        self.config = config
        self.num_worlds = config.num_worlds

        # Initialize warp and device
        torch.manual_seed(config.seed)
        wp.init()
        self.device = "cuda" if wp.get_device().is_cuda else "cpu"

        # Timing
        self.fps = config.fps
        self.frame_dt = config.frame_dt
        self.sim_time = 0.0
        self.sim_substeps = config.sim_substeps
        self.sim_dt = config.sim_dt

        # To be set by subclass in _build_world()
        self.model: Any = None
        self.solver: Any = None
        self.state_0: Any = None
        self.state_1: Any = None
        self.control: Any = None
        self.contacts: Any = None
        self.graph: Any = None

        # Build environment
        self._build_world()
        self._setup_viewer()
        self._capture_graph()

    @property
    @abstractmethod
    def obs_dim(self) -> int:
        """Observation space dimension."""
        pass

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Action space dimension."""
        pass

    @abstractmethod
    def _build_world(self) -> None:
        """
        Build the Newton model, solver, and states.

        Must set:
        - self.model
        - self.solver
        - self.state_0, self.state_1
        - self.control
        - self.contacts (optional, for collision-enabled envs)
        """
        pass

    @abstractmethod
    def _simulate_internal(self) -> None:
        """Run simulation substeps. Called during CUDA graph capture."""
        pass

    @abstractmethod
    def _get_obs(self) -> torch.Tensor:
        """
        Get observations from current state.

        Returns:
            Tensor of shape (num_worlds, obs_dim)
        """
        pass

    @abstractmethod
    def _compute_rewards(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rewards and done flags.

        Args:
            obs: Current observations (num_worlds, obs_dim)

        Returns:
            rewards: Tensor of shape (num_worlds,)
            dones: Boolean tensor of shape (num_worlds,)
        """
        pass

    @abstractmethod
    def reset(self) -> torch.Tensor:
        """
        Reset environment and return initial observations.

        Returns:
            Initial observations (num_worlds, obs_dim)
        """
        pass

    @abstractmethod
    def _apply_actions(self, actions: torch.Tensor) -> None:
        """
        Apply actions to the environment.

        Args:
            actions: Tensor of shape (num_worlds,) or (num_worlds, action_dim)
        """
        pass

    def _setup_viewer(self) -> None:
        """Setup viewer with model. Override for custom viewer setup."""
        self.viewer.set_model(self.model)

    def _capture_graph(self) -> None:
        """Capture CUDA graph for simulation."""
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self._simulate_internal()
            self.graph = capture.graph

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Step the environment.

        Args:
            actions: Tensor of shape (num_worlds,) or (num_worlds, action_dim)

        Returns:
            obs: Tensor of shape (num_worlds, obs_dim)
            rewards: Tensor of shape (num_worlds,)
            dones: Boolean tensor of shape (num_worlds,)
            info: Dictionary with additional info
        """
        # Apply actions
        self._apply_actions(actions)

        # Step simulation
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._simulate_internal()
        self.sim_time += self.frame_dt

        # Get state & rewards
        obs = self._get_obs()
        rewards, dones = self._compute_rewards(obs)

        return obs, rewards, dones, {}

    def render(self) -> None:
        """Render current state."""
        if hasattr(self.viewer, "begin_frame"):
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            if self.contacts is not None and hasattr(self.viewer, "log_contacts"):
                self.viewer.log_contacts(self.contacts, self.state_0)
            if hasattr(self.viewer, "end_frame"):
                self.viewer.end_frame()

    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self.viewer, "close"):
            self.viewer.close()


class TrainerBase(ABC):
    """
    Abstract base class for trainers.

    Trainers operate on NewtonEnvBase instances and implement
    specific training algorithms (GA, PPO, SAC, etc.)
    """

    def __init__(self, env: NewtonEnvBase, config: TrainerConfig):
        self.env = env
        self.config = config
        self.device = env.device

    @abstractmethod
    def train(self) -> None:
        """Run the training loop."""
        pass

    @abstractmethod
    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get action for given observation (inference mode).

        Args:
            obs: Observations (num_worlds, obs_dim)

        Returns:
            Actions (num_worlds,) or (num_worlds, action_dim)
        """
        pass


# =============================================================================
# Concrete Environment Implementations
# =============================================================================


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


# =============================================================================
# Concrete Trainer Implementations
# =============================================================================


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


class RandomTrainer(TrainerBase):
    """
    Random Action trainer (baseline).

    Samples random actions uniformly from the action space.
    Useful as a baseline for comparison with learning algorithms.
    """

    def __init__(self, env: NewtonEnvBase, config: RandomTrainerConfig):
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
            if hasattr(self.env.viewer, "is_running") and not self.env.viewer.is_running():
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
                obs, reward, dones, _ = self.env.step(masked_actions)

                # Update alive mask and accumulate rewards
                alive_mask = alive_mask & (~dones)
                reward = reward * alive_mask
                episode_returns += reward

                # Render
                if cfg.render_every > 0 and gen % cfg.render_every == 0:
                    self.env.render()

            # Logging
            mean_reward = episode_returns.mean().item()
            top_reward = episode_returns.max().item()

            if top_reward > best_reward:
                best_reward = top_reward

            total_reward_sum += mean_reward
            total_episodes += 1

            elapsed = time.perf_counter() - gen_start
            frames = cfg.episode_steps * self.env.num_worlds
            fps = frames / elapsed if elapsed > 0 else float("inf")

            print(
                f"[Episode {gen:03d}] mean_reward={mean_reward:.3f} "
                f"best_episode={top_reward:.3f} "
                f"best_all={best_reward:.3f} time={elapsed:.2f}s fps={fps:.1f}"
            )

        # Final statistics
        avg_reward = total_reward_sum / total_episodes if total_episodes > 0 else 0.0
        print(f"\nRandom baseline results: avg_reward={avg_reward:.3f} best={best_reward:.3f}")


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
    parser = make_parser()
    viewer, args = newton.examples.init(parser)

    if args.viewer == "null":
        args.render_every = 0

    # Create environment config
    env_config = CartpoleConfig(
        num_worlds=args.num_worlds,
        seed=args.seed,
    )

    # Create trainer config
    trainer_config = RandomTrainerConfig()

    # Initialize environment and trainer
    env = CartpoleEnv(viewer, env_config)
    trainer = RandomTrainer(env, trainer_config)

    # Train
    trainer.train()

    # Cleanup
    env.close()
    # parser = make_parser()
    # viewer, args = newton.examples.init(parser)

    # if args.viewer == "null":
    #     args.render_every = 0

    # # Create environment config (using AllegroHand)
    # env_config = AllegroHandConfig(
    #     num_worlds=args.num_worlds,
    #     seed=args.seed,
    # )

    # # Create trainer config
    # # Note: action_scale and force_limit are less relevant for position control
    # trainer_config = GeneticTrainerConfig(
    #     generations=args.generations,
    #     episode_steps=args.episode_steps,
    #     elite_frac=args.elite_frac,
    #     noise_std=args.noise_std,
    #     hidden_size=args.hidden_size,
    #     action_scale=1.0,  # Actions are [-1, 1] for position targets
    #     force_limit=1.0,   # Clamp to [-1, 1]
    #     render_every=args.render_every,
    # )

    # # Initialize environment and trainer
    # env = AllegroHandEnv(viewer, env_config)
    # env = CartpoleEnv(viewer, )
    # trainer = GeneticTrainer(env, trainer_config)

    # # Train
    # trainer.train()

    # # Cleanup
    # env.close()


if __name__ == "__main__":
    main()