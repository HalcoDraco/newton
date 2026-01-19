import newton
from newton.selection import ArticulationView
import warp as wp
import torch

from .base import NewtonBaseEnv, NewtonTorchEnv
from .configs import CartpoleConfig

class CartpoleEnv(NewtonBaseEnv):
    """
    Vectorized Cartpole environment.

    Observation: [cart_pos, pole_angle, cart_vel, pole_vel]
    Action: Force applied to cart (1D)
    """

    def __init__(self, viewer: newton.viewer.ViewerNull | newton.viewer.ViewerGL, config: CartpoleConfig):
        super().__init__(viewer, config)
        self.config = config

    @property
    def obs_dim(self) -> int:
        return 4
    
    @property
    def action_dim(self) -> int:
        return 1
    
    def _build_model(self, config: CartpoleConfig) -> newton.Model:
        
        world = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(world)
        world.default_shape_cfg.density = self.config.density
        world.default_joint_cfg.armature = self.config.armature
        world.default_body_armature = self.config.armature

        world.add_usd(
            newton.examples.get_asset("cartpole_single_pendulum.usda"),
            collapse_fixed_joints=True,
            enable_self_collisions=False,
        )
        world.joint_q[-2:] = [0.0, 0.3]

        # Replicate across worlds
        scene = newton.ModelBuilder()
        scene.replicate(world, num_worlds=self.num_worlds, spacing=self.config.spacing)

        model = scene.finalize()

        return model
    
    def _define_articulation_views(self) -> None:
        """Define articulation views for efficient batch access."""
        self.articulation = ArticulationView(self.model, "/cartpole", verbose=False)

        self._joint_f_template = torch.zeros(
            (self.num_worlds, self.articulation.joint_dof_count),
            device=self.device,
            dtype=self.torch_dtype,
        )

    def _apply_actions(self, actions: wp.array):
        """Apply actions to the control array."""
        # Actions shape: (num_worlds, action_dim)
        actions = wp.to_torch(actions)
        self._joint_f_template[:, 0] = actions[:, 0]
        self.articulation.set_attribute("joint_f", self.control, self._joint_f_template)

    def _get_obs(self) -> torch.Tensor:
        joint_q = wp.to_torch(self.articulation.get_attribute("joint_q", self.state_0))
        joint_qd = wp.to_torch(self.articulation.get_attribute("joint_qd", self.state_0))

        obs = torch.cat([joint_q, joint_qd], dim=1)
        return obs
    
    def _compute_rewards(self, obs: torch.Tensor) -> tuple[wp.array, wp.array]:
        cart_pos, pole_angle, cart_vel, pole_vel = obs.T

        # Reward shaping
        angle_cost = self.config.angle_cost_weight * pole_angle.abs()
        vel_cost = self.config.vel_cost_weight * (cart_vel.abs() + pole_vel.abs())
        pos_cost = self.config.pos_cost_weight * cart_pos.abs()

        reward = 1.0 - (angle_cost + vel_cost + pos_cost)

        # Termination conditions
        dones = (pole_angle.abs() > self.config.pole_angle_threshold) | (cart_pos.abs() > self.config.cart_pos_threshold)
        dones = dones.to(torch.uint8)

        reward = wp.from_torch(reward, dtype=self.wp_dtype)
        dones = wp.from_torch(dones, dtype=wp.uint8)
        return reward, dones
    
    def _reset_state(self):
        cart_positions = self.config.cart_pos_range * (torch.rand(self.num_worlds, device=self.device) - 0.5)
        pole_angles = self.config.pole_angle_range * (torch.rand(self.num_worlds, device=self.device) - 0.5)

        joint_q = torch.stack([cart_positions, pole_angles], dim=1)
        joint_qd = torch.zeros_like(joint_q)

        self.articulation.set_attribute("joint_q", self.state_0, joint_q)
        self.articulation.set_attribute("joint_qd", self.state_0, joint_qd)


class CartpoleTorchEnv(NewtonTorchEnv, CartpoleEnv):
    """Cartpole environment with PyTorch interface."""
    pass