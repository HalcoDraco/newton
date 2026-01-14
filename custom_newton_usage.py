"""
Genetic algorithm training for Newton cartpole with many GPU worlds.
Run with: uv run --extra torch-cu12 custom_newton_usage.py --viewer gl
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import warp as wp
from torch.func import functional_call, vmap

import newton
import newton.examples
from newton.selection import ArticulationView


torch.set_float32_matmul_precision("medium")


@dataclass
class GAParams:
    num_worlds: int
    generations: int
    episode_steps: int
    elite_frac: float
    noise_std: float
    hidden_size: int
    action_scale: float
    force_limit: float
    render_every: int
    seed: int


class Policy(torch.nn.Module):
    def __init__(self, obs_dim: int, hidden: int, action_dim: int = 1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


class GeneticCartpoleTrainer:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.params = GAParams(
            num_worlds=args.num_worlds,
            generations=args.generations,
            episode_steps=args.episode_steps,
            elite_frac=args.elite_frac,
            noise_std=args.noise_std,
            hidden_size=args.hidden_size,
            action_scale=args.action_scale,
            force_limit=args.force_limit,
            render_every=args.render_every,
            seed=args.seed,
        )

        torch.manual_seed(self.params.seed)
        wp.init()
        self.device = "cuda" if wp.get_device().is_cuda else "cpu"

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self._build_world()
        self._build_policy()
        self._init_population()
        self._capture_graph()

    def _build_world(self):
        world = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(world)
        world.default_shape_cfg.density = 100.0
        world.default_joint_cfg.armature = 0.1
        world.default_body_armature = 0.1

        world.add_usd(
            newton.examples.get_asset("cartpole.usda"),
            collapse_fixed_joints=False,
            enable_self_collisions=False,
        )
        world.joint_q[-3:] = [0.0, 0.2, 0.0]

        scene = newton.ModelBuilder()
        scene.replicate(world, num_worlds=self.params.num_worlds, spacing=(2.0, 2.0, 0.0))

        self.model = scene.finalize()
        self.solver = newton.solvers.SolverMuJoCo(self.model, disable_contacts=True)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.cartpoles = ArticulationView(self.model, "/cartPole", verbose=False)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_world_offsets"):
            self.viewer.set_world_offsets((2.0, 2.0, 0.0))

        self.obs_dim = 2 * self.cartpoles.joint_coord_count
        self.joint_f_template = torch.zeros(
            (self.params.num_worlds, self.cartpoles.joint_dof_count), device=self.device, dtype=torch.float32
        )

    def _build_policy(self):
        self.policy = Policy(self.obs_dim, self.params.hidden_size, action_dim=1).to(self.device)
        self.policy.eval()
        base = {k: v.detach().to(self.device) for k, v in self.policy.named_parameters()}
        self.base_params = base

    def _init_population(self):
        self.population_params: dict[str, torch.Tensor] = {}
        for name, param in self.base_params.items():
            noise = self.params.noise_std * torch.randn(
                (self.params.num_worlds,) + param.shape, device=self.device, dtype=param.dtype
            )
            self.population_params[name] = param.unsqueeze(0).expand_as(noise) + noise

        params_in_dims = {k: 0 for k in self.population_params}

        def policy_apply(p, obs):
            return functional_call(self.policy, p, (obs,))

        self.batched_policy = vmap(policy_apply, in_dims=(params_in_dims, 0))

    def _capture_graph(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self._simulate_internal()
            self.graph = capture.graph

    def _reset_worlds(self):
        cart_positions = 0.5 * (torch.rand(self.params.num_worlds, device=self.device) - 0.5)
        pole1_angles = 0.25 * (torch.rand(self.params.num_worlds, device=self.device) - 0.5)
        pole2_angles = 0.25 * (torch.rand(self.params.num_worlds, device=self.device) - 0.5)
        joint_q = torch.stack([cart_positions, pole1_angles, pole2_angles], dim=1)
        joint_qd = torch.zeros_like(joint_q)
        self.cartpoles.set_attribute("joint_q", self.state_0, joint_q)
        self.cartpoles.set_attribute("joint_qd", self.state_0, joint_qd)
        if not isinstance(self.solver, newton.solvers.SolverMuJoCo):
            self.cartpoles.eval_fk(self.state_0)

    def _get_obs(self) -> torch.Tensor:
        joint_q = wp.to_torch(self.cartpoles.get_attribute("joint_q", self.state_0)).to(self.device)
        joint_qd = wp.to_torch(self.cartpoles.get_attribute("joint_qd", self.state_0)).to(self.device)
        return torch.cat([joint_q, joint_qd], dim=1)

    def _compute_actions(self, obs: torch.Tensor, alive_mask: torch.Tensor) -> torch.Tensor:
        actions = self.batched_policy(self.population_params, obs)
        actions = torch.tanh(actions.squeeze(-1)) * self.params.action_scale
        actions = torch.clamp(actions, -self.params.force_limit, self.params.force_limit)
        return actions * alive_mask

    def _apply_actions(self, actions: torch.Tensor, alive_mask: torch.Tensor):
        joint_f = self.joint_f_template.clone()
        joint_f[:, 0] = actions
        joint_f *= alive_mask.unsqueeze(1)
        self.cartpoles.set_attribute("joint_f", self.control, joint_f)

    def _simulate_internal(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            if hasattr(self.viewer, "apply_forces"):
                self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _step_sim(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._simulate_internal()
        self.sim_time += self.frame_dt

    def _compute_reward(self, obs: torch.Tensor, alive_mask: torch.Tensor):
        cart_pos, pole1, pole2, cart_vel, pole1_vel, pole2_vel = obs.T
        angle_cost = 2.0 * pole1.abs() + 1.5 * pole2.abs()
        vel_cost = 0.05 * (cart_vel.abs() + pole1_vel.abs() + pole2_vel.abs())
        pos_cost = 0.1 * cart_pos.abs()
        reward = 1.0 - (angle_cost + vel_cost + pos_cost)
        fail = (pole1.abs() > 0.8) | (pole2.abs() > 0.8) | (cart_pos.abs() > 2.4)
        alive_mask = alive_mask & (~fail)
        reward = reward * alive_mask
        return reward, alive_mask

    def _render(self):
        if hasattr(self.viewer, "begin_frame"):
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            if hasattr(self.viewer, "end_frame"):
                self.viewer.end_frame()

    def train(self):
        best_reward = -float("inf")
        best_params = None

        for gen in range(self.params.generations):
            if hasattr(self.viewer, "is_running") and not self.viewer.is_running():
                break

            self._reset_worlds()
            episode_returns = torch.zeros(self.params.num_worlds, device=self.device)
            alive_mask = torch.ones(self.params.num_worlds, device=self.device, dtype=torch.bool)
            obs = self._get_obs()

            for step in range(self.params.episode_steps):
                actions = self._compute_actions(obs, alive_mask)
                self._apply_actions(actions, alive_mask)
                self._step_sim()
                obs = self._get_obs()
                reward, alive_mask = self._compute_reward(obs, alive_mask)
                episode_returns += reward

                if self.params.render_every > 0 and gen % self.params.render_every == 0:
                    self._render()

            mean_reward = episode_returns.mean().item()
            top_reward, top_idx = torch.max(episode_returns, dim=0)
            if top_reward.item() > best_reward:
                best_reward = top_reward.item()
                best_params = {k: v[top_idx].detach().clone() for k, v in self.population_params.items()}

            elite_count = max(1, int(self.params.elite_frac * self.params.num_worlds))
            elite_scores, elite_idx = torch.topk(episode_returns, k=elite_count)
            self.population_params = self._mutate(elite_idx)

            print(f"[Gen {gen:03d}] mean_reward={mean_reward:.3f} best_gen={elite_scores.max().item():.3f} best_all={best_reward:.3f}")

        if best_params is not None:
            self.population_params = {k: v.unsqueeze(0).expand(self.params.num_worlds, *v.shape) for k, v in best_params.items()}

    def _mutate(self, elite_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        new_params: dict[str, torch.Tensor] = {}
        repeats = math.ceil(self.params.num_worlds / elite_idx.numel())
        for name, tensor in self.population_params.items():
            elites = tensor[elite_idx]
            tiled = elites.repeat((repeats,) + (1,) * (elites.dim() - 1))
            trimmed = tiled[: self.params.num_worlds]
            noise = self.params.noise_std * torch.randn_like(trimmed)
            new_params[name] = trimmed + noise
        return new_params


def make_parser():
    parser = newton.examples.create_parser()
    parser.add_argument("--num-worlds", type=int, default=64, help="Number of parallel cartpoles / policies.")
    parser.add_argument("--generations", type=int, default=50, help="Training generations.")
    parser.add_argument("--episode-steps", type=int, default=300, help="Steps per generation.")
    parser.add_argument("--elite-frac", type=float, default=0.2, help="Top fraction kept each generation.")
    parser.add_argument("--noise-std", type=float, default=0.05, help="Gaussian noise std for mutation.")
    parser.add_argument("--hidden-size", type=int, default=64, help="Policy hidden width.")
    parser.add_argument("--action-scale", type=float, default=10.0, help="Scale for raw policy output.")
    parser.add_argument("--force-limit", type=float, default=40.0, help="Clamp absolute force.")
    parser.add_argument("--render-every", type=int, default=1, help="Render every N generations (0 disables).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser


def main():
    parser = make_parser()
    viewer, args = newton.examples.init(parser)
    if args.viewer == "null":
        args.render_every = 0

    trainer = GeneticCartpoleTrainer(viewer, args)
    trainer.train()

    if hasattr(viewer, "close"):
        viewer.close()


if __name__ == "__main__":
    main()
