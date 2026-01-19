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

from .configs import EnvConfig


class NewtonBaseEnv(ABC):

    def __init__(self, viewer: newton.viewer.ViewerNull | newton.viewer.ViewerGL, config: EnvConfig):

        assert config.physics_hz % config.control_hz == 0, "physics_hz must be multiple of control_hz"

        self.viewer = viewer
        self.config = config

        wp.init()
        self.device = "cuda" if wp.get_device().is_cuda else "cpu"
        assert self.device == "cuda", "Currently only CUDA device is supported"

        self.seed = config.seed
        self.wp_dtype = config.wp_dtype
        self.torch_dtype = config.torch_dtype
        self.num_worlds = config.num_worlds

        self.control_hz = config.control_hz
        self.physics_hz = config.physics_hz
        self.frame_dt = config.frame_dt
        self.sim_substeps = config.sim_substeps
        self.sim_dt = config.sim_dt
        self.sim_time = 0.0

        self.model = self._build_model(config)
        self.contacts = None
        self.solver = self._build_solver(self.model, config)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self._define_articulation_views()
        self.viewer.set_model(self.model)
        self._capture_cuda_graph()

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
    def _build_model(self, config: EnvConfig) -> newton.Model:
        """
        Build the Newton model.
        call finalize()
        """
        pass

    def _build_solver(self, model, config: EnvConfig) -> Any:
        """Helper to build solver from config."""
        if config.solver_class == "SolverMuJoCo":
            solver = newton.solvers.SolverMuJoCo(
                model,
                mjw_model=config.mjw_model,
                mjw_data=config.mjw_data,
                separate_worlds=config.separate_worlds,
                njmax=config.njmax,
                nconmax=config.nconmax,
                iterations=config.iterations,
                ls_iterations=config.ls_iterations,
                solver=config.solver,
                integrator=config.integrator,
                cone=config.cone,
                impratio=config.impratio,
                use_mujoco_cpu=config.use_mujoco_cpu,
                disable_contacts=config.disable_contacts,
                default_actuator_gear=config.default_actuator_gear,
                actuator_gears=config.actuator_gears,
                update_data_interval=config.update_data_interval,
                save_to_mjcf=config.save_to_mjcf,
                ls_parallel=config.ls_parallel,
                use_mujoco_contacts=config.use_mujoco_contacts,
                tolerance=config.tolerance,
                ls_tolerance=config.ls_tolerance,
                include_sites=config.include_sites,
            )

            self.contacts = None # No external collision detection needed - MuJoCo handles it internally
        else:
            raise ValueError(f"Unsupported solver class: {config.solver_class}")
        return solver

    def _define_articulation_views(self) -> None:
        """Define articulation views for efficient batch access."""
        pass

    def _capture_cuda_graph(self) -> None:
        """Capture CUDA graph for simulation."""
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self._simulate()
            self.graph = capture.graph
    
    def _simulate(self) -> None:
        """Run simulation substeps to perform a single control step."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _sim_step(self) -> None:
        """Step the simulation."""
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._simulate()
        self.sim_time += self.frame_dt

    @abstractmethod
    def _apply_actions(self, actions: wp.array):
        pass

    @abstractmethod
    def _get_obs(self) -> torch.Tensor:
        pass

    @abstractmethod
    def _compute_rewards(self, obs: torch.Tensor) -> tuple[wp.array, wp.array]:
        pass

    @abstractmethod
    def _reset_state(self):
        pass

    def _from_warp(self, array: wp.array) -> Any:
        """Override to convert output from warp array to desired format."""
        pass

    def _to_warp(self, array: Any) -> wp.array:
        """Override to convert input from desired format to warp array."""
        pass

    def reset(self):
        self._reset_state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        obs = self._get_obs()
        obs = wp.from_torch(obs, dtype=self.wp_dtype)
        return self._from_warp(obs)

    def step(self, actions):
        actions = self._to_warp(actions)

        self._apply_actions(actions)
        self._sim_step()
        obs = self._get_obs()
        rewards, dones = self._compute_rewards(obs)

        obs = wp.from_torch(obs, dtype=self.wp_dtype)
        obs = self._from_warp(obs)
        rewards = self._from_warp(rewards)
        dones = self._from_warp(dones)
        return obs, rewards, dones
    
    def render(self) -> None:
        if type(self.viewer) is newton.viewer.ViewerNull:
            return # To ensure ViewerNull does nothing on render, since log_contacts may not be overriden there

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()
    
    def close(self) -> None:
        """Clean up resources."""
        self.viewer.close()

class NewtonTorchEnv(NewtonBaseEnv):
    """Base class for Newton environments with PyTorch interface."""

    def _from_warp(self, array: wp.array) -> torch.Tensor:
        return wp.to_torch(array)

    def _to_warp(self, array: torch.Tensor) -> wp.array:
        return wp.from_torch(array, dtype=self.wp_dtype)