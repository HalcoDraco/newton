from __future__ import annotations

import torch

import newton
import newton.examples

torch.set_float32_matmul_precision("medium")

from newton.newton_learning_examples.envs import CartpoleTorchEnv, AllegroHandTorchEnv
from newton.newton_learning_examples.envs.configs import CartpoleConfig, AllegroHandConfig
from newton.newton_learning_examples.trainers.configs import RandomTrainerConfig, GeneticTrainerConfig
from newton.newton_learning_examples.trainers import RandomTrainer, GeneticTrainer


def make_parser():
    """Create argument parser with all options."""
    parser = newton.examples.create_parser()

    # Environment options
    parser.add_argument("--num-worlds", type=int, default=64, help="Number of parallel environments.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Trainer options
    parser.add_argument("--generations", type=int, default=50, help="Training generations.")
    parser.add_argument("--episode-steps", type=int, default=300, help="Steps per episode.")
    parser.add_argument("--render-every", type=int, default=1, help="Render every N generations (0 disables).")

    return parser


def main():

    parser = make_parser()
    viewer, args = newton.examples.init(parser)

    if args.viewer == "null":
        args.render_every = 0

    env_config = CartpoleConfig(
        num_worlds=args.num_worlds,
        seed=args.seed,
    )
    env = CartpoleTorchEnv(viewer, env_config)


    # env_config = AllegroHandConfig(
    #     num_worlds=args.num_worlds,
    #     seed=args.seed,
    # )
    # env = AllegroHandTorchEnv(viewer, env_config)


    # trainer_config = RandomTrainerConfig(
    #     generations=args.generations,
    #     episode_steps=args.episode_steps,
    #     render_every=args.render_every,
    # )
    # trainer = RandomTrainer(env, trainer_config)
    
    trainer_config = GeneticTrainerConfig(
        generations=args.generations,
        episode_steps=args.episode_steps,
        render_every=args.render_every,
    )
    trainer = GeneticTrainer(env, trainer_config)
    

    trainer.train()

    env.close()


if __name__ == "__main__":
    main()