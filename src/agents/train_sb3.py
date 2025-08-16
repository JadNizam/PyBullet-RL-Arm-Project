import argparse, yaml, os
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.envs.arm_reach_env import ArmReachEnv, ReachCfg

def make_env_reach(render: bool, seed: int, cfg_path: str):
    with open(cfg_path, "r") as f:
        y = yaml.safe_load(f)
    e = ArmReachEnv(ReachCfg(
        max_steps=y["env"]["max_steps"],
        dist_threshold=y["env"]["dist_threshold"],
        action_scale=y["env"]["action_scale"],
        render=render,
        seed=seed
    ))
    return Monitor(e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="reach", choices=["reach"])
    ap.add_argument("--cfg", default="src/configs/reach.yaml")
    ap.add_argument("--total-steps", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--logdir", default="runs/reach/seed_42")
    args = ap.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs("checkpoints/reach", exist_ok=True)

    if args.env == "reach":
        env = make_env_reach(args.render, args.seed, args.cfg)
    else:
        raise ValueError("Unknown env")

    model = PPO("MlpPolicy", env, verbose=1, seed=args.seed)
    model.learn(total_timesteps=args.total_steps)
    model.save(os.path.join("checkpoints", args.env, "best_model"))
    env.close()

if __name__ == "__main__":
    main()
