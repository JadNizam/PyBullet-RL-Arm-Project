import argparse, yaml, os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

def make_env_reach(render: bool, seed: int, cfg_path: str):
    from src.envs.arm_reach_env import ArmReachEnv, ReachCfg
    with open(cfg_path, "r") as f:
        y = yaml.safe_load(f)
    e = y["env"]
    cfg = ReachCfg(
        max_steps=e["max_steps"],
        dist_threshold=e["dist_threshold"],
        action_scale=e["action_scale"],
        render=render,
        seed=seed,
    )
    return Monitor(ArmReachEnv(cfg))

def make_env_pickplace(render: bool, seed: int, cfg_path: str):
    from src.envs.arm_pickplace_env import ArmPickPlaceEnv, PickPlaceCfg
    with open(cfg_path, "r") as f:
        y = yaml.safe_load(f)
    e = y["env"]
    cfg = PickPlaceCfg(
        max_steps=e["max_steps"],
        grasp_radius=e["grasp_radius"],
        action_scale=e["action_scale"],
        render=render,
        seed=seed,
        inner_xyh=tuple(e.get("inner_xyh", [0.26, 0.26, 0.16])),
        wall_thickness=float(e.get("wall_thickness", 0.012)),
        auto_grasp=bool(e.get("auto_grasp", True)),
        auto_release=bool(e.get("auto_release", True)),
        release_xy_margin=float(e.get("release_xy_margin", 0.10)),
        release_z_thresh=float(e.get("release_z_thresh", 0.12)),
        over_bin_height=float(e.get("over_bin_height", 0.18)),
    )
    return Monitor(ArmPickPlaceEnv(cfg))

def make_env_sort(render: bool, seed: int, cfg_path: str):
    from src.envs.arm_sort_env import ArmSortEnv, SortCfg
    with open(cfg_path, "r") as f:
        y = yaml.safe_load(f)
    e = y["env"]
    cfg = SortCfg(
        max_steps=e.get("max_steps", 500),
        action_scale=e.get("action_scale", 0.03),
        grasp_radius=e.get("grasp_radius", 0.08),
        render=render,
        seed=seed,
        num_balls=e.get("num_balls", 4),
        ball_radius=e.get("ball_radius", 0.03),
        ball_mass=e.get("ball_mass", 0.05),
        inner_xyh=tuple(e.get("inner_xyh", [0.26, 0.26, 0.16])),
        wall_thickness=float(e.get("wall_thickness", 0.012)),
        auto_grasp=bool(e.get("auto_grasp", True)),
        auto_release=bool(e.get("auto_release", True)),
        release_xy_margin=float(e.get("release_xy_margin", 0.10)),
        release_z_thresh=float(e.get("release_z_thresh", 0.12)),
        over_bin_height=float(e.get("over_bin_height", 0.18)),
    )
    return Monitor(ArmSortEnv(cfg))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="reach", choices=["reach", "pickplace", "sort"])
    ap.add_argument("--cfg", default="src/configs/reach.yaml")
    ap.add_argument("--total-steps", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--logdir", default="runs/reach/seed_42")
    args = ap.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(f"checkpoints/{args.env}", exist_ok=True)

    if args.env == "reach":
        env = make_env_reach(args.render, args.seed, args.cfg)
    elif args.env == "pickplace":
        env = make_env_pickplace(args.render, args.seed, args.cfg)
    else:
        env = make_env_sort(args.render, args.seed, args.cfg)

    model = PPO("MlpPolicy", env, verbose=1, seed=args.seed)
    model.learn(total_timesteps=args.total_steps)
    model.save(os.path.join("checkpoints", args.env, "best_model"))
    env.close()

if __name__ == "__main__":
    main()
