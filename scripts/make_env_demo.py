import argparse, time, numpy as np
from src.envs.arm_reach_env import ArmReachEnv, ReachCfg

# Simple script to sanity check the ArmReachEnv/ReachCfg environment.

def main():
    ap = argparse.ArgumentParser(description="Quick sanity check for ArmReachEnv.")
    ap.add_argument("--render", action="store_true", help="Open PyBullet GUI")
    ap.add_argument("--steps", type=int, default=200, help="Number of random steps")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed for repeatability")
    ap.add_argument("--fps", type=int, default=30, help="GUI pacing (sleep per step)")
    args = ap.parse_args()

    env = ArmReachEnv(ReachCfg(render=args.render, seed=args.seed))
    obs, info = env.reset()
    print("Target:", info["target"])

    for t in range(args.steps):
        
        action = np.random.uniform(-1, 1, size=3).astype("float32")
        obs, reward, terminated, truncated, step_info = env.step(action)

        if args.render:
            time.sleep(1.0 / args.fps)

        if (t % 20) == 0:
            print(f"t={t:03d}  dist={step_info.get('dist', np.nan):.3f}  reward={reward:.3f}")

        if terminated or truncated:
            print(f"Episode ended — terminated={terminated} truncated={truncated} at t={t}, reward={reward:.3f}")
            obs, info = env.reset()
            print("New target:", info["target"])

    env.close()

if __name__ == "__main__":
    main()
