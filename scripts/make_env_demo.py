import argparse, time, numpy as np
from src.envs.arm_reach_env import ArmReachEnv, ReachCfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    env = ArmReachEnv(ReachCfg(render=args.render))
    obs, info = env.reset()
    print("Target:", info["target"])

    for t in range(200):
        # random walk in EE space
        action = np.random.uniform(-1, 1, size=(3,)).astype("float32") * 0.5
        obs, r, d, tr, inf = env.step(action)
        if d or tr:
            print("Terminated:", d, "Truncated:", tr, "Step:", t, "Reward:", r)
            obs, info = env.reset()
    env.close()

if __name__ == "__main__":
    main()
