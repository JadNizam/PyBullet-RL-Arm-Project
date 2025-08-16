import argparse, os, time
import numpy as np
from stable_baselines3 import PPO
from src.envs.arm_reach_env import ArmReachEnv, ReachCfg

def run_episodes(model_path: str, episodes: int = 5, render: bool = True, fps: int = 60, video_path: str | None = None):
    env = ArmReachEnv(ReachCfg(render=render))
    model = PPO.load(model_path)
    returns = []

    writer = None
    if video_path:
        import imageio.v2 as imageio
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        writer = imageio.get_writer(video_path, fps=fps)

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        trunc = False
        ret = 0.0
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(action)
            ret += reward

            if render:
                time.sleep(1.0 / fps)

            if writer is not None:
                import pybullet as p
                w, h, rgb, _, _ = p.getCameraImage(960, 720, renderer=p.ER_TINY_RENDERER, physicsClientId=env.client)
                writer.append_data(rgb)

        print(f"Episode {ep+1}: return = {ret:.3f}")
        returns.append(ret)

    if writer is not None:
        writer.close()
        print(f"Saved video to: {video_path}")

    env.close()
    print(f"Mean return over {episodes} episodes: {np.mean(returns):.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/reach/best_model.zip")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--video", default="", help="Path to save mp4 (e.g., videos/reach_demo.mp4)")
    args = ap.parse_args()

    video_path = args.video if args.video else None
    run_episodes(args.model, args.episodes, args.render, args.fps, video_path)

if __name__ == "__main__":
    main()
