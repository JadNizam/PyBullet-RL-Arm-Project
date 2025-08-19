# src/agents/evaluate.py
import argparse, os, time
import numpy as np
import pybullet as p
from stable_baselines3 import PPO

from src.envs.arm_reach_env import ArmReachEnv, ReachCfg
from src.envs.arm_pickplace_env import ArmPickPlaceEnv, PickPlaceCfg
from src.envs.arm_sort_env import ArmSortEnv, SortCfg


def make_env(env_name: str, render: bool):
    if env_name == "reach":
        return ArmReachEnv(ReachCfg(render=render))
    elif env_name == "pickplace":
        return ArmPickPlaceEnv(PickPlaceCfg(render=render))
    elif env_name == "sort":
        return ArmSortEnv(SortCfg(render=render))
    else:
        raise ValueError("Unknown env")


# camera helpers
def _build_cameras(target=(0.72, 0.0, 0.12), w=640, h=360):
    # 3 nice views around the workspace
    fov, aspect, near, far = 60.0, w / h, 0.01, 3.0
    specs = [
        dict(yaw=110, pitch=-30, dist=1.2),   # left-iso
        dict(yaw= 90, pitch=-35, dist=1.25),  # front
        dict(yaw= 70, pitch=-30, dist=1.2),   # right-iso
    ]
    cams = []
    for s in specs:
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=target, distance=s["dist"],
            yaw=s["yaw"], pitch=s["pitch"], roll=0.0, upAxisIndex=2
        )
        proj = p.computeProjectionMatrixFOV(
            fov=fov, aspect=aspect, nearVal=near, farVal=far
        )
        cams.append((view, proj))
    return cams


def _choose_renderer(render: bool):
    # use hardware GL if GUI is up; otherwise Tiny works headless
    if render:
        try:
            return p.ER_BULLET_HARDWARE_OPENGL
        except Exception:
            return p.ER_TINY_RENDERER
    return p.ER_TINY_RENDERER


def _capture_views(client_id: int, views, w: int, h: int, renderer):
    frames = []
    for (view, proj) in views:
        _, _, rgba, _, _ = p.getCameraImage(
            w, h, viewMatrix=view, projectionMatrix=proj,
            renderer=renderer, physicsClientId=client_id
        )
        arr = np.asarray(rgba)
        if arr.ndim == 1:
            img = arr.reshape(h, w, 4)[..., :3]
        else:
            # pybullet sometimes returns (H,W,4) already
            if arr.shape[-1] == 4:
                img = arr[..., :3]
            else:
                img = arr.reshape(h, w, 4)[..., :3]
        img = img.astype(np.uint8, copy=False)
        frames.append(img)
    # stack side-by-side
    return np.concatenate(frames, axis=1)


def run_episodes(
    env_name: str,
    model_path: str,
    episodes: int = 5,
    render: bool = True,
    fps: int = 60,
    video_path: str | None = None,
    views: int = 3,      # 1 or 3
    width: int = 640,
    height: int = 360,
):
    env = make_env(env_name, render)
    model = PPO.load(model_path)
    returns = []

    writer = None
    if video_path:
        import imageio.v2 as imageio
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        writer = imageio.get_writer(video_path, fps=fps)

    renderer = _choose_renderer(render)
    all_views = _build_cameras(w=width, h=height)
    if views == 1:
        views_to_use = [all_views[1]]  # center/front only
    else:
        views_to_use = all_views       # all 3

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        trunc = False
        ep_ret = 0.0

        if render:
            # nice default GUI camera
            p.resetDebugVisualizerCamera(
                cameraDistance=1.25, cameraYaw=90, cameraPitch=-35,
                cameraTargetPosition=[0.72, 0.0, 0.12],
                physicsClientId=env.client
            )

        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, _ = env.step(action)
            ep_ret += reward

            if render:
                time.sleep(1.0 / fps)

            if writer is not None:
                frame = _capture_views(env.client, views_to_use, width, height, renderer)
                writer.append_data(frame)

        print(f"Episode {ep+1}: return = {ep_ret:.3f}")
        returns.append(ep_ret)

    if writer is not None:
        writer.close()
        print(f"Saved video to: {video_path}")

    env.close()
    mean_ret = float(np.mean(returns)) if returns else 0.0
    print(f"Mean return over {episodes} episodes: {mean_ret:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="reach", choices=["reach", "pickplace", "sort"])
    ap.add_argument("--model", default="checkpoints/reach/best_model.zip")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--video", default="", help="Path to save mp4 (e.g., videos/demo.mp4)")
    ap.add_argument("--views", type=int, default=3, choices=[1, 3])
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    args = ap.parse_args()

    model_path = args.model or f"checkpoints/{args.env}/best_model.zip"
    video_path = args.video if args.video else None

    run_episodes(
        args.env,
        model_path,
        args.episodes,
        args.render,
        args.fps,
        video_path,
        views=args.views,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
