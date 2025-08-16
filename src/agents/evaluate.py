import argparse, os, time
import numpy as np
from stable_baselines3 import PPO
from src.envs.arm_reach_env import ArmReachEnv, ReachCfg
from src.envs.arm_pickplace_env import ArmPickPlaceEnv, PickPlaceCfg
import pybullet as p

# ----- make_env: pick one of our envs -----
def make_env(env_name: str, render: bool):
    if env_name == "reach":
        return ArmReachEnv(ReachCfg(render=render))
    elif env_name == "pickplace":
        return ArmPickPlaceEnv(PickPlaceCfg(render=render))
    else:
        raise ValueError(f"Unknown env: {env_name}")

# ----- camera helpers -----
def _build_cameras(target=(0.72, 0.0, 0.12)):
    """
    Define a few nice views around the workspace.
    Returns list of (viewMatrix, projMatrix) tuples.
    """
    W, H = 640, 360  # per view; final width is W * num_views
    fov, aspect, near, far = 60.0, W / H, 0.01, 3.0

    # yaw, pitch, dist around the target
    specs = [
        # left-iso
        dict(yaw=110, pitch=-30, dist=1.2),
        # front
        dict(yaw=90, pitch=-35, dist=1.25),
        # right-iso
        dict(yaw=70, pitch=-30, dist=1.2),
    ]

    cams = []
    for s in specs:
        v = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=target, distance=s["dist"],
            yaw=s["yaw"], pitch=s["pitch"], roll=0.0, upAxisIndex=2
        )
        proj = p.computeProjectionMatrixFOV(
            fov=fov, aspect=aspect, nearVal=near, farVal=far
        )
        cams.append((v, proj))
    return cams, (W, H)

def _capture_views(client_id: int, views, size, renderer):
    """
    Grab frames from multiple camera views and stack them horizontally.
    Returns uint8 RGB array of shape (H, W*num_views, 3).
    """
    W, H = size
    rgb_rows = []
    for (view, proj) in views:
        w, h, rgba, _, _ = p.getCameraImage(
            W, H, viewMatrix=view, projectionMatrix=proj,
            renderer=renderer, physicsClientId=client_id
        )
        # Convert to (H, W, 3) uint8
        arr = np.asarray(rgba, dtype=np.uint8)
        if arr.ndim == 1:
            frame = arr.reshape(h, w, 4)[..., :3]
        elif arr.shape[-1] == 4:
            frame = arr[..., :3]
        else:
            frame = arr.reshape(h, w, 4)[..., :3]
        rgb_rows.append(frame)
    return np.concatenate(rgb_rows, axis=1)

# ----- evaluation loop -----
def run_episodes(
    env_name: str,
    model_path: str,
    episodes: int = 5,
    render: bool = True,
    fps: int = 60,
    video_path: str | None = None,
    views: int = 3,           # 1 or 3
    width: int = 640,
    height: int = 360,
):
    env = make_env(env_name, render)
    model = PPO.load(model_path)
    returns = []

    writer = None
    imgio = None
    if video_path:
        import imageio.v2 as imageio
        imgio = imageio
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        writer = imageio.get_writer(video_path, fps=fps)

    # Cameras (only used if we’re writing video)
    renderer = p.ER_TINY_RENDERER  # works with/without GUI
    if writer is not None:
        all_views, default_size = _build_cameras(target=(0.72, 0.0, 0.12))
        if views == 1:
            all_views = [all_views[1:2]]  # just the center/front view
        else:
            # keep all 3
            all_views = [all_views[0], all_views[1], all_views[2]]
        size = (width, height)
    else:
        all_views, size = [], (width, height)

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        trunc = False
        ep_ret = 0.0

        # If GUI is on, put a nice default camera in the debug viewer too
        if render:
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
                # capture multi-view (3-wide) or single view
                if views == 1:
                    # build a single center view quickly (reuse the function)
                    # Using the same views list but only the middle one
                    viewset, _ = _build_cameras(target=(0.72, 0.0, 0.12))
                    view = [viewset[1]]  # center/front
                    frame = _capture_views(env.client, view, size, renderer)
                else:
                    # three angles side-by-side
                    frame = _capture_views(env.client, all_views, size, renderer)
                writer.append_data(frame)

        print(f"Episode {ep+1}: return = {ep_ret:.3f}")
        returns.append(ep_ret)

    if writer is not None:
        writer.close()
        print(f"Saved video to: {video_path}")

    env.close()
    mean_ret = float(np.mean(returns)) if returns else 0.0
    print(f"Mean return over {episodes} episodes: {mean_ret:.3f}")

# ----- CLI entrypoint -----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="reach", choices=["reach", "pickplace"])
    ap.add_argument("--model", default="checkpoints/reach/best_model.zip")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--video", default="", help="Path to save mp4 (e.g., videos/demo.mp4)")
    ap.add_argument("--views", type=int, default=3, choices=[1, 3], help="Number of camera angles in the video.")
    ap.add_argument("--width", type=int, default=640, help="Per-view video width")
    ap.add_argument("--height", type=int, default=360, help="Per-view video height")
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
