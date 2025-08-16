from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data

from dataclasses import dataclass
from src.sim.world import connect, reset_world, set_gravity, load_plane
from src.sim.robots import load_arm, reset_neutral, get_state, apply_delta_ee
from src.sim.objects import create_target_sphere

@dataclass
class ReachCfg:
    max_steps: int = 200
    dist_threshold: float = 0.03
    action_scale: float = 0.02
    render: bool = False
    seed: int = 42

class ArmReachEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, cfg: ReachCfg = ReachCfg()):
        super().__init__()
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.client = connect(cfg.render)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        set_gravity(self.client)
        self.plane_id = load_plane(self.client)

        self.arm = load_arm(self.client)
        # obs: q, dq, ee, target, delta
        obs_dim = self.arm.obs_dim + 3 + 3
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self.step_count = 0
        self.target = np.zeros(3, dtype=np.float32)
        self.target_id = None

    def seed(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def _sample_target(self) -> np.ndarray:
        return np.array([0.6 + 0.05*self.rng.standard_normal(),
                         0.0 + 0.20*self.rng.uniform(-1,1),
                         0.15 + 0.25*self.rng.uniform(0,1)], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.seed(seed)
        self.plane_id = reset_world(self.client)
        self.arm = load_arm(self.client)
        reset_neutral(self.arm)

        self.target = self._sample_target()
        self.target_id = create_target_sphere(self.client, self.target.tolist(), radius=0.03)

        self.step_count = 0
        obs = self._get_obs()
        info = {"target": self.target.copy()}
        return obs, info

    def _get_obs(self) -> np.ndarray:
        q, dq, ee = get_state(self.arm)
        delta = self.target - ee
        return np.concatenate([q, dq, ee, self.target, delta]).astype(np.float32)

    def step(self, action: np.ndarray):
        action = np.clip(action, -1, 1) * self.cfg.action_scale
        apply_delta_ee(self.arm, action, max_step=0.05)
        for _ in range(4):
            p.stepSimulation(physicsClientId=self.client)
        q, dq, ee = get_state(self.arm)
        dist = float(np.linalg.norm(self.target - ee))
        reward = -dist - 0.001 * float(np.square(action).sum())
        terminated = dist < self.cfg.dist_threshold
        self.step_count += 1
        truncated = self.step_count >= self.cfg.max_steps
        obs = self._get_obs()
        info = {"dist": dist}
        if terminated:
            reward += 1.0
        return obs, reward, terminated, truncated, info

    def render(self):
        # GUI handled by connect(render=True)
        pass

    def close(self):
        if self.client is not None:
            p.disconnect(self.client)
