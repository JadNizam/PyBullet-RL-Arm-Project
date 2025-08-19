from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
from dataclasses import dataclass

from src.sim.world import connect, reset_world, set_gravity, load_plane
from src.sim.robots import load_arm, reset_neutral, get_state, apply_delta_ee
from src.sim.objects import (
    create_table, create_ball, create_open_container, ball_inside_container
)

@dataclass
class PickPlaceCfg:
    max_steps: int = 500
    grasp_radius: float = 0.10          # strong/easy grasp
    action_scale: float = 0.035          # move a bit faster post-grasp
    render: bool = False
    seed: int = 42

    # container geometry
    inner_xyh: tuple[float, float, float] = (0.26, 0.26, 0.16)
    wall_thickness: float = 0.012

    # helpers
    auto_grasp: bool = True
    auto_release: bool = True
    release_xy_margin: float = 0.10     # XY radius to consider “over bin”
    release_z_thresh: float = 0.12      # Z above bin to allow release
    lift_height: float = 0.16         # lift target right after grasp
    over_bin_height: float = 0.18       # fly-over height above bin

class ArmPickPlaceEnv(gym.Env):
    """Action: [dx, dy, dz, grip] ∈ [-1,1]; grip>0 attach, grip<=0 release."""
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, cfg: PickPlaceCfg = PickPlaceCfg()):
        super().__init__()
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.client = connect(cfg.render)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        set_gravity(self.client)
        self.plane_id = load_plane(self.client)

        self.arm = load_arm(self.client)
        reset_neutral(self.arm)

        # obs: q, dq, ee, ball, bin, grasp_flag, (ball-ee), (bin-ball)
        base_dim = self.arm.obs_dim + 3 + 3 + 1
        rel_dim  = 3 + 3
        obs_dim  = base_dim + rel_dim
        self.observation_space = spaces.Box(low=-10, high=10, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # sim state
        self.ball_id = None
        self.container_part_ids = []
        self.container_center = np.zeros(3, dtype=np.float32)
        self.container_inner = np.zeros(3, dtype=np.float32)
        self.grasped = False
        self.constraint_id = None
        self.prev_ball_goal_dist = 0.0
        self.t = 0

    # ----------------------- world / obs -----------------------

    def seed(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def _spawn_world(self):
        reset_world(self.client)
        self.plane_id = load_plane(self.client)
        create_table(self.client)

        # reload arm
        self.arm = load_arm(self.client)
        reset_neutral(self.arm)

        # Ball: close to base, negative-Y; raised to ease pickup
        ball_pos = np.array([
            self.rng.uniform(0.50, 0.55),
            self.rng.uniform(-0.12, -0.08),
            0.12
        ], dtype=np.float32)
        self.ball_id = create_ball(self.client, pos=ball_pos.tolist(), radius=0.03, mass=0.05)

        # Box: near base (front-right). Keep your ranges as in your snippet.
        min_sep = 0.10
        for _ in range(50):
            c_center = np.array([
                self.rng.uniform(0.70, 0.80),
                self.rng.uniform(0.15, 0.20),
                0.0
            ], dtype=np.float32)
            if np.linalg.norm(c_center[:2] - ball_pos[:2]) >= min_sep:
                break

        parts, center, inner_ret, _ = create_open_container(
            self.client, center=c_center.tolist(), inner=tuple(self.cfg.inner_xyh),
            thickness=self.cfg.wall_thickness, rgba=(0.36, 0.25, 0.20, 1.0)
        )
        self.container_part_ids = parts
        self.container_center = center.astype(np.float32)
        self.container_inner = inner_ret.astype(np.float32)

        self.grasped = False
        self.constraint_id = None
        self.t = 0
        self.prev_ball_goal_dist = float(np.linalg.norm(ball_pos - self.container_center))

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.seed(seed)
        self._spawn_world()
        return self._get_obs(), {"container_center": self.container_center.copy()}

    def _obs_parts(self):
        q, dq, ee = get_state(self.arm)
        ball_pos = np.array(
            p.getBasePositionAndOrientation(self.ball_id, physicsClientId=self.client)[0],
            dtype=np.float32
        )
        return q, dq, ee, ball_pos

    def _get_obs(self) -> np.ndarray:
        q, dq, ee, ball_pos = self._obs_parts()
        grasp_flag = np.array([1.0 if self.grasped else 0.0], dtype=np.float32)
        vec_ball_minus_ee = ball_pos - ee
        vec_bin_minus_ball = self.container_center - ball_pos
        return np.concatenate([
            q, dq, ee, ball_pos, self.container_center, grasp_flag,
            vec_ball_minus_ee, vec_bin_minus_ball
        ]).astype(np.float32)

    # ----------------------- grasp helpers -----------------------

    def _set_ball_arm_collisions(self, enable: bool):
        n = p.getNumJoints(self.arm.body_id, physicsClientId=self.client)
        for j in range(-1, n):  # -1 = base
            p.setCollisionFilterPair(self.arm.body_id, self.ball_id, j, -1,
                                     1 if enable else 0, physicsClientId=self.client)

    def _maybe_grasp(self, want_close: bool, ee_pos: np.ndarray):
        if want_close and not self.grasped:
            ball_pos = np.array(p.getBasePositionAndOrientation(self.ball_id, physicsClientId=self.client)[0], dtype=np.float32)
            if np.linalg.norm(ee_pos - ball_pos) < self.cfg.grasp_radius:
                self.constraint_id = p.createConstraint(
                    self.arm.body_id, self.arm.ee_link_index,
                    self.ball_id, -1, p.JOINT_FIXED, [0,0,0],
                    [0,0,0], [0,0,0], physicsClientId=self.client
                )
                self._set_ball_arm_collisions(False)
                self.grasped = True
                return True
        elif (not want_close) and self.grasped:
            p.removeConstraint(self.constraint_id, physicsClientId=self.client)
            self._set_ball_arm_collisions(True)
            self.constraint_id = None
            self.grasped = False
        return False

    # ----------------------- step -----------------------

    def step(self, action: np.ndarray):
        delta = np.clip(action[:3], -1, 1) * self.cfg.action_scale
        want_close = action[3] > 0.0

        # Workspace widened to cover both ball and bin areas.
        apply_delta_ee(
            self.arm, delta, max_step=0.05,
            workspace_lo=[-0.60, -0.30, 0.06],
            workspace_hi=[ 0.95,  0.30, 0.70],
        )

        for _ in range(4):
            p.stepSimulation(physicsClientId=self.client)

        q, dq, ee, ball_pos = self._obs_parts()

        # Auto-grasp when near ball
        if self.cfg.auto_grasp and not self.grasped:
            if np.linalg.norm(ee - ball_pos) < self.cfg.grasp_radius:
                want_close = True

        grasp_event = self._maybe_grasp(want_close, ee)

        # ---------- reward shaping ----------
        act_pen  = 0.001 * float(np.square(delta).sum())
        time_pen = 0.001
        reward   = -time_pen - act_pen

        if not self.grasped:
            # Reach the ball
            dist_ee_ball = float(np.linalg.norm(ee - ball_pos))
            reward += -2.0 * dist_ee_ball
            if grasp_event:
                reward += 0.8  # grasp bonus
        else:
            # Stage 1: Lift after grasp
            lift_err = max(0.0, self.cfg.lift_height - float(ball_pos[2]))
            reward += -4.0 * lift_err  # push hard to lift

            # Stage 2: Move over the bin
            over_bin_target = self.container_center + np.array([0.0, 0.0, self.cfg.over_bin_height], dtype=np.float32)
            ee_to_overbin = float(np.linalg.norm(ee - over_bin_target))
            reward += -3.0 * ee_to_overbin  # strong attraction over bin

            # Progress toward bin center (ball perspective)
            dist_ball_goal = float(np.linalg.norm(ball_pos - self.container_center))
            progress = self.prev_ball_goal_dist - dist_ball_goal
            reward += 1.2 * progress
            self.prev_ball_goal_dist = dist_ball_goal

            # Auto-release when over bin (optional but helps learning a lot)
            if self.cfg.auto_release:
                xy_ok = np.linalg.norm((ball_pos - self.container_center)[:2]) < self.cfg.release_xy_margin
                z_ok  = bool(ball_pos[2] > self.cfg.release_z_thresh)
                if xy_ok and z_ok:
                    self._maybe_grasp(False, ee)  # release

        # Terminal success: ball inside the container (and released)
        inside = ball_inside_container(ball_pos, self.container_center, self.container_inner)
        terminated = bool(inside and (not self.grasped))
        self.t += 1
        truncated = (self.t >= self.cfg.max_steps)
        if terminated:
            reward += 2.0  # success bonus

        return self._get_obs(), reward, terminated, truncated, {
            "inside_container": inside,
            "grasped": self.grasped
        }

    def render(self):  # GUI handled by connect(render=True)
        pass

    def close(self):
        if self.client is not None:
            p.disconnect(self.client)