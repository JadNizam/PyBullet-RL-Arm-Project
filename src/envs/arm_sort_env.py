from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
from dataclasses import dataclass

from src.sim.world import connect, reset_world, set_gravity, load_plane
from src.sim.robots import load_arm, reset_neutral, get_state, apply_delta_ee
from src.sim.objects import create_table, create_ball, create_open_container, ball_inside_container

RED = 0
BLUE = 1
COLOR2RGBA = {
    RED:  (0.90, 0.15, 0.15, 1.0),
    BLUE: (0.15, 0.25, 0.90, 1.0),
}

@dataclass
class SortCfg:
    max_steps: int = 500
    action_scale: float = 0.03
    grasp_radius: float = 0.08
    render: bool = False
    seed: int = 42
    num_balls: int = 4
    ball_radius: float = 0.03
    ball_mass: float = 0.05
    inner_xyh: tuple[float, float, float] = (0.26, 0.26, 0.16)
    wall_thickness: float = 0.012
    auto_grasp: bool = True
    auto_release: bool = True
    release_xy_margin: float = 0.10
    release_z_thresh: float = 0.12
    over_bin_height: float = 0.18

class ArmSortEnv(gym.Env):
    """
    Action: [dx, dy, dz, grip] in [-1,1]; grip>0 attach, grip<=0 release.
    Obs (fixed-size): [q, dq, ee, cur_ball_pos(3), red_bin(3), blue_bin(3),
                       grasped(1), cur_ball_color_onehot(2),
                       (cur_ball-ee)(3), (target_bin-cur_ball)(3)]
    """
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, cfg: SortCfg = SortCfg()):
        super().__init__()
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.client = connect(cfg.render)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        set_gravity(self.client)
        self.plane_id = load_plane(self.client)

        self.arm = load_arm(self.client)
        reset_neutral(self.arm)

        # obs dim = arm.obs_dim + 18 (see docstring)
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(self.arm.obs_dim + 18,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # sim state
        self.ball_ids: list[int] = []
        self.ball_colors: list[int] = []
        self.sorted_mask: np.ndarray | None = None
        self.cur_idx: int = 0
        self.grasped: bool = False
        self.constraint_id = None
        self.red_bin_center = np.zeros(3, dtype=np.float32)
        self.blue_bin_center = np.zeros(3, dtype=np.float32)
        self.container_inner = np.array(self.cfg.inner_xyh, dtype=np.float32)
        self.t = 0
        self.prev_goal_dist = 0.0

    # utils
    def _colorize(self, body_id: int, rgba):
        try:
            p.changeVisualShape(body_id, -1, rgbaColor=rgba, physicsClientId=self.client)
        except Exception:
            pass

    def _rand_on_ring(self, r_min=0.55, r_max=0.95) -> np.ndarray:
        r = float(self.rng.uniform(r_min, r_max))
        ang = float(self.rng.uniform(0.0, 2.0*np.pi))
        return np.array([r*np.cos(ang), r*np.sin(ang), 0.0], dtype=np.float32)

    def _spawn_bins(self):
        # sample two angles at least ~120° apart so bins are well separated
        c1 = self._rand_on_ring()
        for _ in range(100):
            c2 = self._rand_on_ring()
            v1 = c1[:2] / (np.linalg.norm(c1[:2]) + 1e-6)
            v2 = c2[:2] / (np.linalg.norm(c2[:2]) + 1e-6)
            if np.dot(v1, v2) < -0.5:  # ~>120°
                break

        parts, center_r, inner_r, _ = create_open_container(
            self.client, center=c1.tolist(), inner=tuple(self.cfg.inner_xyh),
            thickness=self.cfg.wall_thickness, rgba=(0.55, 0.25, 0.25, 1.0)
        )
        parts, center_b, inner_b, _ = create_open_container(
            self.client, center=c2.tolist(), inner=tuple(self.cfg.inner_xyh),
            thickness=self.cfg.wall_thickness, rgba=(0.25, 0.30, 0.55, 1.0)
        )
        self.red_bin_center = center_r.astype(np.float32)
        self.blue_bin_center = center_b.astype(np.float32)
        self.container_inner = inner_r.astype(np.float32)

    def _spawn_balls(self):
        self.ball_ids.clear()
        self.ball_colors.clear()
        self.sorted_mask = np.zeros(self.cfg.num_balls, dtype=bool)

        for _i in range(self.cfg.num_balls):
            # keep balls away from bins a bit
            for _ in range(100):
                pos = self._rand_on_ring()
                pos[2] = 0.10
                far_r = np.linalg.norm(pos[:2] - self.red_bin_center[:2]) > 0.18
                far_b = np.linalg.norm(pos[:2] - self.blue_bin_center[:2]) > 0.18
                if far_r and far_b:
                    break
            bid = create_ball(
                self.client, pos=pos.tolist(),
                radius=self.cfg.ball_radius, mass=self.cfg.ball_mass
            )
            color = int(self.rng.integers(0, 2))  # 0=RED, 1=BLUE
            self._colorize(bid, COLOR2RGBA[color])
            self.ball_ids.append(bid)
            self.ball_colors.append(color)

    def _target_bin_center(self, color: int) -> np.ndarray:
        return self.red_bin_center if color == RED else self.blue_bin_center

    def _unsorted_indices(self):
        return [i for i in range(self.cfg.num_balls) if not self.sorted_mask[i]]

    def _select_current_ball(self):
        # choose nearest unsorted ball to the EE
        _, _, ee = get_state(self.arm)
        cand = []
        for i in self._unsorted_indices():
            pos = np.array(
                p.getBasePositionAndOrientation(self.ball_ids[i], physicsClientId=self.client)[0],
                dtype=np.float32
            )
            cand.append((np.linalg.norm(pos - ee), i))
        self.cur_idx = min(cand)[1] if cand else 0

    def _cur_ball_pos(self) -> np.ndarray:
        return np.array(
            p.getBasePositionAndOrientation(self.ball_ids[self.cur_idx], physicsClientId=self.client)[0],
            dtype=np.float32
        )

    # Gym API
    def seed(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def _spawn_world(self):
        reset_world(self.client)
        self.plane_id = load_plane(self.client)

        # big centered table so objects can be anywhere around the base
        create_table(self.client, pos=(0.0, 0.0, 0.005), size=(1.2, 1.2, 0.01), rgba=(0.45, 0.30, 0.20, 1.0))

        self.arm = load_arm(self.client)
        reset_neutral(self.arm)

        self._spawn_bins()
        self._spawn_balls()
        self._select_current_ball()

        self.grasped = False
        self.constraint_id = None
        self.t = 0
        self.prev_goal_dist = float(
            np.linalg.norm(self._cur_ball_pos() - self._target_bin_center(self.ball_colors[self.cur_idx]))
        )

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.seed(seed)
        self._spawn_world()
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        q, dq, ee = get_state(self.arm)
        bpos = self._cur_ball_pos()
        color = self.ball_colors[self.cur_idx]
        color_onehot = np.array([1.0, 0.0], dtype=np.float32) if color == RED else np.array([0.0, 1.0], dtype=np.float32)
        grasp_flag = np.array([1.0 if self.grasped else 0.0], dtype=np.float32)
        vec_ball_minus_ee = bpos - ee
        vec_bin_minus_ball = self._target_bin_center(color) - bpos
        extra = np.concatenate([
            bpos, self.red_bin_center, self.blue_bin_center,
            grasp_flag, color_onehot, vec_ball_minus_ee, vec_bin_minus_ball
        ])
        return np.concatenate([q, dq, ee, extra]).astype(np.float32)

    # grasp/release helpers
    def _set_ball_arm_collisions(self, enable: bool):
        n = p.getNumJoints(self.arm.body_id, physicsClientId=self.client)
        for j in range(-1, n):
            p.setCollisionFilterPair(
                self.arm.body_id, self.ball_ids[self.cur_idx], j, -1,
                1 if enable else 0, physicsClientId=self.client
            )

    def _maybe_grasp(self, want_close: bool, ee_pos: np.ndarray):
        if want_close and not self.grasped:
            bpos = self._cur_ball_pos()
            if np.linalg.norm(ee_pos - bpos) < self.cfg.grasp_radius:
                self.constraint_id = p.createConstraint(
                    self.arm.body_id, self.arm.ee_link_index,
                    self.ball_ids[self.cur_idx], -1,
                    p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0],
                    physicsClientId=self.client
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

    def step(self, action: np.ndarray):
        delta = np.clip(action[:3], -1, 1) * self.cfg.action_scale
        want_close = action[3] > 0.0

        # 360° workspace around base (z kept safe)
        apply_delta_ee(
            self.arm, delta, max_step=0.05,
            workspace_lo=[-1.0, -1.0, 0.06],
            workspace_hi=[ 1.0,  1.0, 0.80],
        )
        for _ in range(4):
            p.stepSimulation(physicsClientId=self.client)

        q, dq, ee = get_state(self.arm)
        bpos = self._cur_ball_pos()
        color = self.ball_colors[self.cur_idx]
        target_bin = self._target_bin_center(color)

        # auto-grasp
        if self.cfg.auto_grasp and not self.grasped and np.linalg.norm(ee - bpos) < self.cfg.grasp_radius:
            want_close = True
        grasp_event = self._maybe_grasp(want_close, ee)

        # reward shaping
        act_pen = 0.001 * float(np.square(delta).sum())
        rew = -0.001 - act_pen
        if not self.grasped:
            rew += -2.0 * float(np.linalg.norm(ee - bpos))
            if grasp_event:
                rew += 0.8
        else:
            dist_to_bin = float(np.linalg.norm(bpos - target_bin))
            progress = self.prev_goal_dist - dist_to_bin
            rew += -2.0 * dist_to_bin + 1.2 * progress
            self.prev_goal_dist = dist_to_bin

            if self.cfg.auto_release:
                xy_ok = np.linalg.norm((bpos - target_bin)[:2]) < self.cfg.release_xy_margin
                z_ok  = bool(bpos[2] > self.cfg.release_z_thresh)
                if xy_ok and z_ok:
                    self._maybe_grasp(False, ee)

        # success for current ball
        inside = ball_inside_container(bpos, target_bin, self.container_inner)
        if inside and not self.grasped:
            self.sorted_mask[self.cur_idx] = True
            rew += 1.5
            unsorted = self._unsorted_indices()
            if unsorted:
                self._select_current_ball()
                self.prev_goal_dist = float(
                    np.linalg.norm(self._cur_ball_pos() - self._target_bin_center(self.ball_colors[self.cur_idx]))
                )

        self.t += 1
        terminated = bool(self.sorted_mask.all())
        truncated  = bool(self.t >= self.cfg.max_steps)
        return self._get_obs(), rew, terminated, truncated, {"done_count": int(self.sorted_mask.sum())}

    def render(self): pass

    def close(self):
        if self.client is not None:
            p.disconnect(self.client)
