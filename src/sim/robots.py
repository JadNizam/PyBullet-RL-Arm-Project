from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pybullet as p
import pybullet_data


@dataclass
class ArmHandles:
    client_id: int
    body_id: int
    joint_indices: list[int]
    ee_link_index: int

    @property
    def action_dim(self) -> int:
        # 3D delta for end-effector xyz control
        return 3

    @property
    def obs_dim(self) -> int:
        # joint pos + joint vel + ee pos
        return len(self.joint_indices) * 2 + 3


def _movable_joint_indices(body_id: int, client_id: int) -> list[int]:
    """Return indices of non-fixed joints (revolute/prismatic/etc.)."""
    js = []
    n = p.getNumJoints(body_id, physicsClientId=client_id)
    for j in range(n):
        jt = p.getJointInfo(body_id, j, physicsClientId=client_id)[2]
        if jt != p.JOINT_FIXED:
            js.append(j)
    return js


def disable_base_collisions(arm: ArmHandles) -> None:
    """Disable collisions between the robot base (-1) and all links."""
    n = p.getNumJoints(arm.body_id, physicsClientId=arm.client_id)
    for j in range(n):
        p.setCollisionFilterPair(
            arm.body_id, arm.body_id, -1, j, 0, physicsClientId=arm.client_id
        )


def load_arm(client_id: int, base_pos=(0, 0, 0), base_orn=(0, 0, 0, 1)) -> ArmHandles:
    """Load KUKA iiwa and return handles (with base↔link collisions disabled)."""
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
    body_id = p.loadURDF(
        "kuka_iiwa/model.urdf",
        basePosition=base_pos,
        baseOrientation=base_orn,
        useFixedBase=True,
        physicsClientId=client_id,
    )
    n_j = p.getNumJoints(body_id, physicsClientId=client_id)
    ee_link_index = n_j - 1  # treat last link as end-effector
    joint_indices = _movable_joint_indices(body_id, client_id)

    # Put joints in a passive state (no default velocity motors)
    for j in joint_indices:
        p.setJointMotorControl2(
            bodyIndex=body_id,
            jointIndex=j,
            controlMode=p.VELOCITY_CONTROL,
            force=0,
            physicsClientId=client_id,
        )

    arm = ArmHandles(client_id, body_id, joint_indices, ee_link_index)
    disable_base_collisions(arm)
    return arm


def set_joint_positions(arm: ArmHandles, q: list[float]) -> None:
    assert len(q) == len(arm.joint_indices)
    p.setJointMotorControlArray(
        bodyUniqueId=arm.body_id,
        jointIndices=arm.joint_indices,
        controlMode=p.POSITION_CONTROL,
        targetPositions=q,
        forces=[200] * len(arm.joint_indices),
        physicsClientId=arm.client_id,
    )


def reset_neutral(arm: ArmHandles) -> None:
    """Move to a safe neutral pose to avoid singularities."""
    q = [0.0, -0.6, 0.0, 1.2, 0.0, 0.6, 0.0][: len(arm.joint_indices)]
    set_joint_positions(arm, q)
    for _ in range(50):
        p.stepSimulation(physicsClientId=arm.client_id)


def get_joint_states(arm: ArmHandles) -> tuple[np.ndarray, np.ndarray]:
    js = p.getJointStates(arm.body_id, arm.joint_indices, physicsClientId=arm.client_id)
    q = np.asarray([s[0] for s in js], dtype=np.float32)
    dq = np.asarray([s[1] for s in js], dtype=np.float32)
    return q, dq


def _joint_limits_and_rest(arm: ArmHandles):
    """Collect per-joint limits/ranges and current angles as rest poses."""
    lows, highs, ranges = [], [], []
    for j in arm.joint_indices:
        info = p.getJointInfo(arm.body_id, j, physicsClientId=arm.client_id)
        low, high = info[8], info[9]
        # Some URDFs use invalid order for unlimited joints; fall back to [-pi, pi]
        if low > high:
            low, high = -np.pi, np.pi
        lows.append(low)
        highs.append(high)
        ranges.append(high - low if high > low else 2 * np.pi)
    q, _ = get_joint_states(arm)
    return lows, highs, ranges, q.tolist()


def get_end_effector_pos(arm: ArmHandles) -> np.ndarray:
    ls = p.getLinkState(
        arm.body_id,
        arm.ee_link_index,
        computeForwardKinematics=True,
        physicsClientId=arm.client_id,
    )
    return np.asarray(ls[4], dtype=np.float32)  # worldLinkFramePosition


def apply_delta_ee(
    arm: ArmHandles,
    delta_xyz: np.ndarray,
    max_step: float = 0.05,
    workspace_lo: list[float] | np.ndarray | None = None,
    workspace_hi: list[float] | np.ndarray | None = None,
) -> None:
    """
    Move EE by a small delta using IK, with:
      - optional workspace clamp (keeps away from base/table)
      - joint limits & rest poses
      - light damping
      - 'hand-down' orientation (roll=0, pitch=pi, yaw=0)
    """
    delta = np.clip(delta_xyz, -max_step, max_step).astype(np.float32)
    ee = get_end_effector_pos(arm)
    target = ee + delta
    if workspace_lo is not None and workspace_hi is not None:
        lo = np.asarray(workspace_lo, dtype=np.float32)
        hi = np.asarray(workspace_hi, dtype=np.float32)
        target = np.minimum(np.maximum(target, lo), hi)

    # Keep wrist roughly pointing down
    ee_down = p.getQuaternionFromEuler([0.0, np.pi, 0.0])

    lows, highs, ranges, rest = _joint_limits_and_rest(arm)
    q_ik = p.calculateInverseKinematics(
        arm.body_id,
        arm.ee_link_index,
        target.tolist(),
        targetOrientation=ee_down,
        lowerLimits=lows,
        upperLimits=highs,
        jointRanges=ranges,
        restPoses=rest,
        jointDamping=[0.08] * len(lows),
        residualThreshold=1e-3,
        physicsClientId=arm.client_id,
    )
    q_cmd = list(q_ik[: len(arm.joint_indices)])
    set_joint_positions(arm, q_cmd)


def get_state(arm: ArmHandles) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q, dq = get_joint_states(arm)
    ee = get_end_effector_pos(arm)
    return q, dq, ee
