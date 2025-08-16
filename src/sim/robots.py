from __future__ import annotations
from dataclasses import dataclass
import pybullet as p
import pybullet_data
import numpy as np

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
    js = []
    n = p.getNumJoints(body_id, physicsClientId=client_id)
    for j in range(n):
        info = p.getJointInfo(body_id, j, physicsClientId=client_id)
        joint_type = info[2]
        # Revolute=0, Prismatic=1, Spherical=2, Planar=3, Fixed=4
        if joint_type != p.JOINT_FIXED:
            js.append(j)
    return js

def load_arm(client_id: int, base_pos=(0,0,0), base_orn=(0,0,0,1)) -> ArmHandles:
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
    body_id = p.loadURDF("kuka_iiwa/model.urdf",
                         basePosition=base_pos,
                         baseOrientation=base_orn,
                         useFixedBase=True,
                         physicsClientId=client_id)
    n_j = p.getNumJoints(body_id, physicsClientId=client_id)
    ee_link_index = n_j - 1  # last link as end-effector
    joint_indices = _movable_joint_indices(body_id, client_id)
    # Dampen motors
    for j in joint_indices:
        p.setJointMotorControl2(bodyIndex=body_id, jointIndex=j,
                                controlMode=p.VELOCITY_CONTROL, force=0,
                                physicsClientId=client_id)
    return ArmHandles(client_id, body_id, joint_indices, ee_link_index)

def reset_neutral(arm: ArmHandles):
    # Neutral pose: small non-zero offsets to avoid singularities
    q = [0.0, -0.6, 0.0, 1.2, 0.0, 0.6, 0.0][:len(arm.joint_indices)]
    set_joint_positions(arm, q)
    for _ in range(50):
        p.stepSimulation(physicsClientId=arm.client_id)

def set_joint_positions(arm: ArmHandles, q: list[float]):
    assert len(q) == len(arm.joint_indices)
    p.setJointMotorControlArray(bodyUniqueId=arm.body_id,
                                jointIndices=arm.joint_indices,
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=q,
                                forces=[200]*len(arm.joint_indices),
                                physicsClientId=arm.client_id)

def get_joint_states(arm: ArmHandles):
    js = p.getJointStates(arm.body_id, arm.joint_indices, physicsClientId=arm.client_id)
    q = np.array([s[0] for s in js], dtype=np.float32)
    dq = np.array([s[1] for s in js], dtype=np.float32)
    return q, dq

def get_end_effector_pos(arm: ArmHandles):
    link_state = p.getLinkState(arm.body_id, arm.ee_link_index, computeForwardKinematics=True,
                                physicsClientId=arm.client_id)
    pos = np.array(link_state[4], dtype=np.float32)  # worldLinkFramePosition
    return pos

def apply_delta_ee(arm: ArmHandles, delta_xyz: np.ndarray, max_step: float = 0.05):
    delta = np.clip(delta_xyz, -max_step, max_step)
    ee = get_end_effector_pos(arm)
    target = ee + delta
    q_ik = p.calculateInverseKinematics(arm.body_id, arm.ee_link_index, target.tolist(),
                                        residualThreshold=1e-3, physicsClientId=arm.client_id)
    q_cmd = list(q_ik[:len(arm.joint_indices)])
    set_joint_positions(arm, q_cmd)

def get_state(arm: ArmHandles):
    q, dq = get_joint_states(arm)
    ee = get_end_effector_pos(arm)
    return q, dq, ee
