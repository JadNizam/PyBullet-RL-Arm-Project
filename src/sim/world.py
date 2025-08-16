from __future__ import annotations
from dataclasses import dataclass
import pybullet as p

@dataclass
class WorldHandles:
    client_id: int
    plane_id: int | None = None

def connect(render: bool = False) -> int:
    cid = p.connect(p.GUI if render else p.DIRECT)
    return cid

def set_gravity(client_id: int):
    p.setGravity(0, 0, -9.81, physicsClientId=client_id)

def load_plane(client_id: int) -> int:
    import pybullet_data
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
    plane_id = p.loadURDF("plane.urdf", physicsClientId=client_id)
    return plane_id

def reset_world(client_id: int) -> int:
    p.resetSimulation(physicsClientId=client_id)
    set_gravity(client_id)
    plane_id = load_plane(client_id)
    return plane_id
