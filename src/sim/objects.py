from __future__ import annotations
import pybullet as p
import numpy as np

def create_table(client_id: int, pos=(0.6, 0.0, -0.02), size=(0.6, 0.6, 0.02), rgba=(0.6,0.45,0.3,1)):
    vs = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=rgba, physicsClientId=client_id)
    cs = p.createCollisionShape(p.GEOM_BOX, halfExtents=size, physicsClientId=client_id)
    return p.createMultiBody(0, cs, vs, basePosition=pos, physicsClientId=client_id)

def create_ball(client_id: int, pos, radius: float = 0.03, mass: float = 0.05, rgba=(1,0.2,0.2,1)) -> int:
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba, physicsClientId=client_id)
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=client_id)
    bid = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                            basePosition=pos, physicsClientId=client_id)
    p.changeDynamics(bid, -1, lateralFriction=0.8, rollingFriction=0.001, spinningFriction=0.001,
                     physicsClientId=client_id)
    return bid

def create_target_sphere(client_id: int, pos, radius: float = 0.03, rgba=(1,0,0,1)) -> int:
    import pybullet as p
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba, physicsClientId=client_id)
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=client_id)
    bid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                            basePosition=pos, physicsClientId=client_id)
    return bid

# open-top stationary container 
def create_open_container(
    client_id: int,
    center=(0.7, 0.0, 0.0),        # table top is z=0
    inner=(0.24, 0.24, 0.14),      # inner X,Y,wall_height (m)
    thickness: float = 0.01,
    rgba=(0.36, 0.25, 0.20, 1.0)   # brown
):
    """
    Builds a static (mass=0) open-top box: base + 4 thin walls.
    Returns (part_ids, center, inner, thickness)
    """
    cx, cy, _ = center
    ix, iy, h = inner
    t = thickness

    # base sits on table (z = t/2)
    base_he = [ix/2, iy/2, t/2]
    base_vs = p.createVisualShape(p.GEOM_BOX, halfExtents=base_he, rgbaColor=rgba, physicsClientId=client_id)
    base_cs = p.createCollisionShape(p.GEOM_BOX, halfExtents=base_he, physicsClientId=client_id)
    base_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=base_cs, baseVisualShapeIndex=base_vs,
                                basePosition=[cx, cy, t/2], physicsClientId=client_id)

    part_ids = [base_id]

    # walls: two along X, two along Y
    wall_zc = h/2.0
    # X walls (left/right)
    x_he = [t/2, iy/2 + t/2, h/2]
    x_vs = p.createVisualShape(p.GEOM_BOX, halfExtents=x_he, rgbaColor=rgba, physicsClientId=client_id)
    x_cs = p.createCollisionShape(p.GEOM_BOX, halfExtents=x_he, physicsClientId=client_id)
    # +X wall center
    wxp = p.createMultiBody(0, x_cs, x_vs, basePosition=[cx + ix/2 + t/2, cy, wall_zc], physicsClientId=client_id)
    # -X wall center
    wxn = p.createMultiBody(0, x_cs, x_vs, basePosition=[cx - ix/2 - t/2, cy, wall_zc], physicsClientId=client_id)
    part_ids += [wxp, wxn]

    # Y walls (front/back)
    y_he = [ix/2 + t/2, t/2, h/2]
    y_vs = p.createVisualShape(p.GEOM_BOX, halfExtents=y_he, rgbaColor=rgba, physicsClientId=client_id)
    y_cs = p.createCollisionShape(p.GEOM_BOX, halfExtents=y_he, physicsClientId=client_id)
    wyp = p.createMultiBody(0, y_cs, y_vs, basePosition=[cx, cy + iy/2 + t/2, wall_zc], physicsClientId=client_id)
    wyn = p.createMultiBody(0, y_cs, y_vs, basePosition=[cx, cy - iy/2 - t/2, wall_zc], physicsClientId=client_id)
    part_ids += [wyp, wyn]

    # make sure it's really static and not sliding
    for bid in part_ids:
        p.changeDynamics(bid, -1, lateralFriction=1.0, restitution=0.0, physicsClientId=client_id)

    return part_ids, np.array(center, dtype=np.float32), np.array(inner, dtype=np.float32), t

def sample_container_center(rng: np.random.Generator, cx_range=(0.55,0.85), cy_range=(-0.22,0.22)) -> np.ndarray:
    return np.array([rng.uniform(*cx_range), rng.uniform(*cy_range), 0.0], dtype=np.float32)

def ball_inside_container(ball_pos: np.ndarray, center: np.ndarray, inner: np.ndarray, margin: float = 0.01) -> bool:
    ix, iy, h = inner
    dx, dy = abs(ball_pos[0]-center[0]), abs(ball_pos[1]-center[1])
    # must be within inner footprint and below wall height
    in_xy = (dx < ix/2 - margin) and (dy < iy/2 - margin)
    in_z  = (0.0 <= ball_pos[2] <= h + 0.02)  # allow slight above base while dropping in
    return in_xy and in_z

# visual suction tip on EE (massless)
def attach_suction_tip(client_id, parent_body, parent_link,
                       radius=0.02, length=0.08,
                       rgba=(0.2,0.2,0.2,1),
                       local_offset=(0, 0, 0.12)):   # farther from the flange
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=length,
                              rgbaColor=rgba, physicsClientId=client_id)
    tip_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,   # NO collision
                               baseVisualShapeIndex=vis, basePosition=[0,0,0],
                               physicsClientId=client_id)
    # fix to EE; place tip forward along local Z of the EE link
    p.createConstraint(parent_body, parent_link, tip_id, -1, p.JOINT_FIXED, [0,0,0],
                       parentFramePosition=[0,0,0], childFramePosition=list(local_offset),
                       physicsClientId=client_id)
    return tip_id

