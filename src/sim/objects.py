from __future__ import annotations
import pybullet as p
import numpy as np

def create_target_sphere(client_id: int, pos, radius: float = 0.03, rgba=(1,0,0,1)) -> int:
    vis = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=radius, rgbaColor=rgba,
                              physicsClientId=client_id)
    col = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=radius,
                                 physicsClientId=client_id)
    bid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                            basePosition=pos, physicsClientId=client_id)
    return bid

def create_table(client_id: int, pos=(0.6, 0.0, -0.02), size=(0.6, 0.6, 0.02), rgba=(0.7,0.7,0.7,1)):
    vs = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=rgba, physicsClientId=client_id)
    cs = p.createCollisionShape(p.GEOM_BOX, halfExtents=size, physicsClientId=client_id)
    return p.createMultiBody(0, cs, vs, basePosition=pos, physicsClientId=client_id)

def create_cube(client_id: int, pos=(0.6, 0.0, 0.05), size=0.04, mass=0.05, rgba=(0,1,0,1)):
    vs = p.createVisualShape(p.GEOM_BOX, halfExtents=[size/2]*3, rgbaColor=rgba, physicsClientId=client_id)
    cs = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size/2]*3, physicsClientId=client_id)
    return p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=cs, baseVisualShapeIndex=vs,
                             basePosition=pos, physicsClientId=client_id)
