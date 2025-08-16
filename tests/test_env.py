import numpy as np
from src.envs.arm_reach_env import ArmReachEnv, ReachCfg

def test_basic_env():
    env = ArmReachEnv(ReachCfg(render=False))
    obs, info = env.reset()
    assert obs.shape[0] > 0
    for _ in range(5):
        a = np.zeros(3, dtype=np.float32)
        obs, r, d, tr, inf = env.step(a)
    env.close()
