# quick tester
from src.envs.arm_pickplace_env import ArmPickPlaceEnv, PickPlaceCfg
import numpy as np, time
env = ArmPickPlaceEnv(PickPlaceCfg(render=True))
obs, _ = env.reset()
for t in range(300):
    a = np.array([-1.0, 0.0, -1.0, -1.0], dtype="float32")  # push toward base & down
    obs, r, d, tr, info = env.step(a)
    time.sleep(0.05)
env.close()
