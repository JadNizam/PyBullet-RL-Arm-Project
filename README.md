PyBullet RL Arm (Reach + Pick-and-Place)

Minimal RL project using PyBullet + Gymnasium + Stable-Baselines3 (PPO) to train a 6-DoF arm for:
1) Reach: move the end-effector to a random 3D target.
2) PickPlace: grasp a ball and place it in a randomly moving goal box.

Quick Start (Windows PowerShell)
1) Create and activate a virtual environment
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

2) Install dependencies
   pip install -r requirements.txt

3) Sanity check the simulator (GUI opens)
   python -m scripts.make_env_demo --render

Train
# Reach
python -m src.agents.train_sb3 --env reach --cfg src/configs/reach.yaml --total-steps 300000 --seed 1 --logdir runs/reach/seed_1

# Pick-and-Place
python -m src.agents.train_sb3 --env pickplace --cfg src/configs/pickplace.yaml --total-steps 1500000 --seed 1 --logdir runs/pickplace/seed_1_overbin

Evaluate (watch in GUI)
# Reach
python -m src.agents.evaluate --env reach --model checkpoints/reach/best_model.zip --episodes 5 --render --fps 5

# Pick-and-Place
python -m src.agents.evaluate --env pickplace --model checkpoints/pickplace/best_model.zip --episodes 5 --render --fps 15



Optional: record a video (MP4)
Install once:
   pip install "imageio[ffmpeg]"
Example:
   python -m src.agents.evaluate --env pickplace --model checkpoints/pickplace/best_model.zip --episodes 5 --render --fps 10 --video videos/pickplace_demo.mp4

Environments
Reach
- Obs: joint pos/vel, end-effector pos, target pos, delta
- Action: 3D delta in EE space (IK step)
- Reward: -||ee - target|| - 0.001*||action||^2, +1.0 on success (within threshold)

Pick-and-Place
- Obs: joint pos/vel, end-effector pos, ball pos, moving goal pos, grasped flag
- Action: [dx, dy, dz, grip] with IK step; grip>0 closes (attach via fixed constraint), grip<0 opens
- Reward:
  - Not grasped: -||ee - ball|| - 0.001*||action||^2 (+0.5 on grasp event)
  - Grasped: -2.0*||ball - goal|| - 0.001*||action||^2 (+1.0 on successful place)
- Success: ball near goal (< threshold) and released

Project Layout
  rl_pybullet_arm/
    README.md
    requirements.txt
    .gitignore
    src/
      configs/
        default.yaml
        reach.yaml
        pickplace.yaml
      envs/
        arm_reach_env.py
        arm_pickplace_env.py
      sim/
        world.py
        robots.py
        objects.py
      agents/
        train_sb3.py
        evaluate.py
      utils/
        wrappers.py
        logger.py
        seeds.py
        paths.py
    scripts/
      make_env_demo.py
    tests/
      test_env.py

Tips & Troubleshooting
- Run module commands from the project root (python -m ...).
- If Activate.ps1 is blocked:
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\.venv\Scripts\Activate.ps1
- GUI too fast? lower --fps (e.g., --fps 5). To slow more, step fewer times per second.
- best_model.zip appears after training finishes (or via periodic eval if you add EvalCallback).
- CPU is fine for this project; a GPU PyTorch build is optional.
