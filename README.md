RL PyBullet Arm (Reach Task)

A minimal end-to-end reinforcement learning project using PyBullet + Gymnasium + Stable-Baselines3 (PPO) to train a 6-DoF robotic arm to reach a 3D target.

Quick Start (Windows PowerShell)
1) Create and activate a virtual environment
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

2) Install dependencies
   pip install -r requirements.txt

3) Sanity-check the simulator (GUI opens)
   python -m scripts.make_env_demo --render

Train PPO
Run from the project root (venv active). This trains for 300k env steps and saves a model.

   python -m src.agents.train_sb3 --env reach --cfg src/configs/reach.yaml --total-steps 300000 --seed 1 --logdir runs/reach/seed_1

Evaluate and watch the trained agent (GUI)
Loads the saved model and runs a few episodes in the PyBullet GUI.

   python -m src.agents.evaluate --model checkpoints/reach/best_model.zip --episodes 5 --render --fps 5

Optional: record a video (MP4)
Install once:
   pip install imageio[ffmpeg]

Then run:
   python -m src.agents.evaluate --model checkpoints/reach/best_model.zip --episodes 5 --render --fps 10 --video videos/reach_demo.mp4

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

What the agent does
- Each episode spawns a target sphere in reachable space.
- The agent moves the end effector toward the target using small IK-based 3D deltas.
- Reward = negative distance to target + small action penalty; success bonus when within threshold.

Troubleshooting
- If Activate.ps1 is blocked:
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\.venv\Scripts\Activate.ps1
- If you see ModuleNotFoundError: run commands as modules from the project root (python -m ...).
- GUI too fast: lower --fps (e.g., --fps 5).
- best_model.zip appears after training finishes (or via periodic eval if you add an EvalCallback).
- CPU is fine for this project; GPU Torch build is optional.
