"""python play.py --vx 0.3 --wz 0   (viewer, real-time)"""
import argparse, time, numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env_for_quadruped import QPEnv

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/qp_ppo_final.zip")
ap.add_argument("--vecnorm", default="models/vecnormalize_final.pkl")
ap.add_argument("--xml", default="quadruped_fixed.xml")
ap.add_argument("--vx", type=float, default=0.3)
ap.add_argument("--wz", type=float, default=0.0)
a = ap.parse_args()

env = QPEnv(a.xml, render_mode="human", cmd_resample_prob=0.0, randomize=False)
vn = VecNormalize.load(a.vecnorm, DummyVecEnv([lambda: env])); vn.training = False; vn.norm_reward = False
model = PPO.load(a.model, device="cpu")
obs, _ = env.reset(); env.set_command(a.vx, a.wz)
while True:
    t = time.time()
    n = vn.normalize_obs(obs[None])
    act, _ = model.predict(n, deterministic=True)
    obs, r, term, trunc, info = env.step(act[0])
    env.render()
    if term or trunc:
        obs, _ = env.reset(); env.set_command(a.vx, a.wz)
    time.sleep(max(0, env.control_dt - (time.time() - t)))
