"""python train.py [--n-envs 12] [--steps 15000000] [--resume check_points/qp_ppo_XXXX_steps.zip]"""
import os, argparse, multiprocessing as mp
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from env_for_quadruped import QPEnv

XML_PATH = os.path.abspath("quadruped_fixed.xml")   # output of patch_xml.py

def make_env(rank):
    def _init():
        env = Monitor(QPEnv(mjcf_path=XML_PATH))
        env.reset(seed=rank)
        return env
    return _init

def linear(initial, final=1e-5):
    return lambda p: final + p * (initial - final)   # p goes 1 -> 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-envs", type=int, default=max(2, min(16, (mp.cpu_count() or 4) - 2)))
    ap.add_argument("--steps", type=int, default=30_000_000)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--vecnorm", type=str, default=None, help="vecnormalize .pkl matching --resume")
    ap.add_argument("--save-every", type=int, default=100_000, help="timesteps between checkpoints")
    a = ap.parse_args()
    for d in ("tb_logs", "check_points", "models"):
        os.makedirs(d, exist_ok=True)

    venv = SubprocVecEnv([make_env(i) for i in range(a.n_envs)])   # MuJoCo is CPU-bound -> real processes
    n_steps = 256
    if a.resume:
        venv = VecNormalize.load(a.vecnorm, venv); venv.training = True
        model = PPO.load(a.resume, env=venv, device="cpu")
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
        model = PPO(
            "MlpPolicy", venv, verbose=1,
            device="cpu",                                  # small MLP: CPU is faster than CUDA here
            n_steps=n_steps,
            batch_size=(a.n_envs * n_steps) // 8,
            n_epochs=5,
            learning_rate=linear(3e-4),
            gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, target_kl=0.03,
            tensorboard_log="./tb_logs",
            # small tanh actor (easy to run on ESP32); bigger critic is free (not deployed)
            policy_kwargs=dict(log_std_init=-1.0, net_arch=dict(pi=[128, 128], vf=[256, 256])),
        )
    ckpt = CheckpointCallback(save_freq=max(a.save_every // a.n_envs, 1), save_path="./check_points",
                              name_prefix="qp_ppo", save_vecnormalize=True)
    model.learn(a.steps, callback=ckpt, tb_log_name="ppo_run", reset_num_timesteps=not a.resume)
    model.save("models/qp_ppo_final")
    venv.save("models/vecnormalize_final.pkl")
    print("Saved models/qp_ppo_final.zip + models/vecnormalize_final.pkl")
