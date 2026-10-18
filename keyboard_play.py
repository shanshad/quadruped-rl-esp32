"""
Drive the trained quadruped with the arrow keys in the MuJoCo viewer.

  python keyboard_play.py --model models/qp_ppo_final.zip --vecnorm models/vecnormalize_final.pkl
  (checkpoint example)
  python keyboard_play.py --model check_points/qp_ppo_15000000_steps.zip \
                          --vecnorm check_points/qp_ppo_vecnormalize_15000000_steps.pkl

Keys (each press / key-repeat changes the command):
  UP / DOWN     forward speed  +/- 0.05 m/s
  LEFT / RIGHT  turn rate      +/- 0.25 rad/s   (LEFT = counter-clockwise)
  0             stop (vx = wz = 0)
  Backspace     (viewer built-in) reset simulation -- we also auto-reset when the robot falls
macOS: run with `mjpython keyboard_play.py ...` instead of `python`.
"""
import argparse, time
import numpy as np
import mujoco, mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env_for_quadruped import QPEnv

# GLFW key codes
KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP, KEY_0 = 262, 263, 264, 265, 48


class Command:
    def __init__(self, vx_range, wz_range, vx_step, wz_step):
        self.vx = self.wz = 0.0
        self.vx_range, self.wz_range = vx_range, wz_range
        self.vx_step, self.wz_step = vx_step, wz_step
        self.dirty = True

    def on_key(self, key):
        if key == KEY_UP:      self.vx += self.vx_step
        elif key == KEY_DOWN:  self.vx -= self.vx_step
        elif key == KEY_LEFT:  self.wz += self.wz_step
        elif key == KEY_RIGHT: self.wz -= self.wz_step
        elif key == KEY_0:     self.vx = self.wz = 0.0
        else: return
        self.vx = float(np.clip(self.vx, *self.vx_range))
        self.wz = float(np.clip(self.wz, *self.wz_range))
        if abs(self.vx) < 1e-6: self.vx = 0.0
        if abs(self.wz) < 1e-6: self.wz = 0.0
        self.dirty = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qp_ppo_final.zip")
    ap.add_argument("--vecnorm", default="models/vecnormalize_final.pkl")
    ap.add_argument("--xml", default="quadruped_fixed.xml")
    ap.add_argument("--no-noise", action="store_true", help="disable simulated sensor noise")
    a = ap.parse_args()

    env = QPEnv(a.xml, cmd_resample_prob=0.0, randomize=False, obs_noise=not a.no_noise)
    vn = VecNormalize.load(a.vecnorm, DummyVecEnv([lambda: env]))
    vn.training = False; vn.norm_reward = False
    model = PPO.load(a.model, device="cpu")

    cmd = Command(env.cmd_vx_range, env.cmd_wz_range, 0.05, 0.25)
    obs, _ = env.reset()
    env.set_command(0, 0)

    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=cmd.on_key) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid = env.torso_id
        v.cam.distance, v.cam.elevation, v.cam.azimuth = 0.7, -20, 135
        print(__doc__)
        while v.is_running():
            t0 = time.time()
            if cmd.dirty:
                env.set_command(cmd.vx, cmd.wz)
                print(f"command: vx={cmd.vx:+.2f} m/s  wz={cmd.wz:+.2f} rad/s")
                cmd.dirty = False
            act, _ = model.predict(vn.normalize_obs(obs[None]), deterministic=True)
            obs, r, term, trunc, info = env.step(act[0])
            v.sync()
            if term or trunc:
                print("episode ended (fell / time limit) -> reset")
                obs, _ = env.reset(); env.set_command(cmd.vx, cmd.wz)
            time.sleep(max(0.0, env.control_dt - (time.time() - t0)))


if __name__ == "__main__":
    main()
