"""
QPEnv v2 - 8-DOF quadruped, velocity-command walking, built for sim2real on ESP32.

Key differences from v1
  * Policy runs at CONTROL_HZ (50 Hz) with `n_substeps` physics steps per action
    (v1 ran the policy at 500 Hz -> gamma=0.99 only "saw" 0.2 s, and an ESP32 can't do that anyway).
  * Command = (forward velocity vx [m/s], yaw rate wz [rad/s]) in the BODY frame.
    (v1 gave a world yaw *angle* target but never put the robot's yaw in the observation.)
  * Velocities for the reward come from the free joint (qvel), rotated to the body frame,
    not from data.cvel (that is referenced to the tree COM, not the torso).
  * Air-time reward bug fixed (air_time was zeroed before it was used, so the bonus was always 0).
  * Reward/termination use true orientation; the *observation* uses the complementary-filter
    roll/pitch (what the ESP32 will actually have).
  * Feet contact from dedicated foot sphere geoms (added by patch_xml.py), not <touch> sensors.
  * Observation = only things you can measure on the real robot (+ previous action, + command).
  * Light domain randomisation + sensor noise + 0/1 step action latency.

Observation (33):  roll, pitch, gyro(3), joint_pos(8), joint_vel(8), prev_action(8), cmd_vx, cmd_wz, sin(phase), cos(phase)
Joint order (obs AND action):
  [LF_hip, LF_dew, BL_hip, BL_dew, RF_hip, RF_dew, RB_hip, RB_dew]
Action (8, in [-1,1]):  target_angle = nominal + ACTION_SCALE * action   (position servo targets)
"""
import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer

JOINT_NAMES = [
    "hip_drive_joint", "dew_drive_joint",
    "hip_drive_joint_BL", "dew_drive_joint_BL",
    "hip_drive_joint_RF", "dew_drive_joint_RF",
    "hip_drive_joint_RB", "dew_drive_joint_RB",
]
ACTUATOR_NAMES = [
    "LF_hip_actuator", "LF_dew_actuator",
    "BL_hip_actuator", "BL_dew_actuator",
    "RF_hip_actuator", "RF_dew_actuator",
    "RB_hip_actuator", "RB_dew_actuator",
]
FOOT_GEOM_NAMES = ["foot_geom_LF", "foot_geom_BL", "foot_geom_RF", "foot_geom_RB"]
OBS_DIM = 2 + 3 + 8 + 8 + 8 + 2  # = 31


class QPEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        mjcf_path="quadruped_fixed.xml",
        render_mode=None,
        control_hz=50,
        max_episode_seconds=20.0,
        alpha=0.98,                      # complementary filter (per CONTROL step, same as firmware)
        action_scale=0.5,                # rad
        nominal_joint_pos=None,
        torso_body_name="torso",
        gyro_sensor="imu_gyro",
        accel_sensor="imu_accel",
        floor_geom_name="floor",
        # commands
        cmd_vx_range=(-0.25, 0.40),
        cmd_wz_range=(-1.0, 1.0),
        cmd_zero_prob=0.10,
        cmd_resample_prob=0.01,          # per control step (~ every 2 s)
        # reward
        alive_bonus=0.5,
        lin_vel_coef=3.0, lin_vel_sigma=0.04,
        yaw_rate_coef=1.0, yaw_rate_sigma=0.25,
        lat_vel_coef=1.0,
        tilt_coef=3.0,
        height_coef=20.0,
        ang_vel_xy_coef=0.02,
        vz_coef=1.0,
        action_rate_coef=0.1,
        energy_coef=0.002,
        pose_coef=0.05,
        air_time_coef=1.0, min_air_time=0.10, max_air_time=0.30,
        foot_slip_coef=0.2,
        no_contact_penalty=0.5,
        fall_penalty=-5.0,
        tip_threshold_deg=45.0,
        # sim2real
        randomize=True,
        obs_noise=True,
        push_prob=0.004,
        nominal_height=None,             # None -> measured from the model at qpos0
        act_lpf=0.5,                     # action low-pass: a_s = lpf*a_s + (1-lpf)*a  (replicate in firmware)
        gait_hz=2.0,                     # trot clock frequency
        gait_coef=1.5,                   # reward for feet matching trot stance/swing pattern
        foot_clear_coef=200.0, foot_clear_target=0.03,
        dof_acc_coef=2.5e-7,
    ):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode

        self.control_dt = 1.0 / control_hz
        self.n_substeps = max(1, int(round(self.control_dt / self.model.opt.timestep)))
        self.control_dt = self.n_substeps * self.model.opt.timestep
        self.max_steps = int(max_episode_seconds / self.control_dt)
        self.alpha = alpha
        self.action_scale = action_scale

        m = self.model
        nm = mujoco.mj_name2id
        self.torso_id = nm(m, mujoco.mjtObj.mjOBJ_BODY, torso_body_name)
        assert self.torso_id >= 0, f"body '{torso_body_name}' not found"
        j_ids = [nm(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
        a_ids = [nm(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ACTUATOR_NAMES]
        f_ids = [nm(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in FOOT_GEOM_NAMES]
        for names, ids in ((JOINT_NAMES, j_ids), (ACTUATOR_NAMES, a_ids), (FOOT_GEOM_NAMES, f_ids)):
            missing = [n for n, i in zip(names, ids) if i < 0]
            assert not missing, f"not found in XML (did you run patch_xml.py?): {missing}"
        self._qadr = np.array([m.jnt_qposadr[j] for j in j_ids])
        self._dadr = np.array([m.jnt_dofadr[j] for j in j_ids])
        self._act_ids = np.array(a_ids)
        self._foot_geoms = np.array(f_ids)
        self._foot_radius = float(m.geom_size[f_ids[0], 0])
        self.floor_id = nm(m, mujoco.mjtObj.mjOBJ_GEOM, floor_geom_name)

        def s_adr(name):
            sid = nm(m, mujoco.mjtObj.mjOBJ_SENSOR, name)
            assert sid >= 0, f"sensor {name} missing"
            return m.sensor_adr[sid], m.sensor_dim[sid]
        self._gyro_adr, _ = s_adr(gyro_sensor)
        self._accel_adr, _ = s_adr(accel_sensor)

        self._ctrl_low = m.actuator_ctrlrange[self._act_ids, 0].copy()
        self._ctrl_high = m.actuator_ctrlrange[self._act_ids, 1].copy()
        self.nominal_joint_pos = (np.zeros(8, np.float32) if nominal_joint_pos is None
                                  else np.asarray(nominal_joint_pos, np.float32))

        # nominal standing height = torso z - lowest foot point at qpos0
        mujoco.mj_resetData(m, self.data)
        mujoco.mj_forward(m, self.data)
        lowest = self.data.geom_xpos[self._foot_geoms, 2].min() - self._foot_radius
        measured = float(self.data.xpos[self.torso_id, 2] - lowest)
        self.nominal_height = measured if nominal_height is None else nominal_height
        self._spawn_dz = -lowest + 0.003     # lift so feet start 3 mm above floor

        # cache defaults for randomisation
        self._base_floor_fric = m.geom_friction[self.floor_id].copy()
        self._base_gain = m.actuator_gainprm[self._act_ids, 0].copy()
        self._base_bias = m.actuator_biasprm[self._act_ids, 1].copy()
        self._base_mass = m.body_mass.copy()

        # command / reward params
        self.cmd_vx_range, self.cmd_wz_range = cmd_vx_range, cmd_wz_range
        self.cmd_zero_prob, self.cmd_resample_prob = cmd_zero_prob, cmd_resample_prob
        self.__dict__.update(dict(
            alive_bonus=alive_bonus, lin_vel_coef=lin_vel_coef, lin_vel_sigma=lin_vel_sigma,
            yaw_rate_coef=yaw_rate_coef, yaw_rate_sigma=yaw_rate_sigma, lat_vel_coef=lat_vel_coef,
            tilt_coef=tilt_coef, height_coef=height_coef, ang_vel_xy_coef=ang_vel_xy_coef,
            vz_coef=vz_coef, action_rate_coef=action_rate_coef, energy_coef=energy_coef,
            pose_coef=pose_coef, air_time_coef=air_time_coef, min_air_time=min_air_time,
            max_air_time=max_air_time, foot_slip_coef=foot_slip_coef,
            no_contact_penalty=no_contact_penalty, fall_penalty=fall_penalty,
            tip_threshold=math.radians(tip_threshold_deg), randomize=randomize,
            obs_noise=obs_noise, push_prob=push_prob, act_lpf=act_lpf, gait_hz=gait_hz,
            gait_coef=gait_coef, foot_clear_coef=foot_clear_coef, foot_clear_target=foot_clear_target,
            dof_acc_coef=dof_acc_coef,
        ))

        obs_dim = 2 + 3 + 8 + 8 + 8 + 2 + 2   # + sin/cos gait phase
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (8,), np.float32)

        self._viewer = None
        self._step = 0
        self._roll = self._pitch = 0.0
        self._prev_action = np.zeros(8, np.float32)
        self._applied_action = np.zeros(8, np.float32)
        self._delay = 0
        self._phase = 0.0
        self._smooth = np.zeros(8)
        self._prev_dq = np.zeros(8)
        self._cmd_vx = self._cmd_wz = 0.0
        self._air_time = np.zeros(4)
        self._prev_contacts = np.zeros(4, bool)

    # ------------------------------------------------------------------ helpers
    def set_command(self, vx, wz):
        self._cmd_vx, self._cmd_wz = float(vx), float(wz)

    def _resample_command(self):
        if self.np_random.random() < self.cmd_zero_prob:
            self._cmd_vx = self._cmd_wz = 0.0
        else:
            self._cmd_vx = self.np_random.uniform(*self.cmd_vx_range)
            self._cmd_wz = self.np_random.uniform(*self.cmd_wz_range)

    def _foot_contacts(self):
        c = np.zeros(4, bool)
        d = self.data
        for i in range(d.ncon):
            con = d.contact[i]
            for k, g in enumerate(self._foot_geoms):
                if con.geom1 == g or con.geom2 == g:
                    c[k] = True
        return c

    def _foot_horizontal_speed(self):
        v = np.zeros(6)
        out = np.zeros(4)
        for k, g in enumerate(self._foot_geoms):
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_GEOM, int(g), v, 0)
            out[k] = math.hypot(v[3], v[4])   # [ang(3), lin(3)] in world frame
        return out

    def _imu_update(self):
        """Complementary filter - identical maths to run on the ESP32 at CONTROL_HZ."""
        d = self.data
        gyro = d.sensordata[self._gyro_adr:self._gyro_adr + 3].copy()
        accel = d.sensordata[self._accel_adr:self._accel_adr + 3].copy()
        if self.obs_noise:
            gyro += self.np_random.normal(0, 0.03, 3)
            accel += self.np_random.normal(0, 0.3, 3)
        a_roll = math.atan2(accel[1], accel[2])
        a_pitch = -math.atan2(accel[0], math.hypot(accel[1], accel[2]))
        dt = self.control_dt
        self._roll = self.alpha * (self._roll + gyro[0] * dt) + (1 - self.alpha) * a_roll
        self._pitch = self.alpha * (self._pitch + gyro[1] * dt) + (1 - self.alpha) * a_pitch
        return gyro, a_roll, a_pitch

    def _get_obs(self):
        gyro, _, _ = self._imu_update()
        q = self.data.qpos[self._qadr].astype(np.float64)
        dq = self.data.qvel[self._dadr].astype(np.float64)
        if self.obs_noise:
            q = q + self.np_random.normal(0, 0.01, 8)
            dq = dq + self.np_random.normal(0, 0.3, 8)
        obs = np.concatenate([[self._roll, self._pitch], gyro, q, dq,
                              self._prev_action, [self._cmd_vx, self._cmd_wz],
                              [math.sin(2 * math.pi * self._phase), math.cos(2 * math.pi * self._phase)]])
        return obs.astype(np.float32), q, dq

    def _randomize(self):
        m = self.model
        rng = self.np_random
        m.geom_friction[self.floor_id] = self._base_floor_fric * np.array([rng.uniform(0.6, 1.3), 1, 1])
        s = rng.uniform(0.85, 1.15, 8)               # servo stiffness
        m.actuator_gainprm[self._act_ids, 0] = self._base_gain * s
        m.actuator_biasprm[self._act_ids, 1] = self._base_bias * s
        m.body_mass[:] = self._base_mass
        m.body_mass[self.torso_id] *= rng.uniform(0.9, 1.15)
        self._delay = int(rng.integers(0, 2))        # 0 or 1 control-step latency

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        if self.randomize:
            self._randomize()
        self.data.qpos[2] += self._spawn_dz
        mujoco.mj_forward(self.model, self.data)
        # let the closed loops / servos settle at the nominal pose
        self.data.ctrl[self._act_ids] = self.nominal_joint_pos
        mujoco.mj_step(self.model, self.data, nstep=int(0.2 / self.model.opt.timestep))
        self.data.qvel[0:3] += self.np_random.uniform(-0.05, 0.05, 3)
        self.data.qvel[3:6] += self.np_random.uniform(-0.1, 0.1, 3)

        self._step = 0
        self._phase = float(self.np_random.uniform(0, 1))
        self._smooth = np.zeros(8)
        self._prev_dq = np.zeros(8)
        self._prev_action[:] = 0
        self._applied_action[:] = 0
        self._air_time[:] = 0
        self._prev_contacts = self._foot_contacts()
        self._resample_command()

        # initialise filter from accelerometer
        acc = self.data.sensordata[self._accel_adr:self._accel_adr + 3]
        self._roll = math.atan2(acc[1], acc[2])
        self._pitch = -math.atan2(acc[0], math.hypot(acc[1], acc[2]))
        obs, _, _ = self._get_obs()
        return obs, {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self._smooth = self.act_lpf * self._smooth + (1 - self.act_lpf) * action
        sa = self._smooth.astype(np.float32)
        applied = self._applied_action if self._delay else sa
        ctrl = np.clip(self.nominal_joint_pos + self.action_scale * applied,
                       self._ctrl_low, self._ctrl_high)
        self.data.ctrl[self._act_ids] = ctrl
        mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        self._applied_action = sa.copy()
        self._step += 1

        if self.np_random.random() < self.cmd_resample_prob:
            self._resample_command()
        if self.np_random.random() < self.push_prob and self.randomize:
            self.data.qvel[0:2] += self.np_random.uniform(-0.2, 0.2, 2)

        moving = abs(self._cmd_vx) > 0.05 or abs(self._cmd_wz) > 0.1
        if moving:
            self._phase = (self._phase + self.gait_hz * self.control_dt) % 1.0
        obs, q, dq = self._get_obs()
        dt = self.control_dt

        # ---- true state for reward / termination
        R = self.data.xmat[self.torso_id].reshape(3, 3)          # body -> world
        v_b = R.T @ self.data.qvel[0:3]                          # body-frame linear vel
        w_b = self.data.qvel[3:6]                                # body-frame angular vel (free joint)
        gz = R.T @ np.array([0.0, 0.0, -1.0])                    # gravity in body frame
        roll_t = math.atan2(R[2, 1], R[2, 2])
        pitch_t = -math.asin(np.clip(R[2, 0], -1, 1))
        z = self.data.xpos[self.torso_id, 2]

        contacts = self._foot_contacts()
        foot_speed = self._foot_horizontal_speed()

        # air-time bookkeeping (bonus is computed BEFORE the timer is reset)
        self._air_time += dt
        landed = contacts & ~self._prev_contacts
        air_bonus = float(np.sum(np.clip(self._air_time - self.min_air_time, 0, self.max_air_time - self.min_air_time) * landed))
        self._air_time[contacts] = 0.0
        self._prev_contacts = contacts

        finite = np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        fell = (not finite) or abs(roll_t) > self.tip_threshold or abs(pitch_t) > self.tip_threshold \
            or z < 0.5 * self.nominal_height

        if fell:
            reward, terminated = self.fall_penalty, True
        else:
            terminated = False
            r = self.alive_bonus
            r += self.lin_vel_coef * math.exp(-(v_b[0] - self._cmd_vx) ** 2 / self.lin_vel_sigma)
            r += self.yaw_rate_coef * math.exp(-(w_b[2] - self._cmd_wz) ** 2 / self.yaw_rate_sigma)
            r -= self.lat_vel_coef * v_b[1] ** 2
            r -= self.tilt_coef * (gz[0] ** 2 + gz[1] ** 2)
            r -= self.height_coef * (z - self.nominal_height) ** 2
            r -= self.ang_vel_xy_coef * (w_b[0] ** 2 + w_b[1] ** 2)
            r -= self.vz_coef * v_b[2] ** 2
            r -= self.action_rate_coef * float(np.sum((action - self._prev_action) ** 2))
            tau = self.data.actuator_force[self._act_ids]
            r -= self.energy_coef * float(np.sum(np.abs(tau * self.data.qvel[self._dadr])))
            r -= self.pose_coef * float(np.sum((self.data.qpos[self._qadr] - self.nominal_joint_pos) ** 2))
            if moving:
                r += self.air_time_coef * air_bonus
                desired_stance = ((self._phase + np.array([0.0, 0.5, 0.5, 0.0])) % 1.0) < 0.5   # LF,BL,RF,RB trot
                r += self.gait_coef * float(np.mean(contacts == desired_stance))
                h = self.data.geom_xpos[self._foot_geoms, 2] - self._foot_radius
                r -= self.foot_clear_coef * float(np.sum(np.clip(self.foot_clear_target - h, 0, None) ** 2 * foot_speed))
            dqt = self.data.qvel[self._dadr]
            r -= self.dof_acc_coef * float(np.sum(((dqt - self._prev_dq) / dt) ** 2))
            r -= self.foot_slip_coef * float(np.sum(foot_speed * contacts))
            if not contacts.any():
                r -= self.no_contact_penalty
            reward = r

        self._prev_dq = self.data.qvel[self._dadr].copy()
        self._prev_action = action
        truncated = self._step >= self.max_steps
        info = {"vx": float(v_b[0]), "cmd_vx": self._cmd_vx, "wz": float(w_b[2]), "cmd_wz": self._cmd_wz}
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.cam.lookat[:] = [0, 0, 0.1]
            self._viewer.cam.distance = 0.8
            self._viewer.cam.elevation = -20
        self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
