# quadruped-rl-esp32

An 8-DOF quadruped that walks in the direction you command. The walking policy is trained with PPO in MuJoCo and is small enough to run on an ESP32.

## How it works

### The robot
Each of the four legs is a five-bar linkage driven by two ST3215 servos (a hip servo and a second "dew" servo), 8 actuated joints in total. The closed linkage loops are modeled in MuJoCo with equality constraints, and the servos are modeled as position actuators with a torque limit that matches the real servo. Only the feet collide with the floor, which keeps the simulation fast and stable.

### Commands
The policy is told how to move with two numbers, both in the robot's own frame:
- **vx**: forward/backward speed (-0.25 to 0.40 m/s)
- **wz**: turn rate (-1 to 1 rad/s)

The legs have no sideways (abduction) joint, so the robot cannot strafe. It moves forward or backward and steers by turning. With the keyboard, Up/Down change `vx`, Left/Right change `wz`, and `0` stops.

### Observation (what the policy sees)
Only things the real robot can measure:
- roll and pitch from a complementary filter (gyro + accelerometer)
- gyroscope (3 axes)
- joint positions and joint velocities (8 each)
- the previous action (8)
- the current command (vx, wz)
- a gait clock (sin and cos of a 2 Hz phase)

### Action
The policy outputs 8 numbers in [-1, 1], one per joint. Each becomes a servo position target:

`target angle = nominal pose + 0.5 rad × action`

The actions are low-pass filtered before they are sent to the servos, which keeps the motion smooth.

### Control loop
The policy runs at **50 Hz**. Between two policy decisions, MuJoCo runs 10 physics steps (2 ms each), and the position actuators track the targets.

### Reward
The robot is rewarded for:
- matching the commanded forward speed and turn rate
- staying level and at its nominal standing height
- stepping in a **trot** (diagonal legs swing together, matched to the gait clock)
- taking proper swing steps, with feet that clear the ground

It is penalized for:
- sliding sideways or bouncing
- feet dragging along the ground
- jerky actions and high joint accelerations
- wasting energy
- falling, which also ends the episode

### Training
PPO (Stable-Baselines3) trains a small tanh MLP actor (2 × 128) with a larger critic, on CPU across many parallel environments. Observations are normalized with running statistics. Each episode starts from a standing pose with a random command, and the command is resampled during the episode.

### Making it transfer to the real robot
During training the simulator is randomized so the policy does not rely on exact values:
- floor friction, servo stiffness and torso mass
- noise on the IMU and joint readings
- one control step of action delay (0 or 1, per episode)
- small random pushes

### Deployment
The trained network is simple enough to run on an ESP32. Its weights, the observation normalization constants, and all control settings (rate, action scale, filter coefficients, gait frequency) are exported to a C header. The firmware repeats the same loop as the simulation at 50 Hz:

1. Read the IMU and servo positions and speeds.
2. Update the roll/pitch filter and the gait phase.
3. Build the observation and run the network.
4. Low-pass filter the actions, convert them to servo ticks and send all 8 targets in one bus packet.

For this to work, the firmware must use exactly the same joint order, signs, filter and constants as the simulation.

## Status and limitations
- Trained and checked in simulation. Running on the physical robot is still in progress.
- Real servos have backlash and latency that the simulation only approximates, so expect some tuning on hardware.
