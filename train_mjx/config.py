"""Training configuration for G1 fine-tuning with MJX."""
import numpy as np

# --- Environment ---
SCENE_XML = "resources/robots/g1_description/scene_slab.xml"
SIM_DT = 0.005           # MuJoCo simulation timestep
CONTROL_DT = 0.02        # Policy control timestep (4x decimation)
CONTROL_DECIMATION = 4
EPISODE_LENGTH_S = 20.0
MAX_EPISODE_STEPS = int(EPISODE_LENGTH_S / CONTROL_DT)  # 1000

# --- Robot ---
NUM_ACTIONS = 12
NUM_OBS = 47
ACTION_SCALE = 0.25
BASE_HEIGHT_TARGET = 0.55   # robot's actual pelvis height with default angles (Playground uses 0.5)
GAIT_PERIOD = 0.8

DEFAULT_ANGLES = np.array([
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,   # left leg
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,   # right leg
], dtype=np.float32)

KP = np.array([100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40], dtype=np.float32)
KD = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2], dtype=np.float32)

# Observation scaling
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)

# Command (forward walking)
CMD = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # v11: sweet spot (0.5 was too conservative, 1.5 broke straight walking)

# --- Termination ---
MAX_ROLL = 0.8    # rad, ~46 degrees
MAX_PITCH = 1.0   # rad, ~57 degrees
CONTACT_FORCE_THRESHOLD = 1.0  # N, for pelvis contact termination

# --- Rewards (scale * dt applied during training) ---
# v7: REPLACE forward_distance (reward-hack magnet) with tracking_lin_vel.
# tracking_lin_vel is exp(-|v - cmd|² / σ), so running past cmd DECREASES
# reward — the policy can't dive-forward to maximize distance. This was the
# Playground recipe for preventing ballistic gaits.
# Lateral drift is now LINEAR (|y|) not quadratic (y²) so outlier envs can't
# explode the advantage.
REWARD_SCALES = {
    # v9: body-frame tracking + Unitree's hip_pos anti-drift term.
    # Lateral drift zeroed — hip_pos attacks the CAUSE, not the symptom.
    "forward_distance": 0.0,
    "tracking_lin_vel": 1.0,     # Unitree/Playground standard weight
    "tracking_ang_vel": 0.5,     # Unitree standard
    "lateral_drift": 0.0,        # disabled — hip_pos replaces it
    "hip_pos": -1.0,             # NEW (v9, Unitree G1): |qpos[hip_roll, hip_yaw] - default|
    "yaw_penalty": -2.0,         # reduced from -5 — outlier envs with high yaw dragged mean negative
    "lin_vel_z": -2.0,           # Unitree standard
    "ang_vel_xy": -0.05,
    "orientation": -1.0,         # Unitree G1 weight
    "base_height": -5.0,         # only fires below 0.55 (actual standing height)
    "dof_acc": -2.5e-7,
    "dof_vel": -1e-3,
    "action_rate": -0.01,
    "alive": 0.5,                # bumped to shift average reward positive
}
TRACKING_SIGMA = 0.25
ONLY_POSITIVE_REWARDS = False  # distance reward is honest; no clipping

# --- Policy ---
LSTM_HIDDEN_SIZE = 64
ACTOR_HIDDEN_SIZE = 32

# --- PPO ---
NUM_ENVS = 64
NUM_STEPS = 24          # steps per env per iteration
ACTOR_LR = 5e-5         # v6: halfway between v2's 1e-5 and the later 1e-4
CRITIC_LR = 1e-3        # higher for randomly initialized critic
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.15         # slightly looser clipping to allow bigger steps
ENTROPY_COEF = 0.01     # 2x entropy for more exploration on terrain
VALUE_COEF = 0.5
BC_LOSS_WEIGHT = 1.0  # v10: stronger BC to keep policy tight to baseline
MAX_GRAD_NORM = 1.0
NUM_EPOCHS = 5
NUM_MINIBATCHES = 4
MAX_ITERS = 500

# --- Actor LR decay ---
# Hold constant during warmup + curriculum, then cosine-decay over the next
# ACTOR_LR_DECAY_ITERS iterations down to ACTOR_LR * ACTOR_LR_DECAY_ALPHA.
# Reduces late-training noise once the policy has diverged from pre-trained.
ACTOR_LR_DECAY_ITERS = 300
ACTOR_LR_DECAY_ALPHA = 0.1

# --- Initial state randomization (v6: minimally tight) ---
# v4 used ±0.1m/±15°/σ=0.05 and the policy lost robustness (fell 8/10 on multi-
# seed benchmark). v6 uses much tighter ranges — just enough noise to prevent
# the policy from memorizing one exact starting pose, but small enough that the
# fine-tuning doesn't corrupt motion.pt's proven flat-ground walking.
INIT_RAND_XY = 0.02          # ± meters (was 0.1)
INIT_RAND_YAW = 0.05         # ± radians, ≈ ±3° (was 0.262 / ±15°)
INIT_RAND_JOINT_SIGMA = 0.01 # was 0.05

# --- Warmup ---
CRITIC_WARMUP_ITERS = 30  # freeze actor, only train critic for first N iters

# --- Curriculum (episode length) ---
CURRICULUM_START_STEPS = 200   # 4 s — short episodes to start
CURRICULUM_END_STEPS = 1000    # 20 s — full length once the robot walks
CURRICULUM_RAMP_ITERS = 150    # iters to ramp from start to end (after warmup)

# --- Monitoring ---
LOG_INTERVAL = 1        # print every N iters
SAVE_INTERVAL = 100     # checkpoint every N iters
EARLY_STOP_PATIENCE = 80  # stop if no improvement for N iters
VALIDATE_INTERVAL = 25  # run validation rollouts every N iters
VALIDATE_STEPS = 1500   # 30 s sim — covers the full 22 m ramp course at cmd=1.0
VALIDATE_NUM_SEEDS = 8  # number of parallel validation envs (multi-seed avg)
