"""MJX-based environment for G1 terrain fine-tuning."""
import os
import functools
from typing import NamedTuple
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from train_mjx import config as cfg

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_model():
    """Load MuJoCo model and create MJX model."""
    xml_path = os.path.join(ROOT_DIR, cfg.SCENE_XML)
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_model.opt.timestep = cfg.SIM_DT

    # MJX pre-allocates for all collision pairs. With many mesh geoms on the
    # robot + many terrain boxes, this explodes memory. Fix:
    # 1. Disable collision on all mesh geoms (visual only)
    # 2. Convert cylinder geoms to capsules (MJX doesn't support cylinder-box)
    # 3. Keep collision only on sphere/capsule geoms (feet) + terrain boxes + floor
    GEOM_PLANE = 0
    GEOM_SPHERE = 2
    GEOM_CAPSULE = 3
    GEOM_CYLINDER = 5
    GEOM_BOX = 6
    GEOM_MESH = 7

    for i in range(mj_model.ngeom):
        gtype = mj_model.geom_type[i]
        if gtype == GEOM_MESH:
            # Disable collision on visual meshes
            mj_model.geom_contype[i] = 0
            mj_model.geom_conaffinity[i] = 0
        elif gtype == GEOM_CYLINDER:
            # Convert cylinders to capsules for MJX compatibility
            mj_model.geom_type[i] = GEOM_CAPSULE

    mx_model = mjx.put_model(mj_model)
    return mj_model, mx_model


def get_pelvis_contact_id(mj_model):
    """Get geom ID for pelvis to detect falls."""
    for i in range(mj_model.ngeom):
        name = mj_model.geom(i).name
        if "pelvis" in name.lower():
            return i
    # If no pelvis geom found, use body 1 (usually the base)
    return -1


def _get_gravity_orientation(quat):
    """Project gravity into body frame from quaternion."""
    qw, qx, qy, qz = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    gx = 2 * (-qz * qx + qw * qy)
    gy = -2 * (qz * qy + qw * qx)
    gz = 1 - 2 * (qw * qw + qz * qz)
    return jnp.stack([gx, gy, gz], axis=-1)


def _quat_to_rpy(quat):
    """Extract approximate roll and pitch from quaternion."""
    qw, qx, qy, qz = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    # Roll (x-axis rotation)
    sinr = 2 * (qw * qx + qy * qz)
    cosr = 1 - 2 * (qx * qx + qy * qy)
    roll = jnp.arctan2(sinr, cosr)
    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    pitch = jnp.arcsin(jnp.clip(sinp, -1.0, 1.0))
    return roll, pitch


class EnvState(NamedTuple):
    """Container for per-env state that MJX doesn't track. JAX pytree compatible.
    All leading-axis fields must be shape [num_envs, ...] so jax.vmap works uniformly."""
    step_count: jnp.ndarray       # [num_envs]
    prev_action: jnp.ndarray      # [num_envs, 12]
    prev_dof_vel: jnp.ndarray     # [num_envs, 12] for dof_acc computation
    prev_base_x: jnp.ndarray      # [num_envs] x-position at prior step, for distance reward
    max_steps: jnp.ndarray        # [num_envs] — same value for all envs (curriculum)
    rng: jnp.ndarray


def init_env_state(num_envs, rng, max_steps=cfg.MAX_EPISODE_STEPS):
    """Create initial environment state."""
    rngs = jax.random.split(rng, num_envs)  # [num_envs, 2]
    return EnvState(
        step_count=jnp.zeros(num_envs, dtype=jnp.int32),
        prev_action=jnp.zeros((num_envs, cfg.NUM_ACTIONS), dtype=jnp.float32),
        prev_dof_vel=jnp.zeros((num_envs, cfg.NUM_ACTIONS), dtype=jnp.float32),
        prev_base_x=jnp.zeros(num_envs, dtype=jnp.float32),
        max_steps=jnp.full((num_envs,), max_steps, dtype=jnp.int32),
        rng=rngs,
    )


def reset_single(mx_model, mj_model):
    """Create initial MJX data for a single environment."""
    mj_data = mujoco.MjData(mj_model)
    # Set default joint positions
    mj_data.qpos[7:] = cfg.DEFAULT_ANGLES
    mujoco.mj_forward(mj_model, mj_data)
    return mjx.put_data(mj_model, mj_data)


def make_randomized_batch(base_data, rng, num_envs,
                          xy_range=None, yaw_range=None, joint_sigma=None):
    """Batch base_data num_envs times, applying per-env initial-state randomization.

    Each env gets a small random xy offset, a yaw rotation around z, and joint-angle
    noise — forcing the policy to be robust to small perturbations of the starting
    pose instead of memorizing a single trajectory.

    Args:
        base_data: single-env mx_data from reset_single (unbatched).
        rng: jax.random.PRNGKey.
        num_envs: batch size.
        xy_range, yaw_range, joint_sigma: defaults to cfg.INIT_RAND_* if None.
    Returns:
        mx_data pytree with leading axis num_envs.
    """
    if xy_range is None:
        xy_range = cfg.INIT_RAND_XY
    if yaw_range is None:
        yaw_range = cfg.INIT_RAND_YAW
    if joint_sigma is None:
        joint_sigma = cfg.INIT_RAND_JOINT_SIGMA

    rngs = jax.random.split(rng, num_envs)

    def randomize_one(rng_i):
        rng_xy, rng_yaw, rng_joints = jax.random.split(rng_i, 3)
        xy = jax.random.uniform(rng_xy, (2,), minval=-xy_range, maxval=xy_range)
        yaw = jax.random.uniform(rng_yaw, (), minval=-yaw_range, maxval=yaw_range)
        joint_noise = jax.random.normal(rng_joints, (cfg.NUM_ACTIONS,)) * joint_sigma

        qpos = base_data.qpos
        qpos = qpos.at[0:2].add(xy)

        # Yaw-only quaternion around z: (cos(y/2), 0, 0, sin(y/2))
        cy = jnp.cos(yaw / 2)
        sy = jnp.sin(yaw / 2)
        qpos = qpos.at[3].set(cy)
        qpos = qpos.at[4].set(0.0)
        qpos = qpos.at[5].set(0.0)
        qpos = qpos.at[6].set(sy)

        qpos = qpos.at[7:7 + cfg.NUM_ACTIONS].add(joint_noise)

        return base_data.replace(
            qpos=qpos,
            qvel=jnp.zeros_like(base_data.qvel),
        )

    return jax.vmap(randomize_one)(rngs)


def get_obs(mx_data, env_state):
    """Compute observation vector (47-dim)."""
    qj = mx_data.qpos[7:]         # joint positions [12]
    dqj = mx_data.qvel[6:]        # joint velocities [12]
    quat = mx_data.qpos[3:7]      # base quaternion [4]
    omega = mx_data.qvel[3:6]     # base angular velocity [3]

    # Scale observations
    dof_pos = (qj - jnp.array(cfg.DEFAULT_ANGLES)) * cfg.DOF_POS_SCALE
    dof_vel = dqj * cfg.DOF_VEL_SCALE
    gravity_orientation = _get_gravity_orientation(quat)
    ang_vel = omega * cfg.ANG_VEL_SCALE
    cmd_scaled = jnp.array(cfg.CMD) * jnp.array(cfg.CMD_SCALE)

    # Gait phase
    time = env_state.step_count * cfg.CONTROL_DT
    phase = (time % cfg.GAIT_PERIOD) / cfg.GAIT_PERIOD
    sin_phase = jnp.sin(2 * jnp.pi * phase)
    cos_phase = jnp.cos(2 * jnp.pi * phase)

    obs = jnp.concatenate([
        ang_vel,                          # [3]
        gravity_orientation,              # [3]
        cmd_scaled,                       # [3]
        dof_pos,                          # [12]
        dof_vel,                          # [12]
        env_state.prev_action,            # [12]
        jnp.array([sin_phase, cos_phase]),  # [2]
    ])
    return obs


def _world_to_body(vel_world, quat):
    """Rotate a world-frame 3-vector into the base body frame using the
    inverse rotation implied by `quat = (w, x, y, z)`. Matches Playground's
    `get_local_linvel` which reads the body-frame `framelinvel` sensor.
    """
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    # Transpose of body-to-world rotation matrix.
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y + w * z)
    r02 = 2 * (x * z - w * y)
    r10 = 2 * (x * y - w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z + w * x)
    r20 = 2 * (x * z + w * y)
    r21 = 2 * (y * z - w * x)
    r22 = 1 - 2 * (x * x + y * y)
    bx = r00 * vel_world[0] + r01 * vel_world[1] + r02 * vel_world[2]
    by = r10 * vel_world[0] + r11 * vel_world[1] + r12 * vel_world[2]
    bz = r20 * vel_world[0] + r21 * vel_world[1] + r22 * vel_world[2]
    return jnp.stack([bx, by, bz])


# Joint indices within the 12-DOF leg vector where hip roll / hip yaw live.
# Default order per leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll.
_HIP_DRIFT_INDICES = jnp.array([1, 2, 7, 8], dtype=jnp.int32)


def compute_rewards(mx_data, env_state, action):
    """Compute reward terms.

    v9 fixes (research-backed):
      - `tracking_lin_vel` now uses body-frame velocity (was world frame — bug)
      - `hip_pos` penalty on hip roll/yaw deviations (official Unitree G1 term)
    """
    qj = mx_data.qpos[7:]
    dqj = mx_data.qvel[6:]
    quat = mx_data.qpos[3:7]
    world_lin_vel = mx_data.qvel[0:3]
    base_lin_vel = _world_to_body(world_lin_vel, quat)   # body frame
    base_ang_vel = mx_data.qvel[3:6]
    base_height = mx_data.qpos[2]
    base_x = mx_data.qpos[0]
    base_y = mx_data.qpos[1]

    cmd = jnp.array(cfg.CMD)
    dt = cfg.CONTROL_DT

    forward_distance = jnp.clip(base_x - env_state.prev_base_x, 0.0, 0.1)
    lateral_drift = jnp.clip(jnp.abs(base_y), 0.0, 2.0)

    # Tracking rewards — now in body frame so drift is actually penalized
    sigma = cfg.TRACKING_SIGMA
    lin_vel_error = jnp.sum(jnp.square(cmd[:2] - base_lin_vel[:2]))
    tracking_lin_vel = jnp.exp(-lin_vel_error / sigma)
    ang_vel_error = jnp.square(cmd[2] - base_ang_vel[2])
    tracking_ang_vel = jnp.exp(-ang_vel_error / sigma)

    # Penalties
    lin_vel_z = jnp.square(base_lin_vel[2])
    ang_vel_xy = jnp.sum(jnp.square(base_ang_vel[:2]))

    gravity_orientation = _get_gravity_orientation(quat)
    orientation = jnp.sum(jnp.square(gravity_orientation[:2]))

    base_height_low = jnp.square(jnp.minimum(base_height - cfg.BASE_HEIGHT_TARGET, 0.0))

    dof_acc = jnp.sum(jnp.square((dqj - env_state.prev_dof_vel) / dt))
    dof_vel_penalty = jnp.sum(jnp.square(dqj))

    action_rate = jnp.sum(jnp.square(action - env_state.prev_action))

    # hip_pos (Unitree G1 anti-drift term): penalize hip roll/yaw deviations
    # from the default pose. Splayed/twisted legs cause lateral drift, so
    # squashing hip asymmetry directly fixes the drift cause.
    default_ang = jnp.array(cfg.DEFAULT_ANGLES)
    hip_pos_err = qj[_HIP_DRIFT_INDICES] - default_ang[_HIP_DRIFT_INDICES]
    hip_pos = jnp.sum(jnp.abs(hip_pos_err))

    # v10: linear |yaw| penalty — directly punishes heading deviation from +x.
    # tracking_ang_vel only catches the DERIVATIVE of yaw; this catches the
    # absolute position. Needed because sub-5° drift is below PPO noise when
    # only tracking velocity/ang_vel.
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = jnp.arctan2(siny_cosp, cosy_cosp)
    yaw_penalty = jnp.abs(yaw)

    alive = 1.0

    scales = cfg.REWARD_SCALES
    reward = (
        scales["forward_distance"] * forward_distance / dt
        + scales["lateral_drift"] * lateral_drift
        + scales["tracking_lin_vel"] * tracking_lin_vel
        + scales["tracking_ang_vel"] * tracking_ang_vel
        + scales["lin_vel_z"] * lin_vel_z
        + scales["ang_vel_xy"] * ang_vel_xy
        + scales["orientation"] * orientation
        + scales["base_height"] * base_height_low
        + scales["dof_acc"] * dof_acc
        + scales["dof_vel"] * dof_vel_penalty
        + scales["action_rate"] * action_rate
        + scales["hip_pos"] * hip_pos
        + scales["yaw_penalty"] * yaw_penalty
        + scales["alive"] * alive
    ) * dt

    if cfg.ONLY_POSITIVE_REWARDS:
        reward = jnp.maximum(reward, 0.0)

    return reward


def check_termination(mx_data):
    """Check if episode should terminate (robot fell)."""
    quat = mx_data.qpos[3:7]
    roll, pitch = _quat_to_rpy(quat)

    terminated = jnp.logical_or(
        jnp.abs(roll) > cfg.MAX_ROLL,
        jnp.abs(pitch) > cfg.MAX_PITCH,
    )

    # Also terminate if base height is too low (fallen)
    base_height = mx_data.qpos[2]
    terminated = jnp.logical_or(terminated, base_height < 0.3)

    return terminated


def step_single(mx_model, mx_data, action, default_angles, kp, kd):
    """Step a single environment: apply action and simulate."""
    # PD control: target = action * scale + default
    target_q = action * cfg.ACTION_SCALE + default_angles

    def sim_step(data, _):
        tau = (target_q - data.qpos[7:]) * kp + (0.0 - data.qvel[6:]) * kd
        data = data.replace(ctrl=tau)
        data = mjx.step(mx_model, data)
        return data, None

    # Run CONTROL_DECIMATION physics steps
    mx_data, _ = jax.lax.scan(sim_step, mx_data, None, length=cfg.CONTROL_DECIMATION)
    return mx_data


# Vectorized versions
v_get_obs = jax.vmap(get_obs)
v_check_termination = jax.vmap(check_termination)


def make_step_fn(mx_model):
    """Create a vectorized step function for the environment."""
    default_angles = jnp.array(cfg.DEFAULT_ANGLES)
    kp = jnp.array(cfg.KP)
    kd = jnp.array(cfg.KD)

    @jax.jit
    def batched_step(mx_data, actions, env_state):
        """Step all environments in parallel."""
        # Step physics for each env
        v_step = jax.vmap(functools.partial(
            step_single,
            mx_model,
            default_angles=default_angles,
            kp=kp,
            kd=kd,
        ))
        mx_data = v_step(mx_data, actions)

        # Compute rewards (after physics step so we see the new x-position)
        v_rewards = jax.vmap(compute_rewards)
        rewards = v_rewards(mx_data, env_state, actions)

        # Check termination
        terminated = v_check_termination(mx_data)

        # Timeout — use env_state.max_steps so curriculum can grow it
        timed_out = env_state.step_count >= env_state.max_steps

        dones = jnp.logical_or(terminated, timed_out)

        # Update env state
        new_env_state = EnvState(
            step_count=env_state.step_count + 1,
            prev_action=actions,
            prev_dof_vel=mx_data.qvel[..., 6:],
            prev_base_x=mx_data.qpos[..., 0],
            max_steps=env_state.max_steps,
            rng=env_state.rng,
        )

        return mx_data, rewards, dones, new_env_state

    return batched_step
