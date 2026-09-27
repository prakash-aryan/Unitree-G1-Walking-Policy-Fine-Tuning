"""Main training script for G1 fine-tuning with MJX."""
import os
import sys
import time
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import optax

from train_mjx import config as cfg
from train_mjx.env import (
    load_model, reset_single, get_obs, make_step_fn, init_env_state, EnvState,
    make_randomized_batch,
)
from train_mjx.policy import ActorCritic, init_carry, load_pretrained_params, init_log_std
from train_mjx.ppo import (
    compute_gae, ppo_loss, gaussian_log_prob, sample_action, create_optimizer,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune G1 policy with MJX")
    parser.add_argument("--num_envs", type=int, default=cfg.NUM_ENVS)
    parser.add_argument("--max_iters", type=int, default=cfg.MAX_ITERS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    num_envs = args.num_envs
    max_iters = args.max_iters

    print(f"JAX devices: {jax.devices()}")
    print(f"Num envs: {num_envs}")
    print(f"Max iterations: {max_iters}")

    # --- Load environment ---
    print("Loading MuJoCo model...")
    mj_model, mx_model = load_model()
    step_fn = make_step_fn(mx_model)
    print(f"  DOFs: {mj_model.nq - 7} joints, {mj_model.nv - 6} velocities")

    # --- Initialize environments ---
    print("Initializing environments...")
    rng = jax.random.PRNGKey(args.seed)

    # Create initial data for all envs with per-env randomization
    single_data = reset_single(mx_model, mj_model)
    rng, init_rng = jax.random.split(rng)
    mx_data = make_randomized_batch(single_data, init_rng, num_envs)

    rng, env_rng = jax.random.split(rng)
    env_state = init_env_state(num_envs, env_rng, max_steps=cfg.CURRICULUM_START_STEPS)
    # prev_base_x must match the (now randomized) initial x so the first
    # distance-reward delta is zero instead of the random offset.
    env_state = env_state._replace(prev_base_x=mx_data.qpos[:, 0])

    # --- Initialize policy ---
    print("Initializing policy...")
    model = ActorCritic()
    rng, init_rng = jax.random.split(rng)
    dummy_obs = jnp.zeros(cfg.NUM_OBS)
    dummy_carry = (jnp.zeros(cfg.LSTM_HIDDEN_SIZE), jnp.zeros(cfg.LSTM_HIDDEN_SIZE))
    params = model.init(init_rng, dummy_obs, dummy_carry)

    # Load pre-trained weights
    params = load_pretrained_params(params)
    log_std = init_log_std()

    # v9: frozen reference for BC loss (PreCi / PPF regularization).
    # A deep-copy immediately after load_pretrained_params ensures the reference
    # stays at the baseline distribution throughout training.
    ref_params = jax.tree.map(lambda x: x, params)

    # --- Optimizer (separate actor/critic LRs) ---
    param_optimizer, log_std_optimizer = create_optimizer()
    param_opt_state = param_optimizer.init(params)
    log_std_opt_state = log_std_optimizer.init(log_std)

    # --- JIT compile key functions ---
    print("JIT compiling (this may take a minute)...")

    @jax.jit
    def policy_forward(params, obs, carry, log_std, rng):
        """Run policy and sample action for a batch of envs."""
        def single_forward(obs_i, carry_i, rng_i):
            action_mean, value, new_carry = model.apply(params, obs_i, carry_i)
            action = sample_action(rng_i, action_mean, log_std)
            log_prob = gaussian_log_prob(action, action_mean, log_std)
            return action, value, log_prob, new_carry

        rngs = jax.random.split(rng, obs.shape[0])
        h, c = carry
        carry_batch = (h, c)
        actions, values, log_probs, (new_h, new_c) = jax.vmap(single_forward)(
            obs, carry_batch, rngs
        )
        return actions, values, log_probs, (new_h, new_c)

    v_get_obs = jax.jit(jax.vmap(get_obs))

    @jax.jit
    def reset_env_where_done(mx_data, env_state, dones, base_data):
        """Reset environments that are done, pulling the fresh state from
        base_data (a batch of per-env randomized initial states).
        prev_base_x is re-seeded from the NEW random x so the first
        post-reset distance delta is zero.
        """
        def reset_single_where(data_i, state_step, state_prev_act, state_prev_dv,
                               state_prev_x, done, base):
            data_out = jax.tree.map(lambda d, b: jnp.where(done, b, d), data_i, base)
            step_out = jnp.where(done, 0, state_step)
            prev_act_out = jnp.where(done, jnp.zeros_like(state_prev_act), state_prev_act)
            prev_dv_out = jnp.where(done, jnp.zeros_like(state_prev_dv), state_prev_dv)
            prev_x_out = jnp.where(done, base.qpos[0], state_prev_x)
            return data_out, step_out, prev_act_out, prev_dv_out, prev_x_out

        new_data, new_steps, new_prev_acts, new_prev_dvs, new_prev_xs = jax.vmap(reset_single_where)(
            mx_data, env_state.step_count, env_state.prev_action,
            env_state.prev_dof_vel, env_state.prev_base_x, dones, base_data,
        )
        new_env_state = EnvState(
            step_count=new_steps,
            prev_action=new_prev_acts,
            prev_dof_vel=new_prev_dvs,
            prev_base_x=new_prev_xs,
            max_steps=env_state.max_steps,
            rng=env_state.rng,
        )
        return new_data, new_env_state

    @jax.jit
    def make_fresh_base(rng_):
        """Sample a fresh batch of randomized per-env initial states."""
        return make_randomized_batch(single_data, rng_, num_envs)

    # --- Multi-seed validation rollout ---
    # Run VALIDATE_NUM_SEEDS parallel envs, each with a deterministic (fixed
    # across iterations) random perturbation, and return (mean_distance,
    # min_distance). Using the same seeds every iter means valX curves are
    # comparable across time, and the min filters out single-env lucky wins.
    val_single_data = reset_single(mx_model, mj_model)
    VAL_N = cfg.VALIDATE_NUM_SEEDS

    @jax.jit
    def validation_rollout(params, log_std):
        """Run VAL_N parallel validation envs and return (mean_dist, min_dist)."""
        val_mx = make_randomized_batch(val_single_data, jax.random.PRNGKey(0), VAL_N)
        es = init_env_state(VAL_N, jax.random.PRNGKey(1), max_steps=cfg.VALIDATE_STEPS + 10)
        es = es._replace(prev_base_x=val_mx.qpos[:, 0])
        initial_x = val_mx.qpos[:, 0]
        cr = (jnp.zeros((VAL_N, cfg.LSTM_HIDDEN_SIZE)),
              jnp.zeros((VAL_N, cfg.LSTM_HIDDEN_SIZE)))

        def body(carry_state, _):
            mx_i, es_i, cr_i = carry_state
            obs_i = v_get_obs(mx_i, es_i)
            def forward_one(o, c):
                a_mean, _v, nc = model.apply(params, o, c)
                return a_mean, nc
            actions_det, (h_new, c_new) = jax.vmap(forward_one)(obs_i, cr_i)
            mx_i, _rew, _done, es_i = step_fn(mx_i, actions_det, es_i)
            return (mx_i, es_i, (h_new, c_new)), None

        (mx_final, _es_final, _cr_final), _ = jax.lax.scan(
            body, (val_mx, es, cr), None, length=cfg.VALIDATE_STEPS,
        )
        distances = mx_final.qpos[:, 0] - initial_x  # [VAL_N]
        return jnp.mean(distances), jnp.min(distances)

    # Warm up JIT
    obs = v_get_obs(mx_data, env_state)
    carry = init_carry(num_envs)
    rng, warmup_rng = jax.random.split(rng)
    _ = policy_forward(params, obs, carry, log_std, warmup_rng)
    _ = validation_rollout(params, log_std)  # compile this too
    _ = make_fresh_base(jax.random.PRNGKey(0))  # compile
    print("JIT compilation done.")
    init_mean, init_min = validation_rollout(params, log_std)
    print(f"Initial validation: mean={float(init_mean):.3f}m min={float(init_min):.3f}m "
          f"(over {VAL_N} seeds)")

    # --- Training loop ---
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    best_mean_reward = -float("inf")
    best_val_distance = -float("inf")
    no_improve_count = 0
    carry = init_carry(num_envs)

    print("\n" + "=" * 120)
    print(f"{'Iter':>6} | {'Mean Rew':>10} | {'Min Rew':>10} | {'Max Rew':>10} | "
          f"{'Ep Len':>8} | {'MaxStep':>8} | {'MeanX':>8} | {'Time':>8} | {'Status'}")
    print("=" * 120)

    def curriculum_max_steps(iteration_idx):
        """Ramp episode length from START to END over CURRICULUM_RAMP_ITERS (after warmup)."""
        post_warmup = max(0, iteration_idx - cfg.CRITIC_WARMUP_ITERS)
        progress = min(1.0, post_warmup / float(cfg.CURRICULUM_RAMP_ITERS))
        span = cfg.CURRICULUM_END_STEPS - cfg.CURRICULUM_START_STEPS
        return int(cfg.CURRICULUM_START_STEPS + progress * span)

    for iteration in range(max_iters):
        iter_start = time.time()

        # Apply curriculum by swapping max_steps in env_state. Shape-preserving
        # update so JIT isn't invalidated.
        target_max = curriculum_max_steps(iteration)
        env_state = env_state._replace(
            max_steps=jnp.full_like(env_state.max_steps, target_max)
        )

        # --- Collect rollout ---
        all_obs = []
        all_actions = []
        all_values = []
        all_log_probs = []
        all_rewards = []
        all_dones = []
        all_carries = []

        for step in range(cfg.NUM_STEPS):
            obs = v_get_obs(mx_data, env_state)
            rng, act_rng = jax.random.split(rng)

            actions, values, log_probs, new_carry = policy_forward(
                params, obs, carry, log_std, act_rng
            )

            all_obs.append(obs)
            all_actions.append(actions)
            all_values.append(values)
            all_log_probs.append(log_probs)
            all_carries.append(carry)

            # Step environment
            mx_data, rewards, dones, env_state = step_fn(mx_data, actions, env_state)

            all_rewards.append(rewards)
            all_dones.append(dones.astype(jnp.float32))

            # Reset done envs using a freshly randomized base batch so each
            # reset gets a new perturbation, and their LSTM carry
            rng, reset_rng = jax.random.split(rng)
            fresh_base = make_fresh_base(reset_rng)
            mx_data, env_state = reset_env_where_done(mx_data, env_state, dones, fresh_base)

            # Reset carry for done envs
            h, c = new_carry
            done_mask = dones[:, None].astype(jnp.float32)
            h = h * (1 - done_mask)
            c = c * (1 - done_mask)
            carry = (h, c)

        # Stack rollout: [T, N, ...]
        all_obs = jnp.stack(all_obs)
        all_actions = jnp.stack(all_actions)
        all_values = jnp.stack(all_values)
        all_log_probs = jnp.stack(all_log_probs)
        all_rewards = jnp.stack(all_rewards)
        all_dones = jnp.stack(all_dones)

        # Bootstrap value for GAE
        last_obs = v_get_obs(mx_data, env_state)
        rng, val_rng = jax.random.split(rng)
        _, last_values, _, _ = policy_forward(params, last_obs, carry, log_std, val_rng)

        # Compute GAE
        advantages, returns = compute_gae(all_rewards, all_values, all_dones, last_values)

        # Normalize advantages
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

        # --- PPO update ---
        # Flatten: [T*N, ...]
        T, N = all_obs.shape[:2]
        flat_obs = all_obs.reshape(T * N, -1)
        flat_actions = all_actions.reshape(T * N, -1)
        flat_log_probs = all_log_probs.reshape(T * N)
        flat_advantages = advantages.reshape(T * N)
        flat_returns = returns.reshape(T * N)

        # Stack carries: [T, N, hidden] -> [T*N, hidden]
        flat_carries_h = jnp.concatenate([c[0] for c in all_carries]).reshape(T * N, -1)
        flat_carries_c = jnp.concatenate([c[1] for c in all_carries]).reshape(T * N, -1)
        flat_carries = (flat_carries_h, flat_carries_c)

        batch_size = T * N
        minibatch_size = batch_size // cfg.NUM_MINIBATCHES

        for epoch in range(cfg.NUM_EPOCHS):
            rng, perm_rng = jax.random.split(rng)
            perm = jax.random.permutation(perm_rng, batch_size)

            for mb in range(cfg.NUM_MINIBATCHES):
                mb_idx = perm[mb * minibatch_size: (mb + 1) * minibatch_size]

                mb_obs = flat_obs[mb_idx]
                mb_actions = flat_actions[mb_idx]
                mb_log_probs = flat_log_probs[mb_idx]
                mb_advantages = flat_advantages[mb_idx]
                mb_returns = flat_returns[mb_idx]
                mb_carries = (flat_carries[0][mb_idx], flat_carries[1][mb_idx])

                # Compute gradients. Differentiate w.r.t. params (argnum 0) and
                # log_std (argnum 1) only — ref_params is constant.
                grad_fn = jax.value_and_grad(ppo_loss, argnums=(0, 1), has_aux=True)
                (loss, loss_info), grads = grad_fn(
                    params, log_std, model,
                    mb_obs, mb_actions, mb_carries,
                    mb_log_probs, mb_advantages, mb_returns,
                    ref_params,
                )

                param_grads, log_std_grads = grads

                # During warmup: zero out actor/LSTM gradients, only train critic
                if iteration < cfg.CRITIC_WARMUP_ITERS:
                    param_grads = jax.tree.map(
                        lambda g, p: jnp.zeros_like(g) if g.shape != p.shape
                        else g,  # fallback, won't match
                        param_grads, param_grads,
                    )
                    # Zero actor grads manually
                    frozen = param_grads["params"]
                    for key in ["lstm", "actor_0", "actor_1"]:
                        if key in frozen:
                            frozen[key] = jax.tree.map(jnp.zeros_like, frozen[key])
                    param_grads = {**param_grads, "params": frozen}
                    log_std_grads = jnp.zeros_like(log_std_grads)

                # Update params
                p_updates, param_opt_state = param_optimizer.update(
                    param_grads, param_opt_state, params
                )
                params = optax.apply_updates(params, p_updates)

                # Update log_std
                ls_updates, log_std_opt_state = log_std_optimizer.update(
                    log_std_grads, log_std_opt_state, log_std
                )
                log_std = optax.apply_updates(log_std, ls_updates)

        # --- Logging ---
        iter_time = time.time() - iter_start
        mean_reward = float(jnp.mean(all_rewards))
        min_reward = float(jnp.min(jnp.sum(all_rewards, axis=0)))
        max_reward = float(jnp.max(jnp.sum(all_rewards, axis=0)))
        mean_ep_len = float(jnp.mean(env_state.step_count))
        mean_x = float(jnp.mean(mx_data.qpos[..., 0]))

        # Periodic multi-seed validation — this is the reliable metric.
        # Early stopping and best-model selection use the MIN across seeds
        # (robustness) rather than mean.
        if iteration % cfg.VALIDATE_INTERVAL == 0 and iteration >= cfg.CRITIC_WARMUP_ITERS:
            val_mean, val_min = validation_rollout(params, log_std)
            val_mean = float(val_mean)
            val_min = float(val_min)
        else:
            val_mean = None
            val_min = None

        if iteration < cfg.CRITIC_WARMUP_ITERS:
            status = f"warmup ({iteration+1}/{cfg.CRITIC_WARMUP_ITERS})"
        elif val_min is not None:
            if val_min > best_val_distance:
                best_val_distance = val_min
                no_improve_count = 0
                status = f"improving (valMin={val_min:.2f} mean={val_mean:.2f})"

                # Save best checkpoint immediately, keyed on min across seeds
                best_path = os.path.join(CHECKPOINT_DIR, "best_weights.npz")
                actor_best = {
                    "lstm_kernel_i": np.array(params["params"]["lstm"]["kernel_i"]),
                    "lstm_kernel_h": np.array(params["params"]["lstm"]["kernel_h"]),
                    "lstm_bias": np.array(params["params"]["lstm"]["bias"]),
                    "actor_w0": np.array(params["params"]["actor_0"]["kernel"]),
                    "actor_b0": np.array(params["params"]["actor_0"]["bias"]),
                    "actor_w1": np.array(params["params"]["actor_1"]["kernel"]),
                    "actor_b1": np.array(params["params"]["actor_1"]["bias"]),
                    "log_std": np.array(log_std),
                }
                np.savez(best_path, **actor_best)
            else:
                no_improve_count += 1
                status = (f"plateau ({no_improve_count}/{cfg.EARLY_STOP_PATIENCE}) "
                          f"valMin={val_min:.2f} mean={val_mean:.2f}")
        else:
            status = "training"

        # Update best_mean_reward as a secondary stat
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward

        if iteration % cfg.LOG_INTERVAL == 0:
            print(f"{iteration:>6} | {mean_reward:>10.4f} | {min_reward:>10.4f} | "
                  f"{max_reward:>10.4f} | {mean_ep_len:>8.1f} | "
                  f"{int(env_state.max_steps[0]):>8d} | {mean_x:>8.2f} | "
                  f"{iter_time:>7.2f}s | {status}")

        # Save checkpoint
        if iteration % cfg.SAVE_INTERVAL == 0 and iteration > 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"iter_{iteration}.npz")
            flat_params = jax.tree.leaves(params)
            np.savez(
                ckpt_path,
                *[np.array(p) for p in flat_params],
                log_std=np.array(log_std),
            )
            print(f"  -> Checkpoint saved: {ckpt_path}")

        # Early stopping
        if no_improve_count >= cfg.EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping: no improvement for {cfg.EARLY_STOP_PATIENCE} iterations.")
            break

    # --- Save final model ---
    print("\nSaving final model...")
    final_path = os.path.join(CHECKPOINT_DIR, "final_weights.npz")

    # Extract actor weights for deployment
    actor_params = {
        "lstm_kernel_i": np.array(params["params"]["lstm"]["kernel_i"]),
        "lstm_kernel_h": np.array(params["params"]["lstm"]["kernel_h"]),
        "lstm_bias": np.array(params["params"]["lstm"]["bias"]),
        "actor_w0": np.array(params["params"]["actor_0"]["kernel"]),
        "actor_b0": np.array(params["params"]["actor_0"]["bias"]),
        "actor_w1": np.array(params["params"]["actor_1"]["kernel"]),
        "actor_b1": np.array(params["params"]["actor_1"]["bias"]),
        "log_std": np.array(log_std),
    }
    np.savez(final_path, **actor_params)
    print(f"Final weights saved to: {final_path}")
    print(f"Best mean reward: {best_mean_reward:.4f}")
    print(f"Best validation distance: {best_val_distance:.3f}m")
    print(f"Best-distance checkpoint: {os.path.join(CHECKPOINT_DIR, 'best_weights.npz')}")
    print("\nTo convert back to TorchScript for MuJoCo deployment, run:")
    print("  python train_mjx/export_to_torch.py")


if __name__ == "__main__":
    main()
