"""PPO algorithm in JAX for fine-tuning."""
import jax
import jax.numpy as jnp
import optax

from train_mjx import config as cfg


def compute_gae(rewards, values, dones, last_value):
    """Compute Generalized Advantage Estimation.

    Args:
        rewards: [T, N] rewards per step
        values: [T, N] value estimates
        dones: [T, N] episode done flags
        last_value: [N] bootstrap value for last step
    Returns:
        advantages: [T, N]
        returns: [T, N]
    """
    T = rewards.shape[0]

    def scan_fn(carry, t):
        gae, next_value = carry
        done = dones[t]
        reward = rewards[t]
        value = values[t]

        delta = reward + cfg.GAMMA * next_value * (1 - done) - value
        gae = delta + cfg.GAMMA * cfg.GAE_LAMBDA * (1 - done) * gae
        return (gae, value), gae

    init = (jnp.zeros_like(last_value), last_value)
    # Scan in reverse order
    _, advantages = jax.lax.scan(
        scan_fn, init,
        jnp.arange(T - 1, -1, -1),  # reverse indices
    )
    advantages = advantages[::-1]  # reverse back
    returns = advantages + values
    return advantages, returns


def ppo_loss(params, log_std, model, obs_batch, action_batch, carry_batch,
             old_log_prob_batch, advantage_batch, return_batch,
             ref_params):
    """Compute PPO clipped loss + value loss + entropy bonus + BC-to-reference loss.

    The BC term is the PreCi / PPF (arxiv 2504.09833) regularizer:
      L_bc = ||mu_new(s) - mu_ref(s)||^2
    where mu_ref is the action mean of a FROZEN copy of the baseline policy.
    This prevents catastrophic forgetting when fine-tuning a working walker.
    We use a fixed weight (cfg.BC_LOSS_WEIGHT) rather than the paper's
    adaptive z-vel-based weight — simpler and sufficient for our use case.

    All inputs are [batch_size, ...] (flattened from [T*N, ...]).
    """
    def forward_single(p, obs, carry):
        return model.apply(p, obs, carry)

    action_means, values, _ = jax.vmap(
        lambda o, c: forward_single(params, o, c)
    )(obs_batch, carry_batch)

    # Gaussian log probability
    std = jnp.exp(log_std)
    log_probs = -0.5 * jnp.sum(
        jnp.square((action_batch - action_means) / std) + 2 * log_std + jnp.log(2 * jnp.pi),
        axis=-1,
    )

    # PPO clipped objective
    ratio = jnp.exp(log_probs - old_log_prob_batch)
    clipped_ratio = jnp.clip(ratio, 1 - cfg.CLIP_EPS, 1 + cfg.CLIP_EPS)
    policy_loss = -jnp.mean(jnp.minimum(ratio * advantage_batch, clipped_ratio * advantage_batch))

    # Value loss
    value_loss = jnp.mean(jnp.square(values - return_batch))

    # Entropy bonus
    entropy = 0.5 * jnp.sum(1 + 2 * log_std + jnp.log(2 * jnp.pi))

    # BC loss vs frozen reference — compute reference action means on same obs
    ref_action_means, _, _ = jax.vmap(
        lambda o, c: forward_single(ref_params, o, c)
    )(obs_batch, carry_batch)
    bc_loss = jnp.mean(jnp.sum(jnp.square(action_means - ref_action_means), axis=-1))

    total_loss = (
        policy_loss
        + cfg.VALUE_COEF * value_loss
        - cfg.ENTROPY_COEF * entropy
        + cfg.BC_LOSS_WEIGHT * bc_loss
    )

    return total_loss, {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "bc_loss": bc_loss,
    }


def gaussian_log_prob(action, mean, log_std):
    """Compute log probability of action under Gaussian."""
    std = jnp.exp(log_std)
    return -0.5 * jnp.sum(
        jnp.square((action - mean) / std) + 2 * log_std + jnp.log(2 * jnp.pi),
        axis=-1,
    )


def sample_action(rng, mean, log_std):
    """Sample action from Gaussian policy."""
    std = jnp.exp(log_std)
    noise = jax.random.normal(rng, shape=mean.shape)
    return mean + std * noise


def _make_actor_lr_schedule():
    """Constant actor LR through warmup + curriculum, then cosine decay.

    The schedule counts optimizer steps (NOT training iterations). Each training
    iteration runs NUM_EPOCHS * NUM_MINIBATCHES inner optimizer steps.
    """
    inner_steps_per_iter = cfg.NUM_EPOCHS * cfg.NUM_MINIBATCHES
    hold_iters = cfg.CRITIC_WARMUP_ITERS + cfg.CURRICULUM_RAMP_ITERS
    hold_steps = hold_iters * inner_steps_per_iter
    decay_steps = cfg.ACTOR_LR_DECAY_ITERS * inner_steps_per_iter

    schedule = optax.join_schedules(
        schedules=[
            optax.constant_schedule(cfg.ACTOR_LR),
            optax.cosine_decay_schedule(
                init_value=cfg.ACTOR_LR,
                decay_steps=decay_steps,
                alpha=cfg.ACTOR_LR_DECAY_ALPHA,
            ),
        ],
        boundaries=[hold_steps],
    )
    return schedule


def create_optimizer():
    """Create optax optimizer with separate actor/critic learning rates.

    Actor (lstm + actor_0 + actor_1): low LR to preserve pre-trained weights,
    with cosine decay once the curriculum ends to damp late-training noise.
    Critic (critic_0 + critic_1): constant higher LR since randomly initialized.
    """
    def label_fn(params_tree):
        """Label parameters as 'actor' or 'critic'."""
        import jax
        flat = jax.tree.map(lambda _: "actor", params_tree)
        # Override critic params
        if isinstance(flat, dict) and "params" in flat:
            p = flat["params"]
            if "critic_0" in p:
                p["critic_0"] = jax.tree.map(lambda _: "critic", p["critic_0"])
            if "critic_1" in p:
                p["critic_1"] = jax.tree.map(lambda _: "critic", p["critic_1"])
        return flat

    actor_schedule = _make_actor_lr_schedule()

    param_optimizer = optax.multi_transform(
        transforms={
            "actor": optax.chain(
                optax.clip_by_global_norm(cfg.MAX_GRAD_NORM),
                optax.adam(actor_schedule),
            ),
            "critic": optax.chain(
                optax.clip_by_global_norm(cfg.MAX_GRAD_NORM),
                optax.adam(cfg.CRITIC_LR),
            ),
        },
        param_labels=label_fn,
    )

    # log_std also decays with the actor
    log_std_optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.MAX_GRAD_NORM),
        optax.adam(actor_schedule),
    )

    return param_optimizer, log_std_optimizer
