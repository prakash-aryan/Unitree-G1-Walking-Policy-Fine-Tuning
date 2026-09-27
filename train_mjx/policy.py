"""Flax LSTM policy for G1 robot."""
import os
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from typing import Tuple

from train_mjx import config as cfg

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(SCRIPT_DIR, "pretrained_weights.npz")


class LSTMCell(nn.Module):
    """Manual LSTM cell matching PyTorch's gate order (i, f, g, o)."""
    hidden_size: int = 64

    @nn.compact
    def __call__(self, x, carry):
        h, c = carry  # [hidden], [hidden]
        input_size = x.shape[-1]
        hidden = self.hidden_size

        # Kernels: [input, 4*hidden] and [hidden, 4*hidden]
        kernel_i = self.param("kernel_i", nn.zeros_init(), (input_size, 4 * hidden))
        kernel_h = self.param("kernel_h", nn.zeros_init(), (hidden, 4 * hidden))
        bias = self.param("bias", nn.zeros_init(), (4 * hidden,))

        gates = x @ kernel_i + h @ kernel_h + bias
        i, f, g, o = jnp.split(gates, 4, axis=-1)

        i = jax.nn.sigmoid(i)
        f = jax.nn.sigmoid(f)
        g = jnp.tanh(g)
        o = jax.nn.sigmoid(o)

        c_new = f * c + i * g
        h_new = o * jnp.tanh(c_new)

        return h_new, (h_new, c_new)


class ActorCritic(nn.Module):
    """Actor-critic with LSTM, matching the pre-trained architecture."""
    lstm_hidden: int = cfg.LSTM_HIDDEN_SIZE
    actor_hidden: int = cfg.ACTOR_HIDDEN_SIZE
    num_actions: int = cfg.NUM_ACTIONS

    @nn.compact
    def __call__(self, obs, carry):
        """
        Args:
            obs: [obs_dim] observation
            carry: (h, c) LSTM state, each [hidden_size]
        Returns:
            action_mean: [num_actions]
            value: scalar
            new_carry: (h, c)
        """
        # LSTM
        lstm = LSTMCell(hidden_size=self.lstm_hidden, name="lstm")
        h_out, new_carry = lstm(obs, carry)

        # Actor: h -> 32 (ELU) -> 12
        actor_h = nn.Dense(self.actor_hidden, name="actor_0")(h_out)
        actor_h = nn.elu(actor_h)
        action_mean = nn.Dense(self.num_actions, name="actor_1")(actor_h)

        # Critic: h -> 32 (ELU) -> 1 (separate head, not pre-trained)
        critic_h = nn.Dense(self.actor_hidden, name="critic_0")(h_out)
        critic_h = nn.elu(critic_h)
        value = nn.Dense(1, name="critic_1")(critic_h)

        return action_mean, jnp.squeeze(value, -1), new_carry


def init_carry(batch_size=1):
    """Create initial LSTM carry (h, c) for a batch."""
    h = jnp.zeros((batch_size, cfg.LSTM_HIDDEN_SIZE))
    c = jnp.zeros((batch_size, cfg.LSTM_HIDDEN_SIZE))
    return (h, c)


def load_pretrained_params(params):
    """Load pre-trained weights into the parameter tree."""
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Pre-trained weights not found at {WEIGHTS_PATH}. "
            "Run convert_weights.py first."
        )

    w = np.load(WEIGHTS_PATH)

    # Build new param dict with pre-trained weights
    new_params = jax.tree.map(lambda x: x, params)  # deep copy

    new_params["params"]["lstm"]["kernel_i"] = jnp.array(w["lstm_kernel_i"])
    new_params["params"]["lstm"]["kernel_h"] = jnp.array(w["lstm_kernel_h"])
    new_params["params"]["lstm"]["bias"] = jnp.array(w["lstm_bias"])

    new_params["params"]["actor_0"]["kernel"] = jnp.array(w["actor_w0"])
    new_params["params"]["actor_0"]["bias"] = jnp.array(w["actor_b0"])
    new_params["params"]["actor_1"]["kernel"] = jnp.array(w["actor_w1"])
    new_params["params"]["actor_1"]["bias"] = jnp.array(w["actor_b1"])

    # Critic is randomly initialized (not pre-trained)
    print("Loaded pre-trained weights (actor + LSTM). Critic randomly initialized.")
    return new_params


def init_log_std():
    """Initial action log standard deviation."""
    return jnp.full(cfg.NUM_ACTIONS, -0.5)  # std ~ 0.6, conservative for fine-tuning
