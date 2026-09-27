"""Convert fine-tuned JAX weights back to TorchScript .pt for MuJoCo deployment.

Reverses the transformations in convert_weights.py. Loads the original motion.pt
as a template, swaps its parameters with the JAX-trained weights, and saves a new
TorchScript file that deploy_mujoco.py can load directly.
"""
import os
import argparse
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
TEMPLATE_PATH = os.path.join(ROOT_DIR, "deploy", "pre_train", "g1", "motion.pt")
DEFAULT_WEIGHTS = os.path.join(SCRIPT_DIR, "checkpoints", "final_weights.npz")
DEFAULT_OUTPUT = os.path.join(ROOT_DIR, "deploy", "pre_train", "g1", "motion_terrain.pt")

HIDDEN_SIZE = 64
INPUT_SIZE = 47


def jax_lstm_to_pytorch(kernel_i, kernel_h, bias):
    """Reverse convert_weights.py's LSTM transformation.

    JAX format:
        kernel_i: [input, 4*hidden] with gates concatenated as [i, f, g, o]
        kernel_h: [hidden, 4*hidden] same gate order
        bias:     [4*hidden]        combined ih+hh biases

    PyTorch format:
        weight_ih_l0: [4*hidden, input]
        weight_hh_l0: [4*hidden, hidden]
        bias_ih_l0:   [4*hidden]  (we put the full combined bias here)
        bias_hh_l0:   [4*hidden]  (zero, so ih + hh = combined)
    """
    gi = np.split(kernel_i, 4, axis=1)
    gh = np.split(kernel_h, 4, axis=1)

    w_ih = np.concatenate([g.T for g in gi], axis=0)
    w_hh = np.concatenate([g.T for g in gh], axis=0)

    b_ih = bias.copy()
    b_hh = np.zeros_like(bias)

    return w_ih, w_hh, b_ih, b_hh


def mutate_state_dict(model, jax_weights):
    """Overwrite the TorchScript model's parameters with the JAX weights."""
    w_ih, w_hh, b_ih, b_hh = jax_lstm_to_pytorch(
        jax_weights["lstm_kernel_i"],
        jax_weights["lstm_kernel_h"],
        jax_weights["lstm_bias"],
    )

    actor0_w = jax_weights["actor_w0"].T
    actor0_b = jax_weights["actor_b0"]
    actor1_w = jax_weights["actor_w1"].T
    actor1_b = jax_weights["actor_b1"]

    replacements = {
        "memory.weight_ih_l0": w_ih,
        "memory.weight_hh_l0": w_hh,
        "memory.bias_ih_l0": b_ih,
        "memory.bias_hh_l0": b_hh,
        "actor.0.weight": actor0_w,
        "actor.0.bias": actor0_b,
        "actor.2.weight": actor1_w,
        "actor.2.bias": actor1_b,
    }

    found = set()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in replacements:
                new_val = torch.from_numpy(replacements[name].astype(np.float32))
                assert param.shape == new_val.shape, (
                    f"Shape mismatch for {name}: model {tuple(param.shape)}, "
                    f"weights {tuple(new_val.shape)}"
                )
                param.copy_(new_val)
                found.add(name)

    missing = set(replacements) - found
    if missing:
        raise RuntimeError(f"These parameters were not found in the model: {missing}")

    for _, buf in model.named_buffers():
        buf.zero_()


def verify(model, jax_weights):
    """Compare PyTorch output against a numpy reference using the JAX weights."""
    test_obs = np.random.default_rng(0).standard_normal((1, INPUT_SIZE)).astype(np.float32)

    for _, buf in model.named_buffers():
        buf.zero_()
    with torch.no_grad():
        torch_out = model(torch.from_numpy(test_obs)).numpy()

    h = np.zeros((1, HIDDEN_SIZE), dtype=np.float32)
    c = np.zeros((1, HIDDEN_SIZE), dtype=np.float32)
    gates = test_obs @ jax_weights["lstm_kernel_i"] + h @ jax_weights["lstm_kernel_h"] + jax_weights["lstm_bias"]
    i_gate, f_gate, g_gate, o_gate = np.split(gates, 4, axis=1)
    i_gate = 1.0 / (1.0 + np.exp(-i_gate))
    f_gate = 1.0 / (1.0 + np.exp(-f_gate))
    g_gate = np.tanh(g_gate)
    o_gate = 1.0 / (1.0 + np.exp(-o_gate))
    c_new = f_gate * c + i_gate * g_gate
    h_new = o_gate * np.tanh(c_new)
    x = h_new @ jax_weights["actor_w0"] + jax_weights["actor_b0"]
    x = np.where(x > 0, x, np.exp(x) - 1)
    ref_out = x @ jax_weights["actor_w1"] + jax_weights["actor_b1"]

    max_diff = float(np.max(np.abs(torch_out - ref_out)))
    print(f"  PyTorch output: {torch_out[0, :4]}")
    print(f"  Numpy reference: {ref_out[0, :4]}")
    print(f"  Max abs diff:   {max_diff:.2e}")
    if max_diff < 1e-5:
        print("  PASS - fine-tuned weights successfully loaded into TorchScript.")
    else:
        print("  WARNING - outputs differ beyond expected numerical noise.")


def main():
    parser = argparse.ArgumentParser(description="Export JAX weights to TorchScript")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS,
                        help="Path to JAX final_weights.npz")
    parser.add_argument("--template", type=str, default=TEMPLATE_PATH,
                        help="Path to the original TorchScript model used as template")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help="Where to save the new TorchScript .pt")
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"JAX weights not found at {args.weights}")
    if not os.path.exists(args.template):
        raise FileNotFoundError(f"Template model not found at {args.template}")

    print(f"Loading template: {args.template}")
    model = torch.jit.load(args.template, map_location="cpu")

    print(f"Loading JAX weights: {args.weights}")
    jax_weights = dict(np.load(args.weights))
    required = {"lstm_kernel_i", "lstm_kernel_h", "lstm_bias",
                "actor_w0", "actor_b0", "actor_w1", "actor_b1"}
    missing = required - set(jax_weights)
    if missing:
        raise RuntimeError(f"JAX weights file is missing keys: {missing}")

    print("Replacing model parameters with fine-tuned weights...")
    mutate_state_dict(model, jax_weights)

    print("Verifying round-trip...")
    verify(model, jax_weights)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.jit.save(model, args.output)
    print(f"\nSaved TorchScript model to: {args.output}")


if __name__ == "__main__":
    main()
