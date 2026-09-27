"""Convert pre-trained PyTorch TorchScript weights to JAX-compatible numpy format."""
import os
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(ROOT_DIR, "deploy", "pre_train", "g1", "motion.pt")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "pretrained_weights.npz")


def convert():
    model = torch.jit.load(MODEL_PATH, map_location="cpu")

    params = {name: p.detach().numpy() for name, p in model.named_parameters()}
    buffers = {name: b.detach().numpy() for name, b in model.named_buffers()}

    print("PyTorch parameters:")
    for name, p in params.items():
        print(f"  {name}: {p.shape}")
    for name, b in buffers.items():
        print(f"  [buf] {name}: {b.shape}")

    # PyTorch LSTM packs 4 gates as [i, f, g, o] each of size hidden_size
    # into weight_ih: [4*hidden, input], weight_hh: [4*hidden, hidden]
    # Flax OptimizedLSTMCell expects:
    #   i (input kernel): [input_size, hidden_size] per gate
    #   h (hidden kernel): [hidden_size, hidden_size] per gate
    #   bias: [hidden_size] per gate
    hidden_size = 64
    input_size = 47

    # Extract and split LSTM weights
    w_ih = params["memory.weight_ih_l0"]  # [256, 47]
    w_hh = params["memory.weight_hh_l0"]  # [256, 64]
    b_ih = params["memory.bias_ih_l0"]    # [256]
    b_hh = params["memory.bias_hh_l0"]    # [256]

    # Split into 4 gates: i, f, g, o (PyTorch order)
    w_ih_i, w_ih_f, w_ih_g, w_ih_o = np.split(w_ih, 4, axis=0)
    w_hh_i, w_hh_f, w_hh_g, w_hh_o = np.split(w_hh, 4, axis=0)
    b_ih_i, b_ih_f, b_ih_g, b_ih_o = np.split(b_ih, 4)
    b_hh_i, b_hh_f, b_hh_g, b_hh_o = np.split(b_hh, 4)

    # Combine biases (PyTorch has separate ih and hh biases)
    b_i = b_ih_i + b_hh_i
    b_f = b_ih_f + b_hh_f
    b_g = b_ih_g + b_hh_g
    b_o = b_ih_o + b_hh_o

    # Transpose weights: PyTorch [out, in] -> JAX [in, out]
    # Stack gates for Flax: [input, 4*hidden] for input kernel
    lstm_kernel_i = np.concatenate([w_ih_i.T, w_ih_f.T, w_ih_g.T, w_ih_o.T], axis=1)  # [47, 256]
    lstm_kernel_h = np.concatenate([w_hh_i.T, w_hh_f.T, w_hh_g.T, w_hh_o.T], axis=1)  # [64, 256]
    lstm_bias = np.concatenate([b_i, b_f, b_g, b_o])  # [256]

    # Actor MLP weights (transpose for JAX: [in, out])
    actor_w0 = params["actor.0.weight"].T  # [64, 32]
    actor_b0 = params["actor.0.bias"]       # [32]
    actor_w1 = params["actor.2.weight"].T  # [32, 12]
    actor_b1 = params["actor.2.bias"]       # [12]

    weights = {
        "lstm_kernel_i": lstm_kernel_i.astype(np.float32),
        "lstm_kernel_h": lstm_kernel_h.astype(np.float32),
        "lstm_bias": lstm_bias.astype(np.float32),
        "actor_w0": actor_w0.astype(np.float32),
        "actor_b0": actor_b0.astype(np.float32),
        "actor_w1": actor_w1.astype(np.float32),
        "actor_b1": actor_b1.astype(np.float32),
    }

    np.savez(OUTPUT_PATH, **weights)
    print(f"\nSaved JAX weights to: {OUTPUT_PATH}")

    # Sanity check: verify forward pass
    print("\nSanity check...")
    test_obs = np.random.randn(1, input_size).astype(np.float32)
    test_obs_torch = torch.from_numpy(test_obs)

    # Reset LSTM state
    for name, buf in model.named_buffers():
        buf.zero_()

    with torch.no_grad():
        torch_out = model(test_obs_torch).numpy()

    # JAX forward pass
    h = np.zeros((1, hidden_size), dtype=np.float32)
    c = np.zeros((1, hidden_size), dtype=np.float32)

    # LSTM step (manual)
    x = test_obs  # [1, 47]
    gates_i = x @ lstm_kernel_i  # [1, 256]
    gates_h = h @ lstm_kernel_h  # [1, 256]
    gates = gates_i + gates_h + lstm_bias  # [1, 256]

    i_gate, f_gate, g_gate, o_gate = np.split(gates, 4, axis=1)
    i_gate = 1.0 / (1.0 + np.exp(-i_gate))  # sigmoid
    f_gate = 1.0 / (1.0 + np.exp(-f_gate))
    g_gate = np.tanh(g_gate)
    o_gate = 1.0 / (1.0 + np.exp(-o_gate))

    c_new = f_gate * c + i_gate * g_gate
    h_new = o_gate * np.tanh(c_new)

    # Actor MLP
    x = h_new @ actor_w0 + actor_b0  # [1, 32]
    x = np.where(x > 0, x, np.exp(x) - 1)  # ELU
    jax_out = x @ actor_w1 + actor_b1  # [1, 12]

    max_diff = np.max(np.abs(torch_out - jax_out))
    print(f"  PyTorch output: {torch_out[0, :4]}...")
    print(f"  JAX output:     {jax_out[0, :4]}...")
    print(f"  Max abs diff:   {max_diff:.8f}")
    if max_diff < 1e-5:
        print("  PASS - outputs match!")
    else:
        print("  WARNING - outputs differ (may be due to LSTM state handling)")


if __name__ == "__main__":
    convert()
