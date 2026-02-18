# Copyright © 2026 Apple Inc.

from typing import Optional, Tuple

import mlx.core as mx


def _make_gla_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / H;
        auto h_idx = n % H;
        constexpr int n_per_t = D / 32;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // q, k, v layout: [B, H, T, D]
        auto base = (b_idx * H + h_idx) * T * D;
        auto q_ = q + base;
        auto k_ = k + base;
        auto v_ = v + base;
        y += base;

        // state layout: [B, H, D, D]
        auto i_state = state_in + n * D * D;
        auto o_state = state_out + n * D * D;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto dk = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[dk * D + dv_idx]);
        }

        auto decay = static_cast<float>(exp_g[h_idx]);
        auto sc = static_cast<float>(scale[0]);

        for (int t = 0; t < T; ++t) {
          auto v_val = static_cast<float>(v_[dv_idx]);

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto dk = n_per_t * dk_idx + i;
            state[i] = state[i] * decay + static_cast<float>(k_[dk]) * v_val;
            out += state[i] * static_cast<float>(q_[dk]);
          }
          out = simd_sum(out) * sc;
          if (thread_index_in_simdgroup == 0) {
            y[dv_idx] = static_cast<InT>(out);
          }
          // Increment data pointers to next time step
          q_ += D;
          k_ += D;
          v_ += D;
          y += D;
        }

        for (int i = 0; i < n_per_t; ++i) {
          auto dk = n_per_t * dk_idx + i;
          o_state[dk * D + dv_idx] = static_cast<InT>(state[i]);
        }
    """
    return mx.fast.metal_kernel(
        name="gla_recurrent",
        input_names=["q", "k", "v", "exp_g", "state_in", "scale", "T"],
        output_names=["y", "state_out"],
        source=source,
    )


_gla_kernel = _make_gla_kernel()


@mx.compile
def _gla_step(
    q_t: mx.array,
    k_t: mx.array,
    v_t: mx.array,
    decay: mx.array,
    state: mx.array,
) -> Tuple[mx.array, mx.array]:
    """Single recurrent GLA step (float32 fallback)."""
    state = state * decay + k_t[..., :, None] * v_t[..., None, :]
    y_t = (q_t[:, :, None, :] @ state).squeeze(-2)
    return y_t, state


def gla_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    exp_g: mx.array,
    scale: float,
    state: mx.array,
) -> Tuple[mx.array, mx.array]:
    """Ops-based GLA fallback (CPU or non-standard head_dim)."""
    q = q * scale
    decay = exp_g[None, :, None, None]

    outputs = []
    for t in range(q.shape[2]):
        y_t, state = _gla_step(q[:, :, t], k[:, :, t], v[:, :, t], decay, state)
        outputs.append(y_t)

    return mx.stack(outputs, axis=2), state


def gla_kernel(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    exp_g: mx.array,
    scale: mx.array,
    state: mx.array,
) -> Tuple[mx.array, mx.array]:
    """Metal kernel GLA — reads/writes input dtype, computes in float32 registers."""
    B, H, T, D = q.shape
    input_type = q.dtype

    return _gla_kernel(
        inputs=[q, k, v, exp_g, state, scale, T],
        template=[("InT", input_type), ("D", D), ("H", H)],
        grid=(32, D, B * H),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, H, T, D), state.shape],
        output_dtypes=[input_type, input_type],
    )


def gla_recurrent(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    exp_g: mx.array,
    scale: mx.array,
    h: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """GLA linear recurrence.

    Recurrence per (b, h):
        h_t = h_{t-1} * exp(g) + k_t^T @ v_t
        y_t = (q_t @ h_t) * scale

    Uses Metal kernel when available for float32 precision in registers
    while maintaining bf16 memory bandwidth for state.

    Args:
        q, k, v: (B, H, T, D)
        exp_g: (H,) — precomputed exp of decay slopes
        scale: scalar attention scale factor (as mx.array)
        h: optional state (B, H, D, D)

    Returns:
        y: (B, H, T, D)
        h: updated state (B, H, D, D)
    """
    B, H, _, D = q.shape

    if h is None:
        h = mx.zeros((B, H, D, D), dtype=q.dtype)

    if _gla_kernel is not None and mx.default_device() == mx.gpu and D % 32 == 0:
        return gla_kernel(q, k, v, exp_g, scale, h)
    return gla_ops(q, k, v, exp_g, scale.item(), h)
