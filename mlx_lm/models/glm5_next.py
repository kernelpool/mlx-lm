# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients

from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, CacheList, KVCache
from .deepseek_v32 import MoEGate
from .gated_delta import gated_delta_update
from .kimi_linear import ShortConv1d
from .mla import MultiLinear
from .switch_layers import SwitchGLU


@dataclass
class TextConfig(BaseModelArgs):
    model_type: str = "glm5_next_text"
    vocab_size: int = 154880
    hidden_size: int = 4096
    intermediate_size: int = 12288
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 45
    num_attention_heads: int = 64
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 256
    qk_rope_head_dim: int = 0
    v_head_dim: int = 256
    n_routed_experts: int = 288
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 2.5
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"
    first_k_dense_replace: int = 3
    swiglu_limit: float = 10.0
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 1048576
    attention_bias: bool = False
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    index_kpool: int = 4
    index_kpool_always_select_tail: bool = True
    index_kpool_compress: bool = True
    indexer_types: Optional[List[str]] = None
    layer_types: Optional[List[str]] = None
    mlp_layer_types: Optional[List[str]] = None
    linear_attn_config: Optional[Dict[str, Any]] = None
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    def __post_init__(self):
        if self.qk_rope_head_dim != 0:
            raise ValueError("glm5_next attention layers must be NoPE.")
        if not self.index_kpool_compress or self.index_kpool < 2:
            raise ValueError("glm5_next requires k-pool indexer compression.")
        if self.layer_types is None:
            self.layer_types = [
                "linear_attention" if i % 4 != 3 else "deepseek_sparse_attention"
                for i in range(self.num_hidden_layers)
            ]
        if self.mlp_layer_types is None:
            self.mlp_layer_types = [
                "dense" if i < self.first_k_dense_replace else "sparse"
                for i in range(self.num_hidden_layers)
            ]


@dataclass
class ModelArgs(BaseModelArgs):
    text_config: Union[TextConfig, dict]
    model_type: str = "glm5_next"

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)


@mx.compile
def _limited_swiglu(gate: mx.array, up: mx.array, limit: float) -> mx.array:
    if limit and limit > 0:
        gate = mx.minimum(gate, limit)
        up = mx.clip(up, -limit, limit)
    return nn.silu(gate) * up


class LimitedSwiGLU(nn.Module):
    def __init__(self, limit: float):
        super().__init__()
        self.limit = limit

    def __call__(self, x, gate):
        return _limited_swiglu(gate, x, self.limit)


@mx.compile
def _hc_split_sinkhorn_ops(
    mixes: mx.array,
    scale: mx.array,
    base: mx.array,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> Tuple[mx.array, mx.array, mx.array]:
    mixes = mixes.astype(mx.float32)
    scale = scale.astype(mx.float32)
    base = base.astype(mx.float32)
    pre_scale, post_scale, comb_scale = scale[0], scale[1], scale[2]

    pre = mx.sigmoid(mixes[..., :hc_mult] * pre_scale + base[:hc_mult]) + eps
    post = 2 * mx.sigmoid(
        mixes[..., hc_mult : 2 * hc_mult] * post_scale + base[hc_mult : 2 * hc_mult]
    )
    comb = mixes[..., 2 * hc_mult :].reshape(
        *mixes.shape[:-1], hc_mult, hc_mult
    ) * comb_scale + base[2 * hc_mult :].reshape(hc_mult, hc_mult)
    comb = mx.softmax(comb, axis=-1, precise=True) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(max(sinkhorn_iters - 1, 0)):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    return pre, post, comb


def _make_hc_split_sinkhorn_kernel():
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None

    source = """
        uint idx = thread_position_in_grid.x;
        constexpr int MIX = (2 + HC) * HC;
        float epsv = static_cast<float>(eps[0]);

        auto mix = mixes + idx * MIX;
        auto pre_out = pre + idx * HC;
        auto post_out = post + idx * HC;
        auto comb_out = comb + idx * HC * HC;

        float pre_scale = static_cast<float>(scale[0]);
        float post_scale = static_cast<float>(scale[1]);
        float comb_scale = static_cast<float>(scale[2]);

        for (int i = 0; i < HC; ++i) {
            float z = static_cast<float>(mix[i]) * pre_scale
                + static_cast<float>(base[i]);
            pre_out[i] = 1.0f / (1.0f + metal::fast::exp(-z)) + epsv;
        }
        for (int i = 0; i < HC; ++i) {
            int off = HC + i;
            float z = static_cast<float>(mix[off]) * post_scale
                + static_cast<float>(base[off]);
            post_out[i] = 2.0f / (1.0f + metal::fast::exp(-z));
        }

        float c[HC * HC];
        for (int i = 0; i < HC; ++i) {
            float row_max = -INFINITY;
            for (int j = 0; j < HC; ++j) {
                int cidx = i * HC + j;
                int off = 2 * HC + cidx;
                float v = static_cast<float>(mix[off]) * comb_scale
                    + static_cast<float>(base[off]);
                c[cidx] = v;
                row_max = metal::max(row_max, v);
            }
            float row_sum = 0.0f;
            for (int j = 0; j < HC; ++j) {
                int cidx = i * HC + j;
                float v = metal::fast::exp(c[cidx] - row_max);
                c[cidx] = v;
                row_sum += v;
            }
            float inv_sum = 1.0f / row_sum;
            for (int j = 0; j < HC; ++j) {
                int cidx = i * HC + j;
                c[cidx] = c[cidx] * inv_sum + epsv;
            }
        }

        for (int j = 0; j < HC; ++j) {
            float col_sum = 0.0f;
            for (int i = 0; i < HC; ++i) {
                col_sum += c[i * HC + j];
            }
            float inv_denom = 1.0f / (col_sum + epsv);
            for (int i = 0; i < HC; ++i) {
                c[i * HC + j] *= inv_denom;
            }
        }

        for (int iter = 1; iter < ITERS; ++iter) {
            for (int i = 0; i < HC; ++i) {
                float row_sum = 0.0f;
                for (int j = 0; j < HC; ++j) {
                    row_sum += c[i * HC + j];
                }
                float inv_denom = 1.0f / (row_sum + epsv);
                for (int j = 0; j < HC; ++j) {
                    c[i * HC + j] *= inv_denom;
                }
            }
            for (int j = 0; j < HC; ++j) {
                float col_sum = 0.0f;
                for (int i = 0; i < HC; ++i) {
                    col_sum += c[i * HC + j];
                }
                float inv_denom = 1.0f / (col_sum + epsv);
                for (int i = 0; i < HC; ++i) {
                    c[i * HC + j] *= inv_denom;
                }
            }
        }

        for (int i = 0; i < HC * HC; ++i) {
            comb_out[i] = c[i];
        }
    """

    return mx.fast.metal_kernel(
        name="glm5_next_hc_split_sinkhorn",
        input_names=["mixes", "scale", "base", "eps"],
        output_names=["pre", "post", "comb"],
        source=source,
    )


_hc_split_sinkhorn_kernel = _make_hc_split_sinkhorn_kernel()


def hc_split_sinkhorn(
    mixes: mx.array,
    scale: mx.array,
    base: mx.array,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> Tuple[mx.array, mx.array, mx.array]:
    if _hc_split_sinkhorn_kernel is None:
        return _hc_split_sinkhorn_ops(mixes, scale, base, hc_mult, sinkhorn_iters, eps)

    if not isinstance(eps, mx.array):
        eps = mx.array([eps], dtype=mx.float32)
    return _hc_split_sinkhorn_kernel(
        inputs=[mixes, scale, base, eps],
        template=[("HC", hc_mult), ("ITERS", sinkhorn_iters)],
        grid=(mixes.size // ((2 + hc_mult) * hc_mult), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[
            (*mixes.shape[:-1], hc_mult),
            (*mixes.shape[:-1], hc_mult),
            (*mixes.shape[:-1], hc_mult, hc_mult),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )


@mx.compile
def _hc_collapse_op(pre: mx.array, x: mx.array) -> mx.array:
    return (pre[..., None] * x.astype(mx.float32)).sum(axis=2).astype(x.dtype)


@mx.compile
def _hc_expand_op(
    post: mx.array,
    block_out: mx.array,
    comb: mx.array,
    residual: mx.array,
) -> mx.array:
    # The Sinkhorn plan reduces over its first hc axis: comb.T @ residual.
    y = post[..., None] * block_out[:, :, None, :].astype(mx.float32)
    y = y + mx.matmul(comb.swapaxes(-1, -2), residual.astype(mx.float32))
    return y.astype(block_out.dtype)


@mx.compile
def _rms_rsqrt(flat: mx.array, eps: float) -> mx.array:
    return mx.rsqrt((flat * flat).mean(axis=-1, keepdims=True) + eps)


class HyperConnection(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self._hc_eps = (mx.array([config.hc_eps], dtype=mx.float32),)
        self.norm_eps = config.rms_norm_eps
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = mx.zeros((mix, self.hc_mult * config.hidden_size), dtype=mx.float32)
        self.base = mx.zeros((mix,), dtype=mx.float32)
        self.scale = mx.ones((3,), dtype=mx.float32)

    def compute_weights(self, x: mx.array):
        B, L, H, D = x.shape
        flat = x.reshape(B, L, H * D).astype(mx.float32)
        rsqrt = _rms_rsqrt(flat, self.norm_eps)
        mixes = (flat @ self.fn.T) * rsqrt
        split_sinkhorn = _hc_split_sinkhorn_ops if self.training else hc_split_sinkhorn
        return split_sinkhorn(
            mixes,
            self.scale,
            self.base,
            self.hc_mult,
            self.sinkhorn_iters,
            self.hc_eps if self.training else self._hc_eps[0],
        )

    def collapse(self, x: mx.array):
        pre, post, comb = self.compute_weights(x)
        return _hc_collapse_op(pre, x), post, comb

    def expand(
        self,
        block_out: mx.array,
        residual: mx.array,
        post: mx.array,
        comb: mx.array,
    ):
        return _hc_expand_op(post, block_out, comb, residual)


class MLP(nn.Module):
    def __init__(self, args: TextConfig, intermediate_size: Optional[int] = None):
        super().__init__()
        dim = args.hidden_size
        hidden = intermediate_size or args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        self.swiglu_limit = args.swiglu_limit

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(
            _limited_swiglu(self.gate_proj(x), self.up_proj(x), self.swiglu_limit)
        )


class MoE(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.gate = MoEGate(args)
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.n_routed_experts,
            activation=LimitedSwiGLU(args.swiglu_limit),
        )
        self.shared_experts = MLP(
            args, intermediate_size=args.moe_intermediate_size * args.n_shared_experts
        )
        self.sharding_group = None

    def __call__(self, x: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        y = y + self.shared_experts(x)

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)
        return y


class KDAAttention(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        cfg = args.linear_attn_config
        self.num_heads = cfg["num_heads"]
        self.head_dim = cfg["head_dim"]
        self.conv_kernel = cfg.get("short_conv_kernel_size", 4)
        self.lower_bound = cfg.get("gate_lower_bound", -5.0)
        self.projection_dim = self.num_heads * self.head_dim
        self.qkv_dim = 3 * self.projection_dim
        self.scale = float(self.head_dim) ** -0.5

        hidden = args.hidden_size
        self.qkv_proj = nn.Linear(hidden, self.qkv_dim, bias=False)
        self.qkv_conv = ShortConv1d(self.qkv_dim, self.conv_kernel)

        self.f_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, self.projection_dim, bias=False)
        self.b_proj = nn.Linear(hidden, self.num_heads, bias=False)

        self.g_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, self.projection_dim, bias=False)

        self.A_log = mx.zeros((self.num_heads,))
        self.dt_bias = mx.zeros((self.projection_dim,))

        self.o_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.o_proj = nn.Linear(self.projection_dim, hidden, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, T, _ = x.shape

        if cache is not None:
            conv_state, ssm_state = cache[0], cache[1]
            lengths = cache.lengths
        else:
            conv_state = ssm_state = lengths = None
        if conv_state is None:
            conv_state = mx.zeros((B, self.conv_kernel - 1, self.qkv_dim), x.dtype)

        qkv, conv_state = self.qkv_conv(self.qkv_proj(x), conv_state, mask, lengths)
        if cache is not None:
            cache[0] = conv_state

        q, k, v = (
            z.reshape(B, T, self.num_heads, self.head_dim)
            for z in mx.split(qkv, 3, axis=-1)
        )

        # Match the reference l2norm: x / sqrt(sum(x^2) + 1e-6). rms_norm adds
        # eps to mean(x^2), so the equivalent eps is 1e-6 / head_dim.
        eps = 1e-6 / self.head_dim
        q = (self.scale**2) * mx.fast.rms_norm(q, None, eps)
        k = self.scale * mx.fast.rms_norm(k, None, eps)

        a = self.f_b_proj(self.f_a_proj(x)).reshape(B, T, self.num_heads, self.head_dim)
        b = self.b_proj(x)

        out, ssm_state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log.reshape(self.num_heads, 1),
            self.dt_bias.reshape(self.num_heads, self.head_dim),
            state=ssm_state,
            mask=mask,
            use_kernel=not self.training,
            lower_bound=self.lower_bound,
        )

        if cache is not None:
            cache[1] = ssm_state
            cache.advance(T)

        gate = self.g_b_proj(self.g_a_proj(x)).reshape(
            B, T, self.num_heads, self.head_dim
        )
        out = (self.o_norm(out) * mx.sigmoid(gate)).reshape(B, T, -1)
        return self.o_proj(out)


class KPoolIndexer(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.index_topk = args.index_topk
        self.kpool = args.index_kpool
        self.softmax_scale = self.head_dim**-0.5

        self.wq_b = nn.Linear(
            args.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(args.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.weights_proj = nn.Linear(args.hidden_size, self.n_heads, bias=False)
        self.index_kpool_compress_ape = mx.zeros((self.kpool, self.head_dim))
        self.index_kpool_compress_gate = mx.zeros((self.head_dim, args.hidden_size))

    def _compress(self, packed: mx.array) -> mx.array:
        B = packed.shape[0]
        keys, gates = mx.split(packed, 2, axis=-1)
        keys = keys.reshape(B, -1, self.kpool, self.head_dim)
        gates = gates.reshape(B, -1, self.kpool, self.head_dim)
        w = mx.softmax(
            gates.astype(mx.float32) + self.index_kpool_compress_ape.astype(mx.float32),
            axis=2,
            precise=True,
        )
        return (w.astype(keys.dtype) * keys).sum(axis=2)

    def __call__(
        self,
        x: mx.array,
        qr: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any] = None,
    ) -> Optional[mx.array]:
        B, L, _ = x.shape
        k = self.k_norm(self.wk(x))
        gate = x @ self.index_kpool_compress_gate.T
        packed = mx.concatenate([k, gate], axis=-1)[:, None]

        if cache is not None:
            packed, _ = cache.update_and_fetch(
                packed, mx.zeros((B, 1, L, 0), packed.dtype)
            )
        packed = packed[:, 0]
        T = packed.shape[1]

        P = T // self.kpool
        select_k = self.index_topk // self.kpool
        if P <= select_k:
            return None

        pool_keys = self._compress(packed[:, : P * self.kpool])

        q = self.wq_b(qr)
        q = q.reshape(B, L, self.n_heads, self.head_dim).swapaxes(1, 2)
        scores = q @ pool_keys[:, None].swapaxes(-1, -2)
        scores = mx.maximum(scores, 0)
        weights = self.weights_proj(x) * (self.n_heads**-0.5 * self.softmax_scale)
        if scores.size <= 2**31:
            scores = weights[..., None, :] @ scores.transpose(0, 2, 1, 3)
            scores = scores.transpose(0, 2, 1, 3)
        else:
            scores = scores * weights.swapaxes(-1, -2)[..., None]
            summed = scores[:, 0:1]
            for h in range(1, scores.shape[1]):
                summed = summed + scores[:, h : h + 1]
            scores = summed

        if mask is not None:
            # A pool is a valid candidate only if the query sees all its tokens.
            pool_mask = mask[..., : P * self.kpool]
            pool_mask = pool_mask.reshape(*pool_mask.shape[:-1], P, self.kpool)
            scores = mx.where(pool_mask.all(axis=-1), scores, -float("inf"))

        top_pools = mx.argpartition(scores, kth=-select_k, axis=-1)[..., -select_k:]
        tokens = top_pools[..., None] * self.kpool + mx.arange(
            self.kpool, dtype=top_pools.dtype
        )
        return mx.flatten(tokens, -2, -1)


class DSAAttention(nn.Module):
    def __init__(self, args: TextConfig, layer_idx: int):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.v_head_dim = args.v_head_dim
        self.kpool = args.index_kpool
        self.always_select_tail = args.index_kpool_always_select_tail
        self.scale = self.qk_nope_head_dim**-0.5
        self.skip_topk = (
            args.indexer_types is not None and args.indexer_types[layer_idx] == "shared"
        )

        hidden = args.hidden_size
        self.q_a_proj = nn.Linear(hidden, self.q_lora_rank, bias=args.attention_bias)
        self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=args.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            self.q_lora_rank, self.num_heads * self.qk_nope_head_dim, bias=False
        )

        self.kv_a_proj_with_mqa = nn.Linear(
            hidden, self.kv_lora_rank, bias=args.attention_bias
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=args.rms_norm_eps)
        self.embed_q = MultiLinear(
            self.qk_nope_head_dim, self.kv_lora_rank, self.num_heads
        )
        self.unembed_out = MultiLinear(
            self.kv_lora_rank, self.v_head_dim, self.num_heads
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim, hidden, bias=args.attention_bias
        )

        if not self.skip_topk:
            self.indexer = KPoolIndexer(args)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        topk_indices: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        B, L, _ = x.shape

        qr = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(qr)
        q = q.reshape(B, L, self.num_heads, self.qk_nope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(self.kv_a_proj_with_mqa(x))[:, None]

        if cache is not None:
            kv_latent, _ = cache[0].update_and_fetch(
                kv_latent, mx.zeros((B, 1, L, 0), kv_latent.dtype)
            )
            # Read after the update: BatchKVCache mutates its offset array in
            # place, so a pre-update read would alias the updated value.
            offset = cache[0].offset
        else:
            cache = [None] * 2
            offset = L

        if not self.skip_topk:
            topk_indices = self.indexer(x, qr, mask, cache=cache[1])

        T = kv_latent.shape[2]
        if topk_indices is not None:
            if L == 1 and not isinstance(offset, mx.array):
                idx = topk_indices
                if self.always_select_tail and T % self.kpool:
                    tail = mx.arange((T // self.kpool) * self.kpool, T)
                    tail = tail.astype(idx.dtype).reshape(1, 1, 1, -1)
                    idx = mx.concatenate(
                        [idx, mx.broadcast_to(tail, (B, 1, 1, tail.shape[-1]))],
                        axis=-1,
                    )
                gather = idx[:, :, 0, :, None]
                kv_latent = mx.take_along_axis(
                    kv_latent,
                    mx.broadcast_to(gather, gather.shape[:-1] + (kv_latent.shape[-1],)),
                    axis=2,
                )
                if mask is not None:
                    mask = mx.take_along_axis(mask, idx, axis=-1)
            else:
                sparse_mask = mx.zeros((B, 1, L, T), dtype=mx.bool_)
                sparse_mask = mx.put_along_axis(
                    sparse_mask, topk_indices, mx.array(True), axis=-1
                )
                if self.always_select_tail:
                    if isinstance(offset, mx.array):
                        q_pos = (offset[:, None] - L + mx.arange(L))[:, None, :, None]
                    else:
                        q_pos = (offset - L + mx.arange(L)).reshape(1, 1, L, 1)
                    kv_pos = mx.arange(T).reshape(1, 1, 1, T)
                    tail_start = ((q_pos + 1) // self.kpool) * self.kpool
                    sparse_mask = sparse_mask | (
                        (kv_pos >= tail_start) & (kv_pos <= q_pos)
                    )
                if mask is not None:
                    sparse_mask = sparse_mask & mask
                mask = sparse_mask

        # Ensure the indexer cache is evaluated even if the topk_indices are
        # unused to keep the graph from getting too large
        if not self.skip_topk and cache[0] is not None:
            cache[0].keys = mx.depends(cache[0].keys, (cache[1].keys, cache[1].values))

        if L == 1:
            q = self.embed_q(q)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        out = scaled_dot_product_attention(
            q, k, v, cache=cache[0], scale=self.scale, mask=mask
        )
        if L == 1:
            out = self.unembed_out(out)

        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out), topk_indices


class Glm5NextDecoderLayer(nn.Module):
    def __init__(self, args: TextConfig, layer_idx: int):
        super().__init__()
        self.is_linear = args.layer_types[layer_idx] == "linear_attention"
        if self.is_linear:
            self.self_attn = KDAAttention(args)
        else:
            self.self_attn = DSAAttention(args, layer_idx)

        if args.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = MoE(args)
        else:
            self.mlp = MLP(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.attn_hc = HyperConnection(args)
        self.ffn_hc = HyperConnection(args)

    def __call__(
        self,
        h: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        topk_indices: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        x, post, comb = self.attn_hc.collapse(h)
        x = self.input_layernorm(x)
        if self.is_linear:
            r = self.self_attn(x, mask, cache)
        else:
            r, topk_indices = self.self_attn(x, mask, cache, topk_indices)
        h = self.attn_hc.expand(r, h, post, comb)

        x, post, comb = self.ffn_hc.collapse(h)
        r = self.mlp(self.post_attention_layernorm(x))
        h = self.ffn_hc.expand(r, h, post, comb)
        return h, topk_indices


class LanguageModel(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Glm5NextDecoderLayer(args, idx) for idx in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ssm_idx = args.layer_types.index("linear_attention")
        self.attn_idx = args.layer_types.index("deepseek_sparse_attention")

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        x = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)

        ssm_mask = create_ssm_mask(x, cache[self.ssm_idx])
        attn_cache = cache[self.attn_idx]
        attn_mask = create_attention_mask(
            x, attn_cache[0] if attn_cache is not None else None, return_array=True
        )

        h = mx.broadcast_to(
            x[:, :, None, :], (*x.shape[:2], self.args.hc_mult, x.shape[-1])
        )

        topk_indices = None
        for layer, c in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else attn_mask
            h, topk_indices = layer(h, mask, c, topk_indices)

        return self.norm(h.mean(axis=2))


class Glm5NextModel(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.language_model = LanguageModel(args)


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = Glm5NextModel(config.text_config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        out = self.model.language_model(inputs, cache)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.language_model.layers

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.is_linear:
                caches.append(ArraysCache(size=2))
            elif layer.self_attn.skip_topk:
                caches.append(CacheList(KVCache()))
            else:
                caches.append(CacheList(KVCache(), KVCache()))
        return caches

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        args = self.args.text_config
        n_layers = args.num_hidden_layers
        prefix = "model.language_model"

        def keep(k):
            if k.startswith("model.visual."):
                return False
            parts = k.split(".")
            if len(parts) > 3 and parts[2] == "layers" and int(parts[3]) >= n_layers:
                return False
            return True

        weights = {k: v for k, v in weights.items() if keep(k)}

        def dequant(weight, scale_inv):
            weight = mx.from_fp8(weight, dtype=mx.bfloat16)
            bs = 128
            m, n = weight.shape
            pad_bottom = (-m) % bs
            pad_side = (-n) % bs
            weight = mx.pad(weight, ((0, pad_bottom), (0, pad_side)))
            weight = weight.reshape(
                (m + pad_bottom) // bs, bs, (n + pad_side) // bs, bs
            )
            weight = (weight * scale_inv[:, None, :, None]).reshape(
                m + pad_bottom, n + pad_side
            )
            return weight[:m, :n].astype(mx.bfloat16)

        if any("weight_scale_inv" in k for k in weights):
            new_weights = {}
            for k, v in weights.items():
                if "weight_scale_inv" in k:
                    wk = k.replace("_scale_inv", "")
                    new_weights[wk] = dequant(weights[wk], v)
                elif k not in new_weights:
                    new_weights[k] = v
            weights = new_weights

        hc_renames = {
            "hc_attn_fn": "attn_hc.fn",
            "hc_attn_base": "attn_hc.base",
            "hc_attn_scale": "attn_hc.scale",
            "hc_ffn_fn": "ffn_hc.fn",
            "hc_ffn_base": "ffn_hc.base",
            "hc_ffn_scale": "ffn_hc.scale",
        }
        for k in list(weights):
            leaf = k.rsplit(".", 1)[-1]
            if leaf in hc_renames:
                weights[k[: -len(leaf)] + hc_renames[leaf]] = weights.pop(k)

        for l, layer in enumerate(self.layers):
            p = f"{prefix}.layers.{l}"
            ap = f"{p}.self_attn"

            if layer.is_linear:
                if f"{ap}.qkv_proj.weight" not in weights:
                    for suffix in ("weight", "scales", "biases"):
                        parts = [f"{ap}.{n}_proj.{suffix}" for n in "qkv"]
                        if all(pp in weights for pp in parts):
                            weights[f"{ap}.qkv_proj.{suffix}"] = mx.concatenate(
                                [weights.pop(pp) for pp in parts], axis=0
                            )
                if f"{ap}.qkv_conv.conv.weight" not in weights:
                    parts = [f"{ap}.{n}_conv1d.weight" for n in "qkv"]
                    if all(pp in weights for pp in parts):
                        w = mx.concatenate([weights.pop(pp) for pp in parts], axis=0)
                        weights[f"{ap}.qkv_conv.conv.weight"] = w.moveaxis(2, 1)
                for name in ("A_log", "dt_bias"):
                    key = f"{ap}.{name}"
                    if key in weights and weights[key].ndim > 1:
                        weights[key] = weights[key].reshape(-1)
            elif f"{ap}.kv_b_proj.weight" in weights:
                quantized = f"{ap}.kv_b_proj.scales" in weights
                v = weights.pop(f"{ap}.kv_b_proj.weight")
                head_dim = args.qk_nope_head_dim + args.v_head_dim

                if quantized:
                    dims = args.kv_lora_rank
                    scales = weights.pop(f"{ap}.kv_b_proj.scales")
                    biases = weights.pop(f"{ap}.kv_b_proj.biases")
                    bits = (v.shape[-1] * 32) // dims
                    group_size = dims // scales.shape[-1]
                    v = mx.dequantize(
                        v, scales, biases, bits=bits, group_size=group_size
                    )
                v = v.reshape(args.num_attention_heads, head_dim, -1)
                wk = mx.contiguous(v[:, : args.qk_nope_head_dim, :].swapaxes(-1, -2))
                wv = mx.contiguous(v[:, args.qk_nope_head_dim :, :])
                if quantized:
                    wk, wk_scales, wk_biases = mx.quantize(
                        wk, bits=bits, group_size=group_size
                    )
                    wv, wv_scales, wv_biases = mx.quantize(
                        wv, bits=bits, group_size=group_size
                    )
                    weights[f"{ap}.embed_q.scales"] = wk_scales
                    weights[f"{ap}.unembed_out.scales"] = wv_scales
                    weights[f"{ap}.embed_q.biases"] = wk_biases
                    weights[f"{ap}.unembed_out.biases"] = wv_biases
                weights[f"{ap}.embed_q.weight"] = wk
                weights[f"{ap}.unembed_out.weight"] = wv

            if isinstance(layer.mlp, MoE):
                for m in ("gate_proj", "up_proj", "down_proj"):
                    for k in ("weight", "scales", "biases"):
                        if f"{p}.mlp.experts.0.{m}.{k}" in weights:
                            to_join = [
                                weights.pop(f"{p}.mlp.experts.{e}.{m}.{k}")
                                for e in range(args.n_routed_experts)
                            ]
                            weights[f"{p}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)

        return weights

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        if N == 1:
            return
        rank = group.rank()

        for layer in self.layers:
            attn = layer.self_attn

            if layer.is_linear:
                D = attn.head_dim
                P = attn.projection_dim
                num_heads = attn.num_heads // N
                sh = rank * num_heads
                eh = sh + num_heads

                attn.qkv_proj = shard_linear(
                    attn.qkv_proj,
                    "all-to-sharded",
                    segments=[1 / 3, 2 / 3],
                    group=group,
                )
                attn.f_b_proj = shard_linear(
                    attn.f_b_proj, "all-to-sharded", group=group
                )
                attn.g_b_proj = shard_linear(
                    attn.g_b_proj, "all-to-sharded", group=group
                )
                attn.b_proj = shard_linear(attn.b_proj, "all-to-sharded", group=group)
                attn.o_proj = shard_linear(attn.o_proj, "sharded-to-all", group=group)

                w = attn.qkv_conv.conv.weight
                attn.qkv_conv.conv.weight = mx.concatenate(
                    [
                        w[sh * D : eh * D],
                        w[P + sh * D : P + eh * D],
                        w[2 * P + sh * D : 2 * P + eh * D],
                    ],
                    axis=0,
                )
                attn.qkv_conv.conv.groups = 3 * num_heads * D

                attn.A_log = attn.A_log.reshape(-1)[sh:eh]
                attn.dt_bias = attn.dt_bias.reshape(-1)[sh * D : eh * D]
                attn.num_heads = num_heads
                attn.projection_dim = num_heads * D
                attn.qkv_dim = 3 * attn.projection_dim
            else:
                attn.q_b_proj = shard_linear(
                    attn.q_b_proj, "all-to-sharded", group=group
                )
                attn.o_proj = shard_linear(attn.o_proj, "sharded-to-all", group=group)
                attn.num_heads //= N
                num_heads = attn.num_heads
                sh = rank * num_heads
                eh = sh + num_heads

                def shard_heads(w):
                    return w[sh:eh]

                attn.embed_q.apply(shard_heads)
                attn.unembed_out.apply(shard_heads)

            if isinstance(layer.mlp, MoE):
                layer.mlp.sharding_group = group
                shard_inplace(
                    layer.mlp.shared_experts.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.shared_experts.up_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.shared_experts.down_proj, "sharded-to-all", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.up_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.down_proj, "sharded-to-all", group=group
                )
            else:
                layer.mlp.gate_proj = shard_linear(
                    layer.mlp.gate_proj, "all-to-sharded", group=group
                )
                layer.mlp.up_proj = shard_linear(
                    layer.mlp.up_proj, "all-to-sharded", group=group
                )
                layer.mlp.down_proj = shard_linear(
                    layer.mlp.down_proj, "sharded-to-all", group=group
                )

    @property
    def cast_predicate(self):
        def predicate(path: str):
            if "e_score_correction_bias" in path:
                return False
            if path.endswith("A_log") or path.endswith("dt_bias"):
                return False
            if ".attn_hc." in path or ".ffn_hc." in path:
                return False
            if path.endswith("index_kpool_compress_ape"):
                return False
            return True

        return predicate
