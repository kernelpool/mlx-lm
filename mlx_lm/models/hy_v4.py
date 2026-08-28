# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import CacheList, KVCache
from .deepseek_v32 import DeepseekV32MLP, DeepseekV32MoE
from .mla import MultiLinear
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "hy_v4"
    vocab_size: int = 120832
    hidden_size: int = 6144
    intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 78
    num_attention_heads: int = 64
    q_lora_rank: int = 2048
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    index_head_dim: int = 128
    index_n_heads: int = 32
    index_topk: int = 2048
    indexer_types: Optional[List[str]] = None
    mlp_layer_types: Optional[List[str]] = None
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    routed_scaling_factor: float = 2.827
    norm_topk_prob: bool = True
    n_group: int = 1
    topk_group: int = 1
    topk_method: str = "noaux_tc"
    swiglu_limit: float = 10.0
    hc_mult: int = 4
    hc_magnitude: float = 2.0
    hc_eps: float = 1e-6
    gating_type: str = "elementwise"
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-5
    rope_parameters: Optional[Dict] = None
    rope_theta: float = 10000000.0
    rope_scaling: Optional[Dict] = None
    attention_bias: bool = False

    def __post_init__(self):
        if self.rope_parameters is not None:
            self.rope_theta = self.rope_parameters["rope_theta"]
            if self.rope_parameters.get("rope_type", "default") != "default":
                self.rope_scaling = self.rope_parameters
        if self.indexer_types is None:
            self.indexer_types = [
                "full" if i == 0 or (i - 1) % 4 == 0 else "shared"
                for i in range(self.num_hidden_layers)
            ]
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] + ["sparse"] * (self.num_hidden_layers - 1)
        if self.gating_type != "elementwise":
            raise ValueError("Only elementwise gated MLA is supported.")


def _pipeline_segment(full, num_layers, pipeline_size, pipeline_rank):
    if pipeline_size > len(full):
        raise NotImplementedError(
            f"pipeline_size ({pipeline_size}) exceeds the number of indexer "
            f"'full' layers ({len(full)}); each shared layer must share a "
            "pipeline stage with the full layer whose top-k it reuses"
        )
    starts = list(full) + [num_layers]
    base, extra = divmod(len(full), pipeline_size)
    bounds = [0]
    g = 0
    for s in range(pipeline_size):
        g += base + (1 if s < extra else 0)
        bounds.append(starts[g])
    seg = pipeline_size - 1 - pipeline_rank
    return bounds[seg], bounds[seg + 1]


@partial(mx.compile, shapeless=True)
def _clamped_swiglu(x, gate, limit):
    gate = mx.minimum(gate, limit)
    x = mx.clip(x, -limit, limit)
    return nn.silu(gate) * x


class ClampedSwiGLU(nn.Module):
    def __init__(self, limit: float):
        super().__init__()
        self.limit = limit

    def __call__(self, x, gate):
        return _clamped_swiglu(x, gate, self.limit)


class Indexer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.hidden_size
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.index_topk = args.index_topk
        self.wq_b = nn.Linear(
            args.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(self.dim, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim**-0.5
        self.rope = initialize_rope(
            dims=args.qk_rope_head_dim,
            base=args.rope_theta,
            traditional=True,
            max_position_embeddings=args.max_position_embeddings,
            scaling_config=args.rope_scaling,
        )

    def _apply_rope(self, x, offset):
        # The checkpoint puts the rope dims last in the indexer head.
        nope, pe = mx.split(x, [self.head_dim - self.rope_head_dim], axis=-1)
        return mx.concatenate([nope, self.rope(pe, offset=offset)], axis=-1)

    def __call__(
        self,
        x: mx.array,
        qr: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any] = None,
    ):
        b, s, _ = x.shape
        k = self.wk(x)
        k = self.k_norm(k)
        k = mx.reshape(k, (b, 1, s, self.head_dim))

        offset = cache.offset if cache is not None else 0

        k = self._apply_rope(k, offset)

        if cache is not None:
            k, _ = cache.update_and_fetch(k, k)
            # Avoid unevaluated graph growing infinitely
            cache.values = mx.zeros_like(cache.keys)
        if k.shape[2] <= self.index_topk:
            return None
        q = self.wq_b(qr)
        q = q.reshape(b, s, self.n_heads, self.head_dim).swapaxes(1, 2)
        q = self._apply_rope(q, offset)
        scores = q @ k.swapaxes(-1, -2)
        scores = mx.maximum(scores, 0)
        weights = self.weights_proj(x) * (self.n_heads**-0.5 * self.softmax_scale)
        scores = weights[..., None, :] @ scores.transpose(0, 2, 1, 3)
        scores = scores.transpose(0, 2, 1, 3)
        if mask is not None:
            scores = mx.where(mask, scores, -float("inf"))
        return mx.argpartition(scores, kth=-self.index_topk, axis=-1)[
            ..., -self.index_topk :
        ]


class HyV4Attention(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.skip_topk = config.indexer_types[layer_idx] == "shared"
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.scale = self.q_head_dim**-0.5

        self.q_a_proj = nn.Linear(
            self.hidden_size, self.q_lora_rank, bias=config.attention_bias
        )
        self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
        )

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.embed_q = MultiLinear(
            self.qk_nope_head_dim, self.kv_lora_rank, self.num_heads
        )
        self.unembed_out = MultiLinear(
            self.kv_lora_rank, self.v_head_dim, self.num_heads
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self.linear_gate = nn.Linear(
            self.hidden_size, self.num_heads * self.v_head_dim, bias=False
        )
        self.learnable_sink_param = mx.zeros((self.num_heads,))

        if not self.skip_topk:
            self.indexer = Indexer(config)
        self.rope = initialize_rope(
            dims=self.qk_rope_head_dim,
            base=config.rope_theta,
            traditional=True,
            max_position_embeddings=config.max_position_embeddings,
            scaling_config=config.rope_scaling,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        topk_indices: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        B, L, D = x.shape

        qr = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(qr)

        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)
        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)

        offset = cache[0].offset if cache is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)

        kv_latent = mx.expand_dims(kv_latent, axis=1)

        if cache is not None:
            kv_latent, k_pe = cache[0].update_and_fetch(kv_latent, k_pe)
        else:
            cache = [None] * 2

        if not self.skip_topk:
            topk_indices = self.indexer(x, qr, mask, cache=cache[1])
        if topk_indices is not None:
            if L == 1:
                idx = topk_indices[:, :, 0, :, None]
                kv_latent = mx.take_along_axis(
                    kv_latent,
                    mx.broadcast_to(idx, idx.shape[:-1] + (kv_latent.shape[-1],)),
                    axis=2,
                )
                k_pe = mx.take_along_axis(
                    k_pe,
                    mx.broadcast_to(idx, idx.shape[:-1] + (k_pe.shape[-1],)),
                    axis=2,
                )
                if mask is not None:
                    mask = mx.take_along_axis(mask, topk_indices, axis=-1)
            else:
                shape = list(topk_indices.shape)
                shape[-1] = kv_latent.shape[2]
                sparse_mask = mx.zeros(shape, dtype=mx.bool_)
                sparse_mask = mx.put_along_axis(
                    sparse_mask, topk_indices, mx.array(True), axis=-1
                )
                if mask is not None:
                    sparse_mask = sparse_mask & mask
                mask = sparse_mask
        # Ensure the indexer cache is evaluated even if the topk_indices are unused
        # to keep the graph from getting too large
        if not self.skip_topk and cache is not None and cache[0] is not None:
            cache[0].keys = mx.depends(cache[0].keys, (cache[1].keys, cache[1].values))

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask,
                pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
            )

        if L == 1:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        output = scaled_dot_product_attention(
            q_nope,
            k,
            v,
            cache=cache,
            scale=self.scale,
            mask=pe_scores,
            sinks=self.learnable_sink_param.astype(q_nope.dtype),
        )
        if L == 1:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        output = output * mx.sigmoid(self.linear_gate(x))
        return self.o_proj(output), topk_indices


class HyV4MoE(DeepseekV32MoE):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        # The SwiGLU clamp applies only to the routed experts.
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
            activation=ClampedSwiGLU(config.swiglu_limit),
        )


@mx.compile
def _ihc_collapse(h, fn, scale, base, hc, magnitude, rms_eps, hc_eps):
    hf = h.astype(mx.float32)
    flat = mx.flatten(hf, -2, -1)
    rms = mx.rsqrt(mx.mean(mx.square(flat), axis=-1, keepdims=True) + rms_eps)
    mixes = (flat @ fn.T) * rms
    pre = mx.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + hc_eps
    post = magnitude * mx.sigmoid(mixes[..., hc:] * scale[1] + base[hc:]) + hc_eps
    x = (pre[..., None] * hf).sum(axis=-2)
    return x.astype(h.dtype), post


@partial(mx.compile, shapeless=True)
def _ihc_expand(out, residual, post):
    y = post[..., None] * out[..., None, :].astype(mx.float32)
    y = y + residual.astype(mx.float32)
    return y.astype(out.dtype)


@partial(mx.compile, shapeless=True)
def _ihc_head(h, fn, scale, base, rms_eps, hc_eps):
    hf = h.astype(mx.float32)
    flat = mx.flatten(hf, -2, -1)
    rms = mx.rsqrt(mx.mean(mx.square(flat), axis=-1, keepdims=True) + rms_eps)
    mixes = (flat @ fn.T) * rms
    pre = mx.sigmoid(mixes * scale + base) + hc_eps
    y = (pre[..., None] * hf).sum(axis=-2)
    return y.astype(h.dtype)


class HyV4HCPre(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        hc = config.hc_mult
        self.hc_fn = mx.zeros((2 * hc, hc * config.hidden_size), dtype=mx.float32)
        self.hc_scale = mx.ones((2,), dtype=mx.float32)
        self.hc_base = mx.zeros((2 * hc,), dtype=mx.float32)


class HyV4HCLayer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.magnitude = config.hc_magnitude
        self.hc_eps = config.hc_eps
        self.rms_norm_eps = config.rms_norm_eps
        self.hc_pre = HyV4HCPre(config)

    def collapse(self, h: mx.array) -> Tuple[mx.array, mx.array]:
        p = self.hc_pre
        return _ihc_collapse(
            h,
            p.hc_fn,
            p.hc_scale,
            p.hc_base,
            self.hc_mult,
            self.magnitude,
            self.rms_norm_eps,
            self.hc_eps,
        )

    def expand(self, out: mx.array, residual: mx.array, post: mx.array) -> mx.array:
        return _ihc_expand(out, residual, post)


class HyV4HCHead(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        hc = config.hc_mult
        self.hc_eps = config.hc_eps
        self.rms_norm_eps = config.rms_norm_eps
        self.hc_head_fn = mx.zeros((hc, hc * config.hidden_size), dtype=mx.float32)
        self.hc_head_scale = mx.ones((1,), dtype=mx.float32)
        self.hc_head_base = mx.zeros((hc,), dtype=mx.float32)

    def __call__(self, h: mx.array) -> mx.array:
        return _ihc_head(
            h,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )


class HyV4DecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = HyV4Attention(config, layer_idx)
        self.mlp = (
            DeepseekV32MLP(config)
            if config.mlp_layer_types[layer_idx] == "dense"
            else HyV4MoE(config)
        )
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.hc_attn_layer = HyV4HCLayer(config)
        self.hc_mlp_layer = HyV4HCLayer(config)

    def __call__(
        self,
        h: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        topk_indices: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        residual = h
        x, post = self.hc_attn_layer.collapse(h)
        r, topk_indices = self.self_attn(
            self.input_layernorm(x), mask, cache, topk_indices
        )
        h = self.hc_attn_layer.expand(r, residual, post)

        residual = h
        x, post = self.hc_mlp_layer.collapse(h)
        r = self.mlp(self.post_attention_layernorm(x))
        return self.hc_mlp_layer.expand(r, residual, post), topk_indices


class HyV4Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            HyV4DecoderLayer(config, idx) for idx in range(config.num_hidden_layers)
        ]
        self.shares_indexer = any(layer.self_attn.skip_topk for layer in self.layers)
        self.start_idx = 0
        self.end_idx = len(self.layers)
        self.num_layers = self.end_idx

        self.hc_head = HyV4HCHead(config)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pipeline_rank = 0
        self.pipeline_size = 1

    def pipeline(self, group):
        # Split layers in reverse so rank=0 gets the last layers and
        # rank=pipeline_size-1 gets the first
        self.pipeline_rank = group.rank()
        self.pipeline_size = group.size()
        if self.shares_indexer and self.pipeline_size > 1:
            full = [
                i
                for i, layer in enumerate(self.layers)
                if not layer.self_attn.skip_topk
            ]
            self.start_idx, self.end_idx = _pipeline_segment(
                full, len(self.layers), self.pipeline_size, self.pipeline_rank
            )
        else:
            layers_per_rank = len(self.layers) // self.pipeline_size
            extra = len(self.layers) - layers_per_rank * self.pipeline_size
            if self.pipeline_rank < extra:
                layers_per_rank += 1
            self.start_idx = (
                self.pipeline_size - self.pipeline_rank - 1
            ) * layers_per_rank
            self.end_idx = self.start_idx + layers_per_rank
        self.layers = self.layers[: self.end_idx]
        self.layers[: self.start_idx] = [None] * self.start_idx
        self.num_layers = len(self.layers) - self.start_idx

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.embed_tokens(x)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * self.num_layers
        mask = create_attention_mask(
            h, cache[0][0] if cache[0] else None, return_array=True
        )

        # Replicate the hidden state over the hc_mult residual channels
        h = mx.broadcast_to(
            h[:, :, None, :],
            (h.shape[0], h.shape[1], self.args.hc_mult, h.shape[2]),
        )
        h = mx.contiguous(h)

        # Receive from the previous process in the pipeline
        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        topk_indices = None
        for i in range(self.num_layers):
            h, topk_indices = self.layers[self.start_idx + i](
                h, mask, cache[i], topk_indices
            )

        # Send to the next process in the pipeline
        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            if cache[-1] is not None:
                cache[-1][0].keys = mx.depends(cache[-1][0].keys, h)

        # Broadcast h while keeping it in the graph
        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(self.hc_head(h))


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = HyV4Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ):
        out = self.model(inputs, cache)
        return self.lm_head(out)

    def sanitize(self, weights):
        # Remove the multi-token prediction layer and any truncated layers
        def keep(k):
            if k.startswith("model.mtp_layers."):
                return False
            parts = k.split(".")
            if len(parts) >= 3 and parts[1] == "layers":
                return int(parts[2]) < self.args.num_hidden_layers
            return True

        weights = {k: v for k, v in weights.items() if keep(k)}

        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}.mlp"
            gate_up = weights.pop(f"{prefix}.experts.gate_up_proj", None)
            if gate_up is not None:
                dims = self.args.moe_intermediate_size
                weights[f"{prefix}.switch_mlp.gate_proj.weight"] = mx.contiguous(
                    gate_up[:, :dims]
                )
                weights[f"{prefix}.switch_mlp.up_proj.weight"] = mx.contiguous(
                    gate_up[:, dims:]
                )
            down = weights.pop(f"{prefix}.experts.down_proj", None)
            if down is not None:
                weights[f"{prefix}.switch_mlp.down_proj.weight"] = down

            prefix = f"model.layers.{l}.self_attn"
            if f"{prefix}.kv_b_proj.weight" in weights:
                quantized = f"{prefix}.kv_b_proj.scales" in weights
                v = weights.pop(f"{prefix}.kv_b_proj.weight")
                head_dim = self.args.qk_nope_head_dim + self.args.v_head_dim

                if quantized:
                    dims = self.args.kv_lora_rank
                    scales = weights.pop(f"{prefix}.kv_b_proj.scales")
                    biases = weights.pop(f"{prefix}.kv_b_proj.biases")
                    # Try to infer bits and group size
                    bits = (v.shape[-1] * 32) // dims
                    group_size = dims // scales.shape[-1]
                    v = mx.dequantize(
                        v, scales, biases, bits=bits, group_size=group_size
                    )
                num_heads = self.args.num_attention_heads
                v = v.reshape(num_heads, head_dim, -1)
                wk = mx.contiguous(
                    v[:, : self.args.qk_nope_head_dim, :].swapaxes(-1, -2)
                )
                wv = mx.contiguous(v[:, self.args.qk_nope_head_dim :, :])
                if quantized:
                    wk, wk_scales, wk_biases = mx.quantize(
                        wk, bits=bits, group_size=group_size
                    )
                    wv, wv_scales, wv_biases = mx.quantize(
                        wv, bits=bits, group_size=group_size
                    )
                    weights[f"{prefix}.embed_q.scales"] = wk_scales
                    weights[f"{prefix}.unembed_out.scales"] = wv_scales
                    weights[f"{prefix}.embed_q.biases"] = wk_biases
                    weights[f"{prefix}.unembed_out.biases"] = wv_biases
                weights[f"{prefix}.embed_q.weight"] = wk
                weights[f"{prefix}.unembed_out.weight"] = wv

        return weights

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        rank = group.rank()
        for layer in self.model.layers:
            layer.self_attn.q_b_proj = shard_linear(
                layer.self_attn.q_b_proj, "all-to-sharded", group=group
            )
            layer.self_attn.linear_gate = shard_linear(
                layer.self_attn.linear_gate, "all-to-sharded", group=group
            )
            layer.self_attn.o_proj = shard_linear(
                layer.self_attn.o_proj, "sharded-to-all", group=group
            )
            layer.self_attn.num_heads //= N
            num_heads = layer.self_attn.num_heads
            sh = rank * num_heads
            eh = sh + num_heads

            def shard_heads(w):
                return w[sh:eh]

            layer.self_attn.embed_q.apply(shard_heads)
            layer.self_attn.unembed_out.apply(shard_heads)
            layer.self_attn.learnable_sink_param = layer.self_attn.learnable_sink_param[
                sh:eh
            ]

            # Shard the MLP
            if isinstance(layer.mlp, DeepseekV32MLP):
                layer.mlp.gate_proj = shard_linear(
                    layer.mlp.gate_proj, "all-to-sharded", group=group
                )
                layer.mlp.down_proj = shard_linear(
                    layer.mlp.down_proj, "sharded-to-all", group=group
                )
                layer.mlp.up_proj = shard_linear(
                    layer.mlp.up_proj, "all-to-sharded", group=group
                )

            # Shard the MoE. Shard in place since the MoE should be responsible
            # for aggregating the results.
            else:
                layer.mlp.sharding_group = group
                shard_inplace(
                    layer.mlp.shared_experts.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.shared_experts.down_proj, "sharded-to-all", group=group
                )
                shard_inplace(
                    layer.mlp.shared_experts.up_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.down_proj, "sharded-to-all", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.up_proj, "all-to-sharded", group=group
                )

    @property
    def layers(self):
        return self.model.layers[self.model.start_idx : self.model.end_idx]

    @property
    def cast_predicate(self):
        def predicate(k):
            return not (
                "e_score_correction_bias" in k
                or "learnable_sink_param" in k
                or ".hc_pre." in k
                or "hc_head" in k
            )

        return predicate

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.self_attn.skip_topk:
                caches.append(CacheList(KVCache()))
            else:
                caches.append(CacheList(KVCache(), KVCache()))
        return caches
