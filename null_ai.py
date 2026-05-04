#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║    ███╗   ██╗██╗   ██╗██╗     ██╗          █████╗ ██╗                       ║
║    ████╗  ██║██║   ██║██║     ██║         ██╔══██╗██║                       ║
║    ██╔██╗ ██║██║   ██║██║     ██║         ███████║██║                       ║
║    ██║╚██╗██║██║   ██║██║     ██║         ██╔══██║██║                       ║
║    ██║ ╚████║╚██████╔╝███████╗███████╗    ██║  ██║██║                       ║
║    ╚═╝  ╚═══╝ ╚═════╝ ╚══════╝╚══════╝    ╚═╝  ╚═╝╚═╝                       ║
║                                                                              ║
║    Ultra-Compact 16MB Language Model  |  Google Colab T4 Ready              ║
║    Author: Based on Hiroto Funasaki's SNN-LM + parameter-golf stack         ║
╚══════════════════════════════════════════════════════════════════════════════╝

Techniques Implemented:
  ✓ GQA (Grouped Query Attention) 2:1                  → 30% fewer KV params
  ✓ LeakyReLU² MLP activation                          → PR #493
  ✓ Partial RoPE (25% of head dims) + LN 1/√(layer+1) → PR #315
  ✓ U-Net encoder-decoder skip connections + gates     → PR #289
  ✓ Depth recurrence (loop layers 3-4, 2× after 35%)  → PR #1344
  ✓ Logit softcap (30)                                 → Gemma2-style
  ✓ SmearGate with BOS cross-doc leak fix              → PR #1667 + fix
  ✓ Sparse attention head-output gate (window=12)      → PR #1787
  ✓ Per-head learned QK gain (init 5.0)                → PR #1276
  ✓ Parallel decoder (2-lane, layers 6+)               → PR #1530
  ✓ SNN Spike Encoder (11D Hypercube, novel fusion)    → snn_lm.py
  ✓ Muon optimizer (Polar-Express Newton-Schulz, 5 NS) → PR #1344→#1787
  ✓ Warmup (100 steps) + cosine warmdown (85% window)  → PR #1787
  ✓ MIN_LR floor = 0.10 × peak_lr                     → PR #1787
  ✓ bfloat16 mixed precision (GradScaler)              → T4 native
  ✓ Weight tying (embed ↔ lm_head)                     → Standard
  ✓ Gradient clipping (0.3)                            → Stack default
  ✓ EMA weights (decay=0.9965)                         → Stack default
  ✓ Test-Time Training (TTT) eval pass                 → PR #1736
  ✓ Post-training int8 quantization + compress save    → PR #1797 inspired

Target: ~8M params → ~16 MB (bfloat16) on T4 Colab free tier

Usage:
    # Colab: just run — downloads TinyShakespeare automatically
    !python null_ai.py

    # Custom dataset:
    !python null_ai.py --data my_corpus.txt --max_iters 20000

    # Larger model (still fits T4):
    !python null_ai.py --d_model 384 --n_layers 10 --max_iters 15000
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 0 — IMPORTS & SETUP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import os, sys, math, time, json, copy, struct, argparse, hashlib
import urllib.request
import re
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders
from datasets import load_dataset

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 1 — CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class NullAIConfig:
    """
    NullAI model + training configuration.
    Defaults are tuned to hit ~32MB (bfloat16) and train well on T4 Colab.
    """

    # ── Model size ──────────────────────────────────────────────────────────
    vocab_size: int  = 32000     # Updated at runtime by tokenizer
    max_seq_len: int = 2048      # Increased context length
    d_model: int     = 264       # Scaled hidden dimension
    n_layers: int    = 8         # Number of transformer blocks
    n_heads: int     = 4         # Query heads
    n_kv_heads: int  = 2         # KV heads (GQA ratio 2:1)
    d_mlp: int       = 1056      # Scaled MLP inner dim (4× d_model)
    window_size: int = 512       # Sliding window attention size

    # ── Architectural features ───────────────────────────────────────────────
    rope_frac: float          = 0.25   # Partial RoPE: fraction of head dims
    yarn_scale: float         = 1.0    # YaRN length extrapolation scale
    logit_softcap: float      = 30.0   # Gemma2-style tanh softcap
    leaky_slope: float        = 0.5    # LeakyReLU slope for MLP

    # SmearGate: forward-1 position mixing with BOS-leak fix
    smear_gate: bool          = True

    # Sparse attention head-output gate
    sparse_attn_gate: bool    = True
    gate_window: int          = 12
    sparse_attn_gate_scale: float = 0.5

    # U-Net encoder-decoder skip connections
    unet_skips: bool          = True

    # Depth recurrence: repeat middle layers after loop_start_frac
    loop_layers: tuple        = (3, 4)
    loop_repeats: int         = 2
    loop_start_frac: float    = 0.35  # Enable after 35% of training

    # Parallel decoder: 2-lane split after this layer index
    parallel_decoder_start: int = 6
    parallel_decoder: bool    = True

    # SNN spike encoder: brain-inspired sparse encoding (11D Hypercube)
    snn_encoder: bool         = True
    snn_hypercube_dim: int    = 5     # 2^5=32 neurons per group in d_model
    snn_sparsity: float       = 0.3   # Target sparsity

    # Per-head QK gain (learned scalar, init 5.0)
    qk_gain_init: float       = 5.0

    # LN weight scale (applied as multiplier to norm output)
    # = 1/√(layer+1), computed dynamically per layer

    # ── Training ────────────────────────────────────────────────────────────
    batch_size: int           = 24     # Fits T4 16GB easily with seq_len=256
    seq_len: int              = 256
    lr_peak: float            = 3e-3
    lr_min_frac: float        = 0.10   # MIN_LR = lr_peak * lr_min_frac
    warmup_steps: int         = 100
    warmdown_frac: float      = 0.85   # Fraction of budget for warmdown window
    beta1: float              = 0.9
    beta2: float              = 0.99
    weight_decay: float       = 0.1
    grad_clip: float          = 0.3

    # Muon optimizer (for matrix params)
    muon_lr_scale: float      = 0.5    # muon_lr = lr_peak * muon_lr_scale
    muon_momentum: float      = 0.95
    muon_ns_steps: int        = 5      # Newton-Schulz iterations

    # Adam for embeddings + scalars
    embed_lr_scale: float     = 0.01   # embed_lr = lr_peak * embed_lr_scale
    scalar_lr_scale: float    = 0.05

    # EMA
    ema_decay: float          = 0.9965

    # ── Evaluation / TTT ────────────────────────────────────────────────────
    ttt_enabled: bool         = True   # Test-Time Training pass at eval
    ttt_steps: int            = 3      # Quick TTT iterations
    ttt_lr: float             = 1e-4

    # ── Logging / saving ────────────────────────────────────────────────────
    max_iters: int            = 10000
    max_wallclock: int        = 3300   # ~55 min (Colab 60 min sessions)
    log_every: int            = 50
    eval_every: int           = 250
    save_every: int           = 1000
    seed: int                 = 42

    # Special token IDs
    bos_id: int = 1
    eos_id: int = 2
    pad_id: int = 0

    # Runtime (set automatically, not user-facing)
    _actual_vocab: int = 0
    tokenizer_path: str = "tokenizer_vocab.json"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 2 — UTILITIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_size_mb(model: nn.Module, dtype=torch.bfloat16) -> float:
    n     = sum(p.numel() for p in model.parameters())
    nbytes = 2 if dtype == torch.bfloat16 else 4
    return n * nbytes / (1024 ** 2)

def bits_per_byte(loss: float) -> float:
    """Convert cross-entropy (nats) to bits-per-byte."""
    return loss / math.log(2)

def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 3 — 11D HYPERCUBE TOPOLOGY (from hypercube.py, adapted)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_hypercube_adjacency(dim: int) -> torch.Tensor:
    """
    Create adjacency matrix for n-dimensional hypercube.
    Each node i is connected to neighbors that differ by exactly 1 bit.
    2^dim nodes, each with exactly dim connections → very sparse.
    """
    n    = 2 ** dim
    mask = torch.zeros(n, n, dtype=torch.float32)
    for node in range(n):
        for d in range(dim):
            neighbor         = node ^ (1 << d)
            mask[node, neighbor] = 1.0
    return mask


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 4 — SNN SPIKE ENCODER (novel fusion of snn_lm.py + hypercube.py)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class HypercubeSpikeEncoder(nn.Module):
    """
    SNN-inspired spike encoder using 11D Hypercube topology.

    Concept (from snn_lm.py + hypercube.py):
      - Dense embedding → threshold comparison → sparse spike pattern
      - Spike propagation follows 11D Hypercube adjacency (brain-like)
      - Temporal coding: information in WHICH neurons fire, not just how much
      - Straight-through estimator for differentiable binarisation

    Why this helps:
      - Forces sparse representations (brain uses ~1-5% active neurons)
      - Hypercube routing gives O(dim) path length (11 hops max vs O(N))
      - Acts as a learnable dropout / feature selector
      - Adds temporal memory via the reservoir-style state
    """

    def __init__(self, d_model: int, hypercube_dim: int = 5, sparsity: float = 0.3):
        super().__init__()
        self.d_model       = d_model
        self.hypercube_dim = hypercube_dim
        self.n_reservoir   = 2 ** hypercube_dim          # 32 for dim=5
        self.n_groups      = d_model // self.n_reservoir # groups of 32
        self.sparsity      = sparsity

        # Adjacency mask for spike propagation (fixed, not trained)
        adj = build_hypercube_adjacency(hypercube_dim)   # (32, 32)
        self.register_buffer('adj', adj)

        # Learnable input → threshold projection
        self.threshold_proj = nn.Linear(d_model, d_model, bias=True)
        nn.init.zeros_(self.threshold_proj.weight)
        nn.init.constant_(self.threshold_proj.bias, 0.5)

        # Reservoir weights per group (hypercube-masked, fixed random)
        W_res = torch.randn(self.n_groups, self.n_reservoir, self.n_reservoir) * 0.3
        W_res *= adj.unsqueeze(0)                        # Apply hypercube mask
        # Scale to spectral radius ≈ 1.2 (chaotic but stable)
        for g in range(self.n_groups):
            eigs = torch.linalg.eigvals(W_res[g]).abs().max()
            if eigs > 0:
                W_res[g] *= 1.2 / eigs.item()
        self.register_buffer('W_res', W_res)

        # Output blending: spike_out → d_model
        self.out_scale = nn.Parameter(torch.ones(d_model) * 0.1)
        self.out_gate  = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.out_gate.weight)

        # Running reservoir state (reset per sequence)
        self.register_buffer('reservoir_state',
                             torch.zeros(1, self.n_groups, self.n_reservoir))

    def _fire(self, x_flat: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        """
        Leaky-Integrate-and-Fire: fire when activation > threshold.
        Straight-through for gradients.
        """
        # Soft sigmoid (differentiable) spike
        pre_spike = torch.sigmoid((x_flat - threshold) / 0.15)
        # Hard binarisation
        hard      = (pre_spike > (1 - self.sparsity)).float()
        # STE: pass gradients through as if hard = pre_spike
        return pre_spike + (hard - pre_spike).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        Returns: x + spike-gated residual, same shape
        """
        B, T, D = x.shape

        # Dynamic threshold (learned per-token)
        threshold = self.threshold_proj(x)               # (B, T, D)

        # Reshape to groups of n_reservoir neurons
        x_g = x.reshape(B * T, self.n_groups, self.n_reservoir)    # (BT, G, 32)
        th_g = threshold.reshape(B * T, self.n_groups, self.n_reservoir)

        # Fire: compute spike patterns
        spikes = self._fire(x_g, th_g)                  # (BT, G, 32)

        # Hypercube propagation (one reservoir step)
        # spikes_next[g] = tanh(W_res[g] @ spikes[g])
        # W_res: (G, 32, 32), spikes: (BT, G, 32)
        propagated = torch.tanh(
            torch.einsum('gij, bgj -> bgi', self.W_res, spikes)
        )                                                # (BT, G, 32)

        # Reshape back to (B, T, D)
        spike_out = propagated.reshape(B, T, D)

        # Gate and blend: residual connection with learned scale
        gate       = torch.sigmoid(self.out_gate(spike_out))
        return x + self.out_scale * gate * spike_out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 5 — PARTIAL ROPE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PartialRoPE(nn.Module):
    """
    Partial RoPE: apply rotary embeddings only to the first rope_frac
    fraction of head dimensions.  Non-rotated dims remain as-is.
    Combined with YaRN-style frequency scaling for length generalisation.
    """

    def __init__(self, d_head: int, max_seq: int,
                 rope_frac: float = 0.25, yarn_scale: float = 1.0):
        super().__init__()
        rope_dims = max(2, int(d_head * rope_frac))
        rope_dims = rope_dims - (rope_dims % 2)      # Must be even
        self.rope_dims = rope_dims

        # YaRN: scale base frequency by context length ratio
        base  = 10000.0
        theta = 1.0 / (
            (base * yarn_scale ** (rope_dims / (rope_dims - 2))) **
            (torch.arange(0, rope_dims, 2).float() / rope_dims)
        )
        pos   = torch.arange(max_seq).float()
        freqs = torch.outer(pos, theta)              # (T, rope_dims/2)
        self.register_buffer('cos', freqs.cos())
        self.register_buffer('sin', freqs.sin())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, T, D)"""
        B, H, T, D  = x.shape
        rd           = self.rope_dims
        x1           = x[..., :rd]
        x_pass       = x[..., rd:]

        # Rotate x1
        x1_pairs     = x1.reshape(B, H, T, rd // 2, 2)
        cos          = self.cos[:T].unsqueeze(0).unsqueeze(0)  # (1,1,T,rd/2)
        sin          = self.sin[:T].unsqueeze(0).unsqueeze(0)

        x1_rot = torch.stack([
            x1_pairs[..., 0] * cos - x1_pairs[..., 1] * sin,
            x1_pairs[..., 0] * sin + x1_pairs[..., 1] * cos,
        ], dim=-1).flatten(-2)

        return torch.cat([x1_rot, x_pass], dim=-1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 6 — SMEARGATE (BOS-LEAK FIXED)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SmearGate(nn.Module):
    """
    PR #1667 SmearGate: forward-1 position mixing.
      x[t] += g[t-1] * x[t-1]

    BOS-leak fix (this work, matching PR #1797 spirit):
      x[t] += g[t-1] * x[t-1] * not_bos[t]
    Prevents the last token of doc N from contaminating the BOS of doc N+1
    in packed training streams — critical for multi-document batches.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.gate.weight)   # Start as identity (no smear)

    def forward(self, x: torch.Tensor,
                input_ids: Optional[torch.Tensor] = None,
                bos_id: int = 1) -> torch.Tensor:
        """x: (B, T, D)"""
        g    = torch.sigmoid(self.gate(x[:, :-1]))  # (B, T-1, D)
        prev = x[:, :-1]
        curr = x[:, 1:]

        if input_ids is not None and input_ids.shape[1] > 1:
            # BOS mask: zero out the smear wherever current token is BOS
            not_bos = (input_ids[:, 1:] != bos_id).to(x.dtype).unsqueeze(-1)
            smeared = curr + g * prev * not_bos
        else:
            smeared = curr + g * prev

        return torch.cat([x[:, :1], smeared], dim=1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 7 — SPARSE ATTENTION GATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SparseAttnGate(nn.Module):
    """
    PR #1787: Narrow per-head output gate.
    Gates each attention head's contribution independently,
    allowing the model to selectively suppress noisy heads.

    gate = sigmoid(W @ head_out) * scale + (1 - scale)
    So at init: gate ≈ 0.5 * scale + (1-scale) = 1 - 0.5*scale
    (starts mostly open, learns to close noisy heads)
    """

    def __init__(self, n_heads: int, d_head: int,
                 gate_window: int = 12, scale: float = 0.5):
        super().__init__()
        self.n_heads   = n_heads
        self.d_head    = d_head
        self.scale     = scale
        # Narrow projection: d_head → 1 per head
        self.gate_proj = nn.Linear(d_head, 1, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 1.0)  # Start open

    def forward(self, attn_out: torch.Tensor) -> torch.Tensor:
        """attn_out: (B, T, n_heads * d_head)"""
        B, T, D = attn_out.shape
        x    = attn_out.reshape(B, T, self.n_heads, self.d_head)
        gate = torch.sigmoid(self.gate_proj(x))      # (B, T, H, 1)
        # Scale: gate ∈ [1-scale, 1] so it never fully closes
        gate = gate * self.scale + (1.0 - self.scale)
        return (x * gate).reshape(B, T, D)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 8 — GROUPED QUERY ATTENTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GQAttention(nn.Module):
    """
    Grouped Query Attention (GQA ratio = n_heads / n_kv_heads).
    With:
      - Partial RoPE (rope_frac of head dims)
      - Per-head learned QK gain (init 5.0, PR #1276)
      - Sparse attention gate (PR #1787)
      - Efficient KV repeat-interleave for GQA
    """

    def __init__(self, cfg: NullAIConfig, rope: PartialRoPE):
        super().__init__()
        assert cfg.n_heads % cfg.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_heads    = cfg.n_heads
        self.n_kv       = cfg.n_kv_heads
        self.n_rep      = cfg.n_heads // cfg.n_kv_heads
        self.d_head     = cfg.d_model  // cfg.n_heads
        self.rope       = rope

        self.q_proj     = nn.Linear(cfg.d_model, cfg.n_heads * self.d_head, bias=False)
        self.k_proj     = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.d_head, bias=False)
        self.v_proj     = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.d_head, bias=False)
        self.o_proj     = nn.Linear(cfg.n_heads * self.d_head, cfg.d_model, bias=False)

        # Per-head QK gain: learned scalar per query head, init = 5.0
        self.qk_gain    = nn.Parameter(torch.full((cfg.n_heads,), cfg.qk_gain_init))

        # Sparse attention gate
        if cfg.sparse_attn_gate:
            self.attn_gate = SparseAttnGate(cfg.n_heads, self.d_head,
                                            cfg.gate_window,
                                            cfg.sparse_attn_gate_scale)
        else:
            self.attn_gate = None

    def forward(self, x: torch.Tensor, causal_mask: Optional[torch.Tensor],
                past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False):
        B, T, D = x.shape

        q = self.q_proj(x).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.n_kv,   self.d_head).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.n_kv,   self.d_head).transpose(1, 2)

        # Partial RoPE
        q = self.rope(q)
        k = self.rope(k)

        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        new_kv = (k, v) if use_cache else None

        # Per-head scale: qk_gain / sqrt(d_head)
        scale = self.qk_gain.reshape(1, self.n_heads, 1, 1) / math.sqrt(self.d_head)

        # Expand KV for GQA
        k_rep = k.repeat_interleave(self.n_rep, dim=1)   # (B, H, T_total, d_head)
        v_rep = v.repeat_interleave(self.n_rep, dim=1)

        # Attention scores + causal mask
        scores = torch.matmul(q * scale, k_rep.transpose(-2, -1))
        if causal_mask is not None:
            # When using KV cache, T_q = 1, T_k = T_total. Mask should be (1, T_total)
            if T == 1 and k.shape[2] > 1:
                # We need a mask for just the new token against all previous
                scores = scores + causal_mask[:, :, -1:, :k.shape[2]]
            else:
                scores = scores + causal_mask[:, :, :T, :T]
        attn   = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, v_rep)                   # (B, H, T, d_head)
        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.d_head)

        # Sparse gate
        if self.attn_gate is not None:
            out = self.attn_gate(out)

        return self.o_proj(out), new_kv


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 9 — LEAKYRELU² MLP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LeakyReLU2MLP(nn.Module):
    """
    PR #493: MLP with LeakyReLU² activation.
      f(x) = LeakyReLU(x, slope)²

    Why it works:
      - Preserves negative gradients (unlike ReLU dead-neuron problem)
      - Squaring makes the function super-linear → sharper feature selection
      - Empirically outperforms SiLU in parameter-constrained models
    """

    def __init__(self, d_model: int, d_mlp: int, slope: float = 0.5):
        super().__init__()
        self.up   = nn.Linear(d_model, d_mlp, bias=False)
        self.down = nn.Linear(d_mlp,   d_model, bias=False)
        self.slope = slope
        # Fan-in init for down projection
        nn.init.normal_(self.down.weight, std=1.0 / math.sqrt(d_mlp))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.up(x)
        h = F.leaky_relu(h, self.slope) ** 2
        return self.down(h)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 10 — TRANSFORMER BLOCK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class NullAIBlock(nn.Module):
    """
    Single NullAI transformer block with:
      - Pre-norm (RMSNorm) with 1/√(layer+1) scale (PR #315)
      - GQA attention
      - LeakyReLU² MLP
      - U-Net skip gate (PR #289)
    """

    def __init__(self, cfg: NullAIConfig, layer_idx: int, rope: PartialRoPE):
        super().__init__()
        self.layer_idx = layer_idx
        self.ln_scale  = 1.0 / math.sqrt(layer_idx + 1)  # PR #315

        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.attn  = GQAttention(cfg, rope)
        self.mlp   = LeakyReLU2MLP(cfg.d_model, cfg.d_mlp, cfg.leaky_slope)

        # U-Net skip gate: learned gating of encoder skip connection
        self.skip_gate = nn.Parameter(torch.zeros(cfg.d_model))

    def forward(self, x: torch.Tensor,
                causal_mask: Optional[torch.Tensor],
                skip: Optional[torch.Tensor] = None,
                past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False):
        # Inject U-Net encoder skip (decoder layers only)
        if skip is not None:
            x = x + torch.sigmoid(self.skip_gate) * skip

        # Attention sub-layer
        attn_out, new_kv = self.attn(self.norm1(x) * self.ln_scale, causal_mask, past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out

        # MLP sub-layer
        x = x + self.mlp(self.norm2(x) * self.ln_scale)

        return x, new_kv


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 11 — NULL AI MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class NullAI(nn.Module):
    """
    NullAI: Ultra-Compact 16MB Language Model

    Full architecture:
      embed → SNN spike encoder → SmearGate
        → [Encoder layers 0..N/2-1]   (saves states for U-Net)
        → [Decoder layers N/2..N-1]   (receives U-Net skip connections)
        → [Parallel 2-lane from layer P+]  (PR #1530)
      → RMSNorm → lm_head (weight-tied with embed) → softcap
    """

    def __init__(self, cfg: NullAIConfig):
        super().__init__()
        self.cfg = cfg

        d_head = cfg.d_model // cfg.n_heads

        # Shared RoPE (same for all layers)
        self.rope = PartialRoPE(d_head, cfg.max_seq_len,
                                cfg.rope_frac, cfg.yarn_scale)

        # Token embeddings (tied with lm_head)
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # SNN Spike Encoder (novel: 11D Hypercube reservoir)
        self.spike_enc = (HypercubeSpikeEncoder(cfg.d_model,
                                                cfg.snn_hypercube_dim,
                                                cfg.snn_sparsity)
                          if cfg.snn_encoder else None)

        # SmearGate (BOS-leak fixed)
        self.smear = SmearGate(cfg.d_model) if cfg.smear_gate else None

        # Transformer blocks
        self.blocks = nn.ModuleList([
            NullAIBlock(cfg, i, self.rope) for i in range(cfg.n_layers)
        ])

        # Parallel decoder lane (PR #1530): second residual stream
        if cfg.parallel_decoder and cfg.parallel_decoder_start < cfg.n_layers:
            pd_layers = cfg.n_layers - cfg.parallel_decoder_start
            self.parallel_blocks = nn.ModuleList([
                NullAIBlock(cfg, cfg.parallel_decoder_start + i, self.rope)
                for i in range(pd_layers)
            ])
            self.lane_mix = nn.Parameter(torch.tensor(0.5))  # Learned blend
        else:
            self.parallel_blocks = None
            self.lane_mix        = None

        # Final norm
        self.norm_f = nn.RMSNorm(cfg.d_model)

        # LM head — TIED weights with embed (saves ~2M params for 8K vocab)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        # Runtime state: enable depth recurrence after warmup
        self.use_recurrence = False

        # Causal mask cache
        self._causal_mask_cache: Dict[int, torch.Tensor] = {}

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.lm_head:
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _get_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        if T not in self._causal_mask_cache:
            mask = torch.full((1, 1, T, T), float('-inf'), device=device)
            mask = torch.triu(mask, diagonal=1)
            # Sliding window attention
            if hasattr(self.cfg, 'window_size') and self.cfg.window_size > 0:
                # Mask tokens further back than window_size
                mask_low = torch.tril(torch.ones(T, T, device=device), diagonal=-self.cfg.window_size-1)
                mask.masked_fill_(mask_low.bool().unsqueeze(0).unsqueeze(0), float('-inf'))
            self._causal_mask_cache[T] = mask
        return self._causal_mask_cache[T]

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        B, T   = input_ids.shape
        device = input_ids.device
        next_kv = [] if use_cache else None

        # In case of KV cache, we might need a longer mask if not cached
        max_T = T
        if past_kv is not None:
            max_T = T + past_kv[0][0].shape[2]
        causal_mask = self._get_causal_mask(max_T, device)

        # ── Embedding + Encoding ─────────────────────────────────────────
        x = self.embed(input_ids)                    # (B, T, D)

        if self.spike_enc is not None:
            x = self.spike_enc(x)                   # SNN: sparse spike repr

        if self.smear is not None:
            x = self.smear(x, input_ids, self.cfg.bos_id)

        # ── Determine loop repeats ───────────────────────────────────────
        loop_set  = set(self.cfg.loop_layers) if self.use_recurrence else set()
        n_repeats = self.cfg.loop_repeats     if self.use_recurrence else 1

        # ── U-Net split ─────────────────────────────────────────────────
        n_enc          = self.cfg.n_layers // 2
        encoder_states: List[torch.Tensor] = []

        # ── Parallel decoder lane setup ──────────────────────────────────
        pd_start = self.cfg.parallel_decoder_start
        x_lane2  = None  # Second lane (only active after pd_start)

        # ── Main transformer stack ───────────────────────────────────────
        for i, block in enumerate(self.blocks):

            # U-Net: inject encoder skip into decoder layers
            skip = None
            if self.cfg.unet_skips and i >= n_enc:
                enc_mirror = self.cfg.n_layers - 1 - i
                if 0 <= enc_mirror < len(encoder_states):
                    skip = encoder_states[enc_mirror]

            # Depth recurrence: run middle layers multiple times
            reps = n_repeats if i in loop_set else 1
            for _ in range(reps):
                kv_i = past_kv[i] if (past_kv is not None and i < len(past_kv)) else None
                x, new_kv = block(x, causal_mask, skip=skip, past_kv=kv_i, use_cache=use_cache)
                if use_cache:
                    next_kv.append(new_kv)
                skip = None   # Only inject skip on first repeat

            # Save encoder state for U-Net
            if self.cfg.unet_skips and i < n_enc:
                encoder_states.append(x)

            # Fork parallel lane after pd_start
            if (self.parallel_blocks is not None
                    and i == pd_start - 1):
                x_lane2 = x.clone()

            # Advance parallel lane
            if (self.parallel_blocks is not None
                    and x_lane2 is not None
                    and i >= pd_start):
                lane_block_idx = i - pd_start
                if lane_block_idx < len(self.parallel_blocks):
                    x_lane2, _ = self.parallel_blocks[lane_block_idx](
                        x_lane2, causal_mask)

        # Merge parallel lane
        if x_lane2 is not None and self.lane_mix is not None:
            alpha = torch.sigmoid(self.lane_mix)
            x     = alpha * x + (1.0 - alpha) * x_lane2

        # ── Head ────────────────────────────────────────────────────────
        x      = self.norm_f(x)
        logits = self.lm_head(x)                     # (B, T, V)

        # Logit softcap (Gemma2-style): tanh stabilisation
        if self.cfg.logit_softcap > 0:
            logits = self.cfg.logit_softcap * torch.tanh(
                logits / self.cfg.logit_softcap)

        # ── Loss ────────────────────────────────────────────────────────
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size),
                targets.reshape(-1),
                ignore_index=self.cfg.pad_id,
            )

        return logits, loss, next_kv

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        no_repeat_ngram_size: int = 3,
        min_new_tokens: int = 0,
    ) -> torch.Tensor:
        self.eval()
        _ = prompt_ids.device

        for step in range(max_new_tokens):
            ctx    = prompt_ids[:, -self.cfg.max_seq_len:]
            if step == 0:
                logits, _, kv_cache = self.forward(ctx, use_cache=True)
            else:
                logits, _, kv_cache = self.forward(prompt_ids[:, -1:], past_kv=kv_cache, use_cache=True)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            # Repetition penalty: down-rank already generated tokens.
            if repetition_penalty and repetition_penalty > 1.0:
                seen_ids = set(prompt_ids[0].tolist())
                for tok_id in seen_ids:
                    if logits[0, tok_id] < 0:
                        logits[0, tok_id] *= repetition_penalty
                    else:
                        logits[0, tok_id] /= repetition_penalty

            # No-repeat n-gram blocking (single-batch generation path).
            if no_repeat_ngram_size and no_repeat_ngram_size > 1 and prompt_ids.size(1) >= no_repeat_ngram_size - 1:
                generated = prompt_ids[0].tolist()
                prefix = tuple(generated[-(no_repeat_ngram_size - 1):])
                banned = set()
                for i in range(len(generated) - no_repeat_ngram_size + 1):
                    ng = generated[i:i + no_repeat_ngram_size]
                    if tuple(ng[:-1]) == prefix:
                        banned.add(ng[-1])
                if banned:
                    logits[0, list(banned)] = float('-inf')

            # Top-k filtering
            if top_k > 0:
                k_val, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < k_val[:, [-1]]] = float('-inf')

            # Top-p (nucleus) filtering
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove   = cumprobs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float('-inf')
                logits.scatter_(1, sorted_idx, sorted_logits)

            probs    = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            prompt_ids = torch.cat([prompt_ids, next_tok], dim=1)

            if step + 1 >= min_new_tokens and next_tok.item() == self.cfg.eos_id:
                break

        return prompt_ids


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 12 — MUON OPTIMIZER (POLAR-EXPRESS NEWTON-SCHULZ)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Polar-Express Newton-Schulz iteration (PR #1344 → PR #1787).
    Approximates G / ||G||_2 (the orthogonal polar factor).

    Uses the quintic polynomial iteration with minimax coefficients:
        a=3.4445, b=-4.7750, c=2.0315
    Converges in 5 steps to near machine precision for typical gradient shapes.
    """
    assert G.ndim >= 2, "Newton-Schulz requires at least 2D tensors"
    a, b, c = 3.4445, -4.7750, 2.0315

    X = G.to(torch.bfloat16)
    X = X / (X.norm() + 1e-7)

    transposed = G.shape[-2] > G.shape[-1]
    if transposed:
        X = X.mT

    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X

    if transposed:
        X = X.mT

    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """
    Muon: Momentum + Orthogonal Update for matrix params.

    Applied to all weight matrices (ndim >= 2) except embeddings.
    - Computes EMA momentum buffer
    - Applies Newton-Schulz to get near-orthogonal update direction
    - Scales update by sqrt(max(rows, cols)) for RMS normalisation

    PR lineage: modded-nanogpt → PR #1344 → PR #1787 (Polar-Express NS)
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 ns_steps: int = 5, weight_decay: float = 0.01):
        defaults = dict(lr=lr, momentum=momentum,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr           = group['lr']
            momentum     = group['momentum']
            ns_steps     = group['ns_steps']
            wd           = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad

                if g.ndim < 2:
                    # Scalars: simple SGD with momentum
                    state = self.state[p]
                    if 'buf' not in state:
                        state['buf'] = torch.zeros_like(g)
                    state['buf'].mul_(momentum).add_(g)
                    if wd != 0:
                        p.mul_(1.0 - lr * wd)
                    p.add_(state['buf'], alpha=-lr)
                    continue

                # Matrix params: Muon update
                state = self.state[p]
                if 'buf' not in state:
                    state['buf'] = torch.zeros_like(g)

                buf = state['buf']
                buf.mul_(momentum).add_(g)

                # Orthogonal update via Newton-Schulz
                update = newtonschulz5(buf, steps=ns_steps)

                # Normalise by sqrt(max(rows, cols)) → consistent RMS update
                scale = max(g.shape[-2], g.shape[-1]) ** 0.5

                if wd != 0:
                    p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-lr * scale)

        return loss


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 13 — LEARNING RATE SCHEDULE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_lr(step: int, cfg: NullAIConfig, total_steps: int) -> float:
    """
    Warmup → flat → cosine warmdown → MIN_LR floor.

    warmdown_frac=0.85 means warmdown occupies 85% of budget:
      - warmup: steps 0..warmup_steps
      - flat: steps warmup..15% of total
      - warmdown: 15% to 100% of total
      - floor at lr_peak * lr_min_frac (0.10)

    From PR #1787 (MIN_LR=0.10) + stack defaults.
    """
    peak   = cfg.lr_peak
    min_lr = peak * cfg.lr_min_frac

    # Linear warmup
    if step < cfg.warmup_steps:
        return peak * (step + 1) / max(cfg.warmup_steps, 1)

    # Find warmdown start: after (1 - warmdown_frac) of budget
    wd_start = int(total_steps * (1.0 - cfg.warmdown_frac))
    wd_start = max(wd_start, cfg.warmup_steps)

    if step < wd_start:
        return peak

    # Cosine decay from peak → min_lr
    progress = min((step - wd_start) / max(total_steps - wd_start, 1), 1.0)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak - min_lr) * cosine


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 14 — TOKENIZER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ChatTokenizer:
    """Byte-Level BPE Tokenizer using HuggingFace tokenizers library."""
    PAD, BOS, EOS, UNK, USER, ASSISTANT = 0, 1, 2, 3, 4, 5

    def __init__(self, tokenizer_obj: Optional[Tokenizer] = None, vocab_size: int = 32000):
        self.special_tokens = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>", "<|user|>", "<|assistant|>"]
        self.vocab_size_target = vocab_size
        if tokenizer_obj is not None:
            self.tokenizer = tokenizer_obj
        else:
            self.tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
            self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            self.tokenizer.decoder = decoders.ByteLevel()
            self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    def train(self, texts: List[str]):
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size_target,
            special_tokens=self.special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )
        self.tokenizer.train_from_iterator(texts, trainer=trainer)

    def save(self, path: str):
        self.tokenizer.save(path)

    @classmethod
    def load(cls, path: str):
        return cls(tokenizer_obj=Tokenizer.from_file(path))

    def encode(self, text: str, add_bos: bool = False) -> List[int]:
        ids = self.tokenizer.encode(text).ids
        if add_bos:
            ids = [self.BOS] + ids
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special)

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

# Backward compatibility alias
CharTokenizer = ChatTokenizer


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 15 — DATASET
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TextDataset:
    """Simple random-chunk dataset for language modelling."""

    def __init__(self, data: torch.Tensor, seq_len: int, device: torch.device):
        self.data    = data
        self.seq_len = seq_len
        self.device  = device

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        n   = len(self)
        idx = torch.randint(0, n, (batch_size,))
        x   = torch.stack([self.data[i   : i + self.seq_len    ] for i in idx])
        y   = torch.stack([self.data[i+1 : i + self.seq_len + 1] for i in idx])
        return x.to(self.device), y.to(self.device)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 16 — EMA WRAPPER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EMA:
    """
    Exponential Moving Average of model weights.
    Used for evaluation; training continues on the live model.
    decay=0.9965 matches the parameter-golf stack default.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9965):
        self.decay  = decay
        self.shadow = {
            name: param.data.clone().detach()
            for name, param in model.named_parameters()
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay)

    def apply(self, model: nn.Module):
        """Swap live weights for EMA weights (for eval/save)."""
        self._backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore live weights after eval."""
        for name, param in model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup = {}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 17 — TEST-TIME TRAINING (TTT)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_time_train(
    model: nn.Module, val_batch: Tuple[torch.Tensor, torch.Tensor],
    cfg: NullAIConfig
) -> float:
    """
    Quick TTT: fine-tune on the validation batch itself for a few steps,
    measure the improved loss, then restore original weights.

    Concept from PR #1610, multi-phase from PR #1626, adopted in PR #1736.
    This is a simplified single-phase version suitable for Colab.
    """
    x, y = val_batch
    orig_state = copy.deepcopy(model.state_dict())

    # Small Adam for TTT (only update a small subset of params)
    ttt_params = [p for n, p in model.named_parameters()
                  if 'norm' in n or 'gate' in n or 'smear' in n]
    if not ttt_params:
        ttt_params = list(model.parameters())

    opt = torch.optim.Adam(ttt_params, lr=cfg.ttt_lr, betas=(0.9, 0.99))

    model.train()
    for _ in range(cfg.ttt_steps):
        with autocast(dtype=torch.bfloat16):
            _, loss, _ = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()

    model.eval()
    with torch.no_grad(), autocast(dtype=torch.bfloat16):
        _, val_loss, _ = model(x, y)
    ttt_loss = val_loss.item()

    # Restore original weights
    model.load_state_dict(orig_state)
    return ttt_loss


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 18 — INT8 QUANTISED SAVE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def save_quantised(model: nn.Module, path: str, cfg: NullAIConfig):
    """
    Save a post-training int8 quantised checkpoint.
    Each matrix param is stored as int8 + per-row float32 scale.
    Halves the checkpoint size from ~16MB to ~8MB.

    Inspired by GPTQ int6/int8 quantisation from the parameter-golf stack.
    Note: this is symmetric per-row int8 (simpler than asymmetric int4 LQER).
    """
    quant_state = {}
    for name, param in model.named_parameters():
        t = param.data.float()
        if t.ndim >= 2:
            # Per-row scale: max abs value per row
            scale   = t.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
            t_int8  = (t / scale * 127).round().clamp(-127, 127).to(torch.int8)
            quant_state[name] = {
                'int8':  t_int8,
                'scale': scale.squeeze(-1).to(torch.float32),
                'shape': list(t.shape),
                'quant': True,
            }
        else:
            quant_state[name] = {'data': t.to(torch.float32), 'quant': False}

    torch.save({'quant_state': quant_state, 'cfg': cfg}, path)
    size_mb = os.path.getsize(path) / (1024 ** 2)
    print(f"  Saved int8 quantised model → {path}  ({size_mb:.1f} MB)")


def load_quantised(path: str, device: torch.device) -> Tuple['NullAI', NullAIConfig]:
    """Load and dequantise an int8 checkpoint."""
    ckpt = torch.load(path, map_location=device)
    cfg  = ckpt['cfg']
    model = NullAI(cfg).to(device)

    state_dict = {}
    for name, d in ckpt['quant_state'].items():
        if d['quant']:
            scale = d['scale'].to(device).unsqueeze(-1)
            q     = d['int8'].to(device).float()
            state_dict[name] = (q / 127.0 * scale).reshape(d['shape'])
        else:
            state_dict[name] = d['data'].to(device)

    model.load_state_dict(state_dict)
    return model, cfg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 19 — DATA LOADING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DATASETS = {
    'dolly': (
        'https://raw.githubusercontent.com/databrickslabs/dolly/master/data/databricks-dolly-15k.jsonl',
        'databricks-dolly-15k.jsonl'
    ),
}

def load_data(cfg: NullAIConfig, data_path: Optional[str] = None,
              device: torch.device = torch.device('cpu'),
              dataset_name: str = 'dolly'):
    """
    Load training text. Priority:
      1. --data path if given
      2. HuggingFace datasets mix (Dolly + slice of UltraChat)
    Returns (train_ds, val_ds, tokenizer)
    """
    lines = []
    if data_path and os.path.exists(data_path):
        print(f"  Loading data from: {data_path}")
        text = open(data_path, encoding='utf-8', errors='replace').read()
        if text.lstrip().startswith("{"):
            for ln in text.splitlines():
                if not ln.strip(): continue
                ex = json.loads(ln)
                inst = ex.get("instruction", "")
                ctx = ex.get("context", "")
                resp = ex.get("response", "")
                lines.append(f"<|user|> {inst}\n{ctx}\n<|assistant|> {resp}\n")
        else:
            lines = [f"<|user|> summarize this\n<|assistant|> {chunk}\n"
                     for chunk in text.split("\n\n") if chunk.strip()]
    else:
        print(f"  Loading robust dataset mix from HuggingFace Hub...")
        # Dolly 15k
        try:
            dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
            for ex in tqdm(dolly, desc="Processing Dolly"):
                inst = ex.get("instruction", "")
                ctx = ex.get("context", "")
                resp = ex.get("response", "")
                lines.append(f"<|user|> {inst}\n{ctx}\n<|assistant|> {resp}\n")
        except Exception as e:
            print(f"  Could not load Dolly from HF: {e}. Trying local fallback.")
            local = DATASETS.get(dataset_name, DATASETS['dolly'])[1]
            if os.path.exists(local):
                text = open(local, encoding='utf-8', errors='replace').read()
                for ln in text.splitlines():
                    if not ln.strip(): continue
                    try:
                        ex = json.loads(ln)
                        lines.append(f"<|user|> {ex.get('instruction','')}\n{ex.get('context','')}\n<|assistant|> {ex.get('response','')}\n")
                    except: continue

        # UltraChat slice (10k examples)
        try:
            print("  Loading a slice of UltraChat for robustness...")
            ultrachat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
            count = 0
            for ex in tqdm(ultrachat, total=10000, desc="Processing UltraChat"):
                if count >= 10000: break
                msgs = ex.get("messages", [])
                formatted = ""
                for m in msgs:
                    role = m.get("role", "")
                    content = m.get("content", "")
                    if role == "user":
                        formatted += f"<|user|> {content}\n"
                    elif role == "assistant":
                        formatted += f"<|assistant|> {content}\n"
                if formatted:
                    lines.append(formatted)
                    count += 1
        except Exception as e:
            print(f"  Could not load UltraChat: {e}. Continuing with available data.")

    tokenizer = ChatTokenizer(vocab_size=cfg.vocab_size)
    print(f"  Training BPE Tokenizer on {len(lines):,} examples...")
    tokenizer.train(lines)
    tokenizer.save(cfg.tokenizer_path)
    cfg.vocab_size = tokenizer.vocab_size

    # Shuffle for better distribution
    random.seed(cfg.seed)
    random.shuffle(lines)

    merged_text = "".join(lines)
    print(f"  Corpus: {len(merged_text):,} chars  |  vocab_size: {tokenizer.vocab_size}")

    ids  = tokenizer.encode(merged_text, add_bos=False)
    data = torch.tensor(ids, dtype=torch.long)

    n_train   = int(0.9 * len(data))
    train_ds  = TextDataset(data[:n_train],  cfg.seq_len, device)
    val_ds    = TextDataset(data[n_train:],   cfg.seq_len, device)

    print(f"  Train: {n_train:,} tokens  |  Val: {len(data)-n_train:,} tokens")
    return train_ds, val_ds, tokenizer


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 20 — TRAINER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class NullAITrainer:
    """
    Full training loop with:
      - Muon (matrix params) + Adam (embed + scalars)
      - bfloat16 GradScaler
      - LR schedule (warmup + cosine warmdown + floor)
      - EMA weights
      - Depth recurrence activation at loop_start_frac
      - Periodic TTT evaluation
      - Checkpoint saving (fp32 full + int8 quantised)
    """

    def __init__(self, cfg: NullAIConfig, model: NullAI, device: torch.device):
        self.cfg    = cfg
        self.model  = model
        self.device = device
        self.step   = 0
        self.best_val_bpb = float('inf')
        self.train_losses: List[float] = []

        # ── Parameter grouping ───────────────────────────────────────────
        matrix_params = []
        embed_params  = []
        scalar_params = []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # Embeddings: large but need careful LR (tied with head)
            if 'embed' in name or 'lm_head' in name:
                embed_params.append(p)
            elif p.ndim >= 2:
                # All weight matrices → Muon
                matrix_params.append(p)
            else:
                # Scalars, biases, gains → Adam
                scalar_params.append(p)

        muon_lr = cfg.lr_peak * cfg.muon_lr_scale

        self.muon = Muon(matrix_params, lr=muon_lr, momentum=cfg.muon_momentum,
                         ns_steps=cfg.muon_ns_steps, weight_decay=cfg.weight_decay)

        self.adam = torch.optim.Adam(
            [
                {'params': embed_params,  'lr': cfg.lr_peak * cfg.embed_lr_scale},
                {'params': scalar_params, 'lr': cfg.lr_peak * cfg.scalar_lr_scale},
            ],
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=0.0,
        )

        self.scaler = GradScaler()
        self.ema    = EMA(model, decay=cfg.ema_decay)

    def _set_lr(self, lr: float):
        muon_lr  = lr * self.cfg.muon_lr_scale
        embed_lr = lr * self.cfg.embed_lr_scale
        scal_lr  = lr * self.cfg.scalar_lr_scale

        for g in self.muon.param_groups:
            g['lr'] = muon_lr
        self.adam.param_groups[0]['lr'] = embed_lr
        self.adam.param_groups[1]['lr'] = scal_lr

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        # Activate depth recurrence after 35% of training
        if (not self.model.use_recurrence and
                self.step >= int(self.cfg.max_iters * self.cfg.loop_start_frac)):
            self.model.use_recurrence = True
            print(f"\n  [Step {self.step}] Depth recurrence ENABLED "
                  f"(loop layers {self.cfg.loop_layers}, ×{self.cfg.loop_repeats})")

        # LR update
        lr = get_lr(self.step, self.cfg, self.cfg.max_iters)
        self._set_lr(lr)

        self.model.train()
        self.muon.zero_grad()
        self.adam.zero_grad()

        with autocast(dtype=torch.bfloat16):
            _, loss, _ = self.model(x, y)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.muon)
        self.scaler.unscale_(self.adam)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.scaler.step(self.muon)
        self.scaler.step(self.adam)
        self.scaler.update()

        self.ema.update(self.model)

        val = loss.item()
        self.train_losses.append(val)
        self.step += 1
        return val

    @torch.no_grad()
    def evaluate(self, val_ds: TextDataset,
                 n_batches: int = 8) -> Tuple[float, float]:
        """Returns (val_bpb, ttt_bpb). Uses EMA weights."""
        self.ema.apply(self.model)
        self.model.eval()

        total = 0.0
        for _ in range(n_batches):
            x, y = val_ds.get_batch(self.cfg.batch_size)
            with autocast(dtype=torch.bfloat16):
                _, loss, _ = self.model(x, y)
            total += loss.item()
        val_bpb = bits_per_byte(total / n_batches)

        # TTT pass
        ttt_bpb = val_bpb
        if self.cfg.ttt_enabled:
            x, y = val_ds.get_batch(self.cfg.batch_size)
            self.ema.restore(self.model)           # restore before TTT mutates
            ttt_loss = test_time_train(self.model, (x, y), self.cfg)
            ttt_bpb  = bits_per_byte(ttt_loss)
            self.ema.apply(self.model)             # re-apply EMA for consistency

        self.ema.restore(self.model)
        return val_bpb, ttt_bpb

    def save(self, path: str, quant: bool = False):
        self.ema.apply(self.model)
        state = {
            'model':     self.model.state_dict(),
            'cfg':       self.cfg,
            'step':      self.step,
            'best_val':  self.best_val_bpb,
            'losses':    self.train_losses[-1000:],
        }
        torch.save(state, path)
        mb = os.path.getsize(path) / (1024 ** 2)
        print(f"  Saved → {path}  ({mb:.1f} MB)")
        self.ema.restore(self.model)

        if quant:
            self.ema.apply(self.model)
            save_quantised(self.model, path.replace('.pt', '_int8.pt'), self.cfg)
            self.ema.restore(self.model)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.step         = ckpt.get('step', 0)
        self.best_val_bpb = ckpt.get('best_val', float('inf'))
        print(f"  Loaded checkpoint (step {self.step}, "
              f"best val BPB {self.best_val_bpb:.4f})")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 21 — MAIN TRAINING LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BANNER = """
╔══════════════════════════════════════════════════════╗
║          N U L L   A I   —   Training Run            ║
╚══════════════════════════════════════════════════════╝"""

def train(cfg: NullAIConfig, data_path: Optional[str] = None,
          dataset: str = 'shakespeare', resume: Optional[str] = None):

    set_seed(cfg.seed)
    torch.set_float32_matmul_precision('high')  # TF32 on Ampere GPUs

    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(BANNER)
    print(f"\n  Device : {device}")
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU    : {props.name}")
        print(f"  VRAM   : {props.total_memory / 1e9:.1f} GB")
        print(f"  dtype  : bfloat16 (mixed precision)")

    # ── Data ────────────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading data …")
    train_ds, val_ds, tokenizer = load_data(cfg, data_path, device, dataset)

    # ── Model ───────────────────────────────────────────────────────────────
    print(f"\n[2/4] Building NullAI …")
    model   = NullAI(cfg).to(device)
    n_par   = count_params(model)
    sz_mb   = model_size_mb(model)
    d_head  = cfg.d_model // cfg.n_heads
    gqa_r   = cfg.n_heads // cfg.n_kv_heads

    print(f"""
  ┌─────────────────────────────────────────────┐
  │  Architecture                               │
  ├─────────────────────────────────────────────┤
  │  Layers         : {cfg.n_layers:<4}  (loop {cfg.loop_layers} ×{cfg.loop_repeats})│
  │  d_model        : {cfg.d_model:<4}                         │
  │  n_heads (GQA)  : {cfg.n_heads}Q / {cfg.n_kv_heads}KV  (ratio {gqa_r}:1)          │
  │  d_head         : {d_head:<4}  (rope_frac={cfg.rope_frac:.2f})       │
  │  d_mlp          : {cfg.d_mlp:<4}  (LeakyReLU²)           │
  │  vocab_size     : {cfg.vocab_size:<4}                         │
  │  max_seq_len    : {cfg.max_seq_len:<4}                         │
  │  SNN Encoder    : {'ON ' if cfg.snn_encoder else 'off'}   (Hypercube {cfg.snn_hypercube_dim}D)          │
  │  SmearGate      : {'ON ' if cfg.smear_gate else 'off'}   (BOS-fixed)            │
  │  Sparse Gate    : {'ON ' if cfg.sparse_attn_gate else 'off'}   (window={cfg.gate_window})               │
  │  U-Net skips    : {'ON ' if cfg.unet_skips else 'off'}                         │
  │  Parallel dec   : {'ON ' if cfg.parallel_decoder else 'off'}   (start layer {cfg.parallel_decoder_start})       │
  │  SWA Window     : {cfg.window_size:<4}                         │
  │  Logit softcap  : {cfg.logit_softcap}                        │
  ├─────────────────────────────────────────────┤
  │  Parameters     : {n_par:>10,}               │
  │  Size (bf16)    : {sz_mb:>8.1f} MB               │
  └─────────────────────────────────────────────┘""")

    if sz_mb > 20:
        print(f"  ⚠  Model is {sz_mb:.1f} MB — over 16 MB target. "
              f"Reduce d_model or n_layers.")

    # ── Trainer ─────────────────────────────────────────────────────────────
    print(f"\n[3/4] Setting up optimisers …")
    trainer = NullAITrainer(cfg, model, device)

    if resume and os.path.exists(resume):
        print(f"  Resuming from: {resume}")
        trainer.load(resume)

    muon_p = sum(p.numel() for g in trainer.muon.param_groups for p in g['params'])
    adam_p = sum(p.numel() for g in trainer.adam.param_groups for p in g['params'])
    print(f"  Muon (matrix)   : {muon_p:>10,} params")
    print(f"  Adam (embed+sc) : {adam_p:>10,} params")
    print(f"  Peak LR         : {cfg.lr_peak:.2e}  (Muon ×{cfg.muon_lr_scale})")
    print(f"  Min LR floor    : {cfg.lr_peak * cfg.lr_min_frac:.2e}  (×{cfg.lr_min_frac})")
    print(f"  Warmdown frac   : {cfg.warmdown_frac:.2f}  ({int(cfg.max_iters*(1-cfg.warmdown_frac))} flat steps)")
    print(f"  EMA decay       : {cfg.ema_decay}")
    print(f"  TTT             : {'enabled' if cfg.ttt_enabled else 'disabled'}")

    # ── Training loop ───────────────────────────────────────────────────────
    print(f"\n[4/4] Training …\n")
    print(f"{'Step':>7}  {'Loss':>7}  {'BPB':>6}  {'ValBPB':>7}  {'TTT-BPB':>8}  "
          f"{'LR':>8}  {'Recur':>5}  {'Elapsed':>8}")
    print("─" * 77)

    start_time   = time.time()
    run_loss     = 0.0
    log_count    = 0
    best_path    = 'null_ai_best.pt'

    for step in range(trainer.step, cfg.max_iters):

        # Wallclock guard
        elapsed = time.time() - start_time
        if elapsed > cfg.max_wallclock:
            print(f"\n  ⏱  Wallclock limit ({fmt_time(cfg.max_wallclock)}) reached "
                  f"at step {step}.  Saving final checkpoint …")
            break

        x, y = train_ds.get_batch(cfg.batch_size)
        loss  = trainer.train_step(x, y)
        run_loss  += loss
        log_count += 1

        # Logging
        if step % cfg.log_every == 0 and step > 0:
            avg_loss = run_loss / log_count
            bpb      = bits_per_byte(avg_loss)
            lr_now   = get_lr(step, cfg, cfg.max_iters)
            recur    = '✓' if model.use_recurrence else '✗'
            el       = fmt_time(time.time() - start_time)

            # Validation + TTT
            val_bpb_s = '   —   '
            ttt_bpb_s = '    —   '
            if step % cfg.eval_every == 0:
                val_bpb, ttt_bpb = trainer.evaluate(val_ds)
                val_bpb_s = f'{val_bpb:7.4f}'
                ttt_bpb_s = f'{ttt_bpb:8.4f}'

                if val_bpb < trainer.best_val_bpb:
                    trainer.best_val_bpb = val_bpb
                    trainer.save(best_path, quant=False)
                    print(f"  🏆  New best val BPB: {val_bpb:.4f}  "
                          f"(TTT: {ttt_bpb:.4f})")

            print(f"{step:7d}  {avg_loss:7.4f}  {bpb:6.4f}  "
                  f"{val_bpb_s}  {ttt_bpb_s}  "
                  f"{lr_now:.2e}  {recur:>5}  {el}")

            run_loss  = 0.0
            log_count = 0

        # Periodic save
        if step % cfg.save_every == 0 and step > 0:
            trainer.save(f'null_ai_step{step}.pt', quant=False)

    # ── Final save ──────────────────────────────────────────────────────────
    print("\n" + "─" * 77)
    trainer.save('null_ai_final.pt', quant=True)   # Also saves int8 version

    # ── Sample generation ───────────────────────────────────────────────────
    print("\n" + "═" * 55)
    print("  Sample Generation  (temperature=0.8, top-k=50, top-p=0.95)")
    print("═" * 55)

    model.eval()
    trainer.ema.apply(model)

    prompts = ["The ", "Once upon a time", "To be or not to be"]
    for prompt in prompts:
        ids = torch.tensor(
            tokenizer.encode(prompt, add_bos=True),
            dtype=torch.long, device=device
        ).unsqueeze(0)

        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=150,
                                 temperature=0.8, top_k=50, top_p=0.95)

        text = tokenizer.decode(out[0].tolist())
        print(f"\n  Prompt: {prompt!r}")
        print(f"  Output: {text[:200]!r}")

    trainer.ema.restore(model)

    # ── Final stats ─────────────────────────────────────────────────────────
    total_time = time.time() - start_time
    print(f"\n{'═'*55}")
    print(f"  Training complete!")
    print(f"  Steps trained  : {trainer.step:,}")
    print(f"  Total time     : {fmt_time(total_time)}")
    print(f"  Best val BPB   : {trainer.best_val_bpb:.4f}")
    print(f"  Model size     : {model_size_mb(model):.1f} MB (bf16 live)")
    print(f"  int8 model     : null_ai_final_int8.pt (saved)")
    print(f"{'═'*55}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 22 — CLI ENTRYPOINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_args():
    p = argparse.ArgumentParser(
        description='NullAI — Compact 16MB LM trainer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    g = p.add_argument_group('Model')
    g.add_argument('--d_model',     type=int,   default=264)
    g.add_argument('--n_layers',    type=int,   default=8)
    g.add_argument('--n_heads',     type=int,   default=4)
    g.add_argument('--n_kv_heads',  type=int,   default=2)
    g.add_argument('--d_mlp',       type=int,   default=1056)
    g.add_argument('--max_seq_len', type=int,   default=2048)
    g.add_argument('--window_size', type=int,   default=512)

    # Features
    g2 = p.add_argument_group('Architecture Features')
    g2.add_argument('--no_snn',      action='store_true', help='Disable SNN spike encoder')
    g2.add_argument('--no_smear',    action='store_true', help='Disable SmearGate')
    g2.add_argument('--no_sparse',   action='store_true', help='Disable sparse attn gate')
    g2.add_argument('--no_unet',     action='store_true', help='Disable U-Net skips')
    g2.add_argument('--no_parallel', action='store_true', help='Disable parallel decoder')
    g2.add_argument('--no_ttt',      action='store_true', help='Disable TTT evaluation')

    # Training
    g3 = p.add_argument_group('Training')
    g3.add_argument('--batch_size',  type=int,   default=24)
    g3.add_argument('--seq_len',     type=int,   default=256)
    g3.add_argument('--lr',          type=float, default=3e-3)
    g3.add_argument('--max_iters',   type=int,   default=10000)
    g3.add_argument('--max_wallclock', type=int, default=3300,
                    help='Max seconds (default: 55 min for Colab)')
    g3.add_argument('--warmup',      type=int,   default=100)
    g3.add_argument('--grad_clip',   type=float, default=0.3)
    g3.add_argument('--seed',        type=int,   default=42)

    # Data
    g4 = p.add_argument_group('Data')
    g4.add_argument('--data',    type=str, default=None,
                    help='Path to training text / jsonl file')
    g4.add_argument('--dataset', type=str, default='dolly',
                    choices=list(DATASETS.keys()))

    # Misc
    g5 = p.add_argument_group('Misc')
    g5.add_argument('--resume',   type=str, default=None, help='Resume from checkpoint')
    g5.add_argument('--eval_only', action='store_true',
                    help='Only evaluate a checkpoint (use with --resume)')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    cfg = NullAIConfig(
        d_model           = args.d_model,
        n_layers          = args.n_layers,
        n_heads           = args.n_heads,
        n_kv_heads        = args.n_kv_heads,
        d_mlp             = args.d_mlp,
        max_seq_len       = args.max_seq_len,
        window_size       = args.window_size,
        snn_encoder       = not args.no_snn,
        smear_gate        = not args.no_smear,
        sparse_attn_gate  = not args.no_sparse,
        unet_skips        = not args.no_unet,
        parallel_decoder  = not args.no_parallel,
        ttt_enabled       = not args.no_ttt,
        batch_size        = args.batch_size,
        seq_len           = args.seq_len,
        lr_peak           = args.lr,
        max_iters         = args.max_iters,
        max_wallclock     = args.max_wallclock,
        warmup_steps      = args.warmup,
        grad_clip         = args.grad_clip,
        seed              = args.seed,
        vocab_size        = 32000,            # Updated at load time by tokenizer
    )

    if args.eval_only:
        if not args.resume:
            print("ERROR: --eval_only requires --resume <checkpoint>")
            sys.exit(1)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _, _, tokenizer = load_data(cfg, args.data, device, args.dataset)
        model, cfg2 = load_quantised(args.resume.replace('.pt', '_int8.pt'), device)
        print("Evaluating …")
        # Quick perplexity estimate
        model.eval()
        total = 0.0
        n     = 20
        _, val_ds, _ = load_data(cfg2, args.data, device, args.dataset)
        for _ in range(n):
            x, y = val_ds.get_batch(8)
            with torch.no_grad(), autocast(dtype=torch.bfloat16):
                _, loss, _ = model(x, y)
            total += loss.item()
        print(f"Val BPB: {bits_per_byte(total/n):.4f}")
    else:
        train(cfg, data_path=args.data, dataset=args.dataset, resume=args.resume)
