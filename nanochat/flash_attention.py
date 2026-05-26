"""
Unified Flash Attention interface with automatic FA3/SDPA switching.

Exports `flash_attn` module that matches the FA3 API exactly, but falls back
to PyTorch SDPA on non-Hopper GPUs (including Blackwell), MPS, and CPU.

Usage (drop-in replacement for FA3):
    from nanochat.flash_attention import flash_attn

    # Training (no KV cache)
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Inference (with KV cache)
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""
import torch
import torch.nn.functional as F


# =============================================================================
# Detection: Try to load FA3 on Hopper+ GPUs
# =============================================================================
def _load_flash_attention_3():
    """Try to load Flash Attention 3 (requires Hopper GPU, sm90)."""
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        # FA3 kernels are compiled for Hopper (sm90) only
        # Ada (sm89), Blackwell (sm100) need SDPA fallback until FA3 is recompiled
        if major != 9:
            return None
        import os
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        return get_kernel('varunneal/flash-attention-3').flash_attn_interface
    except Exception:
        return None


_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None

# Override for testing: set to 'fa3', 'sdpa', or None (auto)
_override_impl = None


def _resolve_use_fa3():
    """Decide once whether to use FA3, based on availability, override, and dtype."""
    if _override_impl == 'fa3':
        assert HAS_FA3, "Cannot override to FA3: not available on this hardware"
        return True
    if _override_impl == 'sdpa':
        return False
    if HAS_FA3:
        # FA3 Hopper kernels only support bf16 and fp8; fp16/fp32 must use SDPA fallback
        from nanochat.common import COMPUTE_DTYPE
        if COMPUTE_DTYPE == torch.bfloat16:
            return True
        return False
    return False

USE_FA3 = _resolve_use_fa3()


# =============================================================================
# SDPA helpers
# SDPA = Scaled Dot-Product Attention，即标准注意力: softmax(Q·K^T / sqrt(d_k)) · V
# - Scaled: 注意力得分除以 sqrt(d_k)，防止维度大时 softmax 梯度消失
# - Dot-Product: 用 Q·K^T（点积）计算注意力得分
# - Attention: 注意力机制
# 对应 PyTorch 内置函数 F.scaled_dot_product_attention()，是 PyTorch 2.0+ 的融合注意力实现。
# 在本项目中作为 FA3 的 fallback：FA3 仅支持 Hopper GPU(SM 9.0)，SDPA 支持所有 GPU。
# =============================================================================
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    Tq = q.size(2)  # query 的序列长度，维度索引 2 = T（如训练时 Tq=2048，推理时 Tq=1）
    Tk = k.size(2)  # key 的序列长度，训练时 Tk=Tq，推理时 Tk=已缓存的历史长度
    window = window_size[0]  # window_size 是 (left, right) 元组，[0] 取左侧窗口大小；-1 表示无限（full context）

    # Full context, same length — 最简单的场景：标准训练时的 causal attention
    # 条件1: window < 0 或 window >= Tq → 窗口为无限或大于序列长度，等于没有滑动窗口限制
    # 条件2: Tq == Tk → query 和 key 长度相同，即训练时整个序列一起计算
    # 两个条件同时满足 → 直接用标准 causal SDPA，无需额外 mask
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single token generation — 推理时逐 token 生成，Tq=1（只有 1 个新 query token）
    # 此时 Tq != Tk（Tk 是已缓存的历史长度），不能用 is_causal，需要显式处理窗口
    if Tq == 1:
        # 如果窗口有限且小于已缓存长度，只取最近 window+1 个 key/value（滑动窗口裁剪）
        if window >= 0 and window < Tk:
            # window is "left" tokens we need to include (window + 1) keys total
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        # is_causal=False: 单 token 对完整历史不需要 causal mask（历史 key 本身就按时间排列）
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Need explicit mask for sliding window/chunk inference — 最复杂的场景
    # 场景: Tq > 1 且 Tq != Tk（prefill 推理: 输入一段新 token，对已有 cache 做 attention）
    # 此时不能用 is_causal=True（因为 cache 前面已有历史 token，causal mask 的对齐位置不对）
    # 也不能走上面 Tq==1 的简单路径（因为 Tq > 1，有多个新 query token 之间也有 causal 关系）
    # 所以需要手动构建一个 bool mask 来同时处理 causal + 滑动窗口
    #
    # 维度理解: 进入 _sdpa_attention() 时，q 只有新来的 Tq 个 token，而 k/v 是完整的 Tk 个 token
    #   q.shape = (B, H, Tq, D)       ← 例如 (B, H, 5, D)
    #   k.shape = (B, H, Tk, D)       ← 例如 (B, H, 1005, D)，包含历史cache+新插入
    #   v.shape = (B, H, Tk, D)       ← 同上
    #   attention score = q @ k^T → 形状 (Tq, Tk)，即 (5, 1005)
    #   因此 mask.shape 也必须是 (Tq, Tk) = (5, 1005)，和 attention score 对齐
    device = q.device
    # row_idx 是每个 query token 在完整序列中的绝对位置（偏移 Tk-Tq 是因为前面已有 Tk-Tq 个历史 token）
    # 例: cache 已有 1000 个 token，新输入 5 个 token → row_idx = [1000, 1001, 1002, 1003, 1004]
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    # col_idx 是每个 key token 的绝对位置
    # 例: Tk=1005 → col_idx = [0, 1, 2, ..., 1004]
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    # causal mask: key 的位置必须 <= query 的位置（不能"看到未来"）
    # 例: row_idx=1001, col_idx=500 → 500<=1001 ✓ 可以 attend
    #     row_idx=1001, col_idx=1002 → 1002<=1001 ✗ 不能 attend（未来的 token）
    mask = col_idx <= row_idx

    # sliding window (left): query 只能 attend 到距离不超过 window 的 key
    # 例: window=512, row_idx=1001 → 只能 attend 到 col_idx >= 1001-512=489 的 key
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)

    # 最终 mask 同时满足 causal + sliding window 两个条件
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)

# =============================================================================
# Public API: Same interface as FA3
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    if USE_FA3:
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)  # back to (B, T, H, D)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.

    FA3 updates k_cache/v_cache in-place. Our SDPA fallback does the same.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    if USE_FA3:
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )

    # SDPA fallback: manually manage KV cache
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()  # assume uniform position across batch

    # Insert new k, v into cache (in-place, matching FA3 behavior)
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    # Get full cache up to current position + new tokens
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    # Transpose to SDPA layout: (B, T, H, D) -> (B, H, T, D)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)

    return y_sdpa.transpose(1, 2)  # back to (B, T, H, D)


# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA3)
# =============================================================================
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
