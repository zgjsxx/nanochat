"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"


def norm(x):
    # x 的形状为 (B, T, n_embd)，维度 0=B(batch), 1=T(序列长度), 2=n_embd(嵌入维度)
    # RMSNorm 计算过程（无 learnable 参数，无 bias）:
    #   1. 对每个向量计算 RMS = sqrt(mean(x_i^2))       — 在最后一个维度 n_embd 上求均值
    #   2. 输出 = x / RMS                                 — 逐元素除法
    #   例: x = [3.0, 1.0, 0.0, 2.0]
    #       mean(x^2) = (9+1+0+4)/4 = 3.5
    #       RMS = sqrt(3.5) ≈ 1.87
    #       输出 = [3.0/1.87, 1.0/1.87, 0.0/1.87, 2.0/1.87] ≈ [1.60, 0.53, 0.0, 1.07]
    #       与 LayerNorm 的区别: LayerNorm 先减均值再除标准差，RMSNorm 不减均值
    # F.rms_norm(x, (x.size(-1),)) 第二个参数指定归一化的维度形状，这里就是 (n_embd,)
    # note that this will run in bf16, seems ok
    return F.rms_norm(x, (x.size(-1),))

class Linear(nn.Linear):
    """自定义 Linear：forward 时将 weight 临时 cast 到输入 x 的 dtype 来加速矩阵乘法。
    替代 autocast：master weights 保持 fp32 用于 optimizer 更新精度，
    但矩阵乘法在激活值的 dtype 下运行。

    x.dtype 由 COMPUTE_DTYPE 决定，分三种情况：
    - Ampere+ GPU (SM 8.0+):  x.dtype = bf16, weight 临时 cast 到 bf16 → 利用 tensor core 加速
    - 旧 GPU (SM < 8.0):     x.dtype = fp32, weight 本身就是 fp32, .to() 是无操作 → 纯 fp32 训练
    - 手动指定 float16:       x.dtype = fp16, weight 临时 cast 到 fp16 → 利用旧 GPU 的 fp16 tensor core（需 GradScaler）"""
    def forward(self, x):
        # self.weight 本身是 fp32，.to(dtype=x.dtype) 临时转成 x 的 dtype 做矩阵乘法
        # 具体转成什么取决于 COMPUTE_DTYPE：bf16 / fp16 / fp32（见类 docstring）
        # .to() 返回新张量，不修改原始 fp32 weight，optimizer 更新时仍用 fp32 保证精度
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included).

    逻辑: layer_idx 与最后一层 (n_layer-1) 奇偶性相同 → True，不同 → False。
    效果: 交替分配 VE，且最后一层一定有 VE。

    例1: n_layer=12, 最后一层 index=11(奇数)
      偶数层(0,2,4,6,8)  → False, 无 VE
      奇数层(1,3,5,7,9,11) → True,  有 VE  ← 最后一层 11 一定包含

    例2: n_layer=6, 最后一层 index=5(奇数)
      奇数层(1,3,5) → True, 有 VE
      偶数层(0,2,4) → False

    例3: n_layer=13, 最后一层 index=12(偶数)
      偶数层(0,2,4,6,8,10,12) → True, 有 VE  ← 最后一层 12 一定包含
      奇数层(1,3,5,7,9,11)    → False
    """
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    """对 Q/K 应用旋转位置编码（Rotary Position Embedding, RoPE）。

    RoPE 的核心思想：通过旋转矩阵让同一位置的两个维度互相混合，
    使得内积 q·k 中自然包含相对位置信息（位置差 m-n 决定旋转角度差）。

    数学上等价于对每对维度 (x_i, x_{i+d}) 应用 2D 旋转：
      [cos θ,  sin θ]   [x1]   [x1·cos + x2·sin]
      [-sin θ, cos θ] × [x2] = [-x1·sin + x2·cos]

    θ 由 token 位置和维度索引决定，不同维度对有不同的频率（低维高频，高维低频）。
    cos/sin 是向量（长度=D/2），每个维度对有自己的 θ，所以代码中逐元素乘法（element-wise）
    相当于同时对所有维度对做旋转。

    例: head_dim=4, 位置 pos=1, 序列长度=2 (pos=0 和 pos=1)
      x (某个 head 的 Q/K 向量) = [1.0, 2.0, 3.0, 4.0]
        拆分: x1=[1.0, 2.0]  x2=[3.0, 4.0]

      cos/sin 是向量，每个维度对有自己的 θ（频率不同）:
        维度对0 的 θ₀ = pos × 1/10000^(0/4) = 1.0          → cos₀=0.54, sin₀=0.84  ← 高频，旋转多
        维度对1 的 θ₁ = pos × 1/10000^(2/4) = 0.01         → cos₁=1.00, sin₁=0.01  ← 低频，几乎不旋转

      所以在 pos=1 时:
        cos = [0.54, 1.00]   ← 向量，两个维度对的 cos 值不同
        sin = [0.84, 0.01]   ← 向量，两个维度对的 sin 值不同

      逐元素乘法（每个维度对独立旋转）:
        y1 = x1 * cos + x2 * sin = [1.0×0.54 + 3.0×0.84,  2.0×1.00 + 4.0×0.01] = [3.06, 2.04]
        y2 = x1 * (-sin) + x2 * cos = [-1.0×0.84 + 3.0×0.54,  -2.0×0.01 + 4.0×1.00] = [0.78, 3.98]

      对比 pos=0 时（θ=0, cos=[1,1], sin=[0,0], 不旋转）:
        y1 = x1 * [1,1] + x2 * [0,0] = [1.0, 2.0]  ← 原样不变
        y2 = x1 * [0,0] + x2 * [1,1] = [3.0, 4.0]  ← 原样不变
        输出 = [1.0, 2.0, 3.0, 4.0] = x  ← pos=0 不旋转

      pos=1 的输出 = [3.06, 2.04, 0.78, 3.98]  ← 已编码了位置1的信息

    实际形状: x=(B,T,H,D), cos/sin=(1,T,1,D/2), 自动广播到所有 batch 和 head。
    """
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    """因果自注意力，支持 Grouped-Query Attention (GQA)。

    GQA 核心思想：多个 Q head 共享同一个 K/V head，减少 KV 参数和推理时的 KV cache 内存。
    三种模式：
      - MHA: n_kv_head == n_head，每个 Q 有独立的 K/V（标准多头注意力）
      - GQA: n_kv_head < n_head，多个 Q 共享一个 K/V（折中方案，如 n_head=6, n_kv_head=2）
      - MQA: n_kv_head == 1，所有 Q 共享一个 K/V（最激进，内存最少但质量略差）

    例: n_head=6, n_kv_head=2:
      Q heads:  q0  q1  q2  q3  q4  q5    ← 6 个独立的 Q
      K/V heads: k0/v0          k1/v1      ← 2 个共享的 K/V
      分组: q0,q1,q2 → k0,v0    q3,q4,q5 → k1,v1
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head          # Q 的头数
        self.n_kv_head = config.n_kv_head    # K/V 的头数（GQA 时小于 n_head）
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head  # 每个头的维度，所有头共用同一个 head_dim
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0  # 保证 n_kv_head 能均匀分组
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)       # Q 投影: n_embd → n_head × head_dim = n_embd
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)    # K 投影: n_embd → n_kv_head × head_dim（GQA 时输出更小）
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)    # V 投影: 同 K
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)                    # 输出投影: n_embd → n_embd
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm
        # QK norm 后向量被归一化到单位长度，attention logits 会偏小 → softmax 分布偏平坦（接近均匀注意力）。
        # 乘以 1.2 放大 Q 和 K 的幅度，使 attention logits 变大 → softmax 分布更尖锐（更集中在重要 token）。
        # 注意力得分 = Q·K^T，Q 和 K 都乘 1.2 → 总缩放 = 1.2² = 1.44。
        # "split scale between Q and K": 把 1.44 的缩放拆分到 Q 和 K 各乘 1.2，而不是只对其中一个乘 1.44，
        # 这样 Q 和 K 的幅度保持对称，数值更稳定。
        # TODO: 1.2 是经验值，最优值还需进一步验证。
        q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        k = k * 1.2

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        #
        # 训练 vs 推理的 attention 计算方式不同：
        #   训练: 整个序列一次性处理，所有 token 的 K/V 同时计算，不需要缓存。
        #         输入 (B, T) 序列，K/V 形状为 (B, T, n_kv_head, head_dim)，包含所有位置的 K/V。
        #         每步计算量 O(T²)，但一次完成，无重复计算。
        #   推理: 自回归逐 token 生成，每步只输入 1 个新 token (B, 1)。
        #         如果每步都重算之前所有 token 的 K/V → 重复计算，极度浪费。
        #         kv_cache 把历史 K/V 存下来，新 token 只算自己的 K/V 然后拼到 cache 里。
        #         每步计算量 O(T)（T 为当前序列长度），但避免了重算历史 K/V。
        #         推理调用链: GPT.generate() → GPT.forward(idx, kv_cache=kv_cache_obj)
        #         训练调用链: train loop → GPT.forward(idx, targets=targets) → kv_cache=None（默认值）
        if kv_cache is None:
            # Training: 整个序列的 K/V 同时计算，causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: 从 cache 读取历史 K/V，只新算当前 token 的 K/V，拼入 cache
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # 将 vocab_size 向上取整到 pad_vocab_size_to(64) 的倍数，纯粹是效率优化，不影响模型行为
        # 原因：GPU Tensor Core 要求维度是 64 的倍数才能高效运算；DDP 分布式训练也需要 vocab 维度对齐
        # ((x + n - 1) // n) * n 等价于 ceil(x / n) * n 的纯整数写法，向上取整到 n 的倍数
        # 输出的 logits 会切片回原始 vocab_size，所以 padding 对模型行为无影响
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout: subtract cached mid-layer residual before final norm to remove low-level features
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)  # 0.4x init scale for c_fc
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        # Per-layer resid init: stronger residual at early layers, weaker at deep layers
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        # Decaying x0 init: earlier layers get more input embedding blending
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init with small positive values so gates start slightly above neutral
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (quarter context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        # long_window: 完整上下文长度（全注意力窗口）
        long_window = config.sequence_len
        # short_window: 1/4 上下文长度（滑动窗口），向上取整到 128 的倍数（FA3 tile size 要求）
        # -(-a // b) 等价于 ceil(a / b) 的纯整数写法：先取反让 floor 变成 ceil，再取反还原符号
        # 例：sequence_len=2048 → 2048//4=512 → -(-512//128)*128 = 4*128 = 512
        short_window = -(-long_window // 4 // 128) * 128
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                          self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(smear_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Embed the tokens
        x = self.transformer.wte(idx) # embed current token
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        x = norm(x)

        # Smear: mix previous token's embedding into current position (cheap bigram info)
        if kv_cache is None:
            # Training / naive generate: full sequence available, use fast slice
            assert T > 1, "Training forward pass should have T > 1"
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            # KV cache inference: read prev embedding from cache, store current for next step
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                # Prefill: apply smear to positions 1+, same as training
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                # Decode: single token, use cached prev embedding
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        # Forward the trunk of the Transformer
        x0 = x  # 保存初始嵌入，全程不变，作为每层的"记忆锚点"
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2  # 在中间层缓存残差，用于最终去除低层特征
        x_backout = None
        for i, block in enumerate(self.transformer.h):
            # 残流调节：x 逐层更新，x0 是原始嵌入（不变）
            # resid_lambdas[i] 缩放当前残差流幅度，防止深层残差爆炸/衰减
            # x0_lambdas[i] 把原始嵌入回混到每层，让深层也能"回头看"初始 token 信息
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
            if i == backout_layer:
                x_backout = x
        # Subtract mid-layer residual to remove low-level features before logit projection
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference: just return the logits directly
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
