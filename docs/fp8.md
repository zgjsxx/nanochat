# FP8 训练加速说明

本文档介绍 nanochat 项目的 FP8（8 位浮点）训练功能，核心代码位于 `nanochat/fp8.py`，使用入口在 `scripts/base_train.py`。

## 什么是 FP8

FP8 是一种 8 位浮点数格式（E4M3 或 E5M2），相比 BF16/FP16 只有 8 位存储，可显著减少显存占用并加速矩阵乘法。本项目实现的 FP8 训练是 torchao `Float8Linear` 的精简替代方案。

## 核心原理：动态量化

FP8 训练的关键步骤是**动态量化**——每次前向传播时将高精度权重/激活值实时量化为 FP8，再通过 FP8 矩阵乘法完成计算。量化函数 `_to_fp8` 的流程如下：

```
1. 计算 scale = fp8_max / amax   → 将原始数值范围映射到 FP8 可表示范围
2. x_scaled = x * scale           → 放大数值到 FP8 范围内
3. x_clamped = clamp(x_scaled)    → 截断溢出值（饱和而非循环溢出）
4. x_fp8 = x_clamped.to(fp8_dtype) → 转为 FP8 类型
5. 返回 inv_scale = 1 / scale     → 逆缩放因子，矩阵乘法后用于还原数值
```

直观类比：`scale` 相当于地图的比例尺（把真实距离映射到像素），`inv_scale` 是从像素还原回真实距离。整个流程是 **放大 → 截断 → FP8 存储 → 矩阵乘法 → 缩小还原**。

## Tensorwise vs Rowwise

本项目仅支持 **tensorwise** 缩放策略：

| 策略 | 说明 | 性能 |
|------|------|------|
| **Tensorwise** | 整个张量共用一个 scale | 快——cuBLAS 自动处理缩放 |
| **Rowwise** | 每行一个 scale，精度更高 | 慢——需要 CUTLASS kernel |

虽然 CLI 接受 `--fp8-recipe rowwise` 参数，但运行时会抛出 `ValueError`，因为 `Float8LinearConfig.from_recipe_name` 仅支持 `"tensorwise"`。

## 使用方式

### 命令行参数

在 `scripts/base_train.py` 中通过两个参数控制：

- `--fp8`：开启 FP8 训练（默认关闭）
- `--fp8-recipe`：量化策略，默认 `"tensorwise"`

### 层过滤规则

并非所有 `nn.Linear` 都会被替换为 `Float8Linear`，只有满足以下条件的层才会：

- `in_features` 和 `out_features` 都是 16 的倍数（硬件要求）
- `min(in_features, out_features) >= 128`（小层用 FP8 反而更慢）

不满足条件的层保持 BF16 计算。

### 硬件要求

- 必须是 **CUDA 设备**（H100 及以上 GPU 才支持 FP8 matmul）
- 非 CUDA 设备上 `--fp8` 参数会被忽略并打印警告

### 评估时自动切换回 BF16

训练用 FP8 加速，但评估（validation、CORE metric、文本采样）时临时切换回 BF16 以保证精度：

```python
with disable_fp8(model):
    # 临时把 Float8Linear 换回普通 Linear（BF16）
    # 评估完毕后自动恢复 FP8
```

`disable_fp8` 的实现细节：
- 用 `device="meta"` 创建替换模块，避免 VRAM 峰值
- 共享 weight/bias 张量（不复制数据）
- `finally` 块中自动恢复所有 `Float8Linear`

## 实测效果

根据开发日志（`dev/LOG.md`）的实测数据：

| 模型规模 | FP8 步速度提升 | 等精度下有效提升 |
|----------|---------------|-----------------|
| d26+ | ~17% | ~5% |
| d12 | 反而更慢 | 不建议使用 |

较大模型（d24/d26）从 FP8 获益明显，小模型的 FP8 开销超过收益。生产训练脚本 `runs/speedrun.sh` 在 d24/d26 级别启用 `--fp8`。

## 公开 API

`nanochat/fp8.py` 导出的接口：

| API | 说明 |
|-----|------|
| `Float8Linear(nn.Linear)` | `nn.Linear` 的 FP8 替代，路由 matmul 通过 `_Float8Matmul` |
| `Float8Linear.from_float(mod)` | 从已有 `nn.Linear` 创建 `Float8Linear`，共享 weight/bias |
| `Float8LinearConfig` | 配置类（API 兼容 torchao），仅支持 tensorwise |
| `convert_to_float8_training(module, *, config, module_filter_fn)` | 遍历模型树，将符合条件的 `nn.Linear` 替换为 `Float8Linear` |
| `_Float8Matmul` (autograd Function) | FP8 矩阵乘法的自定义前向/反向传播 |
| `_to_fp8(x, fp8_dtype)` | 内部量化函数，返回 `(fp8_data, inverse_scale)` |

## 不使用 FP8 的场景

以下脚本始终使用 BF16，不涉及 FP8：

- `scripts/chat_sft.py` — SFT 训练
- `scripts/base_eval.py` — 基础评估
- `scripts/chat_eval.py` — 聊天评估
- 其他所有 `scripts/` 下的脚本

## 与 torchao 的关系

本实现是 torchao `Float8Linear` 的**精简替代**，`base_train.py` 中保留了 torchao 的导入注释：

```python
# from torchao.float8 import Float8LinearConfig, convert_to_float8_training
```

如需切换回 torchao 实现，取消注释并注释掉 `nanochat.fp8` 的导入即可。