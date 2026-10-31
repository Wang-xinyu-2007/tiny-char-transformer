# Tiny Char Transformer

一个用 PyTorch 直接实现的 GPT 风格字符级 Transformer：从 token/位置嵌入、因果多头注意力、Pre-LN 残差块到自回归采样，每一层都能在代码里对上公式。**不加载预训练权重，也不调用 `nn.TransformerEncoder` 之类的黑盒。**

[中文](README.md) | [English](README.en.md)

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c)
![Tests](https://github.com/Wang-xinyu-2007/tiny-char-transformer/actions/workflows/tests.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 为什么写这个

学线性回归时，把公式抄进 NumPy 只能算第一步。Transformer 的注意力、残差、归一化和优化器之间有大量工程细节，只看教程很容易产生“我懂了”的错觉。

这个项目的目标是：**用一个小到能在 CPU 上训练、又完整到能讲清楚每个设计取舍的模型，把“公式 → 张量形状 → 训练 → 评测 → 采样”整条链路跑通。**

它不是一个要刷榜的模型，也不假装是生产级 LLM。它是一份可以被审阅、被复现、被追问的实现。

> 这里的“从零”指**不加载预训练权重、不调用高层 Transformer 模块**；线性层、LayerNorm 和自动求导仍由 PyTorch 提供。注意力、mask、softmax、残差连接和训练循环是显式写出来的。

---

## 结果

在 Tiny Shakespeare（1,115,394 个字符，65 个字符的 vocab）上训练 3,000 步：

| 指标 | 数值 |
| --- | --- |
| 参数量 | 818,048（含 bias / LayerNorm） |
| 层数 / 头数 / 隐藏维度 | 4 / 4 / 128 |
| 上下文长度 | 128 |
| batch size | 32 |
| 训练步数 | 3,000 |
| 最佳验证损失 | 1.8988 |
| 验证困惑度 | 6.68 |
| 训练设备 | CPU（Intel Core Ultra 5 225H，无 GPU） |
| 训练耗时 | 约 15.7 分钟 |

训练曲线：[`results/loss_curve.png`](results/loss_curve.png) · 指标：[`results/metrics.json`](results/metrics.json) · 采样：[`results/samples.txt`](results/samples.txt)

同模型、同数据下，1,200 步的基线（[`results/runs/1200-steps.json`](results/runs/1200-steps.json)）最佳验证损失为 2.2156、困惑度 9.17；继续训练到 3,000 步把验证损失降到 1.8988。这个对比只说明“训练更久在这个小模型上仍有收益”，不是严格的 scaling law 实验。

生成示例（prompt = `ROMEO:`，temperature = 0.8，top-k = 40，截取前 300 个字符）：

```text
ROMEO:
As good can the rich in the what nevers a like.

REOMENENTERS:
Why such were a creation a mores you
Parses as the a the was be land cand spee;
Will undicers the blising me as my sect,
The make to heads may this brabe well the dade
proce tus my doser of where I you my but heaven.

MENER:
Mast peary tan hear, him my cher head down,
And loownar give the with by whath call to me thou calll pay;
My an
```

模型很小，样本会有拼写和逻辑错误；这里展示的是**可复现的训练结果**，不是聊天能力。

---

## 快速开始

```bash
git clone https://github.com/Wang-xinyu-2007/tiny-char-transformer.git
cd tiny-char-transformer

python -m pip install -r requirements.txt

# 首次运行会自动下载 Tiny Shakespeare 并校验 SHA-256
python train.py

# 用最佳 checkpoint 续写
python generate.py --prompt "ROMEO:" --tokens 300

# 跑测试
python -m pytest -q
```

只想快速验证流程，可以只跑 50 步：

```bash
python train.py --max_steps 50 --eval_interval 25 --eval_iters 5
```

默认配置在普通 CPU 上大约需要 16 分钟；脚本会保存最佳验证损失对应的 checkpoint，并生成 loss 曲线和样本。完整参数见 `python train.py --help`。

---

## 架构

```text
字符 id
  → token embedding + learned position embedding
  → 4 × [ Pre-LN → Causal Multi-Head Attention → residual
          Pre-LN → GELU MLP (4×) → residual ]
  → final LayerNorm
  → lm_head（与 token embedding 共享权重）
  → next-character logits
```

| 部件 | 实现 |
| --- | --- |
| Tokenizer | 字符级，65 个字符，无需 BPE |
| Attention | 4 头，`d_head = 32`，手写 `softmax(QKᵀ/√d_k + mask)` |
| Mask | 下三角 causal mask，上三角填 `-inf` |
| Block | Pre-LN、残差连接、GELU、4× MLP |
| 输出层 | weight tying（`lm_head.weight = wte.weight`） |
| 初始化 | 正态 0.02；残差投影按 `1/√(2N)` 缩放 |
| 优化 | AdamW，线性 warmup + cosine decay，梯度裁剪 1.0 |
| 评测 | 固定验证集，取最佳 val loss checkpoint |

实现上刻意没有使用 `F.scaled_dot_product_attention`：注意力矩阵、mask、softmax 和 dropout 都显式写出来，方便逐行对照公式。生产环境里应该用 fused kernel，这是另一回事。

---

## 仓库结构

```text
charformer/
  model.py        # GPTConfig, CausalSelfAttention, MLP, Block, GPT
  data.py         # 字符 tokenizer、数据集下载与校验、batch 采样
train.py          # 训练循环、评测、checkpoint、曲线、采样
generate.py       # 从 checkpoint 续写
docs/math.md      # 从交叉熵到反向传播的推导与直觉
tests/            # 11 个单元测试，含 causal leakage 测试
results/          # 已提交的训练指标、loss 曲线、样本
```

---

## 测试

```bash
python -m pytest -q
```

测试覆盖：

- 前向形状与 loss 有限性；
- **因果性**：改变未来 token，不能改变前缀位置的 logits；
- 生成长度与起始 token；
- 超过 `block_size` 的序列被拒绝；
- weight tying；
- tokenizer 往返与未知字符；
- batch 的 `targets = inputs + 1` 对齐；
- 数据集切分比例。

因果性测试是这里最重要的测试：它不证明整个模型正确，但能抓住“未来信息泄漏”这类最隐蔽的错误。

---

## 数学笔记

[`docs/math.md`](docs/math.md) 里写了：

- 交叉熵与困惑度；
- 缩放点积注意力，以及为什么要除以 `√d_k`；
- causal mask 为什么让未来位置的梯度为 0；
- Pre-LN 与残差的梯度路径；
- 交叉熵 + softmax 的梯度为什么是 `A - onehot(y)`；
- 参数量估算与残差初始化缩放。

如果要在面试里讲这个项目，建议先能不看笔记回答：**“为什么 mask 之后 softmax 的梯度不会流到未来位置？”**

---

## 设计取舍与局限

- **字符级而不是 BPE**：词表小、实现简单，适合验证注意力与训练流程；真实 LLM 会用子词 tokenizer。
- **4 层、128 维**：CPU 可训练，但容量有限；样本会语法错误，不能当作可用文本生成器。
- **没有 KV cache**：`generate.py` 每次重新计算整个上下文，便于阅读，不适合长文本高吞吐推理。
- **没有混合精度 / 分布式**：单机 CPU 教学实现；生产训练会使用 bf16、FlashAttention、梯度累积和分布式并行。
- **验证集只来自同一部作品集**：loss 能说明模型拟合了 Shakespeare 的风格，但不能说明泛化到其他英文文本。

下一步可以做：BPE tokenizer、KV cache、位置编码对比（ learned vs RoPE）、层数/宽度 ablation，以及在更大语料上报告 perplexity。

---

## 数据与致谢

- 语料：[Tiny Shakespeare](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare)，公共领域文本；脚本会校验 SHA-256。
- 设计参考：Vaswani et al. (2017), GPT-2 (Radford et al., 2019), nanoGPT (Karpathy)。代码为独立实现。
- 数学推导参考见 [`docs/math.md`](docs/math.md) 的参考资料一节。

## License

MIT
