# 从公式到代码：这个 Transformer 里发生了什么

这份笔记只覆盖本项目真正实现的部件。目标不是复述论文，而是把每个张量的形状、每个公式的直觉，以及代码里对应的几行写清楚。配套代码在 [`charformer/model.py`](../charformer/model.py)。

---

## 1. 语言模型在优化什么

给定字符序列 $x_1,\dots,x_T$，自回归语言模型做的是

$$
p(x_1,\dots,x_T)=\prod_{t=1}^{T} p_\theta(x_t\mid x_{<t}).
$$

训练时，每个位置 $t$ 都预测下一个字符，损失是平均交叉熵：

$$
\mathcal{L}(\theta)=-\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t}).
$$

困惑度（perplexity）定义为 $\exp(\mathcal{L})$。可以把它理解为模型在每一步“平均要在多少个字符里犹豫”。随机猜测的基线是词表大小 $V$；本项目词表 $V=65$，所以初始困惑度接近 65。

代码对应：`GPT.forward` 里的 `F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))`。

---

## 2. 词元嵌入与位置嵌入

字符 `c` 被映射为整数 $i_c$，再查表得到

$$
e_t = E_{\text{tok}}[x_t] + E_{\text{pos}}[t],
\quad E_{\text{tok}}\in\mathbb{R}^{V\times d},\;
E_{\text{pos}}\in\mathbb{R}^{T_{\max}\times d}.
$$

没有位置嵌入时，自注意力对排列是等变的：模型无法区分“猫追狗”和“狗追猫”。位置嵌入给每个位置一个可学习的向量，破坏这种对称性。

代码对应：`wte(idx) + wpe(arange(T))`。

---

## 3. 因果自注意力

### 3.1 投影

对每个位置的向量 $x_t$，用三个线性变换得到 query、key、value：

$$
q_t = W_Q x_t,\quad k_t = W_K x_t,\quad v_t = W_V x_t.
$$

实际实现把三个投影合并成一个 `c_attn`，输出 `3d` 维后切开，减少一次矩阵乘法。

### 3.2 缩放点积

单头注意力：

$$
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right),\qquad
\operatorname{Attn}(Q,K,V)=AV.
$$

$M$ 是因果掩码：

$$
M_{ij}=
\begin{cases}
0, & j\le i,\\
-\infty, & j>i.
\end{cases}
$$

因此位置 $i$ 的注意力只会落在 $1,\dots,i$ 上。`-∞` 经过 softmax 后权重精确为 0，未来位置既不会贡献值，也不会收到梯度。

**为什么要除以 $\sqrt{d_k}$？** 如果 $q,k$ 各维独立、均值 0、方差 1，那么点积 $q^\top k$ 的方差是 $d_k$。不缩放时，$d_k$ 越大，logits 越大，softmax 越接近 one-hot，梯度越小且训练不稳定。除以 $\sqrt{d_k}$ 让方差回到 1 左右。

### 3.3 多头

把 $d$ 维拆成 $h$ 个头，每个头维度 $d_h=d/h$，各自做注意力后拼接，再过 $W_O$：

$$
\operatorname{MultiHead}(X)=\operatorname{Concat}(\text{head}_1,\dots,\text{head}_h)W_O.
$$

不同头可以学习不同的关系（局部搭配、指代、标点等）。本项目 $d=128$，$h=4$，所以 $d_h=32$。

代码对应：`CausalSelfAttention.forward`。`causal_mask` 是下三角矩阵，`masked_fill(..., -inf)` 把上三角屏蔽。

### 3.4 复杂度

序列长度 $T$、隐藏维度 $d$：

- 时间：$O(T^2 d)$（注意力矩阵）加 $O(Td^2)$（投影）；
- 显存：每个头要存 $T\times T$ 的注意力矩阵，因此是 $O(hT^2)$。

这就是长上下文昂贵的根本原因；KV cache、FlashAttention、稀疏注意力都在处理这一项。

---

## 4. 前馈网络

每个位置独立地过一个两层 MLP：

$$
\operatorname{MLP}(x)=W_2\,\operatorname{GELU}(W_1x+b_1)+b_2,
$$

隐藏维度通常是 $4d$。注意力负责“位置之间交换信息”，MLP 负责“每个位置内部做非线性变换”。两者交替堆叠。

代码对应：`MLP`。`GELU` 比 ReLU 平滑，在 Transformer 中更常见。

---

## 5. 残差连接与 Pre-LayerNorm

每个 block 是

$$
x\leftarrow x+\operatorname{Attn}(\operatorname{LN}(x)),
$$

$$
x\leftarrow x+\operatorname{MLP}(\operatorname{LN}(x)).
$$

LayerNorm 对每个 token 的特征维做归一化：

$$
\operatorname{LN}(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta.
$$

**Pre-LN**（先归一化再进子层）让残差路径保持恒等映射，梯度可以沿着 $x\leftarrow x+\cdots$ 直接回传，深层模型更稳定。原始 Transformer 用 Post-LN，训练深层网络时需要更精细的 warmup。

代码对应：`Block.forward`。注意 `x = x + ...` 是残差流，不是覆盖。

---

## 6. 输出层与权重共享

最后过一层 LayerNorm，再映射到词表：

$$
\text{logits}=H W_{\text{out}}^\top,\qquad
p(\cdot\mid x_{<t})=\operatorname{softmax}(\text{logits}_t).
$$

本项目默认让 $W_{\text{out}}=E_{\text{tok}}$（weight tying）。直觉是：输入字符的嵌入方向和输出字符的分类方向共享同一个语义空间；同时省掉 $Vd$ 个参数。GPT-2 也使用了这种绑定。

代码对应：`self.lm_head.weight = self.transformer["wte"].weight`。

---

## 7. 反向传播经过注意力

设 $A=\operatorname{softmax}(S)$，$S=QK^\top/\sqrt{d_k}+M$。交叉熵对 $S$ 的梯度可以写成

$$
\frac{\partial \mathcal{L}}{\partial S}
=\frac{1}{T}\left(A-\operatorname{onehot}(y)\right),
$$

再乘上掩码（未来位置梯度为 0）。随后

$$
\frac{\partial \mathcal{L}}{\partial Q}=
\frac{\partial \mathcal{L}}{\partial S}K\frac{1}{\sqrt{d_k}},\qquad
\frac{\partial \mathcal{L}}{\partial K}=
\left(\frac{\partial \mathcal{L}}{\partial S}\right)^\top Q\frac{1}{\sqrt{d_k}}.
$$

这也是为什么 softmax + 交叉熵组合在实现上很简洁：概率与 one-hot 的差就是起点。PyTorch 的 autograd 会处理这些细节，但理解它们有助于解释梯度消失、注意力饱和和训练不稳定的现象。

---

## 8. 初始化与残差缩放

- 线性层和嵌入：$\mathcal{N}(0,0.02^2)$；
- 残差分支最后的 `c_proj`：$\mathcal{N}(0,(0.02/\sqrt{2N})^2)$，$N$ 是层数。

深层残差网络里，每个 block 都往残差流里加东西；如果不缩放，方差会随层数增长。除以 $\sqrt{2N}$ 是在初始化时让每个分支的贡献保持可控。这是 GPT-2 的工程技巧，不是理论上的唯一做法。

---

## 9. 优化器与学习率

使用 AdamW：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
$$

$$
\theta_t=\theta_{t-1}-\eta_t\left(\frac{m_t}{1-\beta_1^t}\Big/\left(\sqrt{\frac{v_t}{1-\beta_2^t}}+\epsilon\right)\right)
-\eta_t\lambda\theta_{t-1}.
$$

本项目用线性 warmup + cosine decay：

$$
\eta_t=
\begin{cases}
\eta_{\max}\dfrac{t+1}{T_{\text{warm}}}, & t<T_{\text{warm}},\\
\eta_{\min}+\dfrac{1+\cos(\pi p_t)}{2}(\eta_{\max}-\eta_{\min}), & \text{otherwise},
\end{cases}
$$

其中 $p_t=(t-T_{\text{warm}})/(T_{\max}-T_{\text{warm}})$。warmup 避免训练初期大梯度破坏参数；cosine 让后期步长变小，便于收敛。梯度裁剪把全局梯度范数限制在 1.0。

代码对应：`learning_rate_at` 与训练循环。

---

## 10. 参数量估算

忽略 bias 和 LayerNorm，单个 block：

- QKV 投影：$3d^2$；
- 输出投影：$d^2$；
- MLP：$d\cdot4d+4d\cdot d=8d^2$；

合计 $12d^2$。$N$ 层就是 $12Nd^2$。嵌入为 $Vd+T_{\max}d$，输出层若绑定则不再增加参数。

本项目 $V=65$、$T_{\max}=128$、$d=128$、$N=4$：

$$
12\cdot4\cdot128^2 + 65\cdot128 + 128\cdot128 \approx 0.81\text{M}.
$$

实际参数量会因 bias 和 LayerNorm 略有不同，训练脚本会打印精确值。

---

## 11. 测试证明了什么

`tests/test_model.py::test_causal_attention_cannot_see_future_tokens` 构造两条前缀相同、未来不同的序列，检查前缀位置的 logits 是否完全一致。它验证的是因果性这个可观察不变量，而不是证明整个实现无误。其余测试覆盖形状、损失有限性、生成长度、权重绑定和 tokenizer 往返。

---

## 12. 参考资料

- Vaswani et al., *Attention Is All You Need* (2017)
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2, 2019)
- Ba, Kiros, Hinton, *Layer Normalization* (2016)
- Hendrycks & Gimpel, *Gaussian Error Linear Units (GELUs)* (2016)
- Andrej Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT) 与 [char-rnn](https://github.com/karpathy/char-rnn)：本项目的数据与教学灵感来源；代码为独立实现。
