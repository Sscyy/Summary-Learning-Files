# MMQ 论文笔记

> **论文全称**：MMQ: Multimodal Mixture-of-Quantization Tokenization for Semantic ID Generation and User Behavioral Adaptation
> **机构**：阿里巴巴集团
> **发表时间**：2025年8月

---

## 一、问题背景

### 传统 ItemID 的局限

推荐系统传统上用 ItemID 表示商品，存在两个核心问题：
- **高频更新**：商品库动态变化，静态 ID embedding 难以跟上
- **长尾稀疏**：长尾商品交互数据少，embedding 学不好

### Semantic ID 的思路

用商品的多模态内容（文本 + 图像）生成离散的语义 token，作为商品的"语义身份证"。语义相近的商品会得到相近的 Semantic ID，从而实现知识共享，缓解长尾问题。

### 现有方法的两个核心矛盾

**矛盾一：协同性 vs 独特性**

| 范式 | 做法 | 问题 |
|------|------|------|
| Modality Alignment (MA) | 先融合文本+图像，再量化 | 一个模态压制另一个，丢失各自独特信号 |
| Modality Separation (MS) | 各自独立量化 | 保住独特性，但错过跨模态协同信息 |

**矛盾二：语义空间 vs 行为空间的错位**

Semantic ID 在内容语义空间训练，但推荐目标是预测用户行为。语义相似的商品，用户实际交互可能截然不同，导致 Semantic ID 和下游任务脱节。

---

## 二、向量量化（VQ）基础知识

理解 MMQ 之前，需要先理解 VQ 的工作原理。

### Codebook 是什么

Codebook 是一张"码本"，存了 $K$ 个向量（codeword），维度与 encoder 输出的 latent 向量相同。例如 $K=3000$，每个 codeword 256 维，则 codebook 是一个 $3000 \times 256$ 的参数矩阵，**随机初始化后通过梯度训练**得到。

### Codebook Lookup

encoder 把输入 embedding 映射成 latent 向量 $z$，然后在 codebook 里找"最像"的 codeword：

**L2 距离（传统做法）：**
$$c = \arg\min_{j} \|z - z_{q,j}\|^2$$

**余弦相似度（MMQ 的做法）：**
$$c = \arg\max_{j} \frac{z^\top z_{q,j}}{\|z\| \cdot \|z_{q,j}\|}$$

找到最近的 codeword 后：
- 下标 $c$ → **Semantic ID**
- 对应向量 $z_q$ → **量化后的表示**，替代原来的 $z$ 参与后续计算

### Codebook 如何训练（STE 技巧）

argmax 操作不可微，梯度无法直接流回 encoder。VQ-VAE 用两个损失解决：

**Commitment Loss**（训练 encoder，让 $z$ 靠近 codeword）：
$$\mathcal{L}_{\text{commit}} = \|z - \text{sg}(z_q)\|^2$$

**Codebook Loss**（训练 codebook，让 codeword 靠近 $z$）：
$$\mathcal{L}_{\text{codebook}} = \|\text{sg}(z) - z_q\|^2$$

两个 `sg`（stop gradient）方向相反，交替拉近彼此。训练收敛后，codebook 相当于对语义空间做了 $K$ 类软聚类，每个 Semantic ID 代表一个语义簇。

### RQ-VAE vs OPQ vs MMQ 的区别

| 方法 | 结构 | ID 序列 |
|------|------|---------|
| RQ-VAE | 残差量化，逐层对残差再量化 | 有序，层层递进 |
| OPQ | 均匀切分 embedding 维度，各子空间独立量化 | 无序，并行 |
| MMQ | 专家网络学出语义子空间，各专家独立量化 | 无序，并行 |

---

## 三、MMQ 方法详解

MMQ 是一个两阶段框架：**多模态共享-特定 Tokenizer 训练** → **行为感知微调**。

> **重要前提**：产出 embedding 的大模型（Qwen3-Embedding 7B、Pailitao v8）参数全程冻结，MMQ 只训练 tokenizer（专家网络 + codebook）的参数。

### 3.1 输入

```
文本 embedding e_t  ←  Qwen3-Embedding 7B（冻结），256 维
图像 embedding e_v  ←  Pailitao v8（冻结），256 维
```

### 3.2 第一阶段：多模态共享-特定 Tokenizer 训练

#### 三类专家结构

**模态共享专家（Shared Expert，$N_s$ 个）**
- 输入：拼接后的 $[e_t, e_v]$
- 作用：捕捉跨模态协同信息（如"时尚感"需要图文结合才能感知）
- 分配方式：**确定性**，所有样本都走

$$z_{s,i} = E_{s,i}([e_t, e_v])$$

**模态特定专家（Specific Expert，文本 $N_t$ 个 + 视觉 $N_v$ 个）**
- 输入：各自单模态 embedding
- 作用：保留各模态独有信号
- 分配方式：**门控网络动态加权**

$$z_{t,i} = E_{t,i}(e_t), \quad z_{v,i} = E_{v,i}(e_v)$$

$$g_t = \text{softmax}(\text{MLP}_t(e_t) + b_t), \quad g_v = \text{softmax}(\text{MLP}_v(e_v) + b_v)$$

**融合表示：**
$$z = \sum_{i=1}^{N_s} z_{s,i} + \sum_{i=1}^{N_v} g_{v,i} z_{v,i} + \sum_{i=1}^{N_t} g_{t,i} z_{t,i}$$

实验配置：$N_s=2, N_t=2, N_v=2$，共 6 个专家，生成 6 个 Semantic ID。

#### 余弦量化器（Cosine Quantizer）

**为什么不用 L2？** 不同模态 encoder 的输出值域差异大，L2 距离会被量级主导而非语义方向，导致 codebook 坍缩（利用率从 1.00 降到 0.59）。

**余弦量化：** 只看方向，不看大小，对多模态场景更合理。

$$c_{s,i} = \arg\max_{j} \frac{z_{s,i}^\top z_{q,j}}{\|z_{s,i}\| \cdot \|z_{q,j}\|}, \quad i = 1, \ldots, N_s$$

量化后的融合表示：
$$z_q = \sum_{j=1}^{N_s} z_{q_{s,j}} + \sum_{j=1}^{N_v} g_{v,j} z_{q_{v,j}} + \sum_{j=1}^{N_t} g_{t,j} z_{q_{t,j}}$$

#### 正交正则化（Orthogonal Regularization）

**问题：** 多个专家容易学到重叠信息，造成参数冗余（expert collapse）。

**做法：** 将每个专家权重矩阵 $W_i \in \mathbb{R}^{d \times d}$ 展平成向量 $v_i$，约束这些向量两两正交：

$$\mathcal{L}_{\text{ortho}} = \left\| V_{\text{norm}} V_{\text{norm}}^\top - I \right\|_F^2$$

强迫每个专家"各司其职"，学不同方向的特征。

> **消融实验结论**：去掉正交正则化后，NDCG@5 从 0.2661 降到 0.1111，是所有组件中影响最大的。

#### 第一阶段损失函数

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{recon}} + \beta \cdot \mathcal{L}_{\text{aux}} + \gamma \cdot \mathcal{L}_{\text{ortho}}$$

| 损失项 | 作用 | 权重 |
|--------|------|------|
| $\mathcal{L}_{\text{recon}}$ | 用 $z_q$ 重建原始多模态 embedding $[e_t, e_v]$ | $\alpha=12$ |
| $\mathcal{L}_{\text{aux}}$ | 分别用文本/视觉专家重建 $e_t$/$e_v$，防止特定专家退化 | $\beta=10$ |
| $\mathcal{L}_{\text{ortho}}$ | 正交约束，防止专家冗余 | $\gamma=0.005$ |

重建损失用 STE 绕过离散化：
$$\mathcal{L}_{\text{recon}} = \|e - \text{decoder}(z + \text{sg}(z_q - z))\|^2$$

### 3.3 第二阶段：行为感知微调（Behavior-Aware Fine-tuning）

#### 问题所在

第一阶段训练完后，tokenizer 和下游推荐模型是割裂的：codebook lookup 是离散的 argmax，梯度无法流回 tokenizer，语义空间和行为空间的错位无法被修正。

#### 软索引机制（Soft Indices）

用可微的软索引替代离散的 argmax：

**计算与 codebook 所有向量的余弦相似度：**
$$p_i = \left[\frac{z_i^\top z_{q_1}}{\|z_i\|\|z_{q_1}\|}, \ldots, \frac{z_i^\top z_{q_K}}{\|z_i\|\|z_{q_K}\|}\right] \in \mathbb{R}^K$$

**软索引（可微）：**
$$\text{soft\_ind} = \text{softmax}(p_i / \tau) \in \mathbb{R}^K$$

**硬索引（离散，前向用）：**
$$\text{hard\_ind} = \arg\max_j \frac{z_i^\top z_{q_j}}{\|z_i\|\|z_{q_j}\|}$$

**STE 结合两者：**
$$\text{ind} = \text{soft\_ind} + \text{sg}(\text{hard\_ind} - \text{soft\_ind})$$

- **前向**：等于 hard\_ind，下游推荐模型正常工作
- **反向**：梯度走 soft\_ind，流回 tokenizer 参数

温度系数 $\tau$：越大梯度越平滑，越小越接近离散行为。

#### 联合优化目标

$$\mathcal{L}_{\text{finetune}} = \mathcal{L}_{\text{downstream}} + \alpha' \cdot \mathcal{L}_{\text{recon}} + \beta' \cdot \mathcal{L}_{\text{aux}}$$

重建损失的作用是防止微调过程中 tokenizer 遗忘第一阶段学到的语义知识（知识保留约束）。$\alpha'=0.5, \beta'=0.5$。

> **消融实验结论**：去掉行为感知微调后，NDCG@5 从 0.2661 降到 0.1606，下降约 40%。

---

## 四、实验结果

### 整体性能（工业数据集，生成式检索）

| 方法 | R@5 | R@10 | N@5 | N@10 |
|------|-----|------|-----|------|
| MA-RQ-VAE | 0.0570 | 0.0754 | 0.1362 | 0.1621 |
| MS-RQ-VAE | 0.0779 | 0.0988 | 0.1897 | 0.2191 |
| MS-OPQ | 0.0757 | 0.0927 | 0.1866 | 0.2105 |
| **MMQ** | **0.1034** | **0.1192** | **0.2661** | **0.2883** |
| 提升 | +32.73% | +20.64% | +40.27% | +31.58% |

MS 范式整体优于 MA 范式，说明在大规模数据下强行融合会损失模态独特信息。

### 消融实验（生成式检索，工业数据集）

| 变体 | R@5 | N@5 | 说明 |
|------|-----|-----|------|
| MMQ（完整） | 0.1034 | 0.2661 | — |
| w/o 余弦量化器 | 0.0786 | 0.1936 | codebook 利用率 0.59 |
| w/o 辅助重建损失 | 0.0684 | 0.1695 | codebook 利用率 0.51 |
| **w/o 正交正则化** | 0.0583 | **0.1111** | **影响最大** |
| w/o 行为感知微调 | 0.0792 | 0.1606 | 语义-行为错位 |

### 在线 A/B 测试（30天，东南亚电商平台）

- 广告收入 **+0.90%**
- 转化率（CVR）**+4.33%**
- 订单量 **+3.52%**

### 长尾商品效果

所有使用 Semantic ID 的方法在长尾商品上的 AUC 均显著优于传统 ItemID 方法，MMQ 提升最大，验证了语义 ID 通过知识共享缓解长尾稀疏问题的核心假设。

---

## 五、与相关工作的关系

| 论文 | 关系 |
|------|------|
| TIGER / VQ-Rec | 早期 Semantic ID 工作，单模态文本，tokenizer 和下游割裂 |
| EAGER | 引入行为信号，但 tokenizer 结构简单 |
| OneRec / UTGRec | MA 范式，多模态融合后量化 |
| EGA | MS 范式，各模态独立量化 |
| **MMQ** | 首个同时建模协同性+独特性，并打通语义-行为梯度的统一框架 |

---

## 六、与 MOON 的对比

| 维度 | MOON | MMQ |
|------|------|-----|
| 目标 | 多模态 embedding 表示学习 | 多模态 Semantic ID 生成 |
| 输出 | 连续向量 | 离散 token（Semantic ID） |
| 应用场景 | 召回侧（向量检索） | 生成式检索 + 精排 |
| 多模态处理 | CUBE 双路融合 | 共享+特定专家 MoE |
| 行为对齐 | 后训练阶段用行为数据微调 | Behavior-Aware Fine-tuning（STE 打通梯度） |
| 大模型参数 | 微调 embedding 模型本身 | 冻结，只训练 tokenizer |

---

## 七、核心思想一句话总结

> 用"共享专家捕协同 + 特定专家保独特 + 正交约束防冗余"解决多模态量化的表示质量问题，再用"软索引 + STE"打通 tokenizer 和下游任务之间的梯度通路，让 Semantic ID 能随推荐行为目标动态调整。
