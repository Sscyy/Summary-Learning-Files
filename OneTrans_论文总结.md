# OneTrans 论文总结

> **一句话概括**：将用户行为序列建模与非序列特征交叉统一到单一 Transformer 中，通过 Causal Attention 让非序列特征直接与每个行为 token 交互，同时借助金字塔截断和 KV Caching 保证工业级推理效率，在线 A/B 测试实现 GMV/用户 +5.68%。

---

## 一、背景：为什么要做这件事？

### 1.1 工业推荐的两条独立赛道

以往推荐模型的主流范式是**先做序列建模，再做特征交叉**，两个模块完全分开：


用户行为序列 [点击1, 点击2, ..., 点击N]
        ↓  序列编码器（DIN / BST / LONGER）
    用户兴趣向量 h_user（一个固定的压缩表示）
        ↓
    [h_user, 物品特征, 用户画像, 上下文特征, ...]
        ↓  特征交叉模块（DCN / Wukong / RankMixer）
        ↓
      预测分数


**问题在哪里？** 序列编码结束后，整个用户历史被压缩成了**一个向量**，信息不可逆地损失了。后续的特征交叉模块只能拿到这个"摘要"，**无法看到原始序列中每个行为的细节**，"序列建模"和"特征交叉"也无法相互促进优化。

### 1.2 各自独立发展的瓶颈

| 赛道 | 代表工作 | 共同问题 |
|------|---------|---------|
| **序列建模** | DIN、BST、LONGER | 序列输出作为单向量喂给特征交叉，信息损失 |
| **特征交叉** | DCN、Wukong、RankMixer | 只能看到序列的"摘要"，无法感知细粒度行为 |
| **尝试融合** | InterFormer | 仍是两个独立模块 + 复杂的跨模块信号传递，割裂 |

### 1.3 核心目标

> **用一个统一的 Transformer，同时完成序列建模和特征交叉，让两者相互促进、联合优化，并保证工业级推理效率**

---

## 二、核心思想：把"时序上的先后"变成"空间上的并列"

### 2.1 与以往方法的设计哲学对比

| 维度 | 以往方法（RankMixer + LONGER 等）   |      OneTrans      |
|------|----------------------------------|----------------------------------|
| **序列与特征关系** | 先序列 → 后特征（串行两阶段） | 并列放入同一 Transformer（一阶段） |
| **NS 特征看到序列的粒度** | 看到一个压缩向量 | **直接 Attend 每个行为 token** |
| **优化方式** | 两个独立损失 / 分别优化 | **端到端统一优化** |
| **信息流向** | 单向（序列 → 特征交叉） | **双向（序列 ↔ 特征 相互感知）** |

### 2.2 Attention 本质上就是特征交叉

当 NS-token（物品特征、用户画像、上下文）的 Query 去查询 S-token（行为序列）的 Key/Value 时，本质上在做：

> **"这个物品/用户属性，和用户历史里哪些行为最相关？"**

这正是 DIN 做的目标感知注意力——只是 OneTrans 把它自然地嵌入了统一 Transformer，无需额外设计。

---

## 三、方法详解：数据如何流转

### 3.1 整体架构（宏观视角）

```
输入：多行为序列 S + 非序列特征 NS（用户/物品/上下文）
    ↓
Tokenization
  ├─ Sequential Tokenizer  →  S-tokens（行为序列 token）
  └─ Non-Seq Tokenizer     →  NS-tokens（非序列 token）
    ↓
拼接成统一序列：[S1, S2, ..., S_Ls | NS1, NS2, ..., NS_Lns]
    ↓
OneTrans Pyramid Blocks × N 层：
  ├─ RMSNorm（预归一化）
  ├─ Mixed Causal Attention（混合因果注意力）
  └─ Mixed FFN（混合前馈网络）
  每层结束后砍掉一部分最前面的 S-token（金字塔截断）
    ↓
最终只剩 NS-tokens → Task Tower → CTR / CVR 预测
```

---

### 3.2 第一步：Tokenization（特征 → Token）

#### 非序列特征 Tokenization（NS-tokens）

**为什么**：数百个异构 NS 特征（数值型、类别型）需要统一成固定数量、固定维度的 token。

**两种方案**：

| 方案 | 做法 | 优缺点 |
|------|------|-------|
| **Group-wise Tokenizer**（对齐 RankMixer） | 人工划分语义组，组内 concat → 各组独立 MLP | 可控语义，但需人工设计 |
| **Auto-Split Tokenizer** | 所有特征 concat → 一个大 MLP → 均匀切分成 L_NS 个 token | 减少 Kernel 启动开销，完全自动 |

**最终结果**：L_NS 个 token，每个维度 d。

#### 序列特征 Tokenization（S-tokens）

**为什么**：多种行为序列（点击、购买、收藏）原始维度不同，需要统一维度。

**怎么做**：
1. 每种行为类型 S_i，有自己专属的 `MLP_i`，把该类型所有事件映射到统一维度 d
2. 多种行为序列按以下两种方式之一合并：
   - **时间戳感知**：按时间戳将所有行为交错排列，附加行为类型指示符
   - **时间戳无感知**：按行为意图强度排序（购买 → 加购 → 点击），每段行为之间插入可学习的 `[SEP]` token
3. 消融实验表明：有时间戳时，时间戳感知方式更优

**最终结果**：L_S 个 S-token，L_S = Σ(行为数) + L_SEP（SEP token 数）。

#### 拼接成统一序列

```
X^(0) = [S-tokens ; NS-tokens] ∈ R^{(L_S + L_NS) × d}
```

**关键**：S-token 在前，NS-token 在后——这个顺序对 Causal Mask 的效果至关重要。

---

### 3.3 第二步：Mixed Causal Attention（混合因果注意力）

这是 OneTrans 的核心模块，包含两个"混合"的含义。

#### 混合一：混合参数化（Mixed Parameterization）

每个 token 计算自己的 Q、K、V：

```
q_i, k_i, v_i = W_Q_i · x_i,  W_K_i · x_i,  W_V_i · x_i
```

参数分配策略：

| Token 类型 | 参数方式 | 原因 |
|-----------|---------|------|
| **S-tokens（行为序列）** | 所有 S-token **共享**同一套 Q/K/V 权重 | 行为序列是同质的，共享参数节省内存，还能跨行为类型泛化 |
| **NS-tokens（非序列特征）** | 每个 NS-token **独享**自己的 Q/K/V 权重 | 不同 NS-token 来自完全不同的语义空间（物品 vs 用户画像 vs 上下文），独立参数更灵活 |

#### 混合二：统一 Causal Mask

把所有 token 的 q/k 拼成矩阵 Q, K，计算注意力分数矩阵并叠加 Causal Mask：

```
Scores = Q · Kᵀ / sqrt(d_head)
Scores = Scores + mask        # mask 为下三角：0 保留，-inf 遮蔽
Attn   = softmax(Scores) · V
```

Mask 的具体效果（token 顺序 = [S1..S8 | NS1..NS3]）：

```
         S1  S2  S3  S4  S5  S6  S7  S8 | NS1 NS2 NS3
S1   [   ✓   ✗   ✗   ✗   ✗   ✗   ✗   ✗ |  ✗   ✗   ✗  ]
S2   [   ✓   ✓   ✗   ✗   ✗   ✗   ✗   ✗ |  ✗   ✗   ✗  ]
S3   [   ✓   ✓   ✓   ✗   ✗   ✗   ✗   ✗ |  ✗   ✗   ✗  ]
...
S8   [   ✓   ✓   ✓   ✓   ✓   ✓   ✓   ✓ |  ✗   ✗   ✗  ]
-----|-----------------------------------|------------------
NS1  [   ✓   ✓   ✓   ✓   ✓   ✓   ✓   ✓ |  ✓   ✗   ✗  ]
NS2  [   ✓   ✓   ✓   ✓   ✓   ✓   ✓   ✓ |  ✓   ✓   ✗  ]
NS3  [   ✓   ✓   ✓   ✓   ✓   ✓   ✓   ✓ |  ✓   ✓   ✓  ]
```

三种注意力关系及其背后的设计逻辑：

| 谁 → 看谁 | 能看到 | 设计原因 |
|----------|--------|---------|
| **S → S（前面）** | 只看之前的行为 token | 保留时序因果性，越靠后的 S-token 积累了越多历史信息 |
| **S → NS** | ❌ 看不到 | NS-token 排在后面，因果 mask 自然屏蔽 |
| **NS → S（全部）** | 看到所有行为 token | 相当于对行为序列做 Target-Attention，获取完整历史 |
| **NS → NS（前面）** | 只看之前的 NS-token | NS-token 之间也保持因果性，增加 token 级别的交互多样性 |

**关键洞察**：NS-token 对 S-token 的全量 Attention，既是"目标感知的序列聚合"（序列建模），也是"跨模态特征交叉"（特征交叉）——两件事在同一个 Attention 里同时完成。

---

### 3.4 第三步：Mixed FFN（混合前馈网络）

与 Attention 相同的参数化策略：
- **S-tokens**：共享同一个 FFN
- **NS-tokens**：每个 token 独享自己的 FFN

---

### 3.5 第四步：Pyramid Stack（金字塔截断）

#### 动机

Causal Mask 带来一个性质：**越靠后的 S-token，看过的历史越多，聚合的信息越丰富**。前面的 S-token 信息已经被后面的 S-token"继承"了，保留它们是一种浪费。

#### 做法

每过一个 Block，砍掉最前面的一部分 S-token，只保留后面的"信息最丰富"的 S-token：

```
Block 1 输入：[S1 S2 S3 S4 S5 S6 S7 S8 | NS1 NS2 NS3]   (11 tokens)
Block 1 结束后砍掉前 4 个：
Block 2 输入：         [S5 S6 S7 S8 | NS1 NS2 NS3]        (7 tokens)
Block 2 结束后砍掉前 2 个：
Block 3 输入：               [S7 S8 | NS1 NS2 NS3]        (5 tokens)
...
最终输出：                         [NS1 NS2 NS3]           (3 tokens)
```

NS-token 全程不减少，S-token 逐层被"蒸馏"进 NS-token。

#### 计算效率优化细节

砍掉前面 S-token 后，下一层做 Attention 时：
- **Query**：只有保留下来的后 L' 个 token 发出查询
- **Key/Value**：仍来自**全部 L 个 token**（包含已被砍掉的）

> 被砍掉的 S-token 不再主动提问，但仍作为"被查询的记忆"存在，信息不丢失。

**计算复杂度**：从每层 `O(L²)` 降到 `O(L × L')`，L' 逐层减小，整体 FLOPs 大幅下降。

---

### 3.6 第五步：输出与预测

经过所有 Block 后，序列中只剩下 L_NS 个 NS-token，每个 NS-token 已经充分融合了整个行为序列的信息。

将这 L_NS 个 token 送入任务特定的预测头（Task Tower），输出 CTR / CVR 等预测分数。

---

## 四、关键创新点（技术贡献）

| 创新 | 解决的问题 | 核心价值 |
|------|-----------|---------|
| **统一 Token 序列** | 序列建模与特征交叉分离 | 一个 Transformer 同时完成两件事，端到端优化 |
| **Mixed Causal Attention** | NS 特征只能看到序列摘要 | NS-token 直接 Attend 每个行为 token，信息无损 |
| **Mixed Parameterization** | S/NS token 异质性 | S 共享参数节省开销，NS 独立参数保留语义差异 |
| **Pyramid Stack** | 长序列计算昂贵 | 信息逐层蒸馏进 NS-token，FLOPs 大幅下降 |
| **Cross-Request KV Caching** | S-side 计算被重复 N 次 | S-side 每个 request 只算一次，按增量复用 |

---

## 五、训练与部署优化（3.5 节）

### 5.1 跨候选项 KV Caching

**工业场景结构**：同一个 request 有 N 个候选物品，S-token（用户行为序列）完全相同，NS-token（物品特征）各不相同。

```
无优化：S-token 被重复计算 N 次（完全浪费）

两阶段优化：
  Stage I（每个 request 只做一次）：
    处理全部 S-tokens，缓存所有层的 K/V
  Stage II（每个候选物品做一次）：
    NS-token 的 Q 去 Attend 缓存的 S-side K/V → 轻量计算 → 输出分数
```

**效果**：S-side 计算量从 `O(N × L)` 降到 `O(L)`，省 N 倍。

### 5.2 跨 Request 增量 KV Caching

用户行为序列是"只追加不修改"的，下次 request 只是多了几个新行为 `ΔL`：

```
上次缓存：[S1, S2, S3, S4, S5] 的 K/V
这次请求：[S1, S2, S3, S4, S5, S6, S7]
→ 只需计算 S6, S7 的增量 K/V，拼入已有缓存

计算量：O(L) → O(ΔL)
```

### 5.3 统一 LLM 工程优化

| 优化 | 解决的问题 | 效果 |
|------|-----------|------|
| **FlashAttention-2** | 标准 Attention 显存 O(L²)，I/O 开销大 | 分块在 SRAM 内完成，显存降至 O(L) |
| **混合精度（BF16/FP16）** | FP32 显存消耗大 | 显存减半，计算更快 |
| **激活重计算** | 前向激活值占满显存 | 丢弃中间激活，反向时重算，以少量计算换大量显存 |

---

## 六、实验结果

### 6.1 数据集规模

| 指标 | 数值 |
|------|------|
| 总曝光样本 | 29.1B |
| 唯一用户数 | 27.9M |
| 唯一物品数 | 10.2M |
| 日均曝光 | 118.2M ± 14.3M |

### 6.2 离线效果（CTR/CVR AUC）

- OneTrans 在相同计算量下，**一致优于** DCNv2+DIN、RankMixer+DIN、RankMixer+Transformer 等所有历代生产基线
- 生产部署迭代顺序：DCNv2+DIN → RankMixer+DIN → RankMixer+Transformer → **OneTrans**

### 6.3 消融实验关键结论

| 消融项 | 结论 |
|-------|------|
| **统一 Transformer vs 分离两阶段** | 统一架构在相同计算下持续更优 |
| **Group-wise vs Auto-Split Tokenizer** | 效果相近，Auto-Split 工程更简洁 |
| **时间戳感知 vs 无感知 Merge** | 有时间戳时，时间戳感知更优 |
| **Mixed 参数化 vs 全共享** | Mixed 参数化更优，NS 独立参数贡献显著 |
| **有 Pyramid vs 无 Pyramid** | 有 Pyramid 在效率提升同时效果持平甚至更好 |

### 6.4 Scaling Law

- 序列长度（L）、模型宽度（d_model）、模型深度（层数）三个维度均呈现**对数线性**扩展规律
- 符合 LLM 领域的幂律 Scaling 特性

### 6.5 在线 A/B 测试

| 指标 | 变化 |
|------|------|
| 用户 GMV | **+5.68%** |
| 推理延迟 | 满足生产约束 |

---

## 七、优缺点与局限性

### 7.1 优势

| 优势 | 说明 |
|------|------|
| **真正的端到端统一** | 序列建模与特征交叉同时进行，无信息损失 |
| **信息利用充分** | NS-token 直接看到每个行为 token，不再只看摘要 |
| **LLM 优化可复用** | 标准 Transformer 架构，直接继承 FlashAttention / KV Caching 等成熟优化 |
| **Scaling Law 友好** | 三个维度均有良好扩展性 |
| **在线效果显著** | GMV/用户 +5.68% |

### 7.2 局限性

| 局限 | 说明 |
|------|------|
| **长序列代价** | 虽有 Pyramid 优化，超长序列（>1000）仍是挑战 |
| **Candidate-Specific 序列** | 如 SIM 的物品侧个性化序列无法复用 S-side KV Cache，需单独处理 |
| **NS-token 数量固定** | tokenizer 设计仍需一定人工决策（分组数 L_NS） |
| **多场景未涉及** | 论文未探讨多场景/多任务场景，与 MTmixAtt 的多场景能力有差距 |

---

## 八、与 RankMixer / MTmixAtt 的对比维度

| 对比维度 | RankMixer | MTmixAtt | OneTrans |
|---------|-----------|----------|----------|
| **序列建模** | ❌ 不做序列建模 | ❌ 不做序列建模 | ✅ **统一做序列建模** |
| **特征交叉** | ✅ Token Mixing（零参数） | ✅ 可学习矩阵 | ✅ Causal Attention |
| **两者关系** | 独立（序列另有模块） | 独立（序列另有模块） | **统一在一个 Transformer** |
| **Token 交互方式** | 固定重排 | 可学习矩阵 | **Causal Attention** |
| **参数化策略** | Per-token 独立 FFN | 共享 + 场景稀疏 MoE | **S 共享 / NS 独立** |
| **特征分组** | 人工语义分组 | AutoToken 自动分组 | 人工 or 自动（两种方案） |
| **多场景** | ❌ | ✅ 场景 MoE + MLoRA | ❌ |
| **工程优化** | Op Fusion，MFU 优先 | 未报告 | **KV Caching + FlashAttn** |
| **Scaling Law** | ✅ | ✅ | ✅ |

---

## 九、核心启示

1. **"先后串行"变"并列统一"是架构升级的核心逻辑**：序列建模和特征交叉本质上都可以用 Attention 来表达，统一到同一个框架是自然趋势
2. **Causal Mask 的妙用**：不只是为了自回归生成，Causal Mask + 尾部 token 聚合信息的性质，天然支持金字塔式信息蒸馏
3. **工业优化要从架构层面设计**：KV Caching 不是事后补丁，而是因为 Causal Attention 和"S-token 跨候选共享"的结构被提前设计进去的
4. **LLM 技术向 RecSys 迁移是可行路径**：FlashAttention、混合精度、激活重计算、KV Caching 直接复用，大幅降低工程成本

---

## 十、后续对比重点问题

- 其他论文如何处理序列与特征交叉的关系？是否也走统一路线？
- 如何同时兼顾多场景（MTmixAtt 的优势）和序列-特征统一建模（OneTrans 的优势）？
- Causal Attention 作为特征交叉机制，与 Token Mixing（RankMixer）、可学习矩阵（MTmixAtt）的表达能力和计算效率如何量化比较？
- 金字塔截断策略是否可以与 MoE 结合？