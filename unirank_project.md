# 美团猜喜精排模型重构：UniRank 统一特征建模框架

> 本文档为项目技术详细设计，覆盖背景动机、核心模块设计、实验结果与面试问答要点。

---

## 一、背景与动机

### 1.1 业务背景

美团首页"猜你喜欢"推荐场景覆盖外卖、到餐、酒旅、闪购等多个业务线，精排模型需要同时服务多个场景，特征规模达 400+ 个异构字段（用户画像、物品属性、序列行为、交叉特征、场景上下文等）。

### 1.2 原有方案的三个痛点

| 痛点 | 具体表现 |
|------|---------|
| **人工特征分组** | 特征按语义人工划分为若干组，分组方案依赖领域经验，主观性强，跨场景难以复用，维护成本随特征规模线性增长 |
| **序列与特征交叉割裂** | 序列建模（DIN/BST）与特征交叉（DCN）是两个独立模块，序列编码器输出的单向量作为特征喂给交叉模块，行为细节不可逆损失 |
| **多场景模型碎片化** | 各业务场景维护独立精排模型，参数冗余，跨场景知识无法共享，新场景冷启动需要重新训练 |

### 1.3 核心目标

> 设计 UniRank 统一精排框架，以 **自动化特征建模** 和 **多场景统一建模** 为核心，同时引入序列与特征交叉的统一计算，全面提升精排模型的表达能力与工程可维护性。

---

## 二、整体架构

```
输入层：400+ 异构特征（用户/物品/序列/交叉/场景）
    ↓
【模块一】AutoToken 自动特征分组
    可学习选择矩阵 S → n_g 个语义 token
    ↓
【模块二】Mixed Causal Attention（序列+特征统一建模）
    S-token（行为序列）‖ NS-token（AutoToken 产出）
    → Causal Attention 统一计算，NS-token attend 每个行为 token
    ↓
【模块三】UniRank Block × N 层
    ├─ Learnable Token Mixing（可学习混合矩阵）
    ├─ Shared Dense MoE（跨场景通用知识）
    └─ Scenario Sparse MoE（场景特有知识）
    ↓
Mean Pooling → 场景特定预测头
    ├─ 主场景：DNN(h_global)
    └─ 其他场景：DNN(h_global) + MLoRA_s(h_global)
    ↓
输出：CTR / CTCVR 预测
```

---

## 三、模块一：AutoToken 自动特征分组

### 3.1 动机

原始方案中，特征按照人工语义分组（如"用户画像组"、"物品属性组"），分组边界固定，无法随数据分布自适应调整。当特征数量从 100 增长到 400+ 时，人工维护成本急剧上升，且跨场景时分组方案需要重新设计。

### 3.2 方法

引入可学习选择矩阵 $S \in \mathbb{R}^{n_g \times n_f}$，其中 $n_g$ 为目标分组数，$n_f$ 为特征总数。

**特征对齐**：每个特征 $f_i$ 通过专属投影 $\text{DNN}_i$ 映射到统一维度 $d$，得到对齐后的特征矩阵 $\hat{X} \in \mathbb{R}^{n_f \times d}$。

**可学习分组**：对第 $j$ 个 token，从 $S$ 的第 $j$ 行中选出 Top-k 个特征，加权聚合：

$$X_j = \sum_{i \in \text{TopK}(S_{j,:})} \omega_i \cdot \hat{X}_i$$

$$\omega_i = \text{softmax}(\text{TopK}(S_{j,:}))_i$$

**关键设计**：通过 softmax 使 Top-k 操作可微分，整个分组过程端到端可学习，无需人工干预。

### 3.3 优势

- **数据驱动**：分组边界由数据决定，消融实验表明 AutoToken 效果媲美多年积累的人工分组经验
- **跨场景自适应**：不同场景的 S 矩阵可以学到不同的最优分组，无需人工重新设计
- **正交初始化**：S 矩阵使用正交初始化，训练初期近似恒等映射，稳定收敛

---

## 四、模块二：序列与特征交叉统一建模

### 4.1 动机

传统方案中，序列建模输出单向量后，行为细节已不可逆损失。非序列特征（物品、用户画像、上下文）只能看到序列的"摘要"，无法感知用户历史中每个具体行为的细节。

### 4.2 方法

将 AutoToken 产出的 NS-token（非序列特征 token）与行为序列 S-token 拼接成统一序列，在同一个 Attention 中完成计算：

```
统一序列 = [S_1, S_2, ..., S_Ls | NS_1, NS_2, ..., NS_Lns]
```

通过 Causal Mask 控制注意力方向：

| 谁 → 看谁 | 能看到 | 设计意图 |
|----------|--------|---------|
| S-token → S-token（前面） | 只看之前的行为 | 保留时序因果性 |
| NS-token → S-token（全部） | 看到所有行为 token | 目标感知的序列聚合，信息无损 |
| NS-token → NS-token（前面） | 只看之前的 NS-token | token 间交叉多样性 |

**核心收益**：NS-token 对 S-token 的全量 Attention，同时完成了"序列聚合"和"特征交叉"两件事，无需两个独立模块。

### 4.3 参数化策略

- **S-token**：所有行为 token 共享同一套 Q/K/V 权重（行为序列同质，共享参数节省开销）
- **NS-token**：每个 token 独享 Q/K/V 权重（不同语义空间，独立参数保留差异性）

---

## 五、模块三：共享+场景稀疏 MoE 多场景建模

### 5.1 动机

美团猜喜覆盖首页、团购、外卖、酒旅等多个场景，各场景用户行为分布差异显著，但也存在大量可共享的通用知识（如用户基本偏好、价格敏感度等）。完全共享一个模型无法捕捉场景差异，完全独立则参数冗余、知识无法迁移。

### 5.2 Shared Dense MoE（通用知识）

所有场景共享 $B$ 个 Dense 专家，每个专家始终激活（Sigmoid 门控，非 Top-k 稀疏）：

$$h_l^C = \sum_{k=1}^{B} U_{k,C} \cdot \text{FFN}_k^{\text{shared}} + \sum_{m=1}^{n \cdot c} V_{m,C} \cdot \text{FFN}_m^{\text{fine}}$$

$$U_{k,C} = \sigma(\text{Gate}_1(Z_C)), \quad V_{m,C} = \sigma(\text{Gate}_2(Z_C))$$

**细粒度专家切分**：将基础专家切分为 $n \times c$ 个细粒度小专家，总参数量不变，但组合灵活性大幅提升，减少专家间冗余。

### 5.3 Scenario Sparse MoE（场景特有知识）

在 Shared Dense MoE 之上，叠加场景稀疏专家层，平衡通用与特化：

**场景专家路由**：
$$I_{k,C} = \text{Gate}_4([D_l^C \| D_l^s]) + W \cdot \mathbb{1}[k = k^*]$$

- $k^*$：当前样本所属场景 ID
- $W > 0$：手动 bonus，**强制当前场景专家必被激活**
- Top-k 选择：除强制激活的场景专家外，再自动选择 k-1 个相关场景专家，实现跨场景知识迁移

**关键设计**：手动 bonus 保证了场景专家的"硬路由"，同时 Top-k 的软选择允许模型自动发现跨场景相关性。

### 5.4 MLoRA 新场景快速适配

对于新接入场景，无需全量重训，只需在预测头上挂载低秩适配器：

$$y_s = \text{DNN}(h_{\text{global}}) + \text{MLoRA}_s(h_{\text{global}})$$

$$\text{MLoRA}_s(x) = W_{\text{up}}^s \cdot W_{\text{down}}^s \cdot x, \quad W_{\text{down}} \in \mathbb{R}^{d \times r}, \quad W_{\text{up}} \in \mathbb{R}^{r \times d}, \quad r \ll d$$

只更新约 1% 的参数，新场景冷启动周期从"重新训练"缩短到"适配器微调"。

---

## 六、训练策略

**联合多任务训练**：所有场景的 CTR / CTCVR 损失取平均，端到端联合优化：

$$\mathcal{L} = -\frac{1}{K} \sum_{s=1}^{K} \mathbb{E}\left[ y \log \hat{y} + (1-y) \log(1-\hat{y}) \right]$$

**训练稳定性**：
- AutoToken 选择矩阵 S 使用正交初始化
- 可学习 Token Mixing 矩阵同样使用正交初始化（$M_i \approx I$），训练初期近似恒等映射
- Shared Dense MoE 使用 Sigmoid 激活（始终 Dense），避免专家极化问题

---

## 七、实验结果

### 7.1 离线消融实验

基线：DIN + DCN 两阶段串行方案

| 模型 | CTR-AUC | CTR-GAUC | CTCVR-AUC | CTCVR-GAUC |
|------|---------|----------|-----------|------------|
| Baseline | - | - | - | - |
| + AutoToken | +3bp | +8bp | +2bp | +6bp |
| + 序列特征统一建模 | +5bp | +13bp | +4bp | +9bp |
| + 场景 MoE | +3bp | +6bp | +2bp | +5bp |
| **UniRank（全量）** | **+11bp** | **+27bp** | **+8bp** | **+20bp** |

### 7.2 关键发现

1. **AutoToken vs 人工分组**：随机分组 < 人工分组 ≈ AutoToken，数据驱动的分组效果媲美领域专家经验，且维护成本为零
2. **序列统一建模**：相比"先序列后交叉"的串行方案，CTR-GAUC 额外提升 13bp，验证了信息无损的价值
3. **场景 MoE**：在主场景（首页）效果提升的同时，跨场景（团购、外卖）也有稳定正向，证明了共享知识的迁移能力
4. **MLoRA 适配**：新场景只需微调适配器，离线 GAUC 与全量微调方案差距在 2bp 以内，参数量仅为全量的 1%

---

## 八、与业界方案对比

| 对比维度 | DIN+DCN（原方案） | RankMixer | MTmixAtt | **UniRank（本方案）** |
|---------|-----------------|-----------|----------|----------------------|
| **特征分组** | 人工语义分组 | 人工语义分组 | AutoToken 自动 | **AutoToken 自动** |
| **Token 交互** | 手工交叉算子 | 固定重排（零参数） | 可学习矩阵 | **可学习矩阵** |
| **序列建模** | 独立 DIN 模块 | 不做序列建模 | 不做序列建模 | **统一 Causal Attention** |
| **多场景** | 独立模型 | 不涉及 | 场景 MoE + MLoRA | **场景 MoE + MLoRA** |
| **新场景适配** | 重新训练 | 重新训练 | MLoRA 微调 | **MLoRA 微调** |

---

## 九、面试常见追问与回答要点

**Q：AutoToken 的 Top-k 操作如何做到可微分？**
A：通过 softmax 对 Top-k 选出的特征做加权，softmax 本身可微，梯度可以回传到选择矩阵 S，实现端到端学习。

**Q：为什么 Shared Dense MoE 用 Sigmoid 而不是 Softmax/Top-k？**
A：Top-k 稀疏路由（如 RankMixer 的 DTSI）存在专家极化风险，部分专家长期不被激活。Sigmoid 让所有专家始终参与计算，梯度充分，避免死亡专家问题，代价是计算量略高，但在精排场景可接受。

**Q：场景专家的手动 bonus W 怎么设置？**
A：W 是超参数，需要保证当前场景专家的得分足够高以进入 Top-k。实践中设置为门控得分均值的 2-3 倍即可，不需要精确调参，只要保证"必选"语义成立。

**Q：MLoRA 和 LoRA 有什么区别？**
A：本质相同，都是低秩矩阵分解。MLoRA 这里特指挂在预测头上的场景适配器，区别于多模态微调中的 LoRA（挂在 Transformer block 的 in/out projection）。命名上做区分是为了强调它是"多场景适配"语境下的低秩适配。

**Q：序列统一建模这部分，和 OneTrans 有什么区别？**
A：OneTrans 是完全以 Causal Attention 为核心，配合金字塔截断做了大量工程优化。UniRank 只借鉴了 NS-token attend S-token 的思路，主体架构仍以 MTmixAtt 的 Token Mixing + MoE 为核心，序列统一建模是补充而非主线。

---

## 十、参考论文

- **MTmixAtt**：Automated Feature Tokenization and Scenario-Aware Mixture-of-Experts for Industrial Recommendation（美团，2024）
- **RankMixer**：Scaling Up Ranking Models in Industrial Recommenders（字节跳动，2024）
- **OneTrans**：Unifying Sequential Modeling and Feature Interaction in One Transformer for Recommendation（2024）
