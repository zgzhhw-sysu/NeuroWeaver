# NeuroWeave v2.2 实施计划：基于 MemGen 的修改路线图

本文档为**仅规划文档**，不涉及代码修改。在动手改代码前，请先通读本计划并与白皮书对照。

---

## 一、当前代码与白皮书对照摘要

### 1.1 现有架构（MemGen 现状）

| 模块 | 当前实现 | 白皮书 v2.2 目标 |
|------|----------|-------------------|
| **Weaver** | 静态可学习向量 `prompt_query_latents` / `inference_query_latents`，直接拼到序列后过 Weaver  backbone（LoRA），取最后几维 hidden 作为记忆 | **Hyper-LoRA Weaver**：轻量超网络 H_φ 生成低秩 A_t、B_t，W_t = B_t A_t + W_base，M_t = σ(W_t h_t + b_t) |
| **记忆演化** | 无跨步状态；每次 augment 独立 | **G-Mem**：GRU 维护递归隐状态 S_t，门控更新，抗噪、可做 Truncated BPTT |
| **Trigger** | 仅“是否以 delimiter 结尾”+ 可选 2 类分类头；每步若结尾是 delimiter 则跑 Trigger | **Sparse Varentropy Trigger**：m_t 仅在标点/换行/思考词或 t mod K==0 为 1；Trigger(t)=m_t·I[Var(H(P_t))>τ_v]，τ_v 可训练 |
| **奖励 / RL** | 仅 env.compute_reward(completions, solution)→单标量；无长度/重复惩罚 | **Gated R-GRPO**：R = R_outcome + I_correct·(α R_density − β_t P_len) − γ P_rep；β_t 随 Acc_batch 动态调度 |
| **训练流程** | Weaver SFT → Weaver GRPO；Trigger GRPO；无统一阶段划分 | **Staged Warm-up**：阶段 0 结构热启动 → 阶段 1 策略预热 → 阶段 2 端到端 R-GRPO |

### 1.2 关键代码位置速查

- **Weaver 定义与调用**：`memgen/model/weaver.py`（静态 latents + `_augment`），`modeling_memgen.py` 中 `_forward` / `generate` 里对 weaver 的调用。
- **Trigger 定义与何时调用**：`memgen/model/trigger.py`，`modeling_utils.py` 中 `_should_augment`（delimiter + trigger logits）。
- **增强点选取**：`modeling_utils.py` 中 `_select_augment_points_after_delimiter`（训练）、`_should_augment`（生成）。
- **奖励计算**：`WeaverGRPOTrainer._generate_and_score_completions` 中 `_calculate_rewards` → 各 env 的 `compute_reward`；无长度/重复项。
- **配置**：`configs/latent_memory/gsm8k.yaml`（weaver/trigger/run），`MemGenConfig` 在 `configuration_memgen.py`。

---

## 二、按白皮书章节的详细修改计划

### 2.1 Hyper-LoRA Weaver（2.1 节）

**目标**：用超网络生成低秩矩阵 A_t、B_t，得到 W_t = B_t A_t + W_base，使 M_t = σ(W_t h_t + b_t)，计算量从 O(d²) 降到 O(2dr)。

**涉及文件与改动要点**：

1. **`memgen/model/configuration_memgen.py`**
   - 新增：`hyper_lora_rank`（默认 16 或 32）、`weaver_use_hyper_lora`（bool）、可选 `task_context_dim`（若用 c_task）。
   - 保留现有 `prompt_latents_len` / `inference_latents_len` 等，用于 latent 序列长度。

2. **`memgen/model/weaver.py`**
   - **当前**：仅维护 `prompt_query_latents`、`inference_query_latents`，在 `_augment` 中拼到输入后过 backbone，取最后若干 token 的 hidden。
   - **改动**：
     - 引入 **Hyper-LoRA 分支**（可与现有“静态 latent 拼接”并存，通过配置切换）：
       - 输入：当前上下文在 reasoner 侧的表示（或 weaver 侧等价物）→ 提取 **h_t**（例如最后 token 的 hidden，或经 pooling）；若有任务信息则拼 **c_task**。
       - 轻量 **超网络 H_φ**：例如 2 个小型 MLP/线性层，分别输出：
         - A_t: (d, r)，B_t: (r, d)；或按头/块拆分为多组低秩（依实现选择）。
       - 定义 **W_base**：可学习 d×d 或沿用现有 projection；等效权重 W_t = W_base + B_t @ A_t。
       - 计算 **M_t**：对当前步的 h_t 做 M_t = σ(W_t @ h_t + b_t)，再 reshape 为 (batch, prompt_latents_len 或 inference_latents_len, hidden_size) 作为 latent 序列（或按白皮书逐 token 生成）。
     - 若保留“静态 latent”模式，可保留原 `_augment` 路径，用配置选择 Hyper-LoRA 路径 vs 原路径。
   - **接口**：`augment_prompt` / `augment_inference` 的入参需包含“当前 h_t”（以及可选 c_task）；调用方在 `modeling_memgen.py` 中传入。

3. **`memgen/model/modeling_memgen.py`**
   - **Forward（训练）**：在每次调用 weaver 前，从当前 `current_inputs_embeds`（或 reasoner hidden）取最后一维作为 h_t；若是多段推理，可传“当前段末尾”的 h_t。将 h_t（及 c_task）传入 weaver 的 augment 接口。
   - **Generate（推理）**：同样在每次决定 augment 后，用当前步的 hidden 作为 h_t 传入 weaver。
   - 确保 reasoner_to_weaver / weaver_to_reasoner 的用法与 Hyper-LoRA 输出维度一致（仍为 latent_len × hidden_size）。

4. **配置与脚本**
   - 在 `configs/latent_memory/*.yaml` 的 weaver 段增加：`use_hyper_lora: true`、`hyper_lora_rank: 16`（或 32）等。
   - `MemGenModel.from_config` 中解析上述新配置并写入 `MemGenConfig`。

**工程完善（首席架构师建议）**：  
- **零初始化策略 (Zero-Init)**：超网络 H_φ 训练初期若随机初始化，生成的 A_t、B_t 会使 W_t 产生巨大噪声，破坏 Weaver 预训练表征导致“崩坏”。  
- **必须**：在实现中**强制 H_φ 输出层零初始化**（或生成 A_t、B_t 的最后一层权重初始化为 0）。  
- **效果**：训练开始时 W_t = W_base，模型从“静态 Weaver”状态平滑启动，随训练逐渐引入动态调节，保证数值稳定性。  
- **优先级**：**High**（防崩坏）。

**验收要点**：  
- 显存/FLOPs 明显低于“全连接超网络生成 d×d”的假设实现；  
- 可关闭 Hyper-LoRA 时行为与现有 Weaver 一致（若保留双模式）；  
- Hyper-LoRA 分支输出层已零初始化，冷启动时等效 W_t = W_base。

---

### 2.2 G-Mem：GRU 门控记忆演化（2.2 节）

**目标**：维护递归隐状态 S_t，用 GRU 门控（更新门 z_t、重置门 r_t）决定“保持旧记忆”还是“写入新信息”；训练时支持 Truncated BPTT（最近 k 步）。

**涉及文件与改动要点**：

1. **`memgen/model/configuration_memgen.py`**
   - 新增：`use_gru_memory`（bool）、`gru_hidden_size`（可与 hidden_size 一致或略小）、`bptt_trigger_steps`（k，用于截断 BPTT）。

2. **`memgen/model/weaver.py` 或新建 `memgen/model/gru_memory.py`**
   - 实现 **G-Mem 单元**：
     - 状态 S_t ∈ R^d（或 gru_hidden_size），按 batch 维护。
     - 输入：当前步的 Weaver 输出（或 h_t 与当前 latent 的融合），以及上一步 S_{t-1}。
     - 公式：z_t = σ(W_z· concat)，r_t = σ(W_r· concat)，S̃_t = tanh(W_h· concat(r_t⊙S_{t-1}, input))，S_t = (1-z_t)⊙S_{t-1} + z_t⊙S̃_t。
   - 在 **训练** forward 中：每在一个 augment 点生成 latent 后，用 G-Mem 更新 S_t，并将 S_t 或由其得到的表示反馈到“下一段”的 Weaver/Reasoner 输入（具体方式需与 2.1 的 M_t 设计统一：例如用 S_t 参与生成下一段 latent，或拼入 context）。
   - **推理**：在 generate 循环中同样维护 S_t，每 augment 一次更新一次；不跨 sequence 共享状态。

3. **`memgen/model/modeling_memgen.py`**
   - Forward：在按 augment 点切分的循环内，在每次 weaver 输出后调用 G-Mem 更新；若做 Truncated BPTT，需在 backward 时只对最近 `bptt_trigger_steps` 次 Trigger 激活步回传（需在 trainer 或 forward 里记录“激活步”索引或 mask）。
   - Generate：在每次插入 latent 后更新 G-Mem 状态，供下一步使用。

4. **Trainer / 梯度**
   - 若 Truncated BPTT 在“序列级”实现：在 `_forward` 里对“非最近 k 次”的 G-Mem 输出做 `.detach()` 或等价截断，避免长链梯度。
   - 或在 `WeaverGRPOTrainer` 的 loss 中只对“最近 k 次 augment”的 token 算 loss（需与 GRPO 的 mask 一致）。

**工程完善（首席架构师建议）**：  
- **Scheduled Sampling（计划采样）**：若 G-Mem 每步都喂入“真实”历史状态（Teacher Forcing），推理时模型需用自己生成的历史，误差累积会导致 Exposure Bias 甚至崩溃。  
- **建议**：在 Stage 0 或 Stage 2 的 G-Mem 训练中，以一定概率**不喂真实值**，而是用模型上一步生成的（可能带噪）状态继续，提高推理鲁棒性。  
- **实现**：配置项如 `scheduled_sampling_prob`，每步以该概率用 `model_output.detach()` 或采样结果替代 ground truth 作为下一步输入。  
- **优先级**：Low（优化项）。
  
**验收要点**：  
- 长序列下显存可控、无梯度爆炸；  
- 可关闭 G-Mem 时与原逻辑等价。

---

### 2.3 Sparse Varentropy Trigger（2.3 节）

**目标**：仅在“标点/换行/思考词”或“每 K 步”处计算 Varentropy；Trigger(t) = m_t · I[Var(H(P_t)) > τ_v]，τ_v 可训练或动态调整。

**涉及文件与改动要点**：

1. **`memgen/model/configuration_memgen.py`**
   - 新增：`use_sparse_varentropy_trigger`（bool）、`varentropy_period_k`（每 K 步强制检测）、`varentropy_threshold_tau`（τ_v 初值）、`varentropy_tau_trainable`（bool）、**`varentropy_top_k`**（如 50 或 100，仅用 Top-K logits 近似熵/变熵，见工程完善）。  
   - 可选：`punct_token_ids` 或从 tokenizer 解析的“标点/换行”集合。

2. **`memgen/model/trigger.py`**
   - **当前**：输入 (input_ids, attention_mask, position_ids)，输出 (batch, seq_len, 2) logits；仅在“序列结尾是否 delimiter”时被调用。
   - **改动**：
     - 保留原有“2 类分类头”用于“是否在该点插入记忆”的决策。
     - 新增 **稀疏 Varentropy 逻辑**（在调用 Trigger 的地方使用，见下）：
       - 仅在 **m_t=1** 的位置计算 Varentropy：  
         - 取 reasoner 在该位置的 logits P_t，在词表上做 softmax 得分布，计算熵 H(P_t)；若在 batch 内或时间窗内有多样本，可算 Var(H(P_t))；白皮书为“不确定性”阈值，可用单步熵或方差。
       - 条件：Var(H(P_t)) > τ_v（或 H(P_t) > τ_v，依白皮书表述）时才有资格触发；再与原有“delimiter”条件结合（例如 m_t 仅在这些位置为 1）。
   - 若 τ_v 可训练：将 τ_v 设为 nn.Parameter 或通过小网络从全局状态生成，并在 Trigger 的优化器中更新。
   - **工程完善**：**Top-K 截断计算**——全词表（vocab ~100k）上做 log_softmax 与熵/变熵，显存与数值稳定性均成问题。**只取 Logits 最大的前 K（如 50 或 100）个 token 近似计算熵与 Varentropy**；尾部极低概率 token 对熵贡献极小但计算量大且引入噪声。公式上对 `TopK_Logits` 计算 Varentropy 即可，速度可提升 10 倍以上，并改善数值稳定性。配置项：`varentropy_top_k: 50`（或 100）。**优先级**：Medium（提速度、稳数值）。

3. **`memgen/model/modeling_utils.py`**
   - **`_should_augment`**（生成）：
     - 先算 **稀疏掩码 m_t**：  
       - 若当前 token x_t 属于标点/换行/思考词集合（通过 input_ids 查表），则 m_t=1；  
       - 或若 t mod K == 0，则 m_t=1；否则 m_t=0。
     - 仅当 m_t=1 时，才跑 reasoner 取 logits → 算 Varentropy（或熵）→ 与 τ_v 比较；若通过则再跑 Trigger 分类头（或直接置 1）；否则不跑 Trigger，augment_decision=0 或 -100。
   - **训练时**：`_select_augment_points_after_delimiter` 可扩展为“仅在 m_t=1 的索引集合内”选点，或保持“delimiter + 标点”一致，并在这些点上用 Varentropy 条件过滤（可选）。

4. **`memgen/model/modeling_memgen.py`**
   - Generate 循环中：在每一步得到 reasoner logits 后，若 m_t=1，计算 Varentropy 再决定是否调用 Trigger；避免每步全词表 softmax，仅 m_t=1 的步计算，降低约 90% 的额外计算（相对“每步都算”）。

5. **Token 集合**
   - 在配置或 tokenizer 中维护“标点/换行/思考词”的 token id 集合 D_punct；可从 tokenizer 的词汇表中按字符或关键字筛选，或从数据统计得到。

**验收要点**：  
- 推理时 Trigger 相关计算量显著减少；  
- 行为符合“读完一句再思考”的稀疏检测。

---

### 2.4 Gated R-GRPO（第 3 节）

**目标**：奖励形如  
R(o_i) = R_outcome + I_correct · (α R_density − β_t P_len) − γ P_rep，  
其中 β_t 随 batch 准确率动态调度；仅当“答对”时才施加长度惩罚。

**涉及文件与改动要点**：

1. **`memgen/trainer/weaver_grpo_trainer.py`**
   - **`_generate_and_score_completions`** 中，在得到 `rewards_per_func`（即各 reward 函数的基础结果，如 0/1）之后：
     - 从 reward 函数或单独逻辑得到 **per-sample 正确性**：  
       - 若 env 返回的就是 0/1，则 I_correct = (rewards_per_func[:, 0] == 1) 或等价；若有多个 reward 函数，需约定哪个是“outcome”（如 compute_reward）。
     - 计算 **R_outcome**：例如正确 +1、错误 -1，或直接用现有 rewards_per_func。
     - 计算 **P_len**：如 tanh((L - L_target) / τ)，L 为 completion 长度，L_target、τ 为配置。
     - 计算 **P_rep**：N-gram 重复惩罚（如 2/3-gram 重复率或已有重复惩罚项）。
     - **β_t 动态调度**：  
       - **工程完善**：直接用当前 batch 正确率 Acc_batch 会导致训练中 β_t 剧烈抖动，难以收敛。**使用 EMA（指数移动平均）平滑准确率**：Acc_ema = μ * Acc_ema + (1−μ) * Acc_batch，用平滑后的 Acc_ema 代入下式计算 β_t。  
       - β_t = β_max * σ(k * (Acc_ema - Acc_target))，σ 为 sigmoid，k、Acc_target 为超参；μ 为 EMA 系数（如 0.9）。  
       - **优先级**：Medium（稳训练）。
     - 组合：R = R_outcome + I_correct * (α * R_density - β_t * P_len) - γ * P_rep；若暂无 R_density 可先设为 0。
   - 上述 R 替代原先仅用 `rewards_per_func` 加权求和的 `rewards`，再照常做 group normalize、advantages。

2. **`memgen/trainer/utils.py` 或新工具函数**
   - 实现：`compute_length_penalty(completion_lengths, L_target, tau)`、`compute_repetition_penalty(completion_ids, ngram)`、`dynamic_beta(acc_batch, acc_target, beta_max, k)`。

3. **配置**
   - 在 `run.weaver.grpo` 中增加：`gated_r_grpo: true`、`length_penalty_target`、`length_penalty_tau`、`alpha_density`、`beta_max`、`acc_target`、`k_schedule`、`gamma_rep`、`ngram_rep` 等；**`acc_ema_mu`**（EMA 系数，如 0.9，用于平滑 Acc 后计算 β_t，见工程完善）。与现有 `memory_reward_alpha/beta`、`length_penalty_coef` 区分并文档化。

4. **Env 接口**
   - 若需“正确性”与“标量奖励”分离，可让 `compute_reward` 返回 `(reward_scalar, is_correct)` 或在后处理中由 reward 推断 is_correct（如 reward>0.5 即正确）。

**验收要点**：  
- 答错时长度不惩罚、答对时长度惩罚生效；  
- β_t 随准确率变化符合公式；  
- 不破坏现有 GRPO loss 与 advantage 计算。

---

### 2.5 Staged Warm-up 训练路线图（第 4 节）

**目标**：三阶段——阶段 0 结构热启动（只训 Hypernetwork + GRU）；阶段 1 策略预热（只训 Trigger + τ）；阶段 2 端到端 R-GRPO（全量 + Gated R-GRPO + Truncated BPTT）。

**涉及文件与改动要点**：

1. **`memgen/runner.py`**
   - **当前**：`train()` 先 `_train_weaver()` 再 `_train_trigger()`；weaver 内部按 `train_weaver_method` 选 SFT 或 GRPO。
   - **改动**：支持 **stage** 配置（如 `run.stage: 0 | 1 | 2`）。
     - **Stage 0（结构热启动）**：  
       - 冻结：Trigger（始终激活或固定策略）、Reasoner。  
       - 训练：Hypernetwork（2.1）、GRU（2.2）；若仍用静态 Weaver，则训练 Weaver + 投影层。  
       - 数据：高质量 CoT（如 GSM8K），**强制在每个句号/标点处激活 Weaver**（即 augmentation 点固定为 delimiter，不学 Trigger）。  
       - **Loss（工程完善）**：原计划“重构下一句 Embedding (MSE)”在工程上易导致 latent 学到“平均值”而模糊、丢失语义。**建议引入 InfoNCE 对比学习损失**：  
         - **正样本**：S_t（或当前 latent 表示）与“真实下一句 Embedding”的相似度；  
         - **负样本**：S_t 与“Batch 内其他无关句子 Embedding”的相似度。  
         - 对比学习比生成重构更易收敛，且迫使 Weaver 学习更强语义区分度。可实现为 MSE + InfoNCE 混合，或仅 InfoNCE。  
         - **优先级**：Medium（提效果）。
     - **Stage 1（策略预热）**：  
       - 冻结：Weaver、Reasoner。  
       - 训练：Trigger 分类头、Varentropy 阈值 τ（若可训练）。  
       - 方法：用预训练模型的 Varentropy 分布初始化 τ；只训练 Trigger 头（及 τ）。
     - **Stage 2（端到端 R-GRPO）**：  
       - 解冻：Hypernetwork、GRU、Trigger 等全部参与更新。  
       - 使用 Gated R-GRPO（2.4）、Truncated BPTT（2.2）、动态 β_t。
   - 为 Stage 0 实现 **WeaverStructuralTrainer**（或扩展现有 SFT）：输入为“到某 delimiter 为止的序列”，标签为“下一句”的 embedding 或 token；**loss 建议采用 InfoNCE 对比损失**（或 MSE + InfoNCE），见上文 Stage 0 Loss 完善。

2. **`memgen/trainer/weaver_grpo_trainer.py`**
   - 在 Stage 2 时启用 Gated R-GRPO 与 Truncated BPTT（若 BPTT 在 trainer 内通过 mask 实现）。
   - 可选：在 Stage 0/1 用不同 `fix_component` / `open_component` 组合，由 runner 传入 stage。

3. **配置**
   - `run.stage: 0`、`run.stage_0_epochs`、`run.stage_1_epochs`、`run.stage_2_epochs`；以及各 stage 的 dataloader、max_length 等。

4. **数据**
   - Stage 0：需要“按句/按 delimiter 切分”的 CoT 数据，或沿用现有 GSM8K 等，在 collate 时构造“前缀 → 下一句”的监督信号。

**验收要点**：  
- 三阶段可顺序跑通；  
- Stage 0 不依赖 RL、Stage 2 正确使用 Gated R-GRPO 与 BPTT。

---

## 三、配置与兼容性

- **向后兼容**：通过 `use_hyper_lora=false`、`use_gru_memory=false`、`use_sparse_varentropy_trigger=false`、`gated_r_grpo=false`、`stage=null`（或仅 weaver/trigger 两阶段）保持与现有 MemGen 行为一致。
- **配置层级**：建议在 `configs/latent_memory/` 下增加 `neuroweave_v22.yaml` 或在其内用注释标出 v2.2 新增项，避免破坏现有 gsm8k 等配置。

---

## 四、实施顺序建议

1. **Phase A（核心结构）**  
   - 2.1 Hyper-LoRA Weaver（含配置与 from_config）  
   - 2.2 G-Mem GRU（含 BPTT 占位或简单版）

2. **Phase B（Trigger 与奖励）**  
   - 2.3 Sparse Varentropy Trigger  
   - 2.4 Gated R-GRPO

3. **Phase C（训练流程）**  
   - 2.5 Staged Warm-up（Stage 0/1/2 + StructuralTrainer + runner 分支）

4. **Phase D（联调与优化）**  
   - 端到端测试、显存/速度验证、超参与文档。

---

## 五、风险与依赖

- **显存**：Hyper-LoRA + GRU 会略增显存，通过 BPTT 截断与 r=16 控制。  
- **依赖**：现有 peft、transformers、trl 版本需支持当前用法；若 Trigger 需取 reasoner 中间 logits，需确认 generate 循环中是否方便取到。  
- **正确性门控**：Gated R-GRPO 依赖“正确性”信号，需与各 env 的 `compute_reward` 约定语义（或统一返回 (score, is_correct)）。

---

## 六、工程完善建议总结表（Chief Architect Refinements）

| 模块 | 原计划 | **建议修改/完善** | 优先级 |
|------|--------|-------------------|--------|
| **Hyper-LoRA** | 标准初始化 | **H_φ 输出层零初始化** (Zero-Init)，使训练初 W_t = W_base | **High**（防崩坏） |
| **G-Mem (Stage 0)** | MSE 重构 Loss | **InfoNCE 对比损失**（正：S_t vs 真实下一句；负：S_t vs batch 内其他句）；或 MSE+Contrastive | Medium（提效果） |
| **Varentropy** | 全词表计算 | **Top-K（如 K=50/100）截断计算**熵与变熵，速度与数值稳定性双提升 | Medium（提速度） |
| **R-GRPO** | 基于 Acc_batch 调度 β_t | **基于 EMA(Acc_batch) 平滑调度**，β_t = f(Acc_ema) | Medium（稳训练） |
| **G-Mem 训练** | Teacher Forcing | **Scheduled Sampling**：以一定概率用模型上一步输出替代 GT 作为下一步输入 | Low（优化项） |

实施建议：**先从 Phase A（2.1 & 2.2）入手**，跑通 Stage 0 的对比学习，验证 Weaver 能压缩语义后，再进行后续 RL 对齐。

---

**文档版本**：v1.1  
**对应白皮书**：NeuroWeave v2.2  
**状态**：仅计划，暂不修改代码；已纳入首席架构师工程完善建议。
