# Phase B 功能测试说明（Sparse Varentropy Trigger + Gated R-GRPO）

通过 **脚本里的环境变量** 开启或关闭 Phase B，无需改 YAML。

---

## 1. 脚本开关位置

### 1.1 Trigger 训练：`scripts/trigger_train.sh`

| 功能 | 变量 | 默认值 | 说明 |
|------|------|--------|------|
| **Sparse Varentropy Trigger (2.3)** | `USE_SPARSE_VARENTROPY_TRIGGER` | `false` | 设为 `true` 开启稀疏 + 熵门控 trigger |
| | `VARENTROPY_PERIOD_K` | `1` | 每 K 步强制检测一次 |
| | `VARENTROPY_THRESHOLD_TAU` | `2.0` | 熵阈值 τ，仅当 H(P_t) > τ 才跑 trigger |
| | `VARENTROPY_TAU_TRAINABLE` | `false` | 是否将 τ 作为可训练参数 |
| | `VARENTROPY_TOP_K` | `50` | 用 top-K logits 近似熵（加速 + 数值稳定） |
| **Gated R-GRPO (2.4)** | `GATED_R_GRPO` | `false` | 设为 `true` 开启门控奖励（长度/重复惩罚仅在对时施加） |
| | `LENGTH_PENALTY_TARGET` | `128.0` | 目标长度 L_target |
| | `LENGTH_PENALTY_TAU` | `32.0` | 长度惩罚温度 τ |
| | `BETA_MAX` | `0.1` | β_t 上限 |
| | `ACC_TARGET` | `0.5` | 准确率目标（用于 β_t 调度） |
| | `ACC_EMA_MU` | `0.9` | 准确率 EMA 系数 |
| | `GAMMA_REP` | `0.0` | 重复惩罚系数（0=关闭） |
| | `NGRAM_REP` | `3` | 重复惩罚 n-gram 阶数 |

### 1.2 评估：`scripts/eval.sh`

评估时 **模型配置需与训练一致**。若 trigger 是用 Phase B 训的，请把下面变量设成与训练时相同：

- `USE_SPARSE_VARENTROPY_TRIGGER`（与训练一致，通常 `true`）
- `VARENTROPY_PERIOD_K`、`VARENTROPY_THRESHOLD_TAU`、`VARENTROPY_TOP_K`

`GATED_R_GRPO` 只影响训练时的 reward 计算，评估时不需要改。

---

## 2. 如何测试 Phase B

**基线已完成**：Phase B 全关时已训过 trigger（如 `.../trigger/checkpoint-52`），eval 得到 `compute_reward ≈ 0.54`。**只需做下面 2、3、4 的对照实验。**

训练时终端输出会自动写入 `logs/trigger_train_<实验标签>_<时间戳>.log`，便于对比各次训练参数与 loss/reward（见 2.4）。

### 2.1 对照实验（只做 2、3、4）

2. **只开 Sparse Varentropy Trigger**  
   - `trigger_train.sh`：`USE_SPARSE_VARENTROPY_TRIGGER=true`，`GATED_R_GRPO=false`  
   - 训练 → 评估 → 对比基线的 reward 与推理速度（理论上 trigger 调用次数会减少）

3. **只开 Gated R-GRPO**  
   - `USE_SPARSE_VARENTROPY_TRIGGER=false`，`GATED_R_GRPO=true`  
   - 训练 → 评估 → 对比 reward、平均生成长度（预期答对时更短、答错不额外罚长度）

4. **两个都开**  
   - `USE_SPARSE_VARENTROPY_TRIGGER=true`，`GATED_R_GRPO=true`  
   - 训练 → 评估 → 与基线及 2、3 对比

### 2.2 训练日志（方便对比参数与曲线）

运行 `bash scripts/trigger_train.sh` 时，终端输出会**同时**写入日志文件，便于事后对比：

- **目录**：项目根目录下 `logs/`（不存在会自动创建）
- **文件名**：`trigger_train_<实验标签>_<YYYYMMDD-HHMMSS>.log`
- **实验标签**：根据当前脚本里的 Phase B 开关自动生成  
  - `baseline`：两个都关  
  - `sparse_only`：只开 Sparse Varentropy  
  - `gated_only`：只开 Gated R-GRPO  
  - `phase_b_both`：两个都开  

例如：`logs/trigger_train_sparse_only_20260206-143022.log`。日志里会保留当次运行的命令行、loss、reward、以及脚本里设置的 Phase B 相关变量，便于和另几次实验对比。

### 2.3 快速冒烟测试（确认不报错）

- **Sparse Varentropy**：  
  - 在 `trigger_train.sh` 里设 `USE_SPARSE_VARENTROPY_TRIGGER=true`，其余 Phase B 保持默认。  
  - 跑 1 个 step 或很少几步（如改 `num_train_epochs` 或只跑几步），看能否正常 forward/backward 并出 loss。  
  - 再在 `eval.sh` 里设 `USE_SPARSE_VARENTROPY_TRIGGER=true`，用同一 checkpoint 跑几条样本，确认生成与 reward 能算出来。

- **Gated R-GRPO**：  
  - 在 `trigger_train.sh` 里设 `GATED_R_GRPO=true`，其余不变。  
  - 跑若干 step，看 log 里 reward/advantage 是否正常（无 NaN）、`acc_ema` 相关是否在更新（若你有打 log 可看，否则只要训练不崩即可）。

### 2.4 评估指标

- **主要**：`compute_reward`（如 GSM8k 正确率）  
- **可选**：平均/中位数生成长度、平均 augmentation 次数（若你有统计）、训练曲线（reward/advantage/loss）

---

## 3. 命令速查

```bash
# 训练 trigger（在 MemGen 项目根目录）
bash scripts/trigger_train.sh

# 评估（先改 LOAD_MODEL_PATH 和 Phase B 开关以匹配 checkpoint）
bash scripts/eval.sh
```

评估结果一般在：  
`results/evaluate/<dataset>/<model>/<run_id>/evaluate/answer.json`，汇总在 `summary_metrics`（如 `compute_reward`）。

---

## 4. 常见问题

- **评估时报错或结果异常**  
  检查 `eval.sh` 里 `USE_SPARSE_VARENTROPY_TRIGGER`、`VARENTROPY_*` 是否与训练时一致；`LOAD_MODEL_PATH` 是否指向该次训练的 trigger checkpoint（例如 `.../trigger/checkpoint-52`）。

- **训练 OOM**  
  Phase B 的 Sparse Varentropy 会多一次 prompt 前向（取 logits），可能略增显存。若 OOM，可先关 `USE_SPARSE_VARENTROPY_TRIGGER`，或再减小 `per_device_train_batch_size` / `num_generations`。

- **Gated R-GRPO 想先关长度惩罚**  
  设 `BETA_MAX=0` 或保持 `GATED_R_GRPO=false`；重复惩罚用 `GAMMA_REP=0` 关闭。
