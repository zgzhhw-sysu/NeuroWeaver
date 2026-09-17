# NeuroWeave v2.2 起步路线图

从哪里开始、按什么顺序做，一目了然。

---

## 总体顺序（按 Phase）

```
Phase A（先做） → Phase B → Phase C → Phase D
 2.1 + 2.2        2.3+2.4    2.5      联调
```

- **Phase A**：Hyper-LoRA Weaver（2.1）→ G-Mem（2.2）→ 用现有 SFT 跑通、验证不崩。
- **Phase B**：Sparse Varentropy Trigger（2.3）→ Gated R-GRPO（2.4）。
- **Phase C**：Staged Warm-up（Stage 0 对比学习 → Stage 1 → Stage 2）。
- **Phase D**：端到端测试、显存/速度、文档。

---

## Phase A 内部：建议第一步

**从 2.1 Hyper-LoRA Weaver 开始**，且按下面顺序做，便于每步可测：

| 步骤 | 做什么 | 涉及文件 | 可自检 |
|------|--------|----------|--------|
| **A1** | 加 Hyper-LoRA 相关配置项 | `configuration_memgen.py`，`modeling_memgen.py`（from_config），`gsm8k.yaml` | 配置加载、默认值正确 |
| **A2** | 在 Weaver 里实现 Hyper-LoRA 分支（含零初始化） | `weaver.py` | 单元测试：use_hyper_lora=False 时行为与原版一致 |
| **A3** | 在 forward/generate 中传入 h_t、走 Hyper-LoRA 路径 | `modeling_memgen.py` | SFT 小步数训练不报错、loss 正常 |
| **A4** | 实现 G-Mem（GRU 单元 + 配置） | `configuration_memgen.py`，`gru_memory.py` 或 `weaver.py` | 单步更新 S_t 正确 |
| **A5** | 在 forward/generate 中接入 G-Mem、BPTT 占位 | `modeling_memgen.py` | 训练/推理不崩、显存可控 |

**第一步具体就是 A1**：只改配置与加载，不动 Weaver 逻辑，保证现有流程不受影响。

---

## 第一步（A1）要改的内容

1. **`memgen/model/configuration_memgen.py`**  
   - 在 `__init__` 中新增（并保存到 `self`）：
     - `weaver_use_hyper_lora: bool = False`
     - `hyper_lora_rank: int = 16`
     - 可选 `task_context_dim`（若暂不用 c_task 可先不加）

2. **`memgen/model/modeling_memgen.py` 的 `from_config`**  
   - 从 `config_dict["weaver"]`（或等价）里读取上述字段，传入 `MemGenConfig`，确保 `MemGenConfig(...)` 能收到这些参数。

3. **`configs/latent_memory/gsm8k.yaml`**  
   - 在 `model.weaver` 下增加示例（可先注释或默认 false）：
     - `use_hyper_lora: false`
     - `hyper_lora_rank: 16`

完成 A1 后，运行一次现有训练（如 `weaver_train.sh` 或等价），确认无报错、配置被正确读入，再进入 A2（在 Weaver 内实现 Hyper-LoRA + 零初始化）。

---

## 之后每一步怎么接

- **A2 做完**：用 `use_hyper_lora=True` 跑 1 个 epoch SFT，看 loss 是否平滑、无 NaN。
- **A3 做完**：同上，确认 forward/generate 都走 Hyper-LoRA 且结果合理。
- **A4–A5 做完**：再开 G-Mem，用 `use_gru_memory=True` 小步数训练，确认 S_t 更新与显存可控。

按这个顺序，**从哪里开始** = **先做 A1（配置）**，然后严格按 A1→A2→A3→A4→A5 推进 Phase A。
