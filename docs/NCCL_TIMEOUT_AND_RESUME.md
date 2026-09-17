# NCCL 超时与断点恢复

## 错误含义

出现类似：

```text
Watchdog caught collective operation timeout: WorkNCCL(SeqNum=..., OpType=ALLGATHER/ALLREDUCE, ...) ran for 1800002 milliseconds before timing out.
```

表示 **多卡分布式训练时，有一次集体通信（ALLGATHER/ALLREDUCE）在 30 分钟内没有完成**，NCCL 认为超时并终止训练。

常见原因：

1. **某一 rank 先挂掉**（OOM、异常、SIGABRT 等），其它 rank 一直在等它参与集体通信，直到 30 分钟超时。
2. **各 rank 步数不一致**（例如 dataloader 没 `drop_last`），导致有的 rank 提前结束或多跑一步，集体通信对不齐。
3. **网络/硬件** 偶发问题，导致某次通信卡住。

你这次是 8 卡（nranks 8），报错里 rank 5 收到 SIGABRT、rank 7/1 等报 ALLGATHER/ALLREDUCE 超时，符合「先有一个 rank 出问题，其它 rank 等集体通信超时」的情况。

## 断点恢复

checkpoint 已保存到例如：

```text
/amax/home/lixian/Documents/WorkSpace/MemGen/results/train/gsm8k/Qwen2.5-1.5B-Instruct/pn=8_pl=16_in=5_il=8_20260205-114243/weaver/checkpoint-1680
```

从该 checkpoint 继续训练：

1. 打开 **`scripts/weaver_train.sh`**。
2. 设置 **`RESUME_FROM_CHECKPOINT`** 为上述路径（或相对路径，与当前工作目录一致），例如：
   ```bash
   RESUME_FROM_CHECKPOINT="results/train/gsm8k/Qwen2.5-1.5B-Instruct/pn=8_pl=16_in=5_il=8_20260205-114243/weaver/checkpoint-1680"
   ```
3. 保持 **GPU 数量、batch size、脚本里其它训练参数** 与上次一致（尤其是 8 卡 + DeepSpeed ZeRO-2）。
4. 重新运行 `bash scripts/weaver_train.sh`。

训练会从 checkpoint-1680 的步数/优化器状态继续，而不是从头开始。

## 降低再次超时的建议

- **确认每卡显存**：若某卡 OOM，可适当减小 `BATCH_SIZE` 或序列长度。
- **保持 dataloader 对齐**：yaml 里 `dataloader_drop_last: true` 已设时不要改成 false，避免各 rank 步数不同。
- **需要时可加大 NCCL 超时**（仅缓解，不治本）：
  ```bash
  export TORCH_NCCL_BLOCKING_WAIT=1
  export TORCH_DISTRIBUTED_TIMEOUT=3600
  ```
- **先用少卡验证**：例如 1～2 卡跑通、再上 8 卡，便于区分是 OOM 还是多卡同步问题。
