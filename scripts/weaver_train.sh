#!/bin/bash

export CUDA_HOME="/home/lixian/Documents/WorkSpace/HiAgent/CUDA_12.2.0"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

export DEBUG_MODE=true
export LOG_PATH="./debug_log_2b.txt"
export CUDA_VISIBLE_DEVICES=2,3
export MAIN_PROCESS_PORT=29507
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_DISABLE=1

# options:
# - Qwen/Qwen2.5-1.5B-Instruct
# - HuggingFaceTB/SmolLM3-3B
REASONER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"   
WEAVER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
TRIGGER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"

# Dataset configs
DATASET_NAME="gsm8k"  # options: gsm8k, gpqa, kodcode, triviaqa

# MemGen configs
TRAIN_METHOD="sft"    # options: sft or grpo

# Augmentation configs:
# - For gsm8k, gpqa, kodcode: MAX_PROMPT_AUG_NUM=1, MAX_INFERENCE_AUG_NUM=5
# - For triviaqa:             MAX_PROMPT_AUG_NUM=6, MAX_INFERENCE_AUG_NUM=0
MAX_PROMPT_AUG_NUM=8
MAX_INFERENCE_AUG_NUM=5
PROMPT_LATENTS_LEN=16
INFERENCE_LATENTS_LEN=8

# NeuroWeave v2.2 switches (override yaml; easy to toggle here)
USE_HYPER_LORA=true          # Hyper-LoRA Weaver (A2)
HYPER_LORA_RANK=16
USE_GRU_MEMORY=true        # G-Mem (A4)
GRU_HIDDEN_SIZE=0            # 0 = use weaver hidden size
BPTT_TRIGGER_STEPS=4

BATCH_SIZE=1

# Resume from checkpoint (e.g. after NCCL timeout). Leave empty to start from scratch.
# Example: RESUME_FROM_CHECKPOINT="results/train/gsm8k/Qwen2.5-1.5B-Instruct/pn=8_pl=16_in=5_il=8_20260205-114243/weaver/checkpoint-1680"
RESUME_FROM_CHECKPOINT=""

# train
python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    main.py \
    --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
    --options \
    model.model_name ${REASONER_MODEL} \
    model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
    model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
    model.weaver.model_name ${WEAVER_MODEL} \
    model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
    model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
    model.weaver.use_hyper_lora ${USE_HYPER_LORA} \
    model.weaver.hyper_lora_rank ${HYPER_LORA_RANK} \
    model.weaver.use_gru_memory ${USE_GRU_MEMORY} \
    model.weaver.gru_hidden_size ${GRU_HIDDEN_SIZE} \
    model.weaver.bptt_trigger_steps ${BPTT_TRIGGER_STEPS} \
    model.trigger.model_name ${TRIGGER_MODEL} \
    model.trigger.active False \
    datasets.mode ${TRAIN_METHOD} \
    run.mode train \
    run.train_weaver True \
    run.train_trigger False \
    run.train_weaver_method ${TRAIN_METHOD} \
    run.weaver.sft.per_device_train_batch_size ${BATCH_SIZE} \
    run.weaver.sft.per_device_train_batch_size ${BATCH_SIZE} \
    run.weaver.sft.bf16 True \
    run.interaction.do_sample True \
    run.interaction.temperature 1.0 \
    run.interaction.max_response_length 1024 \
    $([ -n "${RESUME_FROM_CHECKPOINT}" ] && echo "run.weaver.resume_from_checkpoint ${RESUME_FROM_CHECKPOINT}" || true) \