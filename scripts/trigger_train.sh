#!/bin/bash

export DEBUG_MODE=true
export LOG_PATH="./debug_log_2b.txt"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MAIN_PROCESS_PORT=29507
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_DISABLE=1
# Reduce OOM / fragmentation (trigger GRPO generation is very memory-heavy)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# options:
# - Qwen/Qwen2.5-1.5B-Instruct
# - HuggingFaceTB/SmolLM3-3B
REASONER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"   
WEAVER_MODEL="Qwen/Qwen2.5-1.5B-Instruct" 
TRIGGER_MODEL="Qwen/Qwen2.5-1.5B-Instruct" 

# Dataset configs
DATASET_NAME="gsm8k"  # options: gsm8k, gpqa, kodcode, triviaqa
DATASET_MODE="grpo"   # options: sft or grpo

# MemGen configs
TRAIN_METHOD="grpo"   # options: sft or grpo

# Augmentation configs:
# - For gsm8k, gpqa, kodcode: MAX_PROMPT_AUG_NUM=1, MAX_INFERENCE_AUG_NUM=5
# - For triviaqa:             MAX_PROMPT_AUG_NUM=6, MAX_INFERENCE_AUG_NUM=0
MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=5
PROMPT_LATENTS_LEN=16
INFERENCE_LATENTS_LEN=8

# NeuroWeave v2.2: must match the weaver you trained (same as weaver_train.sh)
USE_HYPER_LORA=true
HYPER_LORA_RANK=16
USE_GRU_MEMORY=true
GRU_HIDDEN_SIZE=0
BPTT_TRIGGER_STEPS=4

# Phase B: Sparse Varentropy Trigger (2.3) — turn on to only run trigger at punct / every K steps with entropy gate
# Current run: phase_b_both (sparse + gated)
USE_SPARSE_VARENTROPY_TRIGGER=true
VARENTROPY_PERIOD_K=2
VARENTROPY_THRESHOLD_TAU=2.0
VARENTROPY_TAU_TRAINABLE=false
VARENTROPY_TOP_K=50

# Phase B: Gated R-GRPO (2.4) — turn on for length/rep penalty gated by correctness + EMA beta
# Current run: phase_b_both (sparse + gated)
GATED_R_GRPO=true
LENGTH_PENALTY_TARGET=200.0
LENGTH_PENALTY_TAU=40.0
BETA_MAX=0.06
ACC_TARGET=0.5
K_SCHEDULE=10.0
ACC_EMA_MU=0.9
GAMMA_REP=0.0
NGRAM_REP=3

# Path to trained weaver (Route A: load this weaver, then train trigger only)
# Example: .../weaver/checkpoint-1680  or  .../weaver  if you saved final model there
LOAD_WEAVER_PATH="/amax/home/lixian/Documents/WorkSpace/MemGen/results/train/gsm8k/Qwen2.5-1.5B-Instruct/weaver/weaver/checkpoint-1680"

# Trigger GRPO is memory-heavy (full model + generation). If OOM, lower batch_size / num_generations / max_response_length below.

# --- Training log: terminal output is also written to logs/ for later comparison ---
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
if [ "${USE_SPARSE_VARENTROPY_TRIGGER}" = true ] && [ "${GATED_R_GRPO}" = true ]; then
    PHASE_B_TAG="phase_b_both"
elif [ "${USE_SPARSE_VARENTROPY_TRIGGER}" = true ]; then
    PHASE_B_TAG="sparse_only"
elif [ "${GATED_R_GRPO}" = true ]; then
    PHASE_B_TAG="gated_only"
else
    PHASE_B_TAG="baseline"
fi
TRAIN_LOG="${LOG_DIR}/trigger_train_${PHASE_B_TAG}_${TIMESTAMP}.log"
echo "Phase B: sparse=${USE_SPARSE_VARENTROPY_TRIGGER} gated=${GATED_R_GRPO} -> tag=${PHASE_B_TAG}"
echo "Logging to ${TRAIN_LOG}"

# train (stdout+stderr tee to log; remove 'tee' to only write log and not print to terminal)
(
python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    main.py \
    --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
    --options \
    model.model_name ${REASONER_MODEL} \
    model.load_model_path ${LOAD_WEAVER_PATH} \
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
    model.trigger.active True \
    model.trigger.use_sparse_varentropy_trigger ${USE_SPARSE_VARENTROPY_TRIGGER} \
    model.trigger.varentropy_period_k ${VARENTROPY_PERIOD_K} \
    model.trigger.varentropy_threshold_tau ${VARENTROPY_THRESHOLD_TAU} \
    model.trigger.varentropy_tau_trainable ${VARENTROPY_TAU_TRAINABLE} \
    model.trigger.varentropy_top_k ${VARENTROPY_TOP_K} \
    datasets.mode ${DATASET_MODE} \
    run.mode train \
    run.train_weaver False \
    run.train_trigger True \
    run.train_trigger_method ${TRAIN_METHOD} \
    run.trigger.grpo.per_device_train_batch_size 1 \
    run.trigger.grpo.per_device_eval_batch_size 1 \
    run.trigger.grpo.num_train_epochs 1 \
    run.trigger.grpo.num_generations 2 \
    run.trigger.grpo.gradient_accumulation_steps 16 \
    run.trigger.grpo.gated_r_grpo ${GATED_R_GRPO} \
    run.trigger.grpo.length_penalty_target ${LENGTH_PENALTY_TARGET} \
    run.trigger.grpo.length_penalty_tau ${LENGTH_PENALTY_TAU} \
    run.trigger.grpo.beta_max ${BETA_MAX} \
    run.trigger.grpo.acc_target ${ACC_TARGET} \
    run.trigger.grpo.k_schedule ${K_SCHEDULE} \
    run.trigger.grpo.acc_ema_mu ${ACC_EMA_MU} \
    run.trigger.grpo.gamma_rep ${GAMMA_REP} \
    run.trigger.grpo.ngram_rep ${NGRAM_REP} \
    run.interaction.do_sample True \
    run.interaction.temperature 1.0 \
    run.interaction.max_response_length 1024
) 2>&1 | tee "${TRAIN_LOG}"

