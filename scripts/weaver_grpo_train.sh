#!/bin/bash
# Weaver GRPO training: load SFT weaver, train weaver with GRPO as new weaver baseline.
# After this, set trigger_train.sh LOAD_WEAVER_PATH to this run's weaver output and re-run trigger.
#
# NCCL timeout: GRPO generation can be uneven across ranks -> one rank may take >10min -> ALLREDUCE
# times out. We use max_completion_length 512 (and max_response_length 512) to cap step time.
# If timeout persists, try fewer GPUs (e.g. CUDA_VISIBLE_DEVICES=0,1,2,3) or smaller num_generations.
export CUDA_HOME="/home/lixian/Documents/WorkSpace/HiAgent/CUDA_12.2.0"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

export DEBUG_MODE=true
export LOG_PATH="./debug_log_2b.txt"
export CUDA_VISIBLE_DEVICES=1,3
export MAIN_PROCESS_PORT=29508
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Base model
REASONER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
WEAVER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
TRIGGER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"

DATASET_NAME="gsm8k"

# Must match the weaver you trained with weaver_train.sh (SFT)
MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=5
PROMPT_LATENTS_LEN=16
INFERENCE_LATENTS_LEN=8

USE_HYPER_LORA=true
HYPER_LORA_RANK=16
USE_GRU_MEMORY=true
GRU_HIDDEN_SIZE=0
BPTT_TRIGGER_STEPS=4

# Path to SFT weaver checkpoint (same as current LOAD_WEAVER_PATH in trigger_train.sh)
# After this script, use this run's output as LOAD_WEAVER_PATH for trigger training.
SFT_WEAVER_PATH="/home/lixian/Documents/NeuroWeave/results/train/gsm8k/Qwen2.5-1.5B-Instruct/pn=8_pl=16_in=5_il=8_20260206-222252/weaver"

# Weaver GRPO: keep batch small to avoid OOM; cap length to avoid NCCL timeout (see comment at top)
PER_DEVICE_BATCH=4
GRADIENT_ACCUMULATION_STEPS=16
NUM_GENERATIONS=2
NUM_TRAIN_EPOCHS=1
MAX_COMPLETION_LENGTH=512

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
TRAIN_LOG="${LOG_DIR}/weaver_grpo_train_${TIMESTAMP}.log"
echo "Weaver GRPO: load SFT weaver from ${SFT_WEAVER_PATH}"
echo "Logging to ${TRAIN_LOG}"

(
python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    main.py \
    --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
    --options \
    model.model_name ${REASONER_MODEL} \
    model.load_model_path ${SFT_WEAVER_PATH} \
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
    datasets.mode grpo \
    run.mode train \
    run.train_weaver True \
    run.train_trigger False \
    run.train_weaver_method grpo \
    run.weaver.grpo.num_train_epochs ${NUM_TRAIN_EPOCHS} \
    run.weaver.grpo.per_device_train_batch_size ${PER_DEVICE_BATCH} \
    run.weaver.grpo.per_device_eval_batch_size 4 \
    run.weaver.grpo.num_generations ${NUM_GENERATIONS} \
    run.weaver.grpo.gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    run.weaver.grpo.max_prompt_length 1024 \
    run.weaver.grpo.max_completion_length ${MAX_COMPLETION_LENGTH} \
    run.weaver.grpo.learning_rate 1e-5 \
    run.interaction.do_sample True \
    run.interaction.temperature 1.0 \
    run.interaction.max_response_length ${MAX_COMPLETION_LENGTH}
) 2>&1 | tee "${TRAIN_LOG}"

echo "Done. Check log for the actual run dir. Weaver output: results/train/${DATASET_NAME}/Qwen2.5-1.5B-Instruct/pn=${MAX_PROMPT_AUG_NUM}_pl=${PROMPT_LATENTS_LEN}_in=${MAX_INFERENCE_AUG_NUM}_il=${INFERENCE_LATENTS_LEN}_<timestamp>/weaver/"
echo "Set trigger_train.sh LOAD_WEAVER_PATH to that weaver path, then re-run trigger training."
