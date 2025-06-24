#!/bin/bash

MODEL_SIZE=$1
PRECISION=$2
TP_SIZE=$3

if [[ -z "$MODEL_SIZE" || -z "$PRECISION" || -z "$TP_SIZE" ]]; then
  echo "Usage: $0 <model_size: 8B|70B> <precision: fp8|bf16> <tensor_parallel: int>"
  exit 1
fi

# Supported model keys are: <MODEL_SIZE>_<PRECISION>
declare -A META_MODELS=(
  ["1B_bf16"]="meta-llama/Llama-3.2-1B-Instruct"
  ["8B_bf16"]="meta-llama/Llama-3.1-8B-Instruct"
  ["70B_bf16"]="meta-llama/Llama-3.3-70B-Instruct"
)

declare -A NVIDIA_MODELS=(
  ["8B_fp8"]="nvidia/Llama-3.1-8B-Instruct-FP8"
  ["70B_fp8"]="nvidia/Llama-3.3-70B-Instruct-FP8"
)

declare -A ARGS=(
  ["bf16"]="--dtype bfloat16"
  ["fp8"]="--quantization modelopt"
)

# Composite key for lookup
KEY="${MODEL_SIZE}_${PRECISION}"

# Resolve model name
if [[ -n "${META_MODELS[$KEY]}" ]]; then
  MODEL_NAME="${META_MODELS[$KEY]}"
elif [[ -n "${NVIDIA_MODELS[$KEY]}" ]]; then
  MODEL_NAME="${NVIDIA_MODELS[$KEY]}"
else
  echo "Invalid model size or precision: MODEL_SIZE=$MODEL_SIZE, PRECISION=$PRECISION"
  exit 1
fi

# Resolve extra args
EXTRA_ARGS="${PRECISION_ARGS[$PRECISION]}"

# Pick an unused port between 8000–9000
find_free_port() {
  comm -23 <(seq 8000 9000) <(ss -Htan | awk '{print $4}' | grep -o '[0-9]*$' | sort -u) | head -n 1
}

PORT=$(find_free_port)
if [[ -z "$PORT" ]]; then
  echo "Could not find a free port between 8000–9000."
  exit 1
fi

if [[ "$TP_SIZE" -eq 1 ]]; then
  # Find GPU with lowest memory used
  FREE_GPU=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | \
    awk '{print $1 " " NR-1}' | sort -n | head -n 1 | awk '{print $2}')
  echo "Launching vLLM with TP=1 on GPU $FREE_GPU at port $PORT..."
  CUDA_VISIBLE_DEVICES=$FREE_GPU vllm serve "$MODEL_NAME" \
    --tensor-parallel-size 1 \
    $EXTRA_ARGS \
    --no-enable-prefix-caching \
    --port "$PORT"
else
  echo "Launching vLLM with TP=$TP_SIZE on all GPUs at port $PORT..."
  vllm serve "$MODEL_NAME" \
    --tensor-parallel-size "$TP_SIZE" \
    $EXTRA_ARGS \
    --no-enable-prefix-caching \
    --port "$PORT"
fi