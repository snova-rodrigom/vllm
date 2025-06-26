#!/bin/bash

MODEL_NAME=$1
PRECISION=$2
TP_SIZE=$3
MAX_MODEL_LEN=$4  # Optional

if [[ -z "$MODEL_NAME" || -z "$PRECISION" || -z "$TP_SIZE" ]]; then
  echo "Usage: $0 <model_checkpoint_path> <precision: fp8|bf16> <tensor_parallel: int> [max_model_len: int]"
  exit 1
fi

# Supported precision flags
declare -A ARGS=(
  ["bf16"]="--dtype bfloat16"
  ["fp8"]="--quantization modelopt"
)

# Resolve model checkpoint
echo "Using model checkpoint: $MODEL_NAME"

# Precision arguments
EXTRA_ARGS="${ARGS[$PRECISION]}"

# Max model length handling
if [[ -n "$MAX_MODEL_LEN" ]]; then
  MAX_LEN_ARG="--max-model-len $MAX_MODEL_LEN"
  echo "Using max-model-len: $MAX_MODEL_LEN"
else
  MAX_LEN_ARG=""
fi

# Find a free port between 8000 and 9000
find_free_port() {
  comm -23 <(seq 8000 9000) <(ss -Htan | awk '{print $4}' | grep -o '[0-9]*$' | sort -u) | head -n 1
}

PORT=$(find_free_port)
if [[ -z "$PORT" ]]; then
  echo "Could not find a free port between 8000–9000."
  exit 1
fi

# Run vLLM
if [[ "$TP_SIZE" -eq 1 ]]; then
  # Find GPU with the lowest memory usage
  FREE_GPU=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | \
    awk '{print $1 " " NR-1}' | sort -n | head -n 1 | awk '{print $2}')
  
  echo "Launching vLLM with TP=1 on GPU $FREE_GPU at port $PORT..."
  CUDA_VISIBLE_DEVICES=$FREE_GPU vllm serve "$MODEL_NAME" \
    --tensor-parallel-size 1 \
    $EXTRA_ARGS \
    $MAX_LEN_ARG \
    --no-enable-prefix-caching \
    --port "$PORT"
else
  echo "Launching vLLM with TP=$TP_SIZE on all GPUs at port $PORT..."
  vllm serve "$MODEL_NAME" \
    --tensor-parallel-size "$TP_SIZE" \
    $EXTRA_ARGS \
    $MAX_LEN_ARG \
    --no-enable-prefix-caching \
    --port "$PORT"
fi
