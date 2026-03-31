#!/bin/bash

# Approximate 0.25B nanochat training script.
# This is intentionally conservative compared to runs/speedrun.sh:
# - targets a ~0.28B model (d11 with current scaling rules)
# - uses shorter context and smaller batch sizes
# - avoids Hopper-only paths such as FP8 / FA3 assumptions
#
# Intended use:
# - 4x RTX 4090: should be a reasonable starting point
# - 2x RTX 3090: may also work with a smaller DEVICE_BATCH_SIZE / fewer iters
#
# Usage:
# bash runs/run_025b.sh
#
# Common overrides:
# NPROC_PER_NODE=2 DEVICE_BATCH_SIZE=1 bash runs/run_025b.sh
# WANDB_RUN=my025b bash runs/run_025b.sh
# SKIP_SETUP=1 SKIP_TOKENIZER=1 bash runs/run_025b.sh

set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

# Ampere / Ada cards are typically happier with fp16 in this repo.
export NANOCHAT_DTYPE="${NANOCHAT_DTYPE:-float16}"

# Logging
WANDB_RUN="${WANDB_RUN:-dummy}"

# Hardware / parallelism
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

# Model scale
DEPTH="${DEPTH:-11}"                  # ~0.28B total params with current rules
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
WINDOW_PATTERN="${WINDOW_PATTERN:-L}" # safer on non-Hopper GPUs

# Pretraining
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-2}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-65536}"
BASE_NUM_ITERATIONS="${BASE_NUM_ITERATIONS:-12000}"
BASE_EVAL_EVERY="${BASE_EVAL_EVERY:-500}"
BASE_EVAL_TOKENS="${BASE_EVAL_TOKENS:-2097152}"
BASE_TAG="${BASE_TAG:-d${DEPTH}_025b}"

# SFT
SFT_DEVICE_BATCH_SIZE="${SFT_DEVICE_BATCH_SIZE:-2}"
SFT_TOTAL_BATCH_SIZE="${SFT_TOTAL_BATCH_SIZE:-65536}"
SFT_NUM_ITERATIONS="${SFT_NUM_ITERATIONS:-2500}"
SFT_EVAL_EVERY="${SFT_EVAL_EVERY:-250}"
SFT_EVAL_TOKENS="${SFT_EVAL_TOKENS:-1048576}"
SFT_TAG="${SFT_TAG:-${BASE_TAG}_sft}"

# Data / tokenizer
TOKENIZER_MAX_CHARS="${TOKENIZER_MAX_CHARS:-2000000000}"
TOKENIZER_VOCAB_SIZE="${TOKENIZER_VOCAB_SIZE:-32768}"
INITIAL_SHARDS="${INITIAL_SHARDS:-8}"
PRETRAIN_SHARDS="${PRETRAIN_SHARDS:-64}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "=============================================="
log "nanochat approx-0.25B training run"
log "=============================================="
log "NPROC_PER_NODE=$NPROC_PER_NODE"
log "DEPTH=$DEPTH MAX_SEQ_LEN=$MAX_SEQ_LEN"
log "DEVICE_BATCH_SIZE=$DEVICE_BATCH_SIZE TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE"
log "NANOCHAT_DTYPE=$NANOCHAT_DTYPE"

# -----------------------------------------------------------------------------
# Setup
if [ -z "${SKIP_SETUP:-}" ]; then
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -d ".venv" ] || uv venv
    uv sync --extra gpu
fi
source .venv/bin/activate

# -----------------------------------------------------------------------------
# Report
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer and dataset
if [ -z "${SKIP_TOKENIZER:-}" ]; then
    log "Downloading initial dataset shards..."
    python -m nanochat.dataset -n "$INITIAL_SHARDS"

    log "Downloading more pretraining shards in background..."
    python -m nanochat.dataset -n "$PRETRAIN_SHARDS" &
    DATASET_DOWNLOAD_PID=$!

    log "Training tokenizer..."
    python -m scripts.tok_train \
        --max-chars="$TOKENIZER_MAX_CHARS" \
        --vocab-size="$TOKENIZER_VOCAB_SIZE"
    python -m scripts.tok_eval

    log "Waiting for pretraining shard download to complete..."
    wait "$DATASET_DOWNLOAD_PID"
else
    log "Skipping tokenizer/data setup because SKIP_TOKENIZER=1"
fi

# -----------------------------------------------------------------------------
# Base pretraining
log "Starting base pretraining..."
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
    --depth="$DEPTH" \
    --model-tag="$BASE_TAG" \
    --run="$WANDB_RUN" \
    --max-seq-len="$MAX_SEQ_LEN" \
    --window-pattern="$WINDOW_PATTERN" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --total-batch-size="$TOTAL_BATCH_SIZE" \
    --num-iterations="$BASE_NUM_ITERATIONS" \
    --eval-every="$BASE_EVAL_EVERY" \
    --eval-tokens="$BASE_EVAL_TOKENS" \
    --core-metric-every=-1 \
    --sample-every=1000 \
    --save-every=1000

log "Running lightweight base eval..."
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_eval -- \
    --model-tag="$BASE_TAG" \
    --device-batch-size=1 \
    --split-tokens=16384 \
    --max-per-task=16

# -----------------------------------------------------------------------------
# SFT data
IDENTITY_PATH="$NANOCHAT_BASE_DIR/identity_conversations.jsonl"
if [ ! -f "$IDENTITY_PATH" ]; then
    log "Downloading identity conversation data..."
    curl -L -o "$IDENTITY_PATH" https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
fi

# -----------------------------------------------------------------------------
# SFT
log "Starting SFT..."
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_sft -- \
    --model-tag="$BASE_TAG" \
    --run="$WANDB_RUN" \
    --max-seq-len="$MAX_SEQ_LEN" \
    --device-batch-size="$SFT_DEVICE_BATCH_SIZE" \
    --total-batch-size="$SFT_TOTAL_BATCH_SIZE" \
    --num-iterations="$SFT_NUM_ITERATIONS" \
    --eval-every="$SFT_EVAL_EVERY" \
    --eval-tokens="$SFT_EVAL_TOKENS" \
    --chatcore-every="$SFT_EVAL_EVERY" \
    --mmlu-epochs=2 \
    --gsm8k-epochs=3

log "Running chat eval on SFT model..."
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_eval -- \
    -i sft \
    -g "$BASE_TAG"

log "Generating report..."
python -m nanochat.report generate

cat <<EOF

Run complete.

Try chatting with the model:
python -m scripts.chat_cli -g "$BASE_TAG" -p "Why is the sky blue?"

Or start the web UI:
python -m scripts.chat_web -g "$BASE_TAG"

If you see OOMs, first try:
NPROC_PER_NODE=$NPROC_PER_NODE DEVICE_BATCH_SIZE=1 SFT_DEVICE_BATCH_SIZE=1 bash runs/run_025b.sh
EOF
