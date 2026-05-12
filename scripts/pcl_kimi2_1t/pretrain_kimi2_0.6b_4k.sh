#!/bin/bash

TP=2
PP=4
EP=1
CP=1
CP_TYPE='ulysses_cp_algo'
NUM_LAYERS=8
SEQ_LEN=4096
MBS=1
GBS=256
TRAIN_ITERS=20000
SAVE_ITERS=100

DISTRIBUTED_ARGS="
    --nproc_per_node $LOCAL_WORLD_SIZE \
    --nnodes $server_count \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MOE_ARGS="
    --moe-shared-expert-overlap \
    --moe-grouped-gemm \
    --moe-token-dispatcher-type alltoall \
    --use-fused-moe-token-permute-and-unpermute \
    --moe-permutation-async-comm \
    --first-k-dense-replace 2 \
    --moe-layer-freq 1 \
    --n-shared-experts 1 \
    --num-experts 64 \
    --moe-router-topk 2 \
    --moe-ffn-hidden-size 512 \
    --moe-router-load-balancing-type none \
    --moe-router-num-groups 8 \
    --moe-router-group-topk 2 \
    --moe-router-topk-scaling-factor 2.827 \
    --seq-aux \
    --norm-topk-prob \
    --moe-router-score-function sigmoid \
    --moe-router-enable-expert-bias \
    --moe-router-dtype fp32 \
    --moe-expert-capacity-factor 2.0 \
    --moe-pad-expert-input-to-capacity \
"

BALANCE_ARGS="
    --balanced-moe-experts \
"

SWA_ARGS="
    --swa-windows 128 \
    --full-attention-layers ${MANUAL_FULL_LAYERS} \
    --mla-fa-divide-qk \
"

GQA_ARGS="
    --kv-channels 64 \
    --qk-layernorm \
    --num-attention-heads 8 \
    --num-query-groups 2 \
    --group-query-attention \
"


DUALPIPE_ARGS="
    --moe-fb-overlap \
    --schedules-method dualpipev \
"


ROPE_ARGS="
    --beta-fast 1 \
    --beta-slow 1 \
    --rope-scaling-factor 32 \
    --rope-scaling-mscale 1.0 \
    --rope-scaling-mscale-all-dim  1.0 \
    --rope-scaling-original-max-position-embeddings 4096 \
    --rope-scaling-type yarn
"

GPT_ARGS="
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --gemm-gradient-accumulation-fusion \
    --swap-optimizer \
    --recompute-activation-function \
    --moe-zero-memory level0 \
    --expert-tensor-parallel-size 1 \
    --no-shared-storage \
    --use-distributed-optimizer \
    --use-flash-attn \
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP} \
    --sequence-parallel \
    --context-parallel-size ${CP} \
    --context-parallel-algo  ${CP_TYPE} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 512 \
    --ffn-hidden-size 1024 \
    --tokenizer-type PretrainedFromHF  \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings 131072 \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --make-vocab-size-divisible-by 1 \
    --lr 1.0e-4 \
    --train-iters $TRAIN_ITERS \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.02 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --use-rotary-position-embeddings \
    --swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.0e-5 \
    --weight-decay 1e-1 \
    --lr-warmup-iters 2000 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 65536 \
    --vocab-size 163840 \
    --padded-vocab-size 163840 \
    --rotary-base 50000 \
    --norm-epsilon 1e-6 \
    --seed 2233 \
    --bf16 \
    --distributed-timeout-minutes 120 \
"

DATA_ARGS="
    --data-path $DATA_PREFIXES \
    --data-cache-path ${DATA_DIR}/cache3 \
    --split 100,0,0 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --log-throughput \
    --save-interval $SAVE_ITERS \
    --eval-interval $TRAIN_ITERS \
    --eval-iters 0 \
"

PROFILING_ARGS="
    --profile \
    --profile-step-start  60  \
    --profile-step-end 61 \
    --profile-ranks 0 \
    --profile-level level1 \
    --profile-with-cpu \
    --profile-with-memory \
    --profile-record-shapes \
    --profile-save-path $log_dir/profiling \
"

unset HIGH_AVAILABILITY

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $GQA_ARGS \
    $DUALPIPE_ARGS \
    $ROPE_ARGS \
    $MOE_ARGS \
    $OUTPUT_ARGS \
    $DATA_ARGS \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    --distributed-backend nccl \
    2>&1 | tee ${TRAIN_LOG_PATH}
