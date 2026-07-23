#!/usr/bin/bash

set -euo pipefail

exec ts serve \
    --engine-startup-timeout 3600 \
    --distributed-timeout-seconds 3600 \
    --model amd/Kimi-K2.5-MXFP4 \
    --attn-tp-size 8 \
    --ep-size 8 \
    --max-model-len 80000 \
    --max-num-seqs 16 \
    --max-prefill-tokens 8192 \
    --chunked-prefill-size 8192 \
    --gpu-memory-utilization 0.8 \
    --disable-cuda-graph-padding \
    --disable-prefill-graph \
    --trust-remote-code \
    --attention-backend mla \
    --drafter-attention-backend mha \
    --moe-backend triton \
    --quantization mxfp4 \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path lightseekorg/kimi-k2.5-eagle3 \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --speculative-draft-model-quantization unquant \
    --sampling-backend greedy \
    --disable-kvstore \
    --kvstore-ratio 0.0 \
    --enable-cache-report \
    --host 127.0.0.1 \
    --port "${TOKENSPEED_PORT:-8000}"
