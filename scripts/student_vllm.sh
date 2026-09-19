#!/bin/bash

STUDENT_MODEL="${1:-Qwen/Qwen3.5-0.8B}"
STUDENT_PORT="${2:-8000}"

echo "Starting STUDENT(${STUDENT_MODEL}) on port ${STUDENT_PORT}"

VLLM_SERVER_DEV_MODE=1 vllm serve ${STUDENT_MODEL} \
    --weight-transfer-config '{"backend": "nccl"}' \
    --port ${STUDENT_PORT} 
