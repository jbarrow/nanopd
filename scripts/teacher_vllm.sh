#!/bin/bash

TEACHER_MODEL="${1:-Qwen/Qwen3.5-2B}"
TEACHER_PORT="${2:-8001}"

echo "Starting TEACHER(${TEACHER_MODEL}) on port ${TEACHER_PORT}"

vllm serve ${TEACHER_MODEL} \
    --port ${TEACHER_PORT}
