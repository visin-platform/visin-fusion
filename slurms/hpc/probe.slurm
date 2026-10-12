#!/bin/bash
# Reports whether a compute node can reach Visin and Hugging Face. Started by `hpc.sh probe`.
set -u
reach() { curl -sS -o /dev/null --max-time 15 "$2" 2>/dev/null && echo "$1: reachable" || echo "$1: NOT reachable"; }
echo "node: $(hostname)"
reach visin "${VISIN_URL:?}"
reach huggingface https://huggingface.co
