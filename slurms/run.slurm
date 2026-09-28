#!/bin/bash
# Run a config through its pipeline (train, test, visualize, benchmark) on a SLURM cluster.
#
#   sbatch slurms/run.slurm <config.json> [run.py options, e.g. --stages test,visualize]
#
# With IMAGE set, the job runs in the published container through Apptainer instead of the venv:
#
#   IMAGE=docker://ghcr.io/visin-platform/visin-fusion:latest sbatch slurms/run.slurm my_config.json
#
# Submit from the repository root. Adjust the resources to your cluster; the commented lines are what
# the TalTech HPC A100 nodes need.
#SBATCH --job-name=visin-fusion
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=5-0
#SBATCH --output=slurm-%j.out
##SBATCH --partition=gpu
##SBATCH --gres=gpu:A100:1
##SBATCH --constraint=A100-80
##SBATCH --account=<your allocation>

set -euo pipefail
config="${1:?usage: sbatch slurms/run.slurm <config.json> [run.py options]}"
shift

# Nodes without internet: keep reports on disk and send them later with `visin sync`
# export VISIN_MODE=offline

if [[ -n "${IMAGE:-}" ]]; then
    # The image reads datasets from /data and writes logs under /outputs (docs/running.md)
    apptainer exec --nv \
        --bind "${DATA_DIR:-$PWD/data}:/data" --bind "${OUTPUT_DIR:-$PWD/outputs}:/outputs" \
        --bind "${CACHE_DIR:-$PWD/.cache}:/cache" \
        "$IMAGE" visin-fusion run -c "$config" --upload "$@"  # the submit directory is bound, so relative config paths work
else
    source venv/bin/activate
    visin-fusion run -c "$config" --upload "$@"
fi
