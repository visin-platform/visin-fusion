#!/bin/bash
# One stage group of one run, started by hpc.sh. It carries no cluster settings: hpc.sh passes the ones in env.sh.
#
#   sbatch ... job.slurm <config.json> [visin-fusion run options]      (HPC_DIR and HPC_STAGE in the environment)
set -euo pipefail
: "${HPC_DIR:?started without HPC_DIR: use slurms/hpc/hpc.sh}"
source "$HPC_DIR/common.sh"
load_env
load_venv

config="${1:?usage: job.slurm <config.json> [run options]}"
shift
echo "=== $(date '+%F %T') on $(hostname): $config"
cd "$REPO"
python "$HPC_DIR/preflight.py" "$config" --stage "${HPC_STAGE:?}" || { echo "PREFLIGHT FAILED"; exit 1; }
visin-fusion run -c "$config" "$@"
echo "=== $(date '+%F %T') DONE"
