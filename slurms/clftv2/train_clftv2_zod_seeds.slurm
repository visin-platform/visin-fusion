#!/bin/bash
#SBATCH --job-name=z-swin-seeds        ### 3-seed replication of headline variants
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint="A100-80"
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH -t 2-0
#SBATCH --mail-type=ALL
#SBATCH --account=project_autolab
#SBATCH --array=0-5

# Seeds 1 and 2 for the headline variants of the corrected evaluation
# (Swin-only vs curated baseline vs raw SAM). The existing runs serve as the
# third sample each; torch was unseeded before, so they are one arbitrary
# draw. Discovery variants are excluded here: their annotations are being
# regenerated with exact connected-component masks and retrained separately
# (train_clftv2_zod_new_variants.slurm).
#
# Do NOT evaluate with test_swin.py — its reference is the variant's own
# annotation dir. After syncing checkpoints, evaluate locally against the
# common reference:
#   python dump_frame_metrics.py -c <config> --logdir-suffix _seed<N> \
#       --splits <zod_temp>/splits_good --reference annotation_camera_only

source venv/bin/activate

CONFIGS=(
    config/vlm/clftv2-base/shared/config_swin_only_fusion.json
    config/vlm/clftv2-base/shared/config_gt_fusion.json
    config/vlm/clftv2-base/shared/config_raw_sam_fusion.json
)
SEEDS=(1 2)

CONFIG=${CONFIGS[$((SLURM_ARRAY_TASK_ID / 2))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 2))]}

echo "Training $CONFIG with seed $SEED"
python train_swin.py --config "$CONFIG" --seed "$SEED"
