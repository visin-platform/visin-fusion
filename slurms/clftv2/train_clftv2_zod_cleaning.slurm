#!/bin/bash
#SBATCH --job-name=z-swin-clean        ### human cleaning vs raw SAM
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

# The cleaning contrast: does manually cleaning the labels raise downstream
# mIoU over training on the raw SAM output?
#
# Both arms train on the SAME 519 frames and validate on the SAME 121 — the
# human-verified subset, cut by make_clean_annotations.py in the paper repo.
# They differ by exactly one thing: the masks a human called incorrect are
# erased in one and kept in the other. This is what the 4,110-frame ladder
# could not do — there the human arm had 800 frames of labels and the automated
# arms had 4,110, and training-set size sat on top of the effect.
#
# 3 seeds per arm. The arms are expected to differ by a few mIoU and a single
# unseeded run is one arbitrary draw of the initialisation, which is the same
# reason the headline variants got replicated (train_clftv2_zod_seeds.slurm).
#
# Do NOT evaluate with test_swin.py — its reference is each arm's own
# annotation dir, so raw_sam would be graded against unfiltered SAM labels and
# rewarded for reproducing the very masks this experiment calls wrong. After
# syncing checkpoints, evaluate both arms against the one human reference:
#   python dump_frame_metrics.py -c <config> --logdir-suffix _seed<N> \
#       --splits /mnt/ml/zod_temp/vlm/human_verified/splits \
#       --reference vlm/human_verified/annotation \
#       --out logs/vlm/frame_metrics/cleaning/<arm>_seed<N>.json
# then pair them up with the paper repo's bootstrap_miou.py. See
# config/vlm/clftv2-base/cleaning/README.md.

source venv/bin/activate

CONFIGS=(
    config/vlm/clftv2-base/cleaning/config_human_verified_fusion.json
    config/vlm/clftv2-base/cleaning/config_raw_sam_fusion.json
)
SEEDS=(0 1 2)

CONFIG=${CONFIGS[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}

echo "Training $CONFIG with seed $SEED"
python train_swin.py --config "$CONFIG" --seed "$SEED"
