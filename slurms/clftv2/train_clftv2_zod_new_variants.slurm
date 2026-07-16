#!/bin/bash
#SBATCH --job-name=z-swin-newvar       ### exact-mask discovery + limitation-fix + consensus variants
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

# Retrains every discovery-bearing variant on annotations where discovery
# objects are painted with exact connected-component masks (_ccm) instead of
# filled bounding boxes, plus the limitation-fix and two-VLM consensus
# variants. Requires these annotation dirs synced to the HPC zod_temp first:
#   vlm/llava_34b/annotation_swin_only_discovery_ccm
#   vlm/llava_34b/annotation_swin_only_discovery_noVLM_ccm
#   vlm/llava_34b/annotation_limits_fixed_discovery_ccm
#   vlm/qwen2.5vl_72b/annotation_swin_only_discovery_ccm
#   vlm/merged_llava_34b_qwen2.5vl_72b/annotation_swin_only_disc_both
#   vlm/merged_llava_34b_qwen2.5vl_72b/annotation_union_disc_both
# Do NOT evaluate with test_swin.py (per-variant reference); use
# dump_frame_metrics.py against the common reference after syncing back.

source venv/bin/activate

CONFIGS=(
    config/vlm/clftv2-base/llava/config_swin_discovery_ccm_fusion.json
    config/vlm/clftv2-base/qwen/config_swin_discovery_ccm_fusion.json
    config/vlm/clftv2-base/shared/config_swin_discovery_noVLM_ccm_fusion.json
    config/vlm/clftv2-base/llava/config_limits_fixed_discovery_fusion.json
    config/vlm/clftv2-base/merged/config_consensus_swin_disc_both_fusion.json
    config/vlm/clftv2-base/merged/config_consensus_union_disc_both_fusion.json
)

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
echo "Training $CONFIG"
python train_swin.py --config "$CONFIG"
