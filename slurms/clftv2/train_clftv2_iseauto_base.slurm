#!/bin/bash
#SBATCH --job-name=i-swin-base         ### iSEAuto CLFTv2-Base transfer, exact-mask discovery source
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint="A100-80"
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH -t 2-0
#SBATCH --account=project_autolab

# Only the Swin+disc.\ (no VLM) initialisation needs (re)training: its source
# ZOD checkpoint (logs/vlm/clftv2-base/shared/swin_discovery_noVLM_ccm_fusion)
# was retrained this session with exact connected-component discovery masks
# instead of the earlier bbox-rectangle approximation.
#
# scratch (logs/vlm/iseauto/scratch) and from_zod_raw_sam (unaffected by the
# discovery-mask bug — no discovery involved) are already complete at 99/100
# epochs; config/vlm/iseauto/iseauto_baseline_base.json and
# to_iseauto_from_zod_raw_sam_base.json exist only for reproducibility and do
# NOT need to be (re)run.
#
# The old logs/vlm/iseauto/from_zod_swin_discovery run (no _ccm suffix) was
# almost certainly initialised from the bbox-rectangle-corrupted checkpoint
# (no corrected source existed until this week) but was NOT deleted pending
# confirmation — this run writes to a new logdir
# (from_zod_swin_discovery_ccm) so both can be compared side by side.

source venv/bin/activate

python train_swin.py --config config/vlm/iseauto/to_iseauto_from_zod_swin_discovery_ccm_base.json
