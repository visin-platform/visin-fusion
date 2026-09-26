# Visin integration

Training runs, epochs, test results, benchmarks and visualizations are reported to
[Visin](https://app.visin.eu) through the [`visin`](https://github.com/visin-platform/visin-py) Python
package. This directory adapts it to this project's scripts.

## Setup

```bash
pip install visin
export VISIN_TOKEN=...                 # a project token: the project's Settings -> API Tokens in Visin
visin check --write                    # confirm this machine can report
```

The token can also go in `integrations/.env` (git-ignored; copy `integrations/.env.example`). `VISIN_URL` defaults to
`https://vision-api.visin.eu`; set it to report to another deployment, such as `http://localhost:4010`.

Without `VISIN_TOKEN` nothing is reported, and every script trains, tests and writes its local logs as
before.

## What each script reports

The SLURM jobs run each config through four scripts, and each one adds to the same run:

| Script | Reports | How it finds the run |
| --- | --- | --- |
| `stages/train/*.py` | the run, its config, then every epoch as it finishes | creates it; a resumed training (`General.resume_training`) finds it by the training UUID in `logs/.../epochs/` |
| `stages/test/*.py` | a test result per tested checkpoint | the training UUID in the epoch logs, and the epoch UUID in the checkpoint's name |
| `stages/visualize/*.py --upload` | segment, overlay, compare and correct_only frames | the same |
| `stages/benchmark/*.py` | a benchmark on the measured checkpoint | the same |

- **Epoch UUIDs are deterministic**: `visin.epoch_uuid_for(training_uuid, epoch)`. The epoch log file,
  the checkpoint (`epoch_{n}_{uuid}.pth`) and Visin all carry the same one.
- **Only training sets the run's status.** Leaving training marks the run completed; a crash, or
  SLURM ending the job, marks it failed, after the epochs it produced have been sent. The test,
  visualize and benchmark scripts only add to the run, so a failed test cannot mark a finished
  training as failed.
- **Resuming continues the same run.** A resumed training reports into the run of the checkpoint it
  resumes, marks it running again, and does not attach a second config. `General.create_new_training`,
  `General.transfer_learning` and `General.reset_lr` start a new run instead: with `reset_lr` the
  epochs count from 0 again, and a run keeps the first values recorded for an epoch.
- **A config trained afresh into a used logs directory** leaves the earlier run's logs and
  checkpoints there. The best and latest checkpoints, and checkpoint pruning, consider only the
  newest run, so testing after training tests what was just trained, and the earlier run's
  checkpoints are neither tested nor deleted.
- **Reporting never slows or stops training.** Epochs and frames are sent in the background. If
  Visin cannot be reached, reports are kept in `~/.visin` and sent when it answers again.

## On the cluster

If compute nodes cannot reach Visin, run the jobs offline and send the results from the login node,
which shares your home directory:

```bash
# in the SLURM script
export VISIN_MODE=offline

# afterwards, on the login node
visin sync --list
visin sync
```

## Modules

| Module | Holds |
| --- | --- |
| `vision_service.py` | `start_training_run`, `attach_to_training`, `report_test_results`, `report_benchmark`, `parse_checkpoint_name` |
| `training_logger.py` | `log_epoch_results`: writes the local epoch log and reports the epoch |
| `visualization_uploader.py` | `queue_visualizations`, `get_epoch_uuid_from_model_path` |
