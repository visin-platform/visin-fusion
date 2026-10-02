# Visin integration

Training runs, epochs, test results, benchmarks and visualizations are reported to
[Visin](https://app.visin.eu) through the [`visin`](https://github.com/visin-platform/visin-py) Python
package. The `visin_fusion.integrations` package adapts it to the pipeline event interface.

## Setup

Install the optional integration and give the calling application a pipeline key:

```bash
python -m pip install -e '.[visin]'
export VISIN_TOKEN=...                 # project's Settings → Pipeline keys in Visin
visin check --write                    # confirm this machine can report
```

You can instead keep credentials in **your application's** `.env` (the current working directory),
or in a file anywhere outside the library:

```bash
cp .env.example .env                  # for this checkout; .env is git-ignored
# or, from another application or a cluster job:
export VISIN_ENV_FILE=/path/to/my-app/visin.env
visin-fusion run -c config.json
```

Exported variables take precedence over the file. `VISIN_ENV_FILE` must point to an existing file;
when it is set, that file takes precedence over the current directory's `.env`. The installed
`visin_fusion` package are never searched for secrets.
`VISIN_URL` defaults to `https://vision-api.visin.eu`; set it for another deployment.

For Docker Compose, `.env` beside `compose.yml` is the default `env_file`, or pass an
external file path:

```bash
VISIN_ENV_FILE=/path/to/my-app/visin.env docker compose run --rm fusion-cpu -c configs/quickstart.json
```

A key file looks like this (keep it out of version control):

```dotenv title=".env"
VISIN_URL=https://vision-api.visin.eu
VISIN_TOKEN=replace-with-your-project-pipeline-key
VISIN_DIR=.visin
```

`visin check --write` creates and deletes a small test run to confirm the key works. Export your credential before downloading a private dataset too.

Without `VISIN_TOKEN` the pipeline keeps its local logs and sends no reports.

## What each script reports

The SLURM jobs run each config through four scripts, and each one adds to the same run:

| Script | Reports | How it finds the run |
| --- | --- | --- |
| `visin_fusion/engine/stages/train/*.py` | the run, its config, then every epoch as it finishes | creates it; a resumed training (`General.resume_training`) finds it by the training UUID in `logs/.../epochs/` |
| `visin_fusion/engine/stages/test/*.py` | a test result per tested checkpoint | the training UUID in the epoch logs, and the epoch UUID in the checkpoint's name |
| `visin_fusion/engine/stages/visualize/*.py --upload` | segment, overlay, compare and correct_only frames | the same |
| `visin_fusion/engine/stages/benchmark/*.py` | a benchmark on the measured checkpoint | the same |

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
| `visin_fusion/integrations/visin.py` | `start_training_run`, `attach_to_training`, `report_test_results`, `report_benchmark` |
| `visin_fusion/engine/epoch_logger.py` | `log_epoch_results`: writes the local epoch log |
| `visin_fusion/integrations/visualization_uploader.py` | `queue_visualizations` |
