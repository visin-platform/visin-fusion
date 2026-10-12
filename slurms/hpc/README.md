# Train, evaluate and publish on a cluster

One command per step, run from the cluster's **login node**. It trains a model with SLURM, scores it on a fixed
test suite, uploads it to Hugging Face and shows the result in Visin. Nothing here is specific to one cluster: your
partition, GPU and account live in one settings file.

## First time (about 10 minutes)

```bash
git clone https://github.com/visin-platform/visin-fusion.git && cd visin-fusion
slurms/hpc/hpc.sh setup      # builds the Python environment, then asks for your keys and writes slurms/hpc/env.sh
$EDITOR slurms/hpc/env.sh    # set your cluster's partition, GPU and account (see "Your cluster" below)
slurms/hpc/hpc.sh doctor     # checks everything and says what is missing
slurms/hpc/hpc.sh probe      # asks whether compute nodes have internet; tells you what to set OFFLINE to
```

You need three things before `setup`: a Visin **pipeline key** (the project's *Settings → Pipeline keys*), the Visin
addresses from whoever runs your Visin, and a Hugging Face **token with write access**. `visin login` is not needed:
`setup` stores the same settings in `env.sh`, and the scripts export them to every job.

## Every run

```bash
slurms/hpc/hpc.sh run configs/examples/zod/clftv2/fusion.json   # downloads, then queues train -> test
slurms/hpc/hpc.sh status                                        # what each job is doing
slurms/hpc/hpc.sh publish <run-id>                              # when both jobs show COMPLETED
```

`run` prints the run id. The config you pass is your training recipe (the example trains for one epoch; copy it and
raise `General.epochs` for a real run). Nothing in it needs editing for the cluster.

`publish` sends the reports to Visin, uploads the best checkpoint to your Hugging Face repo, links it to the run and
prints where the model ranks on the suite.

## What happens

```text
login node   run      download the dataset and pretrained weights, push the evaluation suite
compute      train    trains, reporting every epoch
compute      test     scores the best checkpoint on every test condition, renders images, benchmarks speed
login node   publish  sends the reports, uploads the model, shows the ranking
```

The test result is recorded on a **suite**: a written-down protocol (which test sets, how many frames each, which
metrics), so two models scored on it can be ranked fairly. The suite is built for you from the dataset's test splits.

## Your cluster

Every cluster names its partitions and GPUs differently, so none of that is in the scripts. Set what your cluster
needs in `env.sh`; anything you leave empty is left out of the job, and the cluster's own default applies.

| Setting | Meaning | Find it with |
| --- | --- | --- |
| `SLURM_PARTITION` | the queue to submit to | `sinfo -o "%P %G %l"` |
| `SLURM_GRES` | the GPU request, such as `gpu:1` or `gpu:A100:1` | the `%G` column of that |
| `SLURM_ACCOUNT` | the allocation to charge | your cluster's documentation |
| `SLURM_CONSTRAINT` | a node feature, such as a GPU memory size | `sinfo -o "%N %f"` |
| `SLURM_CPUS`, `SLURM_MEM` | resources per job | |
| `SLURM_TIME`, `SLURM_TEST_TIME` | time limits of the train and test jobs | |
| `SLURM_MAIL_TYPE`, `SLURM_MAIL_USER` | email when a job ends or fails | |
| `SLURM_EXTRA` | any other sbatch options, such as `--qos=long` | |
| `WORK` | a folder the login **and** compute nodes both see, with room for datasets (use scratch, not a small home quota) | |
| `OFFLINE` | `1` if compute nodes have no internet, `0` if they have | `hpc.sh probe` |

`clusters/` holds example values for sites we know. Copy the lines you need, or add `source "$HPC_DIR/clusters/<site>.env"` to
`env.sh`. Please add your own cluster's file in a pull request.

## When something goes wrong

| You see | Do this |
| --- | --- |
| `doctor` prints `FAIL` | fix that line in `env.sh`, run `doctor` again |
| `status` shows `PREFLIGHT FAILED` | the job log names the missing file or folder; nothing was trained |
| A job is `PENDING` for long | the partition is busy, or the request is too big: `squeue -u $USER -o "%i %T %r"` shows why |
| A job hit its time limit | raise `SLURM_TIME`, then `hpc.sh submit <config> --run-id <same id> --resume` |
| `publish` says the upload failed | check `HF_TOKEN` has write access and `HF_MODEL_REPO` is yours; run `publish` again |
| The model is on Hugging Face but not linked in Visin | the Visin project's storage must be set to Hugging Face (project settings) |
| `already holds checkpoints from an earlier run` | use a new run id, or `--resume` to continue that run |
| `setup` says Python is too old | load a newer one (`module avail python`), or `PYTHON=python3.11 slurms/hpc/hpc.sh setup` |
| The suite was not pushed | a pipeline key may not create suites: push `$WORK/suite.json` once with an account key (`visin suites push`) |

Job logs are in `$WORK/runs/<run-id>/` (`train-<run-id>-<jobid>.out`), and checkpoints and epoch logs in its `logs/`.
Datasets and the pretrained weights stay in `$WORK`, so a second run downloads nothing.

## For other recipes

`hpc.sh stage <config>` (downloads only) and `hpc.sh submit <config> [--run-id ID] [--resume]` (queue only) are the two
halves of `run`. To run a container instead of the venv, use `slurms/run.slurm` with `IMAGE` set; it does not do the
staging and publishing above.
