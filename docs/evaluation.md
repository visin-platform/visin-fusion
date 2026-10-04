# Ranking models on a suite

A training run says how a model did on validation. To say which *checkpoint* holds up, and under which conditions,
Visin compares test results that were produced the same way. That "same way" is a **suite**: a written-down
protocol (which test sets, how many frames each, which metrics, how the overall figure is formed). The test stage can
record each checkpoint's results as an **evaluation** on a suite, and Visin says whether the result can be ranked
and, if not, exactly why.

This needs a Visin that has suites, and a `visin` package that can record evaluations (`pip install -U visin`).
Visin's own documentation, under *Evaluating models*, covers the suite format and the ranking in full.

## 1. Publish the suite once

A suite for this project's test stage pins what the stage really does:

- each **test set** is a condition, named as in `Dataset.test_splits` (by default `day_fair`, `day_rain`,
  `night_fair`, `night_rain` and `snow`);
- the metrics are the ones in each test set's `overall` block (`mIoU_foreground`, `mean_ap`, …);
- the overall figure is the **equal mean of the test sets**, which is what the stage computes, so say
  `equal-mean-of-conditions`, not a pooled figure.

```json
{
  "slug": "zod-weather",
  "version": 1,
  "name": "ZOD, five weather test sets",
  "protocol": {
    "task": "semantic-segmentation",
    "data": {
      "kind": "external",
      "label": "ZOD test splits",
      "manifestSha256": "dc0b37108a67875eac18dd3dd966d4a064f44afe693c5f8213fb8b246e6dad3b"
    },
    "split": "test",
    "conditions": [
      {"name": "day_fair", "sampleCount": 356},
      {"name": "day_rain", "sampleCount": 59},
      {"name": "night_fair", "sampleCount": 97},
      {"name": "night_rain", "sampleCount": 26},
      {"name": "snow", "sampleCount": 41}
    ],
    "metrics": [
      {"key": "mIoU_foreground", "direction": "max", "range": {"min": 0, "max": 1}, "headline": true},
      {"key": "mean_ap", "direction": "max", "range": {"min": 0, "max": 1}}
    ],
    "aggregation": "equal-mean-of-conditions",
    "evaluator": {"package": "visin-fusion"}
  }
}
```

The counts and the digest come from your split files, so you do not have to work them out. For ZOD's five
test splits:

```text
$ visin suites manifest day_fair=test_day_fair.txt day_rain=test_day_rain.txt \
      night_fair=test_night_fair.txt night_rain=test_night_rain.txt snow=test_snow.txt
manifestSha256  dc0b37108a67875eac18dd3dd966d4a064f44afe693c5f8213fb8b246e6dad3b
  day_fair              356 samples
  day_rain              59 samples
  night_fair            97 samples
  night_rain            26 samples
  snow                  41 samples
```

The digest is of the sorted frames listed under each condition, which is what makes the suite name those exact
samples: change a frame in a split and it changes. Then publish it:

```bash
visin suites push suites/zod-weather.json --project road-seg --public
```

## 2. Record the results on it

Name the suite file in the config, or on the command line:

```json
"General": {"suite_file": "suites/zod-weather.json"}
```

```bash
python -m visin_fusion.engine.stages.test.common -c cfg.json --suite-file suites/zod-weather.json
```

The suite file is the protocol the stage ran against. Its `slug@version` names the suite (set `General.suite`, or
`--suite`, only to override it), and Visin computes its digest, which the evaluation sends as the **protocol that
ran**. The stage also sends the digest of the frame lists it actually tested, as the **data that was scored**, and the
fusion version as the **evaluator**. Visin compares all three with the suite. What is sent as the data that was scored
follows what the suite pins: the digest of the frame lists for a suite pinned by a manifest, the Hub repo and commit the
dataset root resolved to for a suite pinned to a Hub dataset. A suite pinned to a Visin dataset's archive gets no data
evidence from fusion, which cannot observe the archive's digest, so that result is **reported**. Without a suite file
the result is still recorded and ranked, but as **reported** rather than **observed**: it says nothing about which
protocol ran.

With a suite, the test stage records its results as one evaluation of the tested checkpoint: Visin keeps one record of
a test, so the results are not also sent as a plain test result. Without a suite (or when the suite cannot be used)
the results are sent as a test result on the checkpoint's epoch, which Visin keeps as an evaluation with no suite. The
checkpoint is named by the SHA-256 of the `.pth` file it loaded, the evaluation by the test's own UUID (so repeating
the upload is harmless), and the run and epoch are recorded as where it came from. The stage logs Visin's verdict:

```text
Visin evaluation on zod-weather@1: eligible
Visin evaluation on zod-weather@1: incomplete: missing-condition(snow)
```

## Why a result may not be ranked

The most common reason is a **test set the stage skipped**. A default weather split whose file is missing is skipped
with a warning, and a split that lists no frames is skipped too. The suite pins all five, so the evaluation is stored
as `incomplete` with `missing-condition(snow)`, and the result is still saved and sent. Fix the data and test
again; a new evaluation supersedes nothing, it is simply the checkpoint's latest ranked attempt.

The frame count of each test set is sent with the result, so a suite built from a different number of frames than the
stage scored is `incompatible` (`sample-count-mismatch(day_rain)`), never silently compared. So is a test set with the
same number of frames but different ones: the digest of the frame lists no longer matches the suite's
(`data-mismatch(manifestSha256)`). The digest is of the *listed frames*, not of the image or annotation bytes.

Every reason, and how to fix it, is listed in Visin's documentation under *Why is my result unranked?*.

## What never stops a test

A refused evaluation (an unknown suite, no access to the project, a Visin without suites) is logged as a warning.
The test stage still finishes: the results are already on disk, and are sent as a plain test result instead. With
`VISIN_MODE=offline` the evaluation is kept on disk and `visin sync` sends it later.

## Many checkpoints on many suites

Each `python -m visin_fusion.engine.stages.test.common ... --checkpoint X --suite S` records one evaluation, so a
grid is a loop, or a SLURM array, over checkpoints and suites. `visin leaderboard zod-weather@1` then ranks them, with
each model's weakest condition beside its overall score.
