# Cleaning contrast — does manual cleaning of the labels raise mIoU?

Two arms, one controlled comparison.

| Arm | `annotation_path` | What it is |
|---|---|---|
| `human_verified` | `vlm/human_verified/annotation` | The SAM masks a person kept; the ones they rejected erased. |
| `raw_sam` | `annotation_raw_sam` | The same SAM masks, all of them, uncleaned. |

Same frames, different annotations. Both configs read the identical
`train.txt` / `validation.txt` / `test.txt`; only `annotation_path` differs, and
the loader reaches an annotation by replacing `camera` in the image path with
it, so each arm sees the same images labelled by a different set of PNGs.

Verified over the frames in the set, not assumed:

* `annotation_raw_sam` is byte-identical to `annotation_sam` on these frames —
  it is exactly the mask pool `make_clean_annotations.py` starts from.
* the human set is a strict subset of it: every non-background pixel in the
  human annotation carries the same class in `raw_sam`. Cleaning only deletes.

So the arms differ by exactly one thing, and a difference in downstream mIoU is
attributable to the cleaning and to nothing else.

That single-variable property is why the second arm is `raw_sam` and not one of
the VLM pipeline variants. `qwen/annotation_full` differs from the human set by
*two* things at once — its triage rules keep and drop a different set of masks
*and* its discovery stage paints in extra objects — so it answers "is the
pipeline better than a human" rather than "does cleaning help". The ladder in
`../shared`, `../qwen` and `../llava` already covers the pipeline stages on the
4,110-frame set. **Do not add arms here.**

## What the answer turned out to be

Measured on the 1,001-frame export, 201 test frames, both arms scored against
the human reference:

```
human_verified   veh 64.9  sign 31.6  human 25.3   mIoU 37.2
raw_sam          veh 65.4  sign 29.1  human 21.7   mIoU 36.7

Δ mIoU = +0.86  [-1.08, +2.89]   P(Δ≤0) = 0.22
```

**The aggregate is a null and will stay one.** Cleaning removes 5.3 % of the
object pixels, which is too small an intervention to move a weather-averaged
mIoU past the noise of a test split this size. Scaling the test set does not
rescue it: the paired CI narrows as 1/√n, so even all 1,001 frames would land on
the edge of zero. Cross-validation was built for this and then deleted — it
would have cost ten trainings to buy a CI that still touches zero.

**The result is per class**, where the IoU gain tracks how much of each class the
human actually deleted:

| class | pixels removed | % of that class's pixels | ΔIoU |
|---|---|---|---|
| human (cyclist+pedestrian) | 535,244 | 20.9 % | **+3.6** |
| sign | 693,239 | 12.2 % | **+2.5** |
| vehicle | 864,715 | 3.5 % | −0.5 |

Monotonic in the dose. Vehicle is 78 % of the object pixels and barely got
touched, which is exactly why it dominates and flattens the aggregate.

Those numbers come from a `raw_sam` checkpoint that was not its run's best (its
best was rotated away, see below), so the point estimates flatter the clean arm
slightly. The *ordering* does not depend on that: a uniformly handicapped model
would lose on all three classes, and this one wins on vehicle. Restate them from
the retrained checkpoints before quoting them in the paper.

## The frames

Both configs point at `/mnt/ml/zod_temp/vlm/human_verified/splits/` — currently
651 train / 149 validation / 201 test out of the 1,001 verified frames. Weather
is held exactly, each class's *pixel* mass is equalised across the three splits,
and so is **labelling order**: the human reference drifts by about 2 points of
precision per 100 frames judged and has not plateaued, so a split drawn from
early frames is graded against a more permissive standard than one drawn from
late frames.

Holding the frames fixed is the whole reason this group exists separately. On
the 4,110-frame ladder the human arm had 1,001 frames of labels while the
automated arms had all of them, so training-set size sat on top of the
label-quality effect and neither could be read off the other.

**The splits are regenerated, not fixed.** `make_clean_annotations.py` in the
paper repo rewrites both the annotations and `splits/` from the newest export,
and a frame in train today can be in test at the next export. Checkpoints are
therefore only comparable to others trained against the same generation of
`splits/` — **retrain both arms after regenerating, never one of them.**

## Running

```bash
bash slurms/clftv2/run_vlm_campaign.sh cleaning   # both arms, then the metric dumps
```

**Never leave two runs' epoch records in one logdir.** That is what corrupted the
first attempt. `manage_checkpoints_by_miou` keeps the top `max_checkpoints`
files ranked by validation mIoU, so the best checkpoint of a run always survives
— but the ranking is over whatever is in the directory, not over one run. With a
previous splits generation still present, the new run's checkpoints were ranked
against the old run's, the old ones scored higher, and the new ones were pruned.
`get_best_checkpoint_path` then selected a model trained on frames that had since
moved into the test split, and nothing in the output said so.
Move an old logdir aside before re-running an arm; `max_checkpoints: 2` is fine.

## Scoring — read this before quoting a number

**Do not use `test_swin.py` or `test_clft.py` here.** Both score an arm against
the annotation dir it was trained on, so `raw_sam` would be graded against
unfiltered SAM labels — it would be rewarded for reproducing exactly the masks
this experiment claims are wrong, and the contrast would measure nothing.

There is one admissible reference — the human labels — and each config names it
in its own `Eval` block, so the dump takes no arguments beyond the config:

```json
"Eval": {
  "splits":      "/mnt/ml/zod_temp/vlm/human_verified/splits",
  "reference":   "vlm/human_verified/annotation",
  "metrics_out": "logs/vlm/frame_metrics/cleaning/raw_sam_fusion.json"
}
```

```bash
python dump_frame_metrics.py -c config/vlm/clftv2-base/cleaning/config_${arm}_fusion.json
```

That block is why the reference is not a command-line flag. Scoring an arm
against a different reference is a different measurement, and as a flag it is
invisible in the output — two runs of "the same config" could mean different
things. Passing `--reference` / `--splits` / `--out` still overrides it, which is
what the ZOD-GT cross-check below uses.

Then, in the paper repo:

```bash
python bootstrap_miou.py --metrics-dir <fusion-training>/logs/vlm/frame_metrics/cleaning \
    --pair human_verified_fusion:raw_sam_fusion
```

Read the comparison off the **pooled** per-frame bootstrap, not the
weather-averaged mIoU those scripts print. At this size `test_snow` is 6 frames
and `test_night_rain` is 9, and the weather average gives each the same weight as
the 132-frame `test_day_fair`. `bootstrap_miou.py` resamples frames within
condition and uses identical resample indices for both arms, so the paired delta
cancels the shared frame-sampling noise.

One caveat to state in the paper: the reference is the same annotator's work
that the clean arm was trained on. The test frames are held out from both arms,
so neither model has seen its own test labels, but the honest phrasing is *"a
model trained on cleaned labels agrees more with a human than one trained on raw
SAM output"*, not that it is closer to a neutral ground truth.
`annotation_camera_only` (ZOD's own GT) exists for every frame if an independent
reference is wanted:

```bash
python dump_frame_metrics.py -c config/vlm/clftv2-base/cleaning/config_${arm}_fusion.json \
    --reference annotation_camera_only \
    --out logs/vlm/frame_metrics/cleaning_zodgt/${arm}.json
```

Both arms then get scored against labels neither the annotator nor the pipeline
produced. Write them to a separate directory: `bootstrap_miou.py` compares every
JSON in a directory, so mixing two references in one would pair arms that were
never measured against the same thing.
