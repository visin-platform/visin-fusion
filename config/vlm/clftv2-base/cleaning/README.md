# Cleaning contrast — does manual cleaning of the labels raise mIoU?

Two arms, one controlled comparison.

| Arm | `annotation_path` | What it is |
|---|---|---|
| `human_verified` | `vlm/human_verified/annotation` | The SAM masks a person kept; the ones they rejected erased. |
| `raw_sam` | `annotation_raw_sam` | The same SAM masks, all of them, uncleaned. |

Same frames, different annotations. Both configs read the identical
`train.txt` / `validation.txt` / `test.txt`; only `annotation_path` differs, and
the loader reaches an annotation by replacing `camera` in the image path with
it, so each arm sees the same 800 images labelled by a different set of PNGs.

Verified over all 800 frames, not assumed:

* `annotation_raw_sam` is byte-identical to `annotation_sam` on these frames —
  it is exactly the mask pool `make_clean_annotations.py` starts from.
* the human set is a strict subset of it: every non-background pixel in the
  human annotation carries the same class in `raw_sam`, 0 violations. Cleaning
  only ever deletes.
* 1,396,343 object pixels are gone in the human set, 5.3% of the total — the
  2,962 masks (16.7% of 17,775) a labeler called incorrect. On 140 of the 800
  frames the human removed nothing and the two annotations are identical.

So the arms differ by exactly one thing, and a difference in downstream mIoU is
attributable to the cleaning and to nothing else.

That single-variable property is why the second arm is `raw_sam` and not one of
the VLM pipeline variants. `qwen/annotation_full` differs from the human set by
*two* things at once — its triage rules keep and drop a different set of masks
*and* its discovery stage paints in extra objects — so it answers "is the
pipeline better than a human" rather than "does cleaning help". If that
comparison is wanted, copy `config_raw_sam_fusion.json` and change
`annotation_path` to `vlm/qwen2.5vl_72b_v2/annotation_full`; the ladder in
`../shared`, `../qwen` and `../llava` already covers the pipeline stages on the
4,110-frame set.

## The frames

Both configs point at `/mnt/ml/zod_temp/vlm/human_verified/splits/` —
519 train / 121 validation / 160 test, drawn from the 800 frames a human has
verified so far. Weather is held exactly and each class's *pixel* mass is
equalised across the three splits to within 0.2pp.

Holding the frames fixed is the whole reason this group exists separately. On
the 4,110-frame ladder the human arm had 800 frames of labels while the
automated arms had 4,110, so training-set size sat on top of the label-quality
effect and neither could be read off the other.

**The splits are regenerated, not fixed.** Labelling is ongoing; every export is
larger. `make_clean_annotations.py` in the paper repo rewrites both the
annotations and `splits/` from the newest export, and a frame in train today can
be in test at the next export. Checkpoints are therefore only comparable to
others trained against the same generation of `splits/` — **retrain both arms
after regenerating, never one of them.**

## Running

```bash
sbatch slurms/clftv2/train_clftv2_zod_cleaning.slurm   # 2 arms × 3 seeds
# or one arm locally
python train_swin.py --config config/vlm/clftv2-base/cleaning/config_human_verified_fusion.json --seed 0
```

Three seeds per arm because the arms are expected to differ by a few mIoU and a
single unseeded run is one arbitrary draw of the initialisation — the same
reason the headline variants got replicated (`train_clftv2_zod_seeds.slurm`).

## Scoring — read this before quoting a number

**Do not use `test_swin.py` or `test_clft.py` here.** Both score an arm against
the annotation dir it was trained on, so `raw_sam` would be graded against
unfiltered SAM labels — it would be rewarded for reproducing exactly the masks
this experiment claims are wrong, and the contrast would measure nothing.

There is one admissible reference — the human labels — and it exists for all 160
test frames:

```bash
for arm in human_verified raw_sam; do
  for seed in 0 1 2; do
    python dump_frame_metrics.py \
      -c config/vlm/clftv2-base/cleaning/config_${arm}_fusion.json \
      --logdir-suffix _seed${seed} \
      --splits /mnt/ml/zod_temp/vlm/human_verified/splits \
      --reference vlm/human_verified/annotation \
      --out logs/vlm/frame_metrics/cleaning/${arm}_seed${seed}.json
  done
done
```

Then, in the paper repo:

```bash
python bootstrap_miou.py --metrics-dir <fusion-training>/logs/vlm/frame_metrics/cleaning \
    --pair human_verified_seed0:raw_sam_seed0
```

Read the comparison off the **pooled** per-frame bootstrap, not the
weather-averaged mIoU those scripts print. At this size `test_snow` is 5 frames
and `test_night_rain` is 7, and the weather average gives each the same weight as
the 105-frame `test_day_fair`. `bootstrap_miou.py` resamples frames within
condition and uses identical resample indices for both arms, so the paired delta
cancels the shared frame-sampling noise — which is what makes a few mIoU of
difference on 160 frames separable at all.

One caveat to state in the paper: the reference is the same annotator's work
that the clean arm was trained on. The test frames are held out from both arms,
so neither model has seen its own test labels, but the honest phrasing of the
result is *"a model trained on cleaned labels agrees more with a human than one
trained on raw SAM output"*, not that it is closer to a neutral ground truth.
`annotation_camera_only` (ZOD's own GT) is available for all 800 frames if an
independent reference is wanted: pass `--reference annotation_camera_only` to
the same `dump_frame_metrics.py` calls and both arms get scored against labels
neither the annotator nor the pipeline produced.
