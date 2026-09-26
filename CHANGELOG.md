# Changelog

## Unreleased

The training code became a generic tool: one implementation per stage for every model
(`stages/<stage>/common.py`, `models/registry.py`), configs built from presets, datasets described by
`dataset.json` and downloadable from Visin. Results from earlier code differ in these ways:

### Changes to results

- **AP** is computed the same way for every model, over every stored pixel. SwinFusion, MaskFormer,
  Mask2Former and DeepLabV3+ used to rank a random, unseeded sample of 100,000 pixels, so their AP moved
  by up to about 0.02 between runs of one checkpoint. CLFT's AP is unchanged. How AP is computed:
  `docs/metrics.md`.
- **Testing** evaluates every frame of every test set. CLFT used to drop the last partial batch of each
  (at batch 8 on ZOD: 4 of 356 day-fair frames, 3 of 59 day-rain, 2 of 26 night-rain), so CLFT and the
  other models were not scored on the same frames.
- **Validation** during training uses every frame, in order. CLFT and SwinFusion shuffled it and
  dropped the last partial batch; DeepLabV3+ dropped it. The best checkpoint and early stopping follow
  the validation mIoU, so they can change.
- **Labels** are resized and cropped with pixel-centre nearest sampling (`nearest-exact`). Plain
  `nearest` shifted labels against the image by up to a pixel, in opposite directions for flipped and
  unflipped samples. New training runs are not bit-comparable with earlier ones.
- **Mask2Former's validation loss** is its Hungarian loss, like its training loss (it was
  cross-entropy, so the two were not comparable). Only the logged loss changes.
- **Resuming** a SwinFusion, MaskFormer or Mask2Former run continues its learning-rate schedule;
  checkpoints now store it. The warmup and decay used to start over.
- **Training seeds** torch in every model (only SwinFusion did), so CLFT, MaskFormer, Mask2Former and
  DeepLabV3+ runs of one config no longer start from different weights.

### Fixed

- Swin checkpoints loaded with `strict=False`: a checkpoint missing layers was tested with those layers
  random, without a warning. All checkpoints now load strictly.
- DeepLabV3+ could train one fusion strategy and test another (`fusion_strategy` vs `fusion_type`,
  different defaults).
- Visualizations of SwinFusion, MaskFormer and Mask2Former were uploaded to Visin as epoch 0.
- Label values above every mapped class (e.g. 255) crashed relabelling.
- Any dataset not named zod, waymo or iseauto crashed training, and CLFT scored it against the wrong
  classes.
- Test stages failed on machines without CUDA; TensorBoard logs went to `./runs`.
- A failing benchmark config was skipped and an empty result reported; benchmarks timed models in
  training mode after profiling.
- A missing annotation silently became an empty mask; it now prints a warning naming the file.
