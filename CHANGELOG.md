# Changelog

All notable changes to this package are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Until 1.0, a minor version may change
the public API; a patch version never does.

The release workflow writes each version's entry from the conventional commits since the previous
one, together with anything written by hand under Unreleased.

## [Unreleased]

## [0.1.1] - 2026-09-30

No user-facing changes.

[Unreleased]: https://github.com/visin-platform/visin-fusion/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/visin-platform/visin-fusion/compare/v0.1.0...v0.1.1
## [0.1.0] - 2026-09-28

- Packaged the code as `visin_fusion` with model, training and Visin extras, a console command,
  versioned JSON Schema, model-owned training defaults, and optional reporting callbacks.
- Existing `run.py` and legacy import paths remain available in a source checkout.

The training code became a generic tool: one implementation per stage for every model
(`visin_fusion/engine/stages/<stage>/common.py`, `visin_fusion/models/registry.py`), configs built from presets, datasets described by
`dataset.json` and downloadable from Visin. Results from earlier code differ in these ways:

### Changes to results

- **AP** is computed the same way for every model, over every stored pixel. CLFTv2, MaskFormer,
  Mask2Former and DeepLabV3+ used to rank a random, unseeded sample of 100,000 pixels, so their AP moved
  by up to about 0.02 between runs of one checkpoint. CLFT's AP is unchanged. How AP is computed:
  `docs/metrics.md`.
- **Testing** evaluates every frame of every test set. CLFT used to drop the last partial batch of each
  (at batch 8 on ZOD: 4 of 356 day-fair frames, 3 of 59 day-rain, 2 of 26 night-rain), so CLFT and the
  other models were not scored on the same frames.
- **Validation** during training uses every frame, in order. CLFT and CLFTv2 shuffled it and
  dropped the last partial batch; DeepLabV3+ dropped it. The best checkpoint and early stopping follow
  the validation mIoU, so they can change.
- **Labels** are resized and cropped with pixel-centre nearest sampling (`nearest-exact`). Plain
  `nearest` shifted labels against the image by up to a pixel, in opposite directions for flipped and
  unflipped samples. New training runs are not bit-comparable with earlier ones.
- **Mask2Former's validation loss** is its Hungarian loss, like its training loss (it was
  cross-entropy, so the two were not comparable). Only the logged loss changes.
- **Resuming** a CLFTv2, MaskFormer or Mask2Former run continues its learning-rate schedule;
  checkpoints now store it. The warmup and decay used to start over.
- **Training seeds** torch in every model (only CLFTv2 did), so CLFT, MaskFormer, Mask2Former and
  DeepLabV3+ runs of one config no longer start from different weights.

### Fixed

- CLFTv2 checkpoints loaded with `strict=False`: a checkpoint missing layers was tested with those layers
  random, without a warning. All checkpoints now load strictly.
- DeepLabV3+ could train one fusion strategy and test another (`fusion_strategy` vs `fusion_type`,
  different defaults).
- Visualizations of CLFTv2, MaskFormer and Mask2Former were uploaded to Visin as epoch 0.
- Label values above every mapped class (e.g. 255) crashed relabelling.
- Any dataset not named zod, waymo or iseauto crashed training, and CLFT scored it against the wrong
  classes.
- Test stages failed on machines without CUDA; TensorBoard logs went to `./runs`.
- A failing benchmark config was skipped and an empty result reported; benchmarks timed models in
  training mode after profiling.
- A missing annotation silently became an empty mask; it now prints a warning naming the file.

[0.1.0]: https://github.com/visin-platform/visin-fusion/releases/tag/v0.1.0
