# Troubleshooting

Find the message you see. Every config problem is reported before any stage starts, so fix those first.

## Config

| Message | Cause and fix |
| --- | --- |
| `Invalid config:` followed by `General.epcohs: Extra inputs are not permitted` | A misspelled or unknown key. Compare with the [config reference](reference/config.md). Each problem is listed, so fix them all at once. |
| `mode 'fusion' is not one of clftv2's modes [...]` | Use a mode the model supports: `rgb`, `lidar`, or `cross_fusion` (`fusion` for DeepLabV3+). |
| `extends: no preset 'x'; presets: ...` | `extends` is a preset name (`clftv2`) or a file name ending in `.json`, relative to the config file. |
| `Dataset.dataset_root '...': environment variable not set` | Export the variable, e.g. `export ZOD_DATA_DIR=...`, or write the folder in the config. |
| `train_classes indices must be 0..N with no gaps or duplicates` | Number the model's classes `0, 1, 2, ...`; `0` is background. |
| `train class 'x' maps dataset classes [N], which dataset_classes does not have` | List the number in `dataset_classes`, or remove it from `dataset_mapping`. See [class mapping](configs.md#class-mapping). |
| `Dataset.annotation_path 'x' is not one of the annotations of .../dataset.json` | Pick a folder from the manifest's `annotations` list. |
| `needs a 'CLFTv2' section` | Add the model's section, or extend its preset, which has it. |

## Data

| Message | Cause and fix |
| --- | --- |
| `No annotation ...; using an empty mask` | The annotation file for a frame is missing, so it trains as all background. Restore it or remove the frame from the split file. |
| `no 'camera' folder in frame path` | Lines in a split file must contain the camera folder, e.g. `camera/000001.png`, relative to `dataset_root`. |
| `Dataset.test_splits: files not found` | Test split names are looked up next to `val_split` (or in `Dataset.split_dir`); use a full path otherwise. |
| `Skipping test split ...: file not found` | Only a warning: the default weather test sets are used when you name none, and missing ones are skipped. Set `test_splits` for your own. |
| `no test set had any frames` | Every test split file is empty. |

## Running

| Message | Cause and fix |
| --- | --- |
| `No checkpoint in <logdir>/checkpoints; train first` | Test and visualize need the train stage's checkpoint: run `train` first, or point `Log.logdir` at the finished run. |
| `All N epochs are already trained; set General.epochs higher to continue` | Resuming with `epochs` not above the finished count. |
| CUDA out of memory | Lower `General.batch_size`, or train on the CPU with `General.device: "cpu"`. |
| Training is very slow | Check `torch.cuda.is_available()`. Without CUDA the CPU is used. |
| Pretrained weights fail to download | The first run with `pretrained: true` downloads the backbone. On a machine without internet, set `pretrained: false` or copy a populated cache. |

## Visin

| Message | Cause and fix |
| --- | --- |
| `A Visin pipeline key is set; pip install visin-fusion[visin]` | `VISIN_TOKEN` is set but the integration package is missing: `python -m pip install 'visin-fusion[visin]'`. |
| `A visin: dataset needs pip install visin-fusion[visin]` | Same fix, for `"dataset_root": "visin:zod"`. |
| `VISIN_ENV_FILE does not point to a file` | Correct the path, or unset it. |
| Nothing appears in Visin | With no `VISIN_TOKEN` runs stay local by design. See [Visin reporting](visin.md). |

Still stuck? Open an issue at [github.com/visin-platform/visin-fusion/issues](https://github.com/visin-platform/visin-fusion/issues) with the full output and your config.
