# Datasets

For ZOD, Waymo, and Iseauto examples, including older archives without a manifest, follow
[Download and train](download-and-train.md).

## From Visin

ZOD, Waymo and iseAuto are on [Visin](https://app.visin.eu/datasets). Name one in a config and it is
downloaded the first time it is needed, then used from disk:

```json
"Dataset": {"dataset_root": "visin:zod"}
```

or download it beforehand with the `visin` command (from the [visin](https://github.com/visin-platform/visin-py)
package, which does the downloading either way):

```bash
visin datasets        # what is there
visin download zod    # ZOD, 3.6 GB; prints the folder
```

Datasets go to `$VISIN_DATA_DIR`, by default `~/.cache/visin/datasets` (`/data` in the Docker image).
An interrupted download resumes where it stopped, and a complete one is never downloaded again. The
dataset service is `$VISIN_DATASET_URL`, by default `https://dataset-api.visin.eu`.

## Layout

A dataset is a folder with camera images, LiDAR projections, annotations and split files:

```text
my_dataset/
  dataset.json              the manifest (below)
  camera/000001.png         camera images
  lidar_png/000001.png      LiDAR projected onto the camera image, XYZ encoded as RGB
  annotation/000001.png     labels: one channel, pixel value = dataset class index
  train.txt                 split files: one frame per line, relative to the dataset folder
  validation.txt
  test_day_fair.txt ...     one file per test set
  visualizations.txt        frames to render
```

Frames can be nested (`labeled/day/rain/camera/x.png`). The LiDAR projection and annotation of a frame
are found by replacing its `camera` folder with `lidar_png` or the annotation folder. A dataset can
have several annotation folders (e.g. `annotation_camera_only`, `annotation_fusion`); a config picks
one with `Dataset.annotation_path`.

ZOD, Waymo and iseAuto are available at [app.visin.eu/datasets](https://app.visin.eu/datasets) in
this layout.

## Looking at the annotations

Annotation images store class numbers (0, 1, 2, ...) as pixel values, so any image viewer shows them as black. To check a dataset by eye, render the annotations in the training classes' colors:

```bash
visin-fusion dataset preview --root /data/my_dataset --frames 8 --output preview/
```

Each `preview/<frame>_preview.png` shows the camera image, its annotation in color, and the two blended, with a legend of every class and its share of the frame. Values that no training class maps are drawn gray: training treats them as background. Choose frames with `--split val` (or a split file) and a different folder with `--annotation-path`.

## The manifest

`dataset.json` describes the dataset, so a config only has to point at it:

```json
{
  "format": 1,
  "name": "zod",
  "description": "Zenseact Open Dataset, 2,300 labelled frames",
  "layout": {"camera": "camera", "lidar": "lidar_png"},
  "annotations": ["annotation_camera_only", "annotation_lidar_only", "annotation_fusion"],
  "splits": {
    "train": "train.txt",
    "val": "validation.txt",
    "test": {"day_fair": "test_day_fair.txt", "snow": "test_snow.txt"},
    "visualization": "visualizations.txt"
  },
  "classes": [{"name": "background", "index": 0}, {"name": "vehicle", "index": 2}],
  "train_classes": [{"name": "background", "index": 0, "weight": 0.1, "dataset_mapping": [0], "color": [0, 0, 0]}],
  "normalization": {"lidar_mean": [0.25, 0.49, 0.50], "lidar_std": [0.23, 0.03, 0.17]}
}
```

The first annotation folder is the default. Test sets can have any names; each is evaluated and
reported separately, and their average is the overall result.

Write one with `visin-fusion dataset manifest`. It finds the splits and annotation folders, takes the name,
classes and normalization from an existing config for the dataset, and checks that every frame the
splits list has its camera image, LiDAR projection and annotations:

```bash
visin-fusion dataset manifest --root /data/my_dataset --config my_config.json \
    --description "My dataset, 1,000 labelled frames"
```

A missing annotation file is not an error at training time: the frame gets an empty mask and a
warning names the file.

## LiDAR projections for your own data

`visin-fusion dataset project-lidar` turns point clouds (`.npy` or KITTI-style `.bin`, beside the camera images in a
`lidar_points/` folder) into `lidar_png/`, given the camera intrinsics and the LiDAR-to-camera transform:

```bash
visin-fusion dataset project-lidar --root /data/my_dataset --calibration calibration.json
```

Each pixel holds the camera-frame X (right), Y (down) and Z (forward) of the nearest point that lands on
it, mapped from fixed ranges (default -40..40, -5..5 and 0..80 m) onto 1..255; 0 means no point. The
ranges are the same for every frame, so a value means the same distance everywhere.

Then compute the normalization the model needs, and suggested class weights:

```bash
visin-fusion dataset stats --root /data/my_dataset           # print
visin-fusion dataset stats --root /data/my_dataset --write   # store in dataset.json
```

!!! warning "The existing datasets use three different encodings"
    Their `lidar_png` images were made by different converters, so pixel values do not mean the same
    thing across datasets, or even across frames of one dataset:

    | Dataset | Scaling | Pixel without a point |
    | --- | --- | --- |
    | Waymo | per image, min-max of each channel | varies per image |
    | iseAuto | per image, 95th percentile of each channel | 127 |
    | ZOD | converter not in this repository | 0 |

    Keep this in mind when transferring a model between datasets. Their `lidar_mean` / `lidar_std`
    were computed over the pixels that have a point, as `visin-fusion dataset stats` does.

## Without a manifest

Configs can still name everything themselves: `Dataset.name`, `train_split`, `val_split`,
`dataset_classes`, and optionally `split_dir`, `test_splits` and `visualization_split`
(see the [config reference](reference/config.md)). Test and visualization splits default to the
five weather splits and `visualizations.txt` next to `val_split`.
