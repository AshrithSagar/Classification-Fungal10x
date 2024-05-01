# Classification-Fungal10x

![GitHub](https://img.shields.io/github/license/AshrithSagar/Classification-Fungal10x) ![GitHub repo size](https://img.shields.io/github/repo-size/AshrithSagar/Classification-Fungal10x) [![CodeFactor](https://www.codefactor.io/repository/github/AshrithSagar/Classification-Fungal10x/badge)](https://www.codefactor.io/repository/github/AshrithSagar/Classification-Fungal10x)

Fungal classification | Multiple Instance Learning

## Installation

1. Clone the repository.

    ```bash
    git clone https://github.com/AshrithSagar/Classification-Fungal10x.git
    cd Classification-Fungal10x
    ```

2. Optionally, create a virtual environment and activate it.

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

    Or use `conda` to create a virtual environment.

    ```bash
    conda create --name clfx python=3.9
    conda activate clfx
    ```

3. Install the required packages.

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Config file

The configuration file contains various settings and parameters that control the behavior and settings of the project.
Refer to the [config-template.yaml](config-template.yaml) file for more information.

```yaml
---
dataset:
  annot_dir: (path) Slide annotations directory path. Should have the same names as that in slide_dir.
  create_zip: (bool) Bundle the created dataset directory in a ZIP for easier download.
  data_dir_name: (str) Will be used to create dataset/{data_dir_name}/;
  downsample_factor: (int) Downsample slides resolution by this factor. Defaults to preserve aspect ratio.
  downsample_size: (tuple[int, int]) Downsample slides to this size.
  n_splits: (int) Number of splits for cross-validation.
  overlap: (bool) Overlap factor for the extracting patches. Should be between 0 and 1.
  patch_size: (tuple[int, int]) Patch size for the patches.
  save_slides: (bool) Whether to save slides, in dataset/{data_dir_name}-slides/;
  slide_dir: (path) Slides directory path. Corresponding annotations should be in annot_dir.
  use_augment: (bool) Whether to use data augmentation at patch level for the train split. Preferably always use as True.
gpu:
  device_index: (int) Device index for the GPU. Set to -1 to disable GPU and use CPU instead.
heatmaps:
  alpha: (float) Heatmap transparency while overlaying on the slide. Should be between 0 and 1.
  blur: (tuple[int, int]) Gaussian blur kernel size for the heatmap.
  cmap: (str) Colormap for the heatmap. Refer to matplotlib colormaps.
  downsample_factor: (int) Downsample slides resolution by this factor. Will be used when source_dir is provided.
  downsample_size: (tuple[int, int]) Downsample slides to this size. Will be used when source_dir is provided.
  file_extension: (str) File extension for the heatmap images to be saved.
  invert_preds: (bool) Whether to invert the predictions before making the heatmaps. Default is true.
  overlap: (float) Overlap factor for the heatmap patches. Should be between 0 and 1.
  patch_dims: (tuple[int, int, int]) Patch dimensions for the heatmap.
  percentile_scale: (tuple[int, int]) Scale the heatmap values to percentile. Uses numpy.percentile();
  percentile_score: (bool) Percentile score for scaling the heatmap values. Uses scipy.stats.percentileofscore();
  save_dir: (path) Directory to save the heatmap images. Will be saved at {exp_base_dir}/{exp_name}/{fold-*}/{save_dir}/;
  source_dir: (path) Path to the directory containing the slides. Will be used to get the predictions for the heatmap.
  source_dir_annot: (path) Path to the directory containing the annotations corresponding to the slides in source_dir. Slides should have the same names as in source_dir. Will be used to overlap with the heatmap for easier visualisation. Set to null to use source_dir slides itself for heatmaps.
  use_plt: (bool) Use matplotlib to generate the heatmap images. If false, then heatmaps will match original slide dimensions.
model:
  _select: (str) Model to use for training and inference. {CLAM_SB, EfficientNetB0, MobileNet, ResNet50, VGG16}.
  model-CLAM_SB:
    k_sample: null
    dropout: null
    learning_rate: null
  model-EfficientNetB0:
    freeze_ratio: null
    learning_rate: null
    patience: null
    start_from_epoch: null
  model-MobileNet:
    freeze_ratio: null
    learning_rate: null
    patience: null
    start_from_epoch: null
  model-ResNet50:
    freeze_ratio: null
    learning_rate: null
    patience: null
    start_from_epoch: null
  model-VGG16:
    freeze_ratio: null
    learning_rate: null
    patience: null
    start_from_epoch: null
trainer:
  batch_size: (int) Batch size for training.
  data_dir: (path) Path to the directory containing the dataset. Should likely be as some dataset/{data_dir_name}/; Should contain within as /fold-*/{train, val, test}/{fungal, non-fungal}/; Refer to the directory structure.
  exp_base_dir: (path) Base directory containing all the experiment folders. Usually set to experiments/.
  exp_name: (str) Current experiment name. Will be used to create a directory in exp_base_dir. {exp_base_dir}/{exp_name}/;
  folds: (list[int]) List of folds to be considered. Zero-indexed.
  max_epochs: (int) Maximum number of epochs to train the model.
  overwrite_preds: (bool) Overwrite the predictions if already present. Checks for {exp_base_dir}/{exp_name}/{fold-*}/preds.csv;
  patch_dims: (tuple[int, int, int]) Patch dimensions of the dataset.
  predictions_file: (string) Filename of the predictions CSV file, without the extension.
  subset_size: (int) Subset size of the dataset to be used for training. Used for trial purposes. Set to null to use the entire dataset.
  use_augment: (bool) Whether to use the augmented dataset for training present at train/, or use train_unaugmented/ for training. Whole path:- dataset/{data_dir_name}/fold-*/{train,train_unaugmented}/;
```

### Directory structure

The directory structure of the created datasets is as follows:

Slide directory structure:

```markdown
dataset
└── <ds-name>-slides
    ├── train
    │   ├── fungal
    │   │   ├── F-slide-1.png
    │   │   ├── F-slide-2.png
    │   │   └── ...
    │   └── non-fungal
    │       ├── NF-slide-1.png
    │       ├── NF-slide-2.png
    │       └── ...
    └── test
        ├── fungal
        │   ├── F-slide-1.png
        │   ├── F-slide-2.png
        │   └── ...
        └── non-fungal
            ├── NF-slide-1.png
            ├── NF-slide-2.png
            └── ...
```

Patches directory structure:

```markdown
dataset
└── <ds-name>-<downsample_dimensions>
    ├── fold-0
    │   ├── train
    │   │   ├── fungal
    │   │   │   ├── F-patch-1.png
    │   │   │   ├── F-patch-2.png
    │   │   │   └── ...
    │   │   └── non-fungal
    │   │       ├── NF-patch-1.png
    │   │       ├── NF-patch-2.png
    │   │       └── ...
    │   ├── val
    │   │   ├── fungal
    │   │   │   ├── F-patch-1.png
    │   │   │   ├── F-patch-2.png
    │   │   │   └── ...
    │   │   └── non-fungal
    │   │       ├── NF-patch-1.png
    │   │       ├── NF-patch-2.png
    │   │       └── ...
    │   └── test
    │       ├── fungal
    │       │   ├── F-patch-1.png
    │       │   ├── F-patch-2.png
    │       │   └── ...
    │       └── non-fungal
    │           ├── NF-patch-1.png
    │           ├── NF-patch-2.png
    │           └── ...
    ├── fold-1
    │   └── ...
    └── ...
```

MIL directory structure:

```markdown
dataset
└── <ds-name>-MIL-<downsample_dimensions>
    ├── train
    │   ├── F-slide-1
    │   │   ├── F-patch-1.png
    │   │   ├── F-patch-2.png
    │   │   └── ...
    │   ├── ...
    │   ├── NF-slide-1
    │   │   ├── NF-patch-1.png
    │   │   ├── NF-patch-2.png
    │   │   └── ...
    │   └── ...
    └── test
        ├── F-slide-1
        │   ├── F-patch-1.png
        │   ├── F-patch-2.png
        │   └── ...
        ├── ...
        ├── NF-slide-1
        │   ├── NF-patch-1.png
        │   ├── NF-patch-2.png
        │   └── ...
        └── ...
```

### Workflows

Run the different workflows using `python3 workflows/*.py` from the project directory.

1. `create_dataset.py`:
Create the dataset from the slides and annotations.
Creates a patch level dataset from the slides and performs Stratified k-fold at the patch level.
Mainly uses the `dataset` key in config.yaml.

1. `create_dataset-MIL.py`:
Creates an MIL dataset from the slides.

1. `model_train.py`:
Trains the model on the patch level dataset using the selected model.
The `trainer` key in `config.yaml` is used for training configurations, while the `model` key is utilised for selecting models and specifying their parameters.

1. `model_train-MIL.py`:
Train on an MIL dataset.

1. `generate_heatmaps.py`:
Generate heatmaps for the slides using the predictions of the trained model.

1. `exp_summary.py`:
Generate a summary of the experiment over the different folds.
Also packs the results (metrics, plots and heatmaps) in an exportable ZIP file.

## License

This project falls under the [MIT License](LICENSE).
