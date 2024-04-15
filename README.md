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
  annot_dir: (path) Path to the directory containing the annotations. Slides should have the same names as that in slide_dir.
  create_zip: null
  data_dir_name: (str) dataset/{data_dir_name}/;
  downsample_factor: (int) Downsample slides resolution by this factor.
  downsample_size: (tuple) Downsample slides to this size.
  n_splits: (int) Number of splits for cross-validation.
  save_slides: (bool) Save slides, in dataset/{data_dir_name}-slides/;
  slide_dir: (path) Path to the directory containing the slides. Corresponding annotations should be in annot_dir.
  use_augment: (bool) Data augmentation at patch level for the train split. Preferably always use as True.
gpu:
  device_index: (int) Device index for the GPU. Set to -1 to disable GPU and use CPU instead.
heatmaps:
  alpha: (float) Heatmap transparency while overlaying on the slide.
  blur: (tuple) Gaussian blur kernel size for the heatmap.
  cmap: (str) Colormap for the heatmap.
  downsample_factor: (int) Downsample slides resolution by this factor. Will be used when source_dir is provided.
  downsample_size: (tuple) Downsample slides to this size. Will be used when source_dir is provided.
  file_extension: (str) File extension for the heatmap images.
  overlap: (float) Overlap factor for the heatmap patches.
  patch_size: (tuple) Patch size for the heatmap.
  percentile_scale: (tuple) Scale the heatmap values to percentile. Uses numpy.percentile();
  percentile_score: (bool) Percentile score for scaling the heatmap values. Uses scipy.stats.percentileofscore();
  save_dir: (path) Directory to save the heatmap images.
  source_dir: (path) Path to the directory containing the slides. Will be used to get the predictions for the heatmap.
  source_dir_annot: (path) Path to the directory containing the annotations. Will be used to get the ground truth for the heatmap. Should have the same names as that in source_dir.
  use_plt: (bool) Use matplotlib to generate the heatmap images.
model:
  _select: (str) Model to use for training and inference. Choose from the available models:- {CLAM_SB, EfficientNetB0, MobileNet, ResNet50, VGG16}.
  model-CLAM_SB:
    k_sample: null
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
  data_dir: (path) Path to the directory containing the dataset. Should likely be as some dataset/{data_dir_name}/;
  exp_base_dir: (path) Base directory to save the experiment results.
  exp_name: (str) Current experiment name. Will be used to create a directory in exp_base_dir.
  folds: (list) List of folds to be considered. Zero-indexed.
  max_epochs: (int) Maximum number of epochs to train the model.
  overwrite_preds: (bool) Overwrite the predictions if already present.
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

## License

This project falls under the [MIT License](LICENSE).
