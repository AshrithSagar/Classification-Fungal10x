# Configuration file

The configuration file contains various settings and parameters that control the behavior and settings of the project.
Refer to the [config-template.yaml](https://github.com/AshrithSagar/Classification-Fungal10x/blob/main/config-template.yaml) file for more information.

## dataset

| Parameter         | Type              | Description                                                                                         |
| ----------------- | ----------------- | --------------------------------------------------------------------------------------------------- |
| annot_dir         | `path`            | Slide annotations directory path. Should have the same names as that in slide_dir.                  |
| create_zip        | `bool`            | Bundle the created dataset directory in a ZIP for easier download.                                  |
| data_dir_name     | `str`             | Used to create `dataset/{data_dir_name}/`.                                                          |
| downsample_factor | `int`             | Downsample slides resolution by this factor. Defaults to preserve aspect ratio.                     |
| downsample_size   | `tuple[int, int]` | Downsample slides to this size.                                                                     |
| n_splits          | `int`             | Number of splits for cross-validation.                                                              |
| overlap           | `bool`            | Overlap factor for extracting patches. Should be between 0 and 1.                                   |
| patch_size        | `tuple[int, int]` | Patch size for the patches.                                                                         |
| save_slides       | `bool`            | Whether to save slides, in `dataset/{data_dir_name}-slides/`.                                       |
| slide_dir         | `path`            | Slides directory path. Corresponding annotations should be in `annot_dir`.                          |
| use_augment       | `bool`            | Whether to use data augmentation at patch level for the train split. Preferably always use as True. |

## gpu

| Parameter    | Type  | Description                                                             |
| ------------ | ----- | ----------------------------------------------------------------------- |
| device_index | `int` | Device index for the GPU. Set to -1 to disable GPU and use CPU instead. |

## heatmaps

| Parameter         | Type                   | Description                                                                                                                              |
| ----------------- | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| alpha             | `float`                | Heatmap transparency while overlaying on the slide. Should be between 0 and 1.                                                           |
| blur              | `tuple[int, int]`      | Gaussian blur kernel size for the heatmap.                                                                                               |
| cmap              | `str`                  | Colormap for the heatmap. Refer to matplotlib colormaps.                                                                                 |
| downsample_factor | `int`                  | Downsample slides resolution by this factor (when `source_dir` is provided).                                                             |
| downsample_size   | `tuple[int, int]`      | Downsample slides to this size (when `source_dir` is provided).                                                                          |
| file_extension    | `str`                  | File extension for the heatmap images to be saved.                                                                                       |
| invert_preds      | `bool`                 | Whether to invert the predictions before making the heatmaps. Default is true.                                                           |
| overlap           | `float`                | Overlap factor for the heatmap patches. Should be between 0 and 1.                                                                       |
| patch_dims        | `tuple[int, int, int]` | Patch dimensions for the heatmap.                                                                                                        |
| percentile_scale  | `tuple[int, int]`      | Scale the heatmap values to percentile using `numpy.percentile()`.                                                                       |
| percentile_score  | `bool`                 | Use percentile score for scaling the heatmap values using `scipy.stats.percentileofscore()`.                                             |
| save_dir          | `path`                 | Directory to save the heatmap images. Will be saved at `{exp_base_dir}/{exp_name}/{fold-*}/{save_dir}/`.                                 |
| source_dir        | `path`                 | Path to the directory containing the slides. Used to get predictions for the heatmap.                                                    |
| source_dir_annot  | `path`                 | Path to the directory containing annotations corresponding to slides in `source_dir`. Set to null to use slides themselves for heatmaps. |
| use_plt           | `bool`                 | Use matplotlib to generate the heatmap images. If false, heatmaps will match original slide dimensions.                                  |

## model

### _select

| Parameter  | Type  | Description                                                                                              |
| ---------- | ----- | -------------------------------------------------------------------------------------------------------- |
| classifier | `str` | Model to use for training and inference. Options: {CLAM_SB, EfficientNetB0, MobileNet, ResNet50, VGG16}. |

### model-CLAM_SB

| Parameter     | Type   | Description         |
| ------------- | ------ | ------------------- |
| k_sample      | `null` |                     |
| dropout       | `null` |                     |
| learning_rate | `null` |                     |
| loss_weights  | `dict` | Keys: bag, instance |
| patience      | `null` |                     |
| run_eagerly   | `null` |                     |

### model-EfficientNetB0, model-MobileNet, model-ResNet50, model-VGG16

| Parameter        | Type   | Description |
| ---------------- | ------ | ----------- |
| freeze_ratio     | `null` |             |
| learning_rate    | `null` |             |
| patience         | `null` |             |
| start_from_epoch | `null` |             |

## trainer

| Parameter         | Type                   | Description                                                                                        |
| ----------------- | ---------------------- | -------------------------------------------------------------------------------------------------- |
| batch_size        | `int`                  | Batch size for training.                                                                           |
| data_dir          | `path`                 | Path to the directory containing the dataset. Should likely be `dataset/{data_dir_name}/`.         |
| evaluate_only     | `bool`                 | Evaluate the model on the test set only. Useful for evaluating a trained model.                    |
| exp_base_dir      | `path`                 | Base directory containing all the experiment folders. Usually `experiments/`.                      |
| exp_name          | `str`                  | Current experiment name. Will create a directory in `exp_base_dir` (`{exp_base_dir}/{exp_name}/`). |
| features_dir      | `path`                 | Path to the directory containing the features, particularly for MIL datasets.                      |
| folds             | `list[int]`            | List of folds to be considered. Zero-indexed.                                                      |
| max_epochs        | `int`                  | Maximum number of epochs to train the model.                                                       |
| overwrite_preds   | `bool`                 | Overwrite predictions if already present in `{exp_base_dir}/{exp_name}/{fold-*}/preds.csv`.        |
| patch_dims        | `tuple[int, int, int]` | Patch dimensions of the dataset.                                                                   |
| predictions_file  | `string`               | Filename of the predictions CSV file, without the extension.                                       |
| save_weights_only | `bool`                 | Save only the model's weights during checkpointing. Useful for subclassed models in `tf.keras`.    |
| subset_size       | `int`                  | Subset size of the dataset to use for training. Use `null` to use the entire dataset.              |
| use_augment       | `bool`                 | Whether to use augmented dataset for training (`dataset/{data_dir_name}/fold-*/{train}/`).         |
