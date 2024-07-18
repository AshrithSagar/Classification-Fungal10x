# Run workflows

Run the different workflows using

```bash
python3 workflows/{file}.py
```

from the root of the project directory.

## Workflows

| Workflow                | Description                                                                                                                                                                                                                           |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `create_dataset.py`     | Create the dataset from the slides and annotations. Creates a patch level dataset from the slides and performs Stratified k-fold at the patch level. Mainly uses the `dataset` key in config.yaml.                                    |
| `create_dataset-MIL.py` | Creates an MIL dataset from the slides.                                                                                                                                                                                               |
| `model_train.py`        | Trains the model on the patch level dataset using the selected model. The `trainer` key in `config.yaml` is used for training configurations, while the `model` key is utilised for selecting models and specifying their parameters. |
| `model_train-MIL.py`    | Train on an MIL dataset.                                                                                                                                                                                                              |
| `generate_heatmaps.py`  | Generate heatmaps for the slides using the predictions of the trained model.                                                                                                                                                          |
| `exp_summary.py`        | Generate a summary of the experiment over the different folds. Also packs the results (metrics, plots and heatmaps) in an exportable ZIP file.                                                                                        |
