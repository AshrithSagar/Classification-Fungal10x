# Run workflows

Run the different workflows using

```bash
python3 workflows/{file}.py
```

from the root of the project directory.

## Workflows

<table data-full-width="true"><thead><tr><th>Workflow</th><th>Description</th></tr></thead><tbody><tr><td><code>create_dataset.py</code></td><td>Create the dataset from the slides and annotations. Creates a patch level dataset from the slides and performs Stratified k-fold at the patch level. Mainly uses the <code>dataset</code> key in config.yaml.</td></tr><tr><td><code>create_dataset-MIL.py</code></td><td>Creates an MIL dataset from the slides.</td></tr><tr><td><code>model_train.py</code></td><td>Trains the model on the patch level dataset using the selected model. The <code>trainer</code> key in <code>config.yaml</code> is used for training configurations, while the <code>model</code> key is utilised for selecting models and specifying their parameters.</td></tr><tr><td><code>model_train-MIL.py</code></td><td>Train on an MIL dataset.</td></tr><tr><td><code>generate_heatmaps.py</code></td><td>Generate heatmaps for the slides using the predictions of the trained model.</td></tr><tr><td><code>exp_summary.py</code></td><td>Generate a summary of the experiment over the different folds. Also packs the results (metrics, plots and heatmaps) in an exportable ZIP file.</td></tr></tbody></table>
