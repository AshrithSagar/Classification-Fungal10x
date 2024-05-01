"""
trainer.py
"""

import datetime
import os
import sys
import time
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import yaml
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm

sys.path.append(os.getcwd())
from utils.config import line_separator


class ModelTrainer:
    """
    This class is responsible for training, evaluating, and predicting
    using a specified machine learning model. It handles data loading,
    training, evaluation metrics, and result storage.
    """

    def __init__(
        self,
        exp_base_dir=None,
        exp_name=None,
        exp_dir=None,
        data_dir=None,
        model_args=None,
        model_params=None,
        model_name=None,
        image_dims=(224, 224),
        MIL=None,
        seed=42,
    ):
        """
        Parameters:
        - exp_base_dir: Base directory for experiment results.
        - exp_name: Name of the experiment.
        - exp_dir: Directory for the specific experiment, defaults to "exp_base_dir/exp_name".
        - data_dir: Directory containing the dataset.
        - model_args: Arguments for configuring the model.
        - model_params: Model-specific parameters.
        - model_name: Name of the model
        - image_dims: Image dimensions
        - MIL: Whether MIL dataset is used
        - seed: Random seed for reproducibility.
        """

        self.seed = seed
        tf.random.set_seed(seed)
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
        tf.keras.backend.clear_session()
        if exp_dir is not None:
            self.exp_dir = exp_dir
        else:
            self.exp_dir = os.path.join(exp_base_dir, exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        self.data_dir = data_dir
        self.model_args = model_args
        self.model_params = model_params
        self.model_name = model_name
        self.image_dims = image_dims
        self.MIL = MIL
        self.model = None
        self.results = {}
        self.model_args.update({"model_name": model_name})

    def load_dataset(self, subset_size=None, use_augment=False):
        if self.MIL:
            self.load_MIL_dataset(subset_size, use_augment)
        else:
            self.load_patches_dataset(subset_size, use_augment)

        if subset_size:
            self.train_ds = (
                self.train_ds.unbatch()
                .take(subset_size)
                .batch(self.model_args["batch_size"])
            )
            self.val_ds = (
                self.val_ds.unbatch()
                .take(subset_size)
                .batch(self.model_args["batch_size"])
            )
            self.test_ds = (
                self.test_ds.unbatch()
                .take(subset_size)
                .batch(self.model_args["batch_size"])
            )
            print(f"Using subset of data for training of size: {subset_size}")

    def load_patches_dataset(self, subset_size=None, use_augment=False):
        """
        Parameters:
        - Loads the patches dataset from the data directory.
        - subset_size: Optional, subset of the dataset to use.
            Default is to use entire available dataset.
        - use_augment: Optional, whether to use the augmented dataset.
        """

        train_ds_dir = "train_unaugmented" if not use_augment else "train"
        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            os.path.join(self.data_dir, train_ds_dir),
            labels="inferred",
            color_mode="rgb",
            batch_size=self.model_args["batch_size"],
            image_size=self.image_dims[:2],
        )
        self.val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            os.path.join(self.data_dir, "val"),
            labels="inferred",
            color_mode="rgb",
            batch_size=self.model_args["batch_size"],
            image_size=self.image_dims[:2],
        )
        self.test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            os.path.join(self.data_dir, "test"),
            labels="inferred",
            color_mode="rgb",
            batch_size=self.model_args["batch_size"],
            image_size=self.image_dims[:2],
        )
        self.class_names = self.train_ds.class_names
        self.train_ds = self.train_ds.prefetch(tf.data.experimental.AUTOTUNE)

    def load_MIL_dataset(self, subset_size=None, use_augment=False):
        """
        Parameters:
        - Loads the dataset from the MIL data directory.
        - subset_size: Optional, subset of the dataset to use.
            Default is to use entire available dataset.
        """

        def infer_labels(image, name):
            name = tf.strings.as_string(name)
            label = tf.cond(
                tf.strings.regex_full_match(name, "^F.*"),
                lambda: 1,
                lambda: 0,
            )
            return image, label

        def load(data_dir, sub_dir):
            print(f"{sub_dir}:", end=" ")
            dataset_dir = os.path.join(data_dir, sub_dir)
            dataset = tf.keras.preprocessing.image_dataset_from_directory(
                dataset_dir,
                labels="inferred",
                color_mode="rgb",
                batch_size=self.model_args["batch_size"],
                image_size=self.image_dims[:2],
                shuffle=False,
            )
            return dataset

        print(f"Loading MIL dataset from: {self.data_dir}")
        train_ds = load(self.data_dir, "train")
        self.test_ds = load(self.data_dir, "test")

        self.train_ds, self.val_ds = train_ds, train_ds

        self.train_names = self.train_ds.class_names
        self.val_names = self.val_ds.class_names
        self.test_names = self.test_ds.class_names

        batch_size = self.model_args["batch_size"]
        self.train_ds = self.train_ds.map(infer_labels).batch(batch_size)
        self.val_ds = self.val_ds.map(infer_labels).batch(batch_size)
        self.test_ds = self.test_ds.map(infer_labels).batch(batch_size)

    def load_MIL_features(self, subset_size=None, batch_size=32, shuffle=True):
        """
        Parameters:
        - Loads the features from the MIL features directory.
        - subset_size: Optional, subset of the dataset to use.
            Default is to use entire available dataset.
        """

        def infer_label(filename):
            if filename.startswith("F"):
                return 1
            elif filename.startswith("N"):
                return 0
            return None

        def load(data_dir, sub_dir, batch_size=32, shuffle=True):
            dataset_dir = os.path.join(data_dir, sub_dir)
            files = [file for file in os.listdir(dataset_dir) if file.endswith(".npy")]
            features, labels = [], []
            for file in files:
                data = np.load(os.path.join(dataset_dir, file))
                label = infer_label(file)
                if label is not None:
                    features.append(data)
                    labels.append(label)

            dataset = tf.data.Dataset.from_tensor_slices((features, labels))
            if shuffle:
                dataset = dataset.shuffle(buffer_size=len(features))
            dataset = dataset.batch(batch_size)
            return dataset

        print(f"Loading MIL features from: {os.path.basename(self.data_dir)}")
        self.train_ds = load(self.data_dir, "train", batch_size, shuffle)
        self.val_ds = load(self.data_dir, "train", batch_size, shuffle)
        self.test_ds = load(self.data_dir, "test", batch_size, shuffle)

    def info(self, verbose=False):
        if verbose:
            if self.MIL:
                print(f"Train: {self.train_names}")
                print(f"Val: {self.val_names}")
                print(f"Test: {self.test_names}")
            else:
                print(f"Classes: {self.class_names}")

        self.model.summary()
        model_summary_file = os.path.join(self.exp_dir, "model_summary.txt")
        with open(model_summary_file, "w") as f:
            self.model.summary(print_fn=lambda x: f.write(x + "\n"))

    def train(self):
        epoch_time_callback = EpochTimeCallback()

        tic = time.time()
        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            verbose=1,
            callbacks=[epoch_time_callback] + self.callbacks_list,
            initial_epoch=self.epochs_done,
            epochs=self.model_args["max_epochs"],
            batch_size=self.model_args["batch_size"],
        )
        toc = time.time()

        format_time = (
            lambda x: time.strftime("%Hh %Mm %Ss", time.gmtime(x))
            .replace("00h ", "")
            .replace("00m ", "")
            .lstrip("0")
        )
        time_taken = toc - tic
        self.results.update({"training_time": format_time(time_taken)})

        epoch_times = np.array(epoch_time_callback.epoch_times)
        formatted_epoch_times = [format_time(epoch) for epoch in epoch_times]
        epoch_times_path = os.path.join(self.exp_dir, "epoch_times.txt")
        with open(epoch_times_path, "a") as f:
            f.write("\n".join(formatted_epoch_times) + "\n")

        self.results.update({"converging": None})

        csv_logger_path = os.path.join(self.exp_dir, "logger.csv")
        self.csv_data = pd.read_csv(csv_logger_path)
        epochs = len(history.epoch)
        total_epochs = len(self.csv_data["epoch"])
        self.results.update({"epochs": total_epochs})
        print("Epochs trained in current run:", epochs, end="; ")
        print("Epochs trained in total:", total_epochs)
        print(line_separator)

    def evaluate(self):
        def get_accuracy_plot(show_plots=False):
            plt.clf()
            plt.plot(self.csv_data["accuracy"])
            plt.plot(self.csv_data["val_accuracy"])
            plt.title("model accuracy")
            plt.ylabel("accuracy")
            plt.xlabel("epoch")
            plt.legend(["train", "val"], loc="upper left")
            save_file = os.path.join(self.exp_dir, "model_accuracy.jpeg")
            plt.savefig(save_file, dpi=150)
            plt.show() if show_plots else plt.close()

        def get_loss_plot(show_plots=False):
            plt.clf()
            plt.plot(self.csv_data["loss"])
            plt.plot(self.csv_data["val_loss"])
            plt.title("model loss")
            plt.ylabel("loss")
            plt.xlabel("epoch")
            plt.legend(["train", "val"], loc="upper left")
            save_file = os.path.join(self.exp_dir, "model_loss.jpeg")
            plt.savefig(save_file, dpi=150)
            plt.show() if show_plots else plt.close()

        def get_classification_report():
            print(classification_report(self.y_test, self.y_pred, zero_division=np.nan))
            print(line_separator)

            self.results.update(
                {
                    "classification_report": classification_report(
                        self.y_test, self.y_pred, output_dict=True, zero_division=np.nan
                    )
                }
            )

        def get_confusion_matrix(show_plots=False):
            plt.clf()
            test_confusion_matrix = tf.math.confusion_matrix(
                labels=self.y_test, predictions=self.y_pred
            ).numpy()
            LABELS = ["fungal", "nonfungal"]
            plt.figure()
            sns.heatmap(
                test_confusion_matrix,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                cmap=plt.cm.Greens,
                fmt="d",
            )
            plt.ylabel("True label")
            plt.xlabel("Predicted label")
            save_file = os.path.join(self.exp_dir, "confusion_matrix.jpeg")
            plt.savefig(save_file, dpi=150)
            plt.show() if show_plots else plt.close()

        def get_roc_curve(show_plots=False):
            ns_probs = np.zeros(len(self.y_test))
            ns_auc = roc_auc_score(self.y_test, ns_probs)
            lr_auc = roc_auc_score(self.y_test, self.y_pred)
            print(f"No Skill: ROC AUC={ns_auc:.3}")
            print(f"{self.model_name}: ROC AUC={lr_auc:.3}")
            print(line_separator)

            ns_fpr, ns_tpr, _ = roc_curve(self.y_test, ns_probs)
            lr_fpr, lr_tpr, _ = roc_curve(self.y_test, self.y_pred)

            plt.clf()
            plt.plot(ns_fpr, ns_tpr, linestyle="--", label="No Skill")
            plt.plot(lr_fpr, lr_tpr, marker=".", label=self.model_name)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc="lower right")
            save_file = os.path.join(self.exp_dir, "roc_curve.jpeg")
            plt.savefig(save_file, dpi=150)
            plt.show() if show_plots else plt.close()

            self.results.update({"ROC": float(lr_auc)})

        _, train_acc = self.model.evaluate(self.train_ds)
        _, val_acc = self.model.evaluate(self.val_ds)
        _, test_acc = self.model.evaluate(self.test_ds)

        accuracy_verbose = (
            lambda data_set, acc: f"Model accuracy on {data_set} data: {acc * 100}"
        )
        print(line_separator)
        print(accuracy_verbose("train", train_acc))
        print(accuracy_verbose("val", val_acc))
        print(accuracy_verbose("test", test_acc))
        print(line_separator)

        self.results.update(
            {
                "model_accuracy": {
                    "train": train_acc,
                    "val": val_acc,
                    "test": test_acc,
                }
            }
        )

        self.x_test, self.y_test = map(np.array, zip(*self.test_ds.unbatch()))
        self.y_pred = self.model.predict(self.x_test)
        self.y_pred = np.where(
            self.y_pred > 0.5, 1, 0
        )  # Better to use; https://stackoverflow.com/a/73215222

        get_classification_report()
        get_confusion_matrix()
        get_accuracy_plot()
        get_loss_plot()
        get_roc_curve()

        results_file = os.path.join(self.exp_dir, "results.yaml")
        with open(results_file, "w") as outfile:
            yaml.dump(self.results, outfile, default_flow_style=False)

        self.model_args.update(self.model_params)
        params_file = os.path.join(self.exp_dir, "params.yaml")
        with open(params_file, "w") as outfile:
            yaml.dump(self.model_args, outfile, default_flow_style=False)

    def predict(
        self, patched_slides, save_file="preds.csv", overwrite=False, verbose=None
    ):
        file = os.path.join(self.exp_dir, save_file)
        if not overwrite and os.path.exists(file):
            predictions = np.loadtxt(file, delimiter=",")
            return predictions

        predictions = []
        iterator = (
            patched_slides
            if verbose is not None
            else tqdm(patched_slides, desc="Predicting")
        )
        for index, patches in enumerate(iterator):
            preds = self.model.predict_on_batch(patches)
            if verbose:
                print(f"Slide {index+1}: [{', '.join(str(x[0]) for x in preds)}]")
            predictions.append(preds)

        predictions = np.squeeze(np.array(predictions))
        np.savetxt(file, predictions, delimiter=",")

        return predictions


class ModelSummary:
    """This class provides functionality to analyze and summarize results of each trained model."""

    def __init__(self, exp_base_dir, exp_name):
        """
        Parameters:
        - exp_base_dir: Base directory containing experiment directories.
        - exp_name: Name of the experiment.
        """

        self.exp_dir = os.path.join(exp_base_dir, exp_name)

    def check_folds(self, folds=None):
        """Checks and lists fold directories within the experiment."""
        if folds is None:
            self.folds = [
                fold_dir
                for fold_dir in os.listdir(self.exp_dir)
                if os.path.isdir(fold_dir)
            ]
        else:
            self.folds = [f"fold_{i}" for i in folds]
        return self.folds

    def get_cv_results(self, folds=None):
        """
        Computes cross-validation results from fold-wise experiment results.
        mean and standard deviation are returned for each metric.
        """
        self.get_fold_results(folds)

        mean_std_metrics = {}
        classification_report_metrics = ["f1-score", "precision", "recall"]
        for metric in [
            "ROC",
            "train_accuracy",
            "val_accuracy",
            "test_accuracy",
        ] + classification_report_metrics:
            if metric == "train_accuracy":
                metric_values = [
                    result["model_accuracy"]["train"] for result in self.results
                ]
                metric_values = np.array(metric_values)
            elif metric == "val_accuracy":
                metric_values = [
                    result["model_accuracy"]["val"] for result in self.results
                ]
                metric_values = np.array(metric_values)
            elif metric == "test_accuracy":
                metric_values = [
                    result["model_accuracy"]["test"] for result in self.results
                ]
                metric_values = np.array(metric_values)
            elif metric in classification_report_metrics:
                metric_values = [
                    result["classification_report"]["0"][metric]
                    for result in self.results
                ]
            else:
                metric_values = [result[metric] for result in self.results]

            mean = np.mean(metric_values, axis=0)
            std = np.std(metric_values, axis=0)
            mean_std_metrics[metric] = {"mean": mean, "std": std}

        mean_std_df = pd.DataFrame(mean_std_metrics).T
        print("Folds:", self.folds)
        print(mean_std_df)

        mean_std_df.to_csv(os.path.join(self.exp_dir, "cv_results.csv"), index=True)

    def get_fold_results(self, folds=None):
        """Fetches and summarizes results for individual folds."""
        self.check_folds(folds)

        self.results = []
        for fold in self.folds:
            results_file = os.path.join(self.exp_dir, fold, "results.yaml")
            with open(results_file, "r") as infile:
                self.results.append(yaml.load(infile, Loader=yaml.FullLoader))

        results = []
        columns = [
            "ROC",
            "training_time",
            "epochs",
            "converging",
            "train_accuracy",
            "val_accuracy",
            "test_accuracy",
            "f1-score",
            "precision",
            "recall",
        ]
        for fold_result in self.results:
            metrics = []
            classification_report_metrics = ["f1-score", "precision", "recall"]
            for metric in columns:
                if metric == "train_accuracy":
                    value = fold_result["model_accuracy"]["train"]
                elif metric == "val_accuracy":
                    value = fold_result["model_accuracy"]["val"]
                elif metric == "test_accuracy":
                    value = fold_result["model_accuracy"]["test"]
                elif metric in classification_report_metrics:
                    value = fold_result["classification_report"]["0"][metric]
                else:
                    value = fold_result[metric]
                metrics.append(value)
            results.append(metrics)

        results_df = pd.DataFrame(results, columns=columns)
        print("Folds:", self.folds)
        print(results_df)

        results_df.to_csv(os.path.join(self.exp_dir, "fold_results.csv"), index=True)

    def get_plots(self, folds=None):
        """Creates a zip file containing plots for each fold."""
        self.check_folds(folds)

        zip_file_path = os.path.join(
            self.exp_dir, f"{os.path.basename(self.exp_dir)}_plots.zip"
        )
        with zipfile.ZipFile(zip_file_path, "w") as zipf:
            for fold in self.folds:
                fold_dir = os.path.join(self.exp_dir, fold)
                filenames = [
                    "confusion_matrix.jpeg",
                    "model_accuracy.jpeg",
                    "model_loss.jpeg",
                    "roc_curve.jpeg",
                ]
                for filename in filenames:
                    file = os.path.join(fold_dir, filename)
                    zipf.write(file, arcname=os.path.join(fold, filename))

    def get_heatmaps(self, folder="heatmaps", folds=None):
        """Creates a zip file containing heatmaps for each fold."""
        self.check_folds(folds)

        zip_file_path = os.path.join(
            self.exp_dir, f"{os.path.basename(self.exp_dir)}_heatmaps.zip"
        )
        with zipfile.ZipFile(zip_file_path, "w") as zipf:
            for fold in self.folds:
                heatmap_dir = os.path.join(self.exp_dir, fold, folder)
                for filename in os.listdir(heatmap_dir):
                    file = os.path.join(heatmap_dir, filename)
                    zipf.write(file, arcname=os.path.join(fold, filename))

    def get_results(self, heatmaps="heatmaps", folds=None):
        """Creates a zip file containing overall results, including metrics, plots, and heatmaps."""
        self.check_folds(folds)
        self.get_cv_results(folds)

        zip_file_path = os.path.join(
            self.exp_dir, f"{os.path.basename(self.exp_dir)}_results.zip"
        )
        with zipfile.ZipFile(zip_file_path, "w") as zipf:
            # Metrics
            metrics = ["cv_results.csv", "fold_results.csv"]
            for filename in metrics:
                zipf.write(
                    os.path.join(self.exp_dir, filename),
                    arcname=filename,
                )

            for fold in self.folds:
                fold_dir = os.path.join(self.exp_dir, fold)
                heatmap_dir = os.path.join(fold_dir, heatmaps)
                plots = [
                    "confusion_matrix.jpeg",
                    "model_accuracy.jpeg",
                    "model_loss.jpeg",
                    "roc_curve.jpeg",
                ]

                # Plots
                for filename in plots:
                    zipf.write(
                        os.path.join(fold_dir, filename),
                        arcname=os.path.join(fold, filename),
                    )

                # Heatmaps
                for filename in os.listdir(heatmap_dir):
                    zipf.write(
                        os.path.join(heatmap_dir, filename),
                        arcname=os.path.join(fold, heatmaps, filename),
                    )


class EpochTimeCallback(Callback):
    """A custom callback class to log the time
    taken for each epoch during model training."""

    def __init__(self):
        super().__init__()
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        end_time = time.time()
        epoch_time = end_time - self.start_time
        self.epoch_times.append(epoch_time)


class ModelMaker:
    """
    This class handles the creation and configuration of
    machine learning models based on specified parameters.
    """

    def __init__(
        self,
        model_args,
        model_params,
        exp_dir: str = None,
        model: str = None,
    ):
        """
        Parameters:
        - model_args: Arguments for configuring the model.
        - model_params: Model-specific parameters.
        - exp_dir: Directory for experiment results.
        - model: Name of the model to be used.
        """
        self.args = model_args
        self.params = model_params
        self.exp_dir = exp_dir
        self.model = model
        self.callbacks_list = None

        self.paths = {
            "checkpoint": os.path.join(self.exp_dir, "model.h5"),
            "csv_logger": os.path.join(self.exp_dir, "logger.csv"),
            "tensorboard": os.path.join(
                self.exp_dir,
                "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            ),
        }

        self.get()
        self.load_checkpoint()

    def get(self):
        """
        Retrieves the specified model architecture and associated callbacks.
        """
        callbacks_list = []
        if isinstance(self.model, str):
            model_imports = {
                "CLAM_SB": "models.clamSB_tf",
                "EfficientNetB0": "models.EfficientNetB0",
                "MobileNet": "models.MobileNet",
                "ResNet50": "models.ResNet50",
                "VGG16": "models.VGG16",
            }

            if self.model not in model_imports:
                raise ValueError("Invalid model")

            module = __import__(
                model_imports[self.model], fromlist=["model", "model_callbacks"]
            )
            model_func = getattr(module, "model")
            callbacks_func = getattr(module, "model_callbacks")

            self.model = model_func(self.args, self.params)
            callbacks_list = callbacks_func(self.args, self.params)

        self.callbacks_list = [
            keras.callbacks.ModelCheckpoint(
                filepath=self.paths["checkpoint"],
                verbose=1,
                monitor="val_loss",
                save_best_only=True,
            ),
            keras.callbacks.CSVLogger(
                filename=self.paths["csv_logger"],
                separator=",",
                append=True,
            ),
            keras.callbacks.TensorBoard(
                log_dir=self.paths["tensorboard"], histogram_freq=1
            ),
        ]
        self.callbacks_list.extend(callbacks_list)

    def load_checkpoint(self):
        """Loads model weights from checkpoint file, if available."""

        self.epochs_done = 0
        if os.path.exists(self.paths["checkpoint"]):
            print(f'Loading model from checkpoint: {self.paths["checkpoint"]}')
            self.model.load_weights(self.paths["checkpoint"])

            if os.path.exists(self.paths["csv_logger"]):
                csv_data = pd.read_csv(self.paths["csv_logger"])
                if not csv_data.empty:
                    self.epochs_done = len(csv_data["epoch"])

        remaining_epochs = max(0, self.args["max_epochs"] - self.epochs_done)
        print(f"{self.epochs_done} Epochs done; Remaining epochs: {remaining_epochs}")


class ExperimentSummary:
    """Summarize results from multiple experiments"""

    def __init__(self, exp_base_dir, exp_names):
        self.exp_base_dir = exp_base_dir
        self.exp_names = exp_names
        self.exp_dirs = [os.path.join(exp_base_dir, exp_name) for exp_name in exp_names]

    def get_results(self, format=False):
        """Fetches CV results from each experiment directory"""
        results = []
        for exp_dir in self.exp_dirs:
            file = os.path.join(exp_dir, "cv_results.csv")
            df = pd.read_csv(file, index_col=0)
            results.append(df)

        results_df = pd.concat(results, keys=self.exp_names)
        self.results_df = results_df.pivot_table(
            values="mean",
            index=results_df.index.get_level_values(0),
            columns=results_df.index.get_level_values(1),
        )

        if format:
            self.results_df = self.results_df.applymap(lambda x: f"{x * 100:.2f}%")

    def save(self, filename="exp_results.csv"):
        """Saves the summarized results to a CSV file"""
        self.results_df.to_csv(
            os.path.join(self.exp_base_dir, filename), index=self.exp_names
        )
