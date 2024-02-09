"""
trainer.py
"""

import os
import time
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, roc_curve


class ModelTrainer:
    def __init__(
        self, exp_base_dir, exp_name, data_dir, model_args, model_params, seed=42
    ):
        self.seed = seed
        tf.random.set_seed(seed)
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
        tf.keras.backend.clear_session()
        self.exp_dir = os.path.join(exp_base_dir, exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        self.data_dir = data_dir
        self.model_args = model_args
        self.model_params = model_params
        self.image_dims = (224, 224)
        self.model = None
        self.results = {}

    def check_gpu(self):
        print("GPU:", "Enabled" if tf.test.gpu_device_name() else "Disabled")
        print(tf.config.list_physical_devices())

    def set_gpu(self, device_index):
        physical_devices = tf.config.list_physical_devices("GPU")
        if physical_devices:
            tf.config.set_visible_devices(physical_devices[device_index], "GPU")
            tf.config.experimental.set_memory_growth(
                physical_devices[device_index], True
            )
            print(f"Selecting GPU: {physical_devices[device_index]}")
        else:
            print("No GPU devices found.")

    def load_dataset(self, use_augment=False):
        train_ds_dir = "train_unaugmented" if not use_augment else "train"
        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            os.path.join(self.data_dir, train_ds_dir),
            labels="inferred",
            color_mode="rgb",
            batch_size=self.model_args["batch_size"],
            image_size=self.image_dims,
        )
        self.val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            os.path.join(self.data_dir, "val"),
            labels="inferred",
            color_mode="rgb",
            batch_size=self.model_args["batch_size"],
            image_size=self.image_dims,
        )
        self.test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            os.path.join(self.data_dir, "test"),
            labels="inferred",
            color_mode="rgb",
            batch_size=self.model_args["batch_size"],
            image_size=self.image_dims,
        )

    def info(self):
        class_names = self.train_ds.class_names
        print(class_names)

        self.model.summary()
        model_summary_file = os.path.join(self.exp_dir, "model_summary.txt")
        with open(model_summary_file, "w") as f:
            self.model.summary(print_fn=lambda x: f.write(x + "\n"))

    def train(self):
        tic = time.time()
        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            verbose=1,
            callbacks=self.callbacks_list,
            initial_epoch=self.epochs_done,
            epochs=self.model_args["max_epochs"],
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

        self.results.update({"converging": None})

        csv_logger_path = os.path.join(self.exp_dir, "csv_logger.csv")
        self.csv_data = pd.read_csv(csv_logger_path)
        epochs = len(history.epoch)
        total_epochs = len(self.csv_data["epoch"])
        self.results.update({"epochs": total_epochs})
        print("Epochs trained in current run:", epochs, end="; ")
        print("Epochs trained in total:", total_epochs)

    def evaluate(self):
        def get_accuracy_plot(show_plots=False):
            plt.clf()
            plt.plot(self.csv_data["accuracy"])
            plt.plot(self.csv_data["val_accuracy"])
            plt.title("model accuracy")
            plt.ylabel("accuracy")
            plt.xlabel("epoch")
            plt.legend(["train", "val"], loc="upper left")
            plt.savefig(os.path.join(self.exp_dir, "model_accuracy.jpeg"), dpi=150)
            if show_plots:
                plt.show()

        def get_loss_plot(show_plots=False):
            plt.clf()
            plt.plot(self.csv_data["loss"])
            plt.plot(self.csv_data["val_loss"])
            plt.title("model loss")
            plt.ylabel("loss")
            plt.xlabel("epoch")
            plt.legend(["train", "val"], loc="upper left")
            plt.savefig(os.path.join(self.exp_dir, "model_loss.jpeg"), dpi=150)
            if show_plots:
                plt.show()

        def get_classification_report():
            y_pred = np.where(
                y_pred > 0.5, 1, 0
            )  # Better to use; https://stackoverflow.com/a/73215222
            print(classification_report(self.y_test, y_pred))
            self.results.update(
                {
                    "classification_report": classification_report(
                        self.y_test, y_pred, output_dict=True
                    )
                }
            )

        def get_confusion_matrix(show_plots=False):
            plt.clf()
            test_confusion_matrix = tf.math.confusion_matrix(
                labels=self.y_test, predictions=y_pred
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
            if show_plots:
                plt.show()

        def get_roc_curve(show_plots=False):
            ns_probs = np.zeros(len(self.y_test))
            ns_auc = roc_auc_score(self.y_test, ns_probs)
            lr_auc = roc_auc_score(self.y_test, y_pred)
            print("No Skill: ROC AUC=%.3f" % (ns_auc))
            print("efficientnet: ROC AUC=%.3f" % (lr_auc))

            ns_fpr, ns_tpr, _ = roc_curve(self.y_test, ns_probs)
            lr_fpr, lr_tpr, _ = roc_curve(self.y_test, y_pred)

            plt.clf()
            plt.plot(ns_fpr, ns_tpr, linestyle="--", label="No Skill")
            plt.plot(lr_fpr, lr_tpr, marker=".", label="efficientnet")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc="lower right")
            save_file = os.path.join(self.exp_dir, "roc_curve.jpeg")
            plt.savefig(save_file, dpi=150)
            if show_plots:
                plt.show()

            self.results.update({"ROC": float(lr_auc)})

        _, train_acc = self.model.evaluate(self.train_ds)
        _, val_acc = self.model.evaluate(self.val_ds)
        _, test_acc = self.model.evaluate(self.test_ds)

        accuracy_verbose = (
            lambda data_set, acc: f"Model accuracy on {data_set} data: {acc * 100}"
        )
        print(accuracy_verbose("train", train_acc))
        print(accuracy_verbose("val", val_acc))
        print(accuracy_verbose("test", test_acc))

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
        y_pred = self.model.predict(self.x_test)

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
