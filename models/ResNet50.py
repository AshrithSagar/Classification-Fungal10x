"""
ResNet50 model
"""

import datetime
import os

import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model


def freeze_layers(model, freeze, verbose=True):
    """
    freeze: Ratio in which to freeze the first few layers
    """
    model.trainable = True

    freeze_upto = round(freeze * len(model.layers))
    train_rest = len(model.layers) - freeze_upto

    for layer in model.layers[:freeze_upto]:
        layer.trainable = False
    for layer in model.layers[train_rest:]:
        layer.trainable = True

    if verbose:
        print(f"{freeze_upto} layers freezed; {train_rest} layers trainable")

    return model


def get_ResNet50(args):
    model_checkpoint_path = os.path.join(
        args["model_args"]["exp_dir"], "resnet50-model.h5"
    )
    csv_logger_path = os.path.join(args["model_args"]["exp_dir"], "csv_logger.csv")
    log_dir = os.path.join(
        args["model_args"]["exp_dir"],
        "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )

    base_model = ResNet50(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3), pooling="max"
    )

    base_model = freeze_layers(base_model, args["model_params"]["freeze_ratio"])

    model = models.Sequential()
    model.add(layers.Rescaling(1.0 / 255))
    model.add(base_model)
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.optimizers.Adam(
            learning_rate=float(args["model_params"]["learning_rate"])
        ),
        metrics=["accuracy"],
    )

    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath=model_checkpoint_path,
            verbose=1,
            monitor="val_loss",
            save_best_only=True,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            verbose=1,
            factor=0.2,
            patience=5,
            mode="min",
            min_lr=1e-8,
        ),
        callbacks.CSVLogger(
            filename=csv_logger_path,
            separator=",",
            append=True,
        ),
        callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            verbose=1,
            patience=args["model_params"]["patience"],
        ),
    ]

    model.build(input_shape=(None, 224, 224, 3))

    # Load the model from checkpoint if available
    epochs_done = 0
    if os.path.exists(model_checkpoint_path):
        print(f"Loading model from checkpoint: {model_checkpoint_path}")
        model.load_weights(model_checkpoint_path)

        if os.path.exists(csv_logger_path):
            csv_data = pd.read_csv(csv_logger_path)
            if not csv_data.empty:
                epochs_done = len(csv_data["epoch"])

    remaining_epochs = max(0, args["model_args"]["max_epochs"] - epochs_done)
    print(f"{epochs_done} Epochs done; Remaining epochs: {remaining_epochs}")

    return model, callbacks_list, epochs_done
