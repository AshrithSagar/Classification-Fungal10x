"""
EfficientNetB0 model
"""

import datetime
import os
import sys

import pandas as pd
import tensorflow as tf
from tensorflow import keras

sys.path.append(os.getcwd())
from models.model import freeze_layers


def get_EfficientNetB0(args):
    model_checkpoint_path = os.path.join(
        args["model_args"]["exp_dir"], "efficientnet-finetune-model.h5"
    )
    csv_logger_path = os.path.join(args["model_args"]["exp_dir"], "csv_logger.csv")
    log_dir = os.path.join(
        args["model_args"]["exp_dir"],
        "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )

    base_model = keras.applications.EfficientNetB0(
        input_shape=(224, 224, 3), include_top=True, weights="imagenet"
    )

    base_model = freeze_layers(base_model, args["model_params"]["freeze_ratio"])

    model = keras.models.Sequential()
    model.add(keras.layers.Rescaling(1.0 / 255))
    model.add(base_model)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(1024, activation="relu"))
    model.add(keras.layers.Dense(512, activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.optimizers.Adam(
            learning_rate=float(args["model_params"]["learning_rate"])
        ),
        metrics=["accuracy"],
    )

    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath=model_checkpoint_path,
            verbose=1,
            monitor="val_loss",
            save_best_only=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            verbose=1,
            factor=0.2,
            patience=5,
            mode="min",
            min_lr=1e-8,
        ),
        keras.callbacks.CSVLogger(
            filename=csv_logger_path,
            separator=",",
            append=True,
        ),
        keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        keras.callbacks.EarlyStopping(
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
