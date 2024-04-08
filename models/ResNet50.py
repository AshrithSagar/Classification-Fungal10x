"""
ResNet50 model
"""

import os
import sys

import tensorflow as tf
from tensorflow import keras

sys.path.append(os.getcwd())
from models.model_utils import freeze_layers


def model(args, params):
    base_model = tf.keras.applications.resnet50.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling="max",
    )

    base_model = freeze_layers(base_model, params["freeze_ratio"])

    model = keras.models.Sequential()
    model.add(keras.layers.Rescaling(1.0 / 255))
    model.add(base_model)
    model.add(keras.layers.Dense(1024, activation="relu"))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1024, activation="relu"))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.optimizers.Adam(learning_rate=float(params["learning_rate"])),
        metrics=["accuracy"],
    )

    model.build(input_shape=(None, 224, 224, 3))

    return model


def model_callbacks(args, params):
    return [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            verbose=1,
            factor=0.2,
            patience=5,
            mode="min",
            min_lr=1e-8,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            verbose=1,
            patience=params["patience"],
        ),
    ]
