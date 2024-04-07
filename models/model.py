"""
model.py
"""

import datetime
import os

import pandas as pd
import tensorflow as tf
from tensorflow import keras


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
