"""
config.py
"""

import os
import yaml
import tensorflow as tf


def load_config(config_file="config.yaml", key=None):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    if key:
        return config.get(key, {})
    else:
        return config


class GPUHandler:
    def __init__(self):
        pass

    def check(self):
        print("GPU:", "Enabled" if tf.test.gpu_device_name() else "Disabled")
        print(tf.config.list_physical_devices())

    def set(self, device_index):
        physical_devices = tf.config.list_physical_devices("GPU")
        if physical_devices:
            tf.config.set_visible_devices(physical_devices[device_index], "GPU")
            tf.config.experimental.set_memory_growth(
                physical_devices[device_index], True
            )
            print(f"Selecting GPU: {physical_devices[device_index]}")
        else:
            print("No GPU devices found.")
