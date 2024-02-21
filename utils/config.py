"""
config.py
"""

import os
import yaml
import tensorflow as tf

line_separator = "\u2500" * 50


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
        def set_device(device_name):
            tf.config.set_visible_devices([], "GPU")
            with tf.device(device_name):
                print(f"Selecting device: {device_name}")
                tf.constant(0)

        if device_index == -1:  # Use CPU
            set_device("/cpu:0")
        elif device_index >= 0:  # Use GPU
            set_device(f"/gpu:{device_index}")
        else:
            print("Invalid device index provided.")
        print(line_separator)
