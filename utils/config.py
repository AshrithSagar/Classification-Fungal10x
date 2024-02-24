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
        devices = tf.config.list_physical_devices()
        print("Devices:", devices)

        gpu_devices = tf.config.list_physical_devices("GPU")
        if gpu_devices:
            print("GPU: Enabled;", gpu_devices)
            print("GPU Device Name:", tf.test.gpu_device_name())
        else:
            print("GPU: Disabled;")

    def set(self, device_index):
        if device_index == -1:  # Use CPU
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            tf.config.set_visible_devices([], "GPU")
            tf.device("/cpu:0")
        elif device_index >= 0:  # Use GPU
            physical_devices = tf.config.list_physical_devices("GPU")
            if physical_devices:
                try:
                    tf.config.set_visible_devices(physical_devices[device_index], "GPU")
                    tf.config.experimental.set_memory_growth(
                        physical_devices[device_index], True
                    )
                    print(f"Selecting device: {physical_devices[device_index]}")
                except RuntimeError as e:
                    print(e)
            else:
                print("No GPU devices found.")
        else:
            print("Invalid device index provided.")
        print(line_separator)
