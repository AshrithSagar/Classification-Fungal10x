"""
config.py
"""

import argparse
import os

import tensorflow as tf
import yaml

line_separator = "\u2500" * 50


class Config:
    def __init__(self, file="config.yaml", verbose=True):
        if not os.path.exists(file):
            raise FileNotFoundError(f"Configuration file not found: {file}")
        self.file = file
        self.config = self.load()
        if verbose:
            print(f"Configuration loaded from {self.file}")

    def __getitem__(self, key):
        return self.config[key]

    def load(self):
        with open(self.file, "r") as f:
            if self.file.endswith(".yaml") or self.file.endswith(".yml"):
                config = yaml.safe_load(f)
            else:
                raise ValueError("Invalid configuration file format")
        return config

    @classmethod
    def from_args(cls, description=None):
        """Create a Config instance from command-line arguments"""
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument(
            "--config",
            metavar="config_file",
            type=str,
            nargs="?",
            default="config.yaml",
            help="Path to the configuration file [.yaml/.yml] (default: config.yaml)",
        )
        args = parser.parse_args()
        config_file = getattr(args, "config", "config.yaml")
        return cls(config_file)


class GPUHandler:
    def __init__(self):
        self.check()

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
