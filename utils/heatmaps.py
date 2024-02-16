"""
heatmaps.py
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.colors import Normalize
from PIL import Image
from scipy.stats import percentileofscore
from tensorflow import keras
from tqdm import tqdm


class Heatmaps:
    def __init__(self, exp_dir):
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
        tf.keras.backend.clear_session()
        self.exp_dir = exp_dir

    def make_tilemap(
        self,
        predictions,
        tilemap_shape,
        patch_size=(224, 224, 3),
        cmap="coolwarm",
        overlap=0.5,
        percentile_scale=None,
        percentile_score=False,
    ):
        """
        Generate a tilemap from model predictions.

        Parameters:
        - predictions (numpy.ndarray): Model predictions for patches.
        - tilemap_shape (tuple): Shape of the tilemap (rows, cols).
        - patch_size (tuple): Size of each patch (height, width, channels).
        - cmap (str): Colormap for coloring the tiles.
        - overlap (float): Overlap factor.

        Returns:
        - tilemap (numpy.ndarray): Image with overlaid colored tiles.
        """
        num_patches = len(predictions)
        cmap = plt.cm.get_cmap(cmap)
        norm = Normalize(vmin=0, vmax=1)
        tilemap_size = (
            tilemap_shape[0] * patch_size[0]
            - int(overlap * (tilemap_shape[0] - 1) * patch_size[0]),
            tilemap_shape[1] * patch_size[1]
            - int(overlap * (tilemap_shape[1] - 1) * patch_size[1]),
            patch_size[2],
        )
        tilemap = np.zeros(tilemap_size, dtype=np.float32)
        tilemap_counter = np.zeros(tilemap_size[:2], dtype=np.int32)
        stride = self.calculate_stride(patch_size[:2], overlap)

        if percentile_scale is not None:
            # Apply percentile scaling to predictions
            min_percentile, max_percentile = percentile_scale
            min_value = np.percentile(predictions, min_percentile)
            max_value = np.percentile(predictions, max_percentile)
            predictions = np.clip(predictions, min_value, max_value)
            predictions = (predictions - min_value) / (max_value - min_value)

        if percentile_score:
            percentiles = []
            for score in predictions:
                percentile = percentileofscore(predictions, score)
                percentiles.append(percentile / 100)
            predictions = percentiles

        for i, pred in enumerate(predictions):
            row, col = divmod(i, tilemap_shape[1])
            top_left_row = row * stride[0]
            top_left_col = col * stride[1]

            pred = norm(pred)
            color = np.squeeze(cmap(pred))[:-1]  # Exclude alpha channel

            # Normalize the prediction value to the range [0, 1]
            # normalized_pred = (pred - 0.5) * 2  # Map [0.5, 1] to [0, 1]
            # normalized_pred = np.clip(normalized_pred, 0, 1)  # Clip to [0, 1]

            # Create a red color with intensity based on the normalized prediction value
            # color = np.array([normalized_pred, 0, 0])

            tilemap[
                top_left_row : top_left_row + patch_size[0],
                top_left_col : top_left_col + patch_size[1],
                :,
            ] += color
            tilemap_counter[
                top_left_row : top_left_row + patch_size[0],
                top_left_col : top_left_col + patch_size[1],
            ] += 1

        # Normalize by the number of patches contributing to each region
        tilemap /= np.maximum(tilemap_counter, 1)[:, :, np.newaxis]
        return tilemap

    def superimpose(self, background, overlay, alpha=0.4, blur=None):
        """
        Superimpose an overlay image onto a background image.

        Parameters:
        - background (numpy.ndarray): Background image.
        - overlay (numpy.ndarray): Overlay image.
        - alpha (float): Transparency factor for the overlay (0.0 to 1.0).

        Returns:
        - superimposed_image (numpy.ndarray): Resulting superimposed image.
        """
        background = background[: overlay.shape[0], : overlay.shape[1], :]

        if background.shape != overlay.shape:
            raise ValueError("Background and overlay images must have the same shape.")

        if blur:
            overlay = cv2.blur(overlay, blur)

        # Convert images to PIL format
        # background_pil = Image.fromarray((background * 255).astype(np.uint8))
        background_pil = Image.fromarray(background.numpy().astype(np.uint8))
        overlay_pil = Image.fromarray((overlay * 255).astype(np.uint8))

        superimposed_pil = Image.blend(background_pil, overlay_pil, alpha)
        superimposed_image = np.array(superimposed_pil) / 255.0

        return superimposed_image

    def save(
        self,
        save_dir,
        slides_annot,
        slide_names,
        predictions,
        slide_labels,
        patches_shape,
        patch_size=(224, 224, 3),
        cmap="coolwarm",
        overlap=0.5,
        percentile_scale=None,
        percentile_score=False,
        alpha=0.4,
        blur=(112, 112),
    ):
        self.save_dir = save_dir
        print(f"Saving in {self.save_dir}")

        for slide, slide_name, preds, label in tqdm(
            zip(slides_annot, slide_names, predictions, slide_labels)
        ):
            cmap_labels = [(1 - pred) for pred in preds]

            tilemap = self.make_tilemap(
                cmap_labels,
                patches_shape,
                patch_size=patch_size,
                cmap=cmap,
                overlap=overlap,
                percentile_scale=percentile_scale,
                percentile_score=percentile_score,
            )
            heatmap = self.superimpose(slide, tilemap, alpha=alpha, blur=blur)

            plt.clf()
            plt.imshow(heatmap)

            label_ch = "F" if label == 0 else "N"
            filename = f"{label_ch}_{slide_name}"
            heatmap_path = os.path.join(self.save_dir, filename)
            plt.savefig(heatmap_path)
