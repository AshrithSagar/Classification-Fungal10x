"""
dataset.py
"""

class FungalDataLoader:
    def __init__(self,
     data_dir_name,
     slide_dir,
     annot_dir,
     test_split = 0.2,
     random_state=42
    ):
        self.slide_dims = (1200, 1600)

    def downsample(self, size=None, factor=None, preserve_aspect_ratio=None):
        if factor:
            downsample_size = tuple([int(x/factor) for x in self.slide_dims])
        else:
            downsample_size = size

        print(f"Downsample size: {downsample_size}")
        downsampled_slides = tf.image.resize(
            self.slides,
            downsample_size,
            preserve_aspect_ratio=preserve_aspect_ratio,
        )
        
        return downsampled_slides