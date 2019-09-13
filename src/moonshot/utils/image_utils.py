"""Utility functions for manipulating image data.

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: July 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
from skimage.io import imread


def load_image_array(image_path):
    """Read image from file to ndarray."""
    return np.asarray(imread(image_path))


# TODO(rpeloff) old code, remove if sure not using this
# def resize_square_crop(image_arr, size=(224, 224), resample=Image.LANCZOS):
#     h, w, _ = image_arr.shape
#     short_edge = min(w, h)
#     h_shift = int((h - short_edge) / 2)
#     w_shift = int((w - short_edge) / 2)
#     image_resize = Image.fromarray(image_arr).resize(
#         size, box=(w_shift, h_shift, w - w_shift, h - h_shift), resample=resample)
#     return np.asarray(image_resize)
