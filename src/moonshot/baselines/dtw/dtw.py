"""TODO(rpeloff)

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: September 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import functools


from absl import logging


import numpy as np
from scipy.interpolate import interp2d
from scipy.spatial.distance import cdist


from moonshot.baselines import base
from moonshot.baselines.dtw import fast_dtw


class DynamicTimeWarping(base.Model):
    """TODO(rpeloff).
    """

    def __init__(self, metric="cosine", preprocess=None):
        super(DynamicTimeWarping, self).__init__()

        self.metric = metric

        if callable(metric):
            self.dist_func = metric
        else:  # compute pair distance using dynamic time warping and specified metric
            self.dist_func = functools.partial(fast_dtw.dtw_cdist, metric=self.metric)

        self.preprocess = preprocess  # remove ?

        self.memory = None

    def train(self):
        """Train and validate model on background data."""
        pass  # no prior training

    def _adapt_model(self, train_x, train_y):
        """Adapt model on samples from a few-shot learning task."""
        self.memory = (train_x, train_y)

    def predict(self, test_x, k_neighbours=1):
        """Make predictions on samples from a few-shot learning task."""
        if self.memory is None:
            raise IndexError("Call adapt_model(...) first!")

        distances = self.dist_func(test_x, self.memory[0])

        k_nn_idx = np.argsort(distances, axis=-1)[:, :k_neighbours]
        k_nn_labels = self.memory[1][k_nn_idx]

        predictions = np.apply_along_axis(base.majority_vote, 1, k_nn_labels)

        return predictions


def dtw_reinterp2d(x, interp_length, interp="quintic"):
    """Reinterpolate a sequence of N-dimensional vectors to a fixed length.
    
    Parameter `interp_kind` is one of "linear", "cubic" or "quintic".

    See here for more information on the interpolation:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp2d.html#scipy.interpolate.interp2d
    
    Motivation for same length sequences via reinterpolation when using DTW:
    http://www.cs.ucr.edu/~eamonn/DTW_myths.pdf and
    https://www.cs.unm.edu/~mueen/DTW.pdf

    NOTE:
    For TIDIGITS train data `interp="linear"` is the only valid option with the
    minimum required sequence length (1+1)**2 = 4.
    """
    n_frames, n_feats = np.shape(x)
    # axes for frames (i.e. vectors) in the original and interpolated sequences
    frame_orig_axis = np.arange(n_frames)
    frame_interp_axis = np.linspace(0, n_frames, interp_length, endpoint=False)
    # axis for features dimension remains unchanged (i.e. not interpolated)
    feature_axis = np.arange(n_feats)
    # approximate x = f(features, frames) and apply to interpolation axis
    try:
        f_interp = interp2d(feature_axis, frame_orig_axis, x, kind=interp)
        x_interp = f_interp(feature_axis, frame_interp_axis)
    except Exception as e:
        x_interp = None  # return None for sequence that cannot be interpolated
        logging.log(
            logging.INFO,
            "Caught exception interpolating sequence with shape {}: {}".format(
                np.shape(x), e))
    finally:
        return x_interp
