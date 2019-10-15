"""TODO(rpeloff)

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: September 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import abc
import copy


from absl import flags
from absl import logging


import numpy as np


FLAGS = flags.FLAGS


def majority_vote(labels):
    """Get majority label (random between a tied majority)."""
    counts = np.bincount(labels)
    max_idx = np.where(counts == np.max(counts))[0]
    if len(max_idx) > 1:  # choose random from tied majority labels
        if "debug" in FLAGS and FLAGS.debug:
            logging.log(
                logging.DEBUG,
                "Choosing randomly from tied labels: {}".format(np.asarray(labels[max_idx])))
        majority_label = max_idx[
            np.random.choice(len(max_idx), size=1, replace=False)[0]]
    else:
        majority_label = max_idx[0]
    return majority_label


class Model(abc.ABC):
    """TODO(rpeloff).

    training
    - initialise model (random or pretrained weights)
    - train/fine-tune model on some external background data (disjoint from one-shot!)

    one-shot evaluation (or training/validation)
    - initialise trained model and make a deep copy for state reset
    - receive episode support set for one-shot learning
    - receive episode evaluation task
    - emit model predictions/actions
    - receive results (may be used for training or validation)
    - reset model to initial state
    - repeat for next episode
    - analyse model at the end of the experiment (or store trained model)
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def train(self):
        """Train and validate model on background data."""

    def adapt_model(self, *learning_task, copy_model=True, **kwargs):
        """Adapt model on samples from a few-shot learning task."""
        model = self
        if copy_model:
            model = copy.deepcopy(model)
        model._adapt_model(*learning_task, **kwargs)
        return model

    @abc.abstractmethod
    def _adapt_model(self, *learning_task, **kwargs):
        """Adapt model on samples from a few-shot learning task."""

    @abc.abstractmethod
    def predict(self, *evaluation_task, **kwargs):
        """Make predictions on samples from a few-shot learning task."""
