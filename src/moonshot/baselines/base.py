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
import os


from absl import flags
from absl import logging


import numpy as np
import tensorflow as tf


from moonshot.utils import file_io

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


def save_model(model, output_dir, epoch, step, metric, current_score,
               best_score, name="model"):
    assert hasattr(model, "loss") and model.loss is not None
    assert hasattr(model, "optimizer") and model.optimizer is not None

    model.save(os.path.join(output_dir, f"{name}.h5"))

    file_io.write_csv(
        os.path.join(output_dir, f"{name}.step"),
        [epoch, step, metric, current_score, best_score])


def load_model(model_file, model_step_file, loss):
    logging.log(logging.INFO, f"Loading model: {model_file}")

    model = tf.keras.models.load_model(
        model_file, custom_objects={"loss": loss})

    model_epochs, global_step, metric, val_score, best_score = file_io.read_csv(
        model_step_file)[0]

    model_epochs = int(model_epochs)
    global_step = int(global_step)
    val_score = float(val_score)
    best_score = float(best_score)

    logging.log(
        logging.INFO,
        f"Model trained for {model_epochs} epochs ({global_step} steps)")
    logging.log(
        logging.INFO,
        f"Validation: current {metric}: {val_score:.5f}, previous best "
        f"{metric}: {best_score:.5f}")

    return model, (global_step, model_epochs, val_score, best_score)


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
