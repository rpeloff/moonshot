"""TODO(rpeloff)

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: September 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import functools


from absl import flags
from absl import logging


import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cdist


from moonshot.baselines import fast_dtw


FLAGS = flags.FLAGS


def get_metric(metric="cosine", dtw=False):
    """Get distance metric function."""

    if callable(metric):
        dist_func = metric
    elif dtw:  # compute pair distance using dynamic time warping and specified metric
        dist_func = functools.partial(fast_dtw.dtw_cdist, metric=metric)
    else:  # compute pair distances using scipy and specified metric
        dist_func = functools.partial(cdist, metric=metric)

    return dist_func


def majority_vote(labels):
    """Get majority label among `labels` (random between a tied majority)."""
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


def knn(x_query, x_memory, y_memory=None, k_neighbours=1, metric="cosine",
        dtw=False):
    """Compute k-Nearest Neighbours of `x_query` in `x_memory`.

    If `y_memory` is not specified, index of the nearest template in `x_memory`
    is returned.
    """

    dist_func = get_metric(metric=metric, dtw=dtw)

    distances = dist_func(x_query, x_memory)

    k_nn_idx = np.argsort(distances, axis=-1)[:, :k_neighbours]

    if y_memory is None:
        k_nn_labels = np.arange(len(x_memory))[k_nn_idx]
    else:
        k_nn_labels = y_memory[k_nn_idx]

    predictions = np.apply_along_axis(majority_vote, 1, k_nn_labels)

    return predictions


def gradient_descent_optimizer(lr, apply_updates_func=None):
    """Create an optimizer that applies the basic gradient descent algorithm.

    `lr` is the gradient descent learning rate hyperparameter.

    `apply_updates_func` is a function that should replace current model weights
    with the gradient descent updated weights (default uses `variable.assign`).
    """

    def optimizer(model, gradients):
        """Apply gradient descent to model weights with corresponding grads."""
        updates = tf.nest.map_structure(lambda grad, var: (var - lr*grad),
                                        gradients, model.trainable_variables)

        if apply_updates_func is None:
            for variable, update in zip(model.trainable_variables, updates):
                variable.assign(update)
        else:
            apply_updates_func(model, updates)

    return optimizer


class FewShotModel:
    """Few-shot model class with functions for model training and prediction."""

    def __init__(self, model, loss, mc_dropout=False, stochastic_steps=100):
        self.model = model
        self.loss = loss

        self.mc_dropout = mc_dropout
        self.stochastic_steps = stochastic_steps
        self._predictive_variance = None

    def predict(self, inputs, training=False):
        """Apply model on inputs and return outputs."""

        # compute Monte-Carlo dropout prediction
        if not training and self.mc_dropout:

            mc_prediction = []
            for _ in range(self.stochastic_steps):
                mc_prediction.append(self.model(inputs, training=True))
            mc_prediction = tf.convert_to_tensor(mc_prediction)

            predictive_mean, self._predictive_variance = tf.nn.moments(
                mc_prediction, axes=[0])

            return predictive_mean

        # otherwise, simply compute model prediction
        else:
            return self.model(inputs, training=training)

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, x, y, optimizer=None, training=True,
                   stop_gradients=False, clip_norm=None):
        """Train model for one gradient step on data.

        `optimizer` should be one of the tf.keras.optimizers (e.g. Adam) or
        a callable that takes arguments (model, gradients).

        Set `training=False` to update parameters in inference mode. Useful
        when fine-tuning weights using moving statistics learned during training
        or MC Dropout (i.e. model averaging) predictions in the objective.

        Set `stop_gradients=True` to prevent gradients from backpropagating
        through the training step gradients. Useful when this is being computed
        as part of an inner-optimization of a meta-objective and we would like
        to use a first order approximation (i.e. ommitting second derivatives).
        """

        with tf.GradientTape() as train_tape:
            # watch vars in case of tf.Tensor's which are not tracked by default
            train_tape.watch(self.model.trainable_variables)
            # compute prediction for `x` and evaluate objective
            y_predict = self.predict(x, training=training)
            loss_value = self.loss(y, y_predict)

        train_gradients = train_tape.gradient(
            loss_value, self.model.trainable_variables)

        if "debug" in FLAGS and FLAGS.debug:
            for grad in train_gradients:
                if tf.math.count_nonzero(tf.math.is_nan(grad)) >= 1:
                    tf.print("NaN grad encountered:", grad)
                    tf.print("Loss:", loss_value)
                    tf.print("Predictions:", y_predict)

        # stop gradients through `train_gradients` if specified
        if stop_gradients:
            train_gradients = [
                tf.stop_gradient(grad) for grad in train_gradients]

        # clip gradients by global norm if specified
        if clip_norm is not None:
            train_gradients, global_norm = tf.clip_by_global_norm(
                train_gradients, clip_norm)

            # debugging in eager mode
            if "debug" in FLAGS and FLAGS.debug and global_norm > clip_norm:
                tf.print(
                    "Clipping gradients with global norm", global_norm, "to",
                    "clip norm", clip_norm)

        if isinstance(optimizer, tf.keras.optimizers.Optimizer):
            optimizer.apply_gradients(
                zip(train_gradients, self.model.trainable_variables))
        elif callable(optimizer):
            optimizer(self.model, train_gradients)
        else:
            raise ValueError(
                "Argument `optimizer` should be a tf.keras optimizer or a "
                "callable that takes arguments (model, gradients).")

        return loss_value, y_predict

    @tf.function(experimental_relax_shapes=True)
    def train_steps(self, x, y, num_steps, **train_step_kwargs):
        """Train model for `num_steps` gradient steps (see `train_step`)."""
        for _ in range(num_steps):
            loss_value, y_predict = self.train_step(
                x, y, **train_step_kwargs)

        return loss_value, y_predict

    @property
    def predictive_variance(self):
        """Predictive variance of the last Monte-Carlo dropout prediction."""
        if not self.mc_dropout:
            raise NotImplementedError(
                "Model with `mc_dropout=False` has no predictive variance.")
        return self._predictive_variance

    @property
    def weights(self):
        return self.model.get_weights()

    @weights.setter
    def weights(self, weights):
        self.model.set_weights(weights)


# class Model(abc.ABC):
#     """TODO(rpeloff).

#     training
#     - initialise model (random or pretrained weights)
#     - train/fine-tune model on some external background data (disjoint from one-shot!)

#     one-shot evaluation (or training/validation)
#     - initialise trained model and make a deep copy for state reset
#     - receive episode support set for one-shot learning
#     - receive episode evaluation task
#     - emit model predictions/actions
#     - receive results (may be used for training or validation)
#     - reset model to initial state
#     - repeat for next episode
#     - analyse model at the end of the experiment (or store trained model)
#     """

#     def __init__(self):
#         pass

#     @abc.abstractmethod
#     def train(self):
#         """Train and validate model on background data."""

#     def adapt_model(self, *learning_task, copy_model=True, **kwargs):
#         """Adapt model on samples from a few-shot learning task."""
#         model = self
#         if copy_model:
#             model = copy.deepcopy(model)
#         model._adapt_model(*learning_task, **kwargs)
#         return model

#     @abc.abstractmethod
#     def _adapt_model(self, *learning_task, **kwargs):
#         """Adapt model on samples from a few-shot learning task."""

#     @abc.abstractmethod
#     def predict(self, *evaluation_task, **kwargs):
#         """Make predictions on samples from a few-shot learning task."""
