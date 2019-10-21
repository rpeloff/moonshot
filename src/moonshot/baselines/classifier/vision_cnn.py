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


from moonshot.baselines import base


FLAGS = flags.FLAGS


class Conv2DBlock(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, activation, normalization=None,
                 pooling=None, **kwargs):
        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.normalization = normalization
        self.pooling = pooling

        self.conv_layer = tf.keras.layers.Conv2D(
            filters, kernel_size, strides=(1, 1), padding="valid",
            use_bias=True, kernel_initializer="glorot_uniform",
            bias_initializer="zeros")

        self.activation_layer = tf.keras.layers.Activation(activation)

        if normalization == "batch_norm":
            self.bn_layer = tf.keras.layers.BatchNormalization()

        if pooling == "maxpool":
            self.pool_layer = tf.keras.layers.MaxPool2D()

    def call(self, inputs, training=False):
        x = self.conv_layer(inputs)

        if self.normalization == "batch_norm":
            x = self.bn_layer(x, training=training)

        x = self.activation_layer(x)

        if self.pooling == "maxpool":
            x = self.pool_layer(x)

        return x

    def get_config(self):
        """Return config so that layer is serializable."""
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "activation": self.activation,
            "normalization": self.normalization,
            "pooling": self.pooling})
        return config


def small_vision_cnn(input_shape=None, filters=32, kernel_size=(3, 3), blocks=4,
                     batch_norm=False, dropout_rate=None, drop_channels=False):
    input_kwargs = {}
    if input_shape is not None:  # specifying input shape builds layers
        input_kwargs["input_shape"] = input_shape

    conv_block = functools.partial(
        Conv2DBlock, filters, kernel_size, tf.nn.relu,
        normalization="batch_norm" if batch_norm else None, pooling="maxpool")

    layers = []
    for _ in range(blocks):
        layers.append(blocks)

        if dropout_rate is not None:
            noise_shape = (None, 1, 1, None) if drop_channels else None
            layers.append(tf.keras.layers.Dropout(dropout_rate, noise_shape))

    return tf.keras.Sequential([
        conv_block(**input_kwargs), conv_block(), conv_block(), conv_block()])


def load_small_vision_cnn(model_path, custom_objects=None, compile_model=False):
    if custom_objects is None:
        custom_objects = {}
    custom_objects.update({"Conv2DBlock": Conv2DBlock})

    return tf.keras.models.load_model(
        model_path, custom_objects=custom_objects, compile=compile_model)


class FewShotModel:

    def __init__(self, model, loss, mc_dropout=False, stochastic_steps=100):

        self.model = model
        self.loss = loss

        self.mc_dropout = mc_dropout
        self.stochastic_steps = stochastic_steps
        self._predictive_variance = None

    def predict(self, inputs, training=False):
        if not training and self.mc_dropout:
            mc_prediction = []
            for _ in range(self.stochastic_steps):
                mc_prediction.append(self.model(inputs, training=True))
            mc_prediction = tf.convert_to_tensor(mc_prediction)

            predictive_mean, self._predictive_variance = tf.nn.moments(
                mc_prediction, axes=[0])

            return predictive_mean
        else:
            return self.model(inputs, training=training)

    # @tf.function
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

        if "debug" in FLAGS and FLAGS.debug and tf.math.count_nonzero(
                tf.math.is_nan(loss_value)) >= 1:
            import pdb; pdb.set_trace()

        train_gradients = train_tape.gradient(
            loss_value, self.model.trainable_variables)

        # stop gradients through `train_gradients` if specified
        if stop_gradients:
            train_gradients = [
                tf.stop_gradient(grad) for grad in train_gradients]

        # clip gradients by global norm if specified
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

    @property
    def predictive_variance(self):
        if not self.mc_dropout:
            raise NotImplementedError(
                "Model with `mc_dropout=False` has no predictive variance.")
        return self._predictive_variance


def gradient_descent_optimizer(lr, apply_updates_func=None):

    def optimizer(model, gradients):
        updates = tf.nest.map_structure(lambda grad, var: (var - lr*grad),
                                        gradients, model.trainable_variables)

        if apply_updates_func is None:
            for variable, update in zip(model.trainable_variables, updates):
                variable.assign(update)
        else:
            apply_updates_func(model, updates)

    return optimizer


class VisionCNN(base.Model):
    """TODO(rpeloff).
    """

    def __init__(self, metric="cosine", preprocess=None):
        super().__init__()

        self.metric = metric

        if callable(metric):
            self.dist_func = metric
        else:  # compute pair distances using scipy and specified metric
            self.dist_func = functools.partial(cdist, metric=metric)

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
