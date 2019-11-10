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
from moonshot.baselines import model_utils


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


class BaseModel:
    """Base model with functions for network training and inference."""

    def __init__(self, base_network, loss, mc_dropout=False, stochastic_steps=100):
        self.model = base_network
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
                mc_prediction.append(self.model.predict(inputs, training=True))
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


class WeaklySupervisedModel:
    """Weakly supervised model with two branches for learning from co-occuring paired inputs."""

    def __init__(self, speech_network, vision_network, loss, mc_dropout=False,
                 stochastic_steps=100):

        self.speech_model = BaseModel(
            speech_network, None, mc_dropout=mc_dropout,
            stochastic_steps=stochastic_steps)

        self.vision_model = BaseModel(
            vision_network, None, mc_dropout=mc_dropout,
            stochastic_steps=stochastic_steps)

        self.loss = loss

    # @tf.function(experimental_relax_shapes=True)
    def train_step(self, x_speech, x_image, optimizer, training=True,
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
            train_tape.watch(self.speech_model.model.trainable_variables)
            train_tape.watch(self.vision_model.model.trainable_variables)
            # compute transformations for `x_speech` and `x_image` and evaluate objective
            y_speech = self.speech_model.predict(x_speech, training=training)
            y_image = self.vision_model.predict(x_image, training=training)
            loss_value = self.loss(y_speech, y_image)

        network_s_variables = self.speech_model.model.trainable_variables
        network_i_variables = self.vision_model.model.trainable_variables

        train_gradients_s, train_gradients_i = train_tape.gradient(
            loss_value, [network_s_variables, network_i_variables])

        if "debug" in FLAGS and FLAGS.debug:  # TODO: flatten grads first
            for grad in train_gradients_s + train_gradients_i:
                if tf.math.count_nonzero(tf.math.is_nan(grad)) >= 1:
                    tf.print("NaN grad encountered:", grad)
                    tf.print("Loss:", loss_value)
                    tf.print("Speech network output:", y_speech)
                    tf.print("Image network output:", y_image)

        # stop gradients if specified
        if stop_gradients:
            train_gradients_s = [
                tf.stop_gradient(grad) for grad in train_gradients_s]
            train_gradients_i = [
                tf.stop_gradient(grad) for grad in train_gradients_i]

        # clip gradients by global norm if specified
        if clip_norm is not None:
            train_gradients_s, global_norm = tf.clip_by_global_norm(
                train_gradients_s, clip_norm)

            train_gradients_i, global_norm = tf.clip_by_global_norm(
                train_gradients_i, clip_norm)

            # debugging in eager mode
            if "debug" in FLAGS and FLAGS.debug and global_norm > clip_norm:
                tf.print(
                    "Clipping gradients with global norm", global_norm, "to",
                    "clip norm", clip_norm)

        if isinstance(optimizer, tf.keras.optimizers.Optimizer):
            optimizer.apply_gradients(
                zip(train_gradients_s + train_gradients_i,
                    network_s_variables + network_i_variables))

        elif callable(optimizer):
            optimizer(self.speech_model.model, self.vision_model.model,
                      train_gradients_s, train_gradients_i)

        else:
            raise ValueError(
                "Argument `optimizer_a` should be a tf.keras optimizer or a "
                "callable that takes arguments "
                "(network_a, network_b, gradients_a, gradients_b).")

        return loss_value, y_speech, y_image

    @tf.function(experimental_relax_shapes=True)
    def train_steps(self, x_speech, x_image, num_steps, **train_step_kwargs):
        """Train model for `num_steps` gradient steps (see `train_step`)."""
        for _ in range(num_steps):
            loss_value, y_speech, y_image = self.train_step(
                x_speech, x_image, **train_step_kwargs)

        return loss_value, y_speech, y_image


def multimodal_gradient_descent_optimizer(
        lr, speech_weights_structure, vision_weights_structure):
    """Create an optimizer that applies the basic gradient descent algorithm.

    `lr` is the gradient descent learning rate hyperparameter.
    """

    def optimizer(speech_network, vision_network, speech_gradients, vision_gradients):
        """Apply gradient descent to model weights with corresponding grads."""
        speech_weight_updates = tf.nest.map_structure(
            lambda grad, var: (var - lr*grad), speech_gradients,
            speech_network.trainable_variables)

        vision_weight_updates = tf.nest.map_structure(
            lambda grad, var: (var - lr*grad), vision_gradients,
            vision_network.trainable_variables)

        model_utils.update_model_weights(
            speech_network, speech_weight_updates, speech_weights_structure)

        model_utils.update_model_weights(
            vision_network, vision_weight_updates, vision_weights_structure)

    return optimizer


class WeaklySupervisedMAML(WeaklySupervisedModel):

    def __init__(self, speech_network, vision_network, loss,
                 inner_optimizer_lr=1e-2, mc_dropout=False, stochastic_steps=100):
        super().__init__(
            speech_network, vision_network, loss, mc_dropout=mc_dropout,
            stochastic_steps=stochastic_steps)

        speech_network_clone = tf.keras.models.clone_model(speech_network)
        vision_network_clone = tf.keras.models.clone_model(vision_network)

        self.adapt_model = WeaklySupervisedModel(
            speech_network_clone, vision_network_clone, loss,
            mc_dropout=mc_dropout, stochastic_steps=stochastic_steps)

        self.speech_weights_structure = model_utils.get_model_weights_structure(
            speech_network)
        self.vision_weights_structure = model_utils.get_model_weights_structure(
            vision_network)

        self.inner_optimizer = multimodal_gradient_descent_optimizer(
            inner_optimizer_lr, self.speech_weights_structure,
            self.vision_weights_structure)


    # make our MAML fast with tf.function autograph!
    @tf.function(experimental_relax_shapes=True)
    def maml_train_step(
            self, x_speech_train, x_image_train, x_speech_test, x_image_test,
            num_steps, meta_optimizer, training=True,
            stop_gradients=False, clip_norm=None):

        meta_batch_size = tf.shape(x_speech_train)[0]

        with tf.GradientTape() as meta_tape:

            # watch vars in case of tf.Tensor's which are not tracked by default
            meta_tape.watch(self.speech_model.model.trainable_variables)
            meta_tape.watch(self.vision_model.model.trainable_variables)

            # use tf.TensorArray to accumulate results in dynamically unrolled loop
            inner_losses = tf.TensorArray(tf.float32, size=meta_batch_size)
            meta_losses = tf.TensorArray(tf.float32, size=meta_batch_size)

            # train and evaluate meta-objective on each task in the batch
            for batch_index in tf.range(meta_batch_size):
                x_s_1 = x_speech_train[batch_index]
                x_i_1 = x_image_train[batch_index]
                x_s_2 = x_speech_test[batch_index]
                x_i_2 = x_image_test[batch_index]

                # accumulate train and test losses per update for each task
                train_losses = tf.TensorArray(tf.float32, size=num_steps)
                test_losses = tf.TensorArray(tf.float32, size=num_steps)

                # initial "weight update" with current model weights
                speech_weight_updates = self.speech_model.model.trainable_weights
                vision_weight_updates = self.vision_model.model.trainable_weights

                # # create a model copy starting with the exact weight variables from
                # # the base model so we can update the model on the current task and
                # # then take gradients w.r.t. the base weights on the meta-objective
                # # NOTE: not using variable assign which has no grad ... solutions?
                # self.adapt_model.speech_model.model = self.clone_speech_network_func(
                #     self.speech_model.model)

                # self.adapt_model.vision_model.model = self.clone_vision_network_func(
                #     self.vision_model.model)

                for update_step in tf.range(num_steps):
                    # make sure model has previous updates (python state issue .. ?)
                    model_utils.update_model_weights(
                        self.adapt_model.speech_model.model,
                        speech_weight_updates, self.speech_weights_structure)

                    model_utils.update_model_weights(
                        self.adapt_model.vision_model.model,
                        vision_weight_updates, self.vision_weights_structure)

                    # update model on task training samples
                    inner_task_loss, y_s_1, y_i_1 = self.adapt_model.train_step(
                        x_s_1, x_i_1, optimizer=self.inner_optimizer,
                        training=training, stop_gradients=stop_gradients,
                        clip_norm=clip_norm)

                    # compute transformations for `x_speech` and `x_image` and
                    # evaluate meta-objective of updated model on task test samples
                    y_s_2 = self.adapt_model.speech_model.predict(x_s_2, training=training)
                    y_i_2 = self.adapt_model.vision_model.predict(x_i_2, training=training)
                    meta_task_loss = self.loss(y_s_2, y_i_2)

                    train_losses = train_losses.write(update_step, inner_task_loss)
                    test_losses = test_losses.write(update_step, meta_task_loss)

                inner_losses = inner_losses.write(batch_index, train_losses.stack())
                meta_losses = meta_losses.write(batch_index, test_losses.stack())

            # get stacked tensors from the array
            inner_losses = inner_losses.stack()
            meta_losses = meta_losses.stack()

            # average across task meta-objectives (at the final updates)
            meta_loss = tf.reduce_mean(tf.stack(meta_losses)[:, -1])

        # compute gradient of meta-objective and update MAML model
        network_s_variables = self.speech_model.model.trainable_variables
        network_i_variables = self.vision_model.model.trainable_variables

        meta_gradients_s, meta_gradients_i = meta_tape.gradient(
            meta_loss, [network_s_variables, network_i_variables])

        if "debug" in FLAGS and FLAGS.debug:
            for grad in meta_gradients_s + meta_gradients_i:
                if tf.math.count_nonzero(tf.math.is_nan(grad)) >= 1:
                    tf.print("NaN grad encountered:", grad)
                    tf.print("Loss:", meta_loss)

        # clip gradients by global norm if specified
        if clip_norm is not None:
            meta_gradients_s, global_norm = tf.clip_by_global_norm(
                meta_gradients_s, clip_norm)

            meta_gradients_i, global_norm = tf.clip_by_global_norm(
                meta_gradients_i, clip_norm)

            # debugging in eager mode
            if "debug" in FLAGS and FLAGS.debug and global_norm > clip_norm:
                tf.print(
                    "Clipping gradients with global norm", global_norm, "to",
                    "clip norm", clip_norm)

        if isinstance(meta_optimizer, tf.keras.optimizers.Optimizer):
            meta_optimizer.apply_gradients(
                zip(meta_gradients_s + meta_gradients_i,
                    network_s_variables + network_i_variables))

        elif callable(meta_optimizer):
            meta_optimizer(self.speech_model.model, self.vision_model.model,
                           meta_gradients_s, meta_gradients_i)

        else:
            raise ValueError(
                "Argument `meta_optimizer` should be a tf.keras optimizer or a "
                "callable that takes arguments "
                "(network_a, network_b, gradients_a, gradients_b).")

        return meta_loss, inner_losses, meta_losses



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
