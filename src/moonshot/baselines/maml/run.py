"""TODO(rpeloff)

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: September 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import copy


from absl import app
from absl import flags
from absl import logging


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


from moonshot.baselines.maml import maml
from moonshot.experiments.sinusoids import sinusoids


FLAGS = flags.FLAGS
flags.DEFINE_integer("episodes", 70000, "number of one-shot learning episodes")
# flags.DEFINE_integer("L", 5, "number of classes to sample in a task episode (L-way)")
flags.DEFINE_integer("K", 20, "number of task learning samples per class (K-shot)")
flags.DEFINE_integer("N", 20, "number of task evaluation samples")


# required flags
# flags.mark_flag_as_required("...")


def main(argv):
    del argv  # unused

    logging.log(logging.INFO, "Logging application {}".format(__file__))

    maml_model = maml.MAML()

    sinusoids_exp = sinusoids.Sinusoids()

    total_correct = 0
    total_tests = 0

    # tensorflow 2.0 implementation of maml based on:
    # https://blog.evjang.com/2019/02/maml-jax.html

    # gradients and maml numerics
    # ===========================

    def grad(func):
        def grad_func(*args, targets=None):
            if targets is None:
                targets = args[0]  # differentiate w.r.t first arg
            with tf.GradientTape() as tape:
                func_value = func(*args)
            grad_value = tape.gradient(func_value, targets)
            return grad_value
        return grad_func

    g = lambda x, y: tf.square(x) + y
    x0 = tf.Variable(2.)
    y0 = tf.Variable(1.)

    dg = grad(g)
    dg_dxy = dg(x0, y0, targets=[x0, y0])

    print("g(x, y) = x*x + y")
    print("grad(g)(x0) = {}".format(dg_dxy[0]))  # 2x = 4
    print("grad(g)(y0) = {}".format(dg_dxy[1]))  # 1
    print("x0 - grad(g)(x0) = {}".format(x0 - dg_dxy[0]))  # x - 2x = -2

    def maml_objective(g, x, y):
        return g(x - grad(g)(x, y), y)

    print("maml_objective(g(x, y), x, y) = {}".format(
        maml_objective(g, x0, y0)))  # (x - 2x)**2 + y = x**2 + y = 5

    print("x0 - grad(maml_objective)(x0) = {}".format(
        x0 - grad(maml_objective)(g, x0, y0, targets=x0)))  # x - (2x) = -2.

    # setup sinusoid problem and develop vanilla Adam SGD baseline
    # ============================================================

    regression_mlp = tf.keras.Sequential([
        tf.keras.layers.Dense(40, activation=tf.nn.relu, input_shape=(1, )),
        tf.keras.layers.Dense(40, activation=tf.nn.relu),
        tf.keras.layers.Dense(1)])

    def mse_loss(model, inputs, targets):
        # compute average mean squared error loss for the inputs batch
        predictions = model(inputs)
        return tf.reduce_mean((targets - predictions)**2)

    x_range_inputs = np.linspace(-5, 5, 100, dtype=np.float32).reshape((100, 1))
    y_targets = np.sin(x_range_inputs)

    def predict_and_plot(fig_name):
        predictions = regression_mlp(x_range_inputs)
        losses = tf.stack(list(map(  # apply loss per-input (no averaging)
            lambda x_input, y_target: mse_loss(
                regression_mlp,
                tf.expand_dims(x_input, -1),
                tf.expand_dims(y_target, -1)),
            x_range_inputs, y_targets)))

        plt.figure()
        plt.plot(x_range_inputs, predictions, label="prediction")
        plt.plot(x_range_inputs, losses, label="loss")
        plt.plot(x_range_inputs, y_targets, label="target")
        plt.legend()
        plt.savefig(fig_name)

    predict_and_plot("sinusoids_untrained.png")

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    def step(model, loss, optimizer, x_inputs, y_targets):
        model_params = model.trainable_variables

        loss_value = loss(model, x_inputs, y_targets)
        loss_grads = grad(loss)(
            model, x_inputs, y_targets, targets=model_params)

        optimizer.apply_gradients(zip(loss_grads, model_params))

        return loss_value

    logging.log(logging.INFO, "Training MLP to regress sinusoids")
    for i in range(100):
        loss_value = step(regression_mlp, mse_loss, optimizer, x_range_inputs, y_targets)

        logging.log(
            logging.INFO,
            "Step: {:03d}, Loss: {:.3f}".format(
                i, loss_value.numpy()))

    predict_and_plot("sinusoids_baseline_trained.png")

    # implement maml (with TensorFlow 2.0)
    # ====================================

    def inner_update(model, loss, x1, y1, alpha=.1):
        params = model.trainable_variables
        loss_grads = grad(loss)(model, x1, y1, targets=params)
        inner_sgd_fn = lambda g, state: (state - alpha*g)
        # apply sgd inner function elementwise grads and params (i.e. to each leaf of the trees)
        return tf.nest.map_structure(inner_sgd_fn, loss_grads, params)

    def maml_loss(model, loss, x1, y1, x2, y2):
        # store copy of params prior to meta-update
        params = copy.deepcopy(model.trainable_variables)

        # update params on task learning samples
        params_updated = inner_update(model, loss, x1, y1)

        # apply meta-optimized params to model
        tf.nest.map_structure(
            lambda variable, update: variable.assign(update),
            model.trainable_variables, params_updated)

        # evaluate loss of meta-updated model on task test samples
        test_loss = loss(model, x2, y2)

        # restore model params prior to meta-update
        tf.nest.map_structure(
            lambda variable, update: variable.assign(update),
            model.trainable_variables, params)

        return test_loss

    def maml_step(model, loss, optimizer, x1, y1, x2, y2):

        loss_value = maml_loss(model, loss, x1, y1, x2, y2)

        model_params = model.trainable_variables
        loss_grads = grad(maml_loss)(
            model, loss, x1, y1, x2, y2, targets=model_params)

        optimizer.apply_gradients(zip(loss_grads, model_params))

        return loss_value

    regression_mlp = tf.keras.Sequential([
        tf.keras.layers.Dense(40, activation=tf.nn.relu, input_shape=(1, )),
        tf.keras.layers.Dense(40, activation=tf.nn.relu),
        tf.keras.layers.Dense(1)])

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    K = 10
    np_maml_loss = []

    logging.log(logging.INFO, "Training MAML to regress sinusoids")
    for i in range(FLAGS.episodes):
        # define the task
        amplitude = np.random.uniform(low=0.1, high=.5)
        phase = np.random.uniform(low=0., high=np.pi)
        # meta-training inner split (K examples)
        x1 = np.random.uniform(low=-5., high=5., size=(K, 1)).astype(np.float32)
        y1 = amplitude * np.sin(x1 + phase)
        # meta-training outer split (1 example). Like cross-validating with respect to one example.
        x2 = np.random.uniform(low=-5., high=5., size=(K, 1)).astype(np.float32)
        y2 = amplitude * np.sin(x2 + phase)

        loss_value = maml_step(
            regression_mlp, mse_loss, optimizer, x1, y1, x2, y2)

        np_maml_loss.append(loss_value)
        if i % 500 == 0:
            logging.log(
                logging.INFO,
                "Step: {:03d}, Loss: {:.3f}".format(
                    i, loss_value.numpy()))

    predict_and_plot("sinusoids_maml_trained.png")

    predictions = regression_mlp(x_range_inputs)

    plt.figure()
    plt.plot(x_range_inputs, predictions, label="pre-update prediction")

    y_targets = 2. * np.sin(x_range_inputs + 0.)
    plt.plot(x_range_inputs, y_targets, label="target")

    # unseen task (quadruple previously seen highest amplitude)
    K = 10

    x1 = np.random.uniform(low=-5., high=5., size=(K, 1)).astype(np.float32)
    y1 = 2. * np.sin(x1 + 0.)

    plt.plot(x1, y1, "bd", label="points for grad update")

    for i in range(1, 11):
        # update params on task learning samples
        params_updated = inner_update(
            regression_mlp, mse_loss, x1, y1)
        # apply meta-update params
        tf.nest.map_structure(
            lambda variable, update: variable.assign(update),
            regression_mlp.trainable_variables, params_updated)

        if i % 5 == 0 or i == 1:
            predictions = regression_mlp(x_range_inputs)
            plt.plot(x_range_inputs, predictions,
                     label="{} grad steps".format(i))

    plt.ylim([-3, 3])
    plt.legend()
    plt.savefig("sinusoids_maml_train_adapted_{}-shot.png".format(K))

    # implement batch maml
    # ======================
    def batch_maml_loss(model, loss, x1_b, y1_b, x2_b, y2_b):

        n_tasks = x1_b.shape[0]
        task_losses = []
        for i in range(n_tasks):
            x1, y1, x2, y2 = x1_b[i], y1_b[i], x2_b[i], y2_b[i]
            task_loss = maml_loss(model, loss, x1, y1, x2, y2)

            task_losses.append(task_loss)

        return tf.reduce_mean(task_losses)

    def batch_maml_step(model, loss, optimizer, x1, y1, x2, y2):

        loss_value = batch_maml_loss(model, loss, x1, y1, x2, y2)

        model_params = model.trainable_variables
        loss_grads = grad(batch_maml_loss)(
            model, loss, x1, y1, x2, y2, targets=model_params)

        optimizer.apply_gradients(zip(loss_grads, model_params))

        return loss_value

    regression_mlp = tf.keras.Sequential([
        tf.keras.layers.Dense(40, activation=tf.nn.relu, input_shape=(1, )),
        tf.keras.layers.Dense(40, activation=tf.nn.relu),
        tf.keras.layers.Dense(1)])

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    K = 10
    np_batch_maml_loss = []

    logging.log(logging.INFO, "Training MAML (with batch tasks) to regress sinusoids")
    for i in range(FLAGS.episodes):

        train_batch, test_batch = sinusoids_exp.batch_episodes(4, K, K)

        x_train, y_train = train_batch
        x_test, y_test = test_batch

        loss_value = batch_maml_step(
            regression_mlp, mse_loss, optimizer, x_train, y_train, x_test, y_test)

        np_batch_maml_loss.append(loss_value)
        if i % 500 == 0:
            logging.log(
                logging.INFO,
                "Step: {:03d}, Loss: {:.3f}".format(
                    i, loss_value.numpy()))

    predict_and_plot("sinusoids_maml_batch_trained.png")

    plt.figure()

    predictions = regression_mlp(x_range_inputs)
    plt.plot(x_range_inputs, predictions, label="pre-update prediction")

    y_targets = 2. * np.sin(x_range_inputs + 0.)
    plt.plot(x_range_inputs, y_targets, label="target")

    # unseen task (quadruple previously seen highest amplitude)
    K = 10

    x1 = np.random.uniform(low=-5., high=5., size=(K, 1)).astype(np.float32)
    y1 = 2. * np.sin(x1 + 0.)

    plt.plot(x1, y1, "bd", label="points for grad update")

    for i in range(1, 11):
        # update params on task learning samples
        params_updated = inner_update(
            regression_mlp, mse_loss, x1, y1)
        # apply meta-update params
        tf.nest.map_structure(
            lambda variable, update: variable.assign(update),
            regression_mlp.trainable_variables, params_updated)

        if i % 5 == 0 or i == 1:
            predictions = regression_mlp(x_range_inputs)
            plt.plot(x_range_inputs, predictions,
                     label="{} grad steps".format(i))

    plt.ylim([-3, 3])
    plt.legend()
    plt.savefig("sinusoids_maml_batch_train_adapted_{}-shot.png".format(K))

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    app.run(main)
