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
flags.DEFINE_integer("episodes", 50000, "number of meta-learning episodes")
# flags.DEFINE_integer("L", 5, "number of classes to sample in a task episode (L-way)")
flags.DEFINE_integer("K", 10, "number of task learning samples per class (K-shot)")
flags.DEFINE_integer("N", None, "number of task evaluation samples (set to K if not specified)")

flags.DEFINE_integer("meta_batch_size", 25, "number of tasks sampled per meta-update")
flags.DEFINE_float("meta_lr", 1e-3, "the base learning rate of the generator")
flags.DEFINE_float("update_lr", 1e-3, "step size alpha for inner gradient update")
flags.DEFINE_integer("num_updates", 1, "number of inner gradient updates during training")
flags.DEFINE_bool("first_order", False, "use first order approximation (no second derivatives)")

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
        """Return function that takes gradient of `func` w.r.t. some variables."""
        def grad_func(*args, targets=None, watch_list=None):
            if targets is None:
                targets = args[0]  # differentiate w.r.t first if no targets
            with tf.GradientTape() as tape:
                if watch_list is not None:
                    for var in watch_list:
                        tape.watch(watch_list)
                func_value = func(*args)
            grad_value = tape.gradient(func_value, targets)
            return grad_value
        return grad_func


    def f_func(x):
        return 3 * x

    def g_func(x):
        return x * x

    def inner_func(x):

        y0 = f_func(x)

        y = tf.Variable(0.)
        y.assign(y0)

        return g_func(y)


    x = tf.Variable(3.)

    print(f_func(x))
    print(g_func(x))
    print(inner_func(x))

    print(grad(f_func)(x))
    print(grad(g_func)(x))
    print(grad(inner_func)(x))

    x = tf.Variable([3.])

    @tf.function
    @tf.custom_gradient
    def square_x(x):
        z = x * x
        def grad(dy):
            return dy * tf.gradients(z, x)[0]
        return z, grad


    print(square_x(x))
    print(grad(square_x)(x, targets=x))

    def f(x):
        return square_x(3 * x)

    print(grad(f)(x, targets=x))


    # @tf.function
    @tf.custom_gradient
    def custom_assign(variable, value):
        assert isinstance(variable, tf.Variable)
        variable.assign(value)
        def grad(dy):
            return dy * 1., dy * 1.  # pass through for variable (TODO no grad for value?)
        return variable, grad

    print(custom_assign(x, [5.]))
    print(grad(custom_assign)(x, [5.], targets=x))
    print(grad(x.assign)(x, [5.], targets=x))

    def square(x):
        return x * x

    def times_three(x):
        return 3 * x

    def model(x):

        f_x = times_three(x)
        y = tf.Variable([0.])
        # custom_assign(y, f_x)
        y = tf.grad_pass_through(y.assign)(f_x)
        return square(y)

    x = tf.Variable([3.])
    print(model(x))
    print(grad(model)(x, targets=x))


    x = tf.Variable(1.0, name="x")
    z = tf.Variable(3.0, name="z")

    with tf.GradientTape() as tape:
        # y will evaluate to 9.0
        y = tf.grad_pass_through(x.assign)(z**2)
    # grads will evaluate to 6.0
    grads = tape.gradient(y, z)


    def lg(x, w):
        return x * w
    
    with tf.GradientTape() as tape:
        x = tf.constant(3.)
        w = tf.Variable(1.)

        loss_p = lg(x, w)

        w_p = tf.Variable(1.)
        w_p, loss_p

        loss = lg(x, w_p)

    print(tape.gradient(loss, w))


    x_input = tf.constant([[1.], [3.]])
    y_target = tf.constant([1., 0.])

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, use_bias=False)])

    def mse(y_pred, y_target):
        return tf.reduce_mean((y_pred - y_target)**2)

    with tf.GradientTape() as tape_outer:
        with tf.GradientTape() as tape_inner:
            mse_loss_inner = mse(model(x_input), y_target)

        variables = model.trainable_weights
        gradients = tape_inner.gradient(mse_loss_inner, variables)

        model_clone = tf.keras.models.clone_model(model)
        model_clone(x_input)  # forward pass to build model

        for var_clone, var, grad in zip(
                model_clone.trainable_weights, variables, gradients):
            var_clone.assign(var - 1e-3 * grad)

        mse_loss_outer = mse(model_clone(x_input), y_target)

    variables = model.trainable_weights
    gradients = tape_outer.gradient(mse_loss_outer, variables)
    # gradients results in [None] ... :(


    def update_model_weights_by_reference(model, new_weights):
        # fetch all model weights
        old_weights = model.trainable_weights
        assert len(old_weights) == len(new_weights)

        # create dictionary lookup for [layer name->weight name->new weight]
        weights_lookup = {}
        for old_weight, new_weight in zip(old_weights, new_weights):
            name_split = old_weight.name.split("/")
            layer_name = name_split[-2]
            weight_name = name_split[-1].split(":")[0]
            if layer_name not in weights_lookup:
                weights_lookup[layer_name] = {}
            weights_lookup[layer_name][weight_name] = new_weight

        def update_layer_weights(top_layer, weights_lookup):
            # iterate through each layer in top_layer
            for layer in top_layer.layers:
                # check for layer name in weights_lookup
                if layer.name in weights_lookup:
                    # iterate through each weight in _trainable_weights
                    for index, weight in enumerate(layer._trainable_weights):
                        weight_name = weight.name.split("/")[-1].split(":")[0]
                        if (weight_name in weights_lookup[layer.name]
                                and hasattr(layer, weight_name)):
                            # set weight attribute and variable tracking
                            layer._trainable_weights[index] = (
                                weights_lookup[layer.name][weight_name])
                            setattr(layer, weight_name, weights_lookup[layer.name][weight_name])
                            weights_lookup[layer.name].pop(weight_name)
                            if len(weights_lookup[layer.name]) == 0:
                                weights_lookup.pop(layer.name)
                    # check if current layer has sub-layers
                    if hasattr(layer, "layers"):
                        update_layer_weights(layer, weights_lookup)

        update_layer_weights(model, weights_lookup)

        assert len(weights_lookup) == 0, "Could not update all weights!"

    model = tf.keras.applications.ResNet50(weights=None)

    import time
    start = time.time()
    update_model_weights_by_reference(model, model.trainable_weights)
    end = time.time()
    print("ResNet 50 update by reference time: {:.6f}".format(end - start))


    x_input = tf.constant([[1.], [3.]])
    y_target = tf.constant([1., 0.])

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, use_bias=False)])
    model_clone = tf.keras.models.clone_model(model)

    model(x_input)
    model_clone(x_input)

    variables = model.trainable_weights

    with tf.GradientTape() as tape_outer:
        with tf.GradientTape() as tape_inner:
            mse_loss_inner = mse(model(x_input), y_target)

        gradients = tape_inner.gradient(mse_loss_inner, variables)

        variables_adapted = []
        for index, (var, grad) in enumerate(zip(variables, gradients)):
            variables_adapted.append(var - 1e-3 * grad)

        # adapt model weights
        update_model_weights_by_reference(model_clone, variables_adapted)

        mse_loss_outer = mse(model_clone(x_input), y_target)

        # # restore current model weights for meta-optimisation
        # update_model_weights_by_reference(model, variables)

    variables = model.trainable_weights
    gradients = tape_outer.gradient(mse_loss_outer, variables)


    # def get_model_weights_structure(model):

    #     layer_weights = []  # list of (layer name, weight name) pairs
    #     for weight in model.trainable_weights:
    #         name_split = weight.name.split("/")
    #         layer_name = name_split[-2]
    #         weight_name = name_split[-1].split(":")[0]

    #         layer_weights.append((layer_name, weight_name))

    #     return layer_weights


    # def get_model_weights_structure(model)

    #     def update_layer_weights(top_layer, weights_lookup):
    #         # iterate through each layer in top_layer
    #         for layer in top_layer.layers:
    #             for index, weight in enumerate(layer._trainable_weights):
    #                 name_split = weight.name.split("/")
    #                 layer_name = name_split[-2]  # layer.name doesn't include "_#copy"
    #                 weight_name = name_split[-1].split(":")[0]
    #                 # assume layer_name in weights_lookup ...
    #                 if (weight_name in weights_lookup[layer_name]
    #                         and hasattr(layer, weight_name)):
    #                     # set weight attribute and variable tracking
    #                     layer._trainable_weights[index] = (
    #                         weights_lookup[layer_name][weight_name])
    #                     setattr(layer, weight_name,
    #                             weights_lookup[layer_name][weight_name])
    #                     weights_lookup[layer_name].pop(weight_name)
    #                     if len(weights_lookup[layer_name]) == 0:
    #                         weights_lookup.pop(layer_name)
    #                 # check if current layer has sub-layers
    #                 if hasattr(layer, "layers"):
    #                     update_layer_weights(layer, weights_lookup)


    # def update_model_weights_by_structure(model, updates, weights_structure):

    #     assert len(updates) == len(weights_structure)

    #     # create dictionary lookup for [layer name->weight name->(update, index)]
    #     weights_lookup = {}
    #     for weight, (layer_name, weight_name) in zip(updates,
    #                                                  weights_structure):
    #         if layer_name not in weights_lookup:
    #             weights_lookup[layer_name] = {}
    #         weights_lookup[layer_name][weight_name] = weight

    #     def update_layer_weights(top_layer, weights_lookup):
    #         # iterate through each layer in top_layer
    #         for layer in top_layer.layers:
    #             for index, weight in enumerate(layer._trainable_weights):
    #                 # name_split = weight.name.split("/")
    #                 # layer_name = name_split[-2]  # layer.name doesn't include "_#copy"
    #                 # weight_name = name_split[-1].split(":")[0]
    #                 weight_name = weights_lookup[layer.name]
    #                 # assume layer_name in weights_lookup ...
    #                 if (weight_name in weights_lookup[layer_name]
    #                         and hasattr(layer, weight_name)):
    #                     # set weight attribute and variable tracking
    #                     layer._trainable_weights[index] = (
    #                         weights_lookup[layer_name][weight_name])
    #                     setattr(layer, weight_name,
    #                             weights_lookup[layer_name][weight_name])
    #                     weights_lookup[layer_name].pop(weight_name)
    #                     if len(weights_lookup[layer_name]) == 0:
    #                         weights_lookup.pop(layer_name)
    #                 # check if current layer has sub-layers
    #                 if hasattr(layer, "layers"):
    #                     update_layer_weights(layer, weights_lookup)

    #     # update model weights
    #     update_layer_weights(model, weights_lookup)

    #     assert len(weights_lookup) == 0, "Could not update all weights!"


    def get_model_weights_structure(model):

        def find_variable_layers(top_layer, variables, layer_names=None,
                                 layer_indices=None):
            if layer_names is None:
                layer_names = {}
            if layer_indices is None:
                layer_indices = {}

            # iterate through each layer in top_layer
            for layer in top_layer.layers:

                # map layer weights to model weights
                layer_weights = layer._trainable_weights
                for layer_index, layer_weight in enumerate(layer_weights):

                    for model_index, model_weight in enumerate(variables):

                        if layer_weight is model_weight:  # check if same object
                            # get layer.name (doesn't include the "_#copy")
                            layer_names[model_index] = layer.name
                            layer_indices[model_index] = layer_index
                            break

                    else:  # didn't find anything ... should not happen?
                        raise ValueError(
                            "Could not map model weights to trainable weight "
                            "{} in layer {}!".format(layer_weight.name,
                                                     layer.name))

                # check if current layer has sub-layers
                    if hasattr(layer, "layers"):
                        find_variable_layers(
                            layer, variables, layer_names, layer_indices)

            # finally return layer weights and indices map
            return layer_names, layer_indices

        # get model variable indices -> layer names & variable indices maps
        layer_names, layer_indices = find_variable_layers(
            model, model.trainable_variables)

        layer_weights = []  # list of (layer name, weight name, index) pairs
        for model_index, model_weight in enumerate(model.trainable_weights):
            weight_name = model_weight.name.split("/")[-1].split(":")[0]
            layer_name = layer_names[model_index]
            layer_index = layer_indices[model_index]
            layer_weights.append((layer_name, weight_name, layer_index))

        return layer_weights


    def update_model_weights_by_structure(model, updates, weights_structure):

        assert len(updates) == len(weights_structure)

        # create dictionary lookup for [layer name->weight name->(update, index)]
        weights_lookup = {}
        for update, (layer_name, weight_name, index) in zip(updates,
                                                            weights_structure):
            if layer_name not in weights_lookup:
                weights_lookup[layer_name] = {}
            weights_lookup[layer_name][weight_name] = (update, index)

        def update_layer_weights(top_layer, weights_lookup):
            # iterate through each layer in top_layer
            for layer in top_layer.layers:
                # check for layer name in weights_lookup
                if layer.name in weights_lookup:
                    for weight_name in list(weights_lookup[layer.name].keys()):
                        if hasattr(layer, weight_name):
                            update, index = weights_lookup[layer.name][weight_name]

                            layer._trainable_weights[index] = update
                            setattr(layer, weight_name, update)

                            weights_lookup[layer.name].pop(weight_name)
                            if len(weights_lookup[layer.name]) == 0:
                                weights_lookup.pop(layer.name)

                    # check if current layer has sub-layers
                    if hasattr(layer, "layers"):
                        update_layer_weights(layer, weights_lookup)

        # update model weights
        update_layer_weights(model, weights_lookup)

        assert len(weights_lookup) == 0, "Could not update all weights!"


    model = tf.keras.applications.ResNet50(weights=None)
    weights_structure = get_model_weights_structure(model)

    start = time.time()
    update_model_weights_by_structure(
        model, model.trainable_weights, weights_structure)
    end = time.time()
    print("ResNet 50 update by reference time: {:.6f}".format(end - start))

    x_input = tf.constant([[1.], [3.]])
    y_target = tf.constant([1., 0.])

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, use_bias=False)])

    model(x_input)  # forward pass to build model
    weights_structure = get_model_weights_structure(model)

    variables_current = model.trainable_weights

    def inner_optimizer(gradient, variable, lr=1e-3):
        """Apply gradient descent to variable with gradient."""
        return variable - lr * gradient

    import functools
    inner_optimizer = functools.partial(inner_optimizer, lr=1e-3)

    with tf.GradientTape() as tape_outer:
        variables_adapted = model.trainable_variables

        for i in range(5):  # num updates
            with tf.GradientTape() as tape_inner:
                tape_inner.watch(variables_adapted)  # need to track when this becomes a tf.Tensor
                mse_loss_inner = mse(model(x_input), y_target)

            gradients = tape_inner.gradient(mse_loss_inner, variables_adapted)

            variables_adapted = tf.nest.map_structure(
                inner_optimizer, gradients, variables_adapted)

            # update model with adapted weights and apply to test data from task
            update_model_weights_by_structure(
                model, variables_adapted, weights_structure)

            mse_loss_outer = mse(model(x_input), y_target)

    # restore current model weights for meta-optimisation
    update_model_weights_by_structure(
        model, variables_current, weights_structure)

    gradients = tape_outer.gradient(mse_loss_outer, variables_current)

    import pdb; pdb.set_trace()


    # implement maml (with TensorFlow 2.0)
    # ====================================

    def inner_update(model, loss, x1, y1, alpha=.1):
        params = model.trainable_variables
        loss_grads = grad(loss)(model, x1, y1, targets=params)
        # loss_grads = [tf.stop_gradient(grad) for grad in loss_grads]
        inner_sgd_fn = lambda g, state: (state - alpha*g)
        # apply sgd inner function elementwise grads and params (i.e. to each leaf of the trees)
        return tf.nest.map_structure(inner_sgd_fn, loss_grads, params)

    def maml_loss(model, loss, x1, y1, x2, y2, num_updates=1, alpha=.1):
        # store model parameters prior to meta-update
        current_weights = model.get_weights()


        # NOTE: baseline network should be trained on the initial train loss
        # meta_train_losses[0] and MAML should be optimized on the final test
        # loss meta_test_losses[-1]
        meta_train_losses, meta_test_losses = [], []
        for _ in range(num_updates):
            # update parameters on task learning samples
            meta_train_losses.append(loss(model, x1, y1))
            adapted_weights = inner_update(model, loss, x1, y1, alpha)

            # apply meta-optimized params to model
            model.set_weights(adapted_weights)
            model.var1.assihg(var1 - lr * grad)
            model.var1.add()

            # evaluate meta-updated model on task test samples
            meta_test_losses.append(loss(model, x2, y2))

        tmp_g = grad(loss)(model, x2, y2, targets=adapted_weights)

        # restore model parameters prior to meta-update
        model.set_weights(current_weights)

        return meta_train_losses, meta_test_losses  # TODO return forward pass?

    def maml_step(model, loss, optimizer, x1, y1, x2, y2, num_updates=1, alpha=.1):

        loss_value = maml_loss(model, loss, x1, y1, x2, y2, num_updates, alpha)

        model_params = model.trainable_variables
        maml_loss_func = lambda *args: maml_loss(*args)[1][-1]
        loss_grads = grad(maml_loss_func)(
            model, loss, x1, y1, x2, y2, num_updates, alpha, targets=model_params)

        optimizer.apply_gradients(zip(loss_grads, model_params))

        return loss_value

    regression_mlp = create_model()

    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.meta_lr)

    np_maml_loss = []

    logging.log(logging.INFO, "Training MAML to regress sinusoids")
    for i in range(FLAGS.episodes):
        # define the task
        amplitude = np.random.uniform(low=0.1, high=.5)
        phase = np.random.uniform(low=0., high=np.pi)
        # meta-training inner split (K examples)
        x1 = np.random.uniform(low=-5., high=5., size=(FLAGS.K, 1)).astype(np.float32)
        y1 = amplitude * np.sin(x1 + phase)
        # meta-training outer split (1 example). Like cross-validating with respect to one example.
        x2 = np.random.uniform(low=-5., high=5., size=(FLAGS.K, 1)).astype(np.float32)
        y2 = amplitude * np.sin(x2 + phase)

        loss_value = maml_step(
            regression_mlp, mse_loss, optimizer, x1, y1, x2, y2, FLAGS.num_updates, FLAGS.update_lr)

        np_maml_loss.append(loss_value)
        if i % 500 == 0:
            logging.log(
                logging.INFO,
                "Step: {:03d}, Loss: {:.3f}".format(
                    i, loss_value[1][-1].numpy()))
            print(tuple(x.numpy() for x in loss_value[0]))
            print(tuple(x.numpy() for x in loss_value[1]))

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
