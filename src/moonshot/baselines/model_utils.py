"""TODO(rpeloff)

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: September 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os


from absl import logging


import tensorflow as tf


from moonshot.utils import file_io


def save_model(model, output_dir, epoch, step, metric, current_score,
               best_score, name="model"):
    """Save a model and its training progress."""
    assert hasattr(model, "loss") and model.loss is not None
    assert hasattr(model, "optimizer") and model.optimizer is not None

    model.save(os.path.join(output_dir, f"{name}.h5"))

    file_io.write_csv(
        os.path.join(output_dir, f"{name}.step"),
        [epoch, step, metric, current_score, best_score])


def load_model(model_file, model_step_file, loss):
    """Load a model and its training progress."""
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


def get_model_weights_structure(model, trainable=True):
    """Find (layer, weights, index) aligned to `model.<non_>trainable_weights`."""

    def find_weights(top_layer, model_weights, layer_trace=None,
                     layer_traces=None, weight_indices=None):
        """Recurse through model layers and sub-layers finding layer weights."""
        if layer_trace is None:
            layer_trace = ()
        if layer_traces is None:
            layer_traces = {}
        if weight_indices is None:
            weight_indices = {}
        # iterate through each of the sub-layers in `top_layer`
        for layer_index, layer in enumerate(top_layer.layers):
            # map current layer weights to the model weights
            layer_weights = (layer._trainable_weights
                             if trainable else layer._non_trainable_weights)
            sub_layer_trace = layer_trace + (layer_index, )
            for weight_index, layer_weight in enumerate(layer_weights):
                for model_index, model_weight in enumerate(model_weights):
                    # check if the layer weight matches any model weight
                    if layer_weight is model_weight:
                        layer_traces[model_index] = sub_layer_trace
                        weight_indices[model_index] = (weight_index)
                        break
                else:  # didn't find anything ... should not happen?
                    raise ValueError(
                        "Could not map model weights to trainable weight "
                        "{} in layer {}.".format(layer_weight.name,
                                                 layer.name))
            # check if current layer has any sub-layers to recurse
            if hasattr(layer, "layers"):
                find_weights(layer, model_weights, sub_layer_trace,
                             layer_traces, weight_indices)
        # return the layer weights and indices lookup tables
        return layer_traces, weight_indices

    # get map of model variable indices -> layer traces & variable indices
    model_weights = (model.trainable_weights
                     if trainable else model.non_trainable_weights)
    layer_traces, weight_indices = find_weights(model, model_weights)

    # return (layer trace, weight name, weight index) pairs aligned to model_weights
    weights_structure = []
    for model_index, model_weight in enumerate(model_weights):
        weight_name = model_weight.name.split("/")[-1].split(":")[0]
        layer_trace = layer_traces[model_index]
        weight_index = weight_indices[model_index]
        weights_structure.append((layer_trace, weight_name, weight_index))

    return weights_structure


def update_model_weights(model, updates, weights_structure=None,
                         force_update=False, trainable=True):
    """Update model weights directly inserting `updates` at correct locations.

    `updates` must be aligned to `model.(non_)trainable_weights` and is inserted
    according to model `weights_structure` (see `get_model_weights_structure`).

    NOTE: model variable names will likely change after the weight update, thus
    `weights_structure` will not be retrievable and should be stored prior.
    """
    if weights_structure is None:
        weights_structure = get_model_weights_structure(model)

    assert len(updates) == len(weights_structure)

    for update, structure in zip(updates, weights_structure):
        layer_trace, weight_name, weight_index = structure

        # trace to the possibly nested layer to insert the next weight update
        trace_layer = model
        for trace_location in layer_trace:
            trace_layer = trace_layer.layers[trace_location]

        if hasattr(trace_layer, weight_name) or force_update:
            trace_layer_weights = (trace_layer._trainable_weights if trainable
                                   else trace_layer._non_trainable_weights)
            # set (_non)_trainable_weights used to track layer weights
            n_weights = len(trace_layer_weights)
            if n_weights < weight_index + 1:  # create weight list if not set
                new_weights = [None] * (weight_index + 1)
                new_weights[:n_weights] = trace_layer_weights
                trace_layer_weights = new_weights
            trace_layer_weights[weight_index] = update
            # set the weight attribute used to apply the layer
            setattr(trace_layer, weight_name, update)
        else:  # didn't find attribute... should not happen!
            raise ValueError(
                "Could not find weight attribute {} in layer {}.".format(
                    weight_name, trace_layer.name))


def build_model(top_layer, input_shape):
    """Set all model layers (and sub-layers) to be 'built'."""
    top_layer.built = True
    for layer in top_layer._layers:
        layer.built = True

        # fix batch norm building without calling build ... see:
        # https://github.com/tensorflow/tensorflow/blob/d3b421bc5c86b4dcce8470721c6e24055a4b3ef1/tensorflow/python/keras/layers/normalization.py#L985
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            ndims = len(input_shape)
            # Convert axis to list and resolve negatives
            if isinstance(layer.axis, int):
                layer.axis = [layer.axis]
            for idx, x in enumerate(layer.axis):
                if x < 0:
                    layer.axis[idx] = ndims + x

        # build model recursively
        if hasattr(layer, "layers"):
            build_model(layer, input_shape)


def create_and_copy_model(model, create_model_func, **kwargs):
    """Create a model with `create_model_func` and copy weights from `model`.

    For meta-optimization `create_model_func` should not create new variables.
    """
    new_model = create_model_func(**kwargs)

    update_model_weights(  # copy trainable weights
        new_model, model.trainable_weights,
        weights_structure=get_model_weights_structure(model, trainable=True),
        trainable=True, force_update=True)

    update_model_weights(  # copy non-trainable weights
        new_model, model.non_trainable_weights,
        weights_structure=get_model_weights_structure(model, trainable=False),
        trainable=False, force_update=True)

    # make sure that model is "built" and new variables are not created
    build_model(new_model, model.input_shape)

    return new_model
