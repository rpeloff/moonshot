"""Functions to create and use the DaveNet speech model.

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: September 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf


def create_davenet_audio_network(
        input_shape=None, batch_norm_spectrogram=True, batch_norm_conv=False,
        downsample=True, embedding_dim=1024, padding="same"):
    """Create DAVEnet (Deep Audio-Visual Embedding network) audio branch model.

    Based on:
    https://github.com/dharwath/DAVEnet-pytorch/blob/23a6482859dd2221350307c9bfb5627a5902f6f0/models/AudioModels.py#L7-L33
    """

    input_kwargs = {}
    if input_shape is not None:  # specifying input shape builds layers
        input_kwargs["input_shape"] = input_shape

    model_layers = []

    if batch_norm_spectrogram:  # TODO make sure this results in [1, num_channels] params
        model_layers.append(tf.keras.layers.BatchNormalization(**input_kwargs))
        if "input_shape" in input_kwargs:
            input_kwargs.pop("input_shape")

    model_layers.append(
        tf.keras.layers.Conv1D(
            128, kernel_size=1, strides=1, padding=padding, **input_kwargs))

    if batch_norm_conv:
        model_layers.append(tf.keras.layers.BatchNormalization())

    model_layers.append(tf.keras.layers.ReLU())

    model_layers.append(
        tf.keras.layers.Conv1D(256, kernel_size=11, strides=1, padding=padding))

    if batch_norm_conv:
        model_layers.append(tf.keras.layers.BatchNormalization())

    model_layers.append(tf.keras.layers.ReLU())

    if downsample:
        model_layers.append(tf.keras.layers.MaxPool1D(
            pool_size=3, strides=2, padding=padding))

    model_layers.append(
        tf.keras.layers.Conv1D(512, kernel_size=17, strides=1, padding=padding))

    if batch_norm_conv:
        model_layers.append(tf.keras.layers.BatchNormalization())

    model_layers.append(tf.keras.layers.ReLU())

    if downsample:
        model_layers.append(tf.keras.layers.MaxPool1D(
            pool_size=3, strides=2, padding=padding))

    model_layers.append(
        tf.keras.layers.Conv1D(512, kernel_size=17, strides=1, padding=padding))

    if batch_norm_conv:
        model_layers.append(tf.keras.layers.BatchNormalization())

    model_layers.append(tf.keras.layers.ReLU())

    if downsample:
        model_layers.append(tf.keras.layers.MaxPool1D(
            pool_size=3, strides=2, padding=padding))

    model_layers.append(
        tf.keras.layers.Conv1D(
            embedding_dim, kernel_size=17, strides=1, padding=padding))

    if batch_norm_conv:
        model_layers.append(tf.keras.layers.BatchNormalization())

    model_layers.append(tf.keras.layers.ReLU())

    audio_network = tf.keras.Sequential(model_layers)

    return audio_network


def freeze_weights(audio_network, trainable=None):
    """Freeze weights of DAVEnet audio model before the `trainable` index.

    `trainable` specifies which layer (and layers above) is trainable, one of
    [None, 'final_conv'] or an integer index.
    """
    # freeze all layers
    if trainable is None:
        train_index = len(audio_network.layers)
    # freeze audio network layers before final convolutional layer
    # i.e. fine tune only the final convolutional layer
    elif trainable == "final_conv":
        train_index = -2
    # freeze layers up to `trainable` index
    elif isinstance(trainable, int):
        train_index = trainable
    # freeze layers
    for layer in audio_network.layers[:train_index]:
        layer.trainable = False
    # make sure layers above this are not frozen
    for layer in audio_network.layers[train_index:]:
        layer.trainable = True
