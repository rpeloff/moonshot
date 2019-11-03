"""Functions to create and use the Inception V3 model.

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: September 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf


def create_inceptionv3_network(
        input_shape=(299, 299, 3), pretrained=False, include_top=False):
    """Create Inception V3 model.

    `pretrained` should be False for one-shot experiments; cannot use "imagenet"
    pretrained weights since these are trained on data that is not disjoint from
    the Flickr 8k one-shot task.

    `include_top` specifies whether to include the top global average pooling
    and fully-connected logits layers.
    """
    weights = "imagenet" if pretrained else None
    input_shape = None if input_shape is None else tuple(input_shape)

    # NOTE: inject `layers=tf.keras.layers` to use TF 2.0 behaviour since
    # keras_applications seems to be using the old TF 1.0 layers.
    # Specifically, this is useful to replace the v1.0 batch normalisation layer
    # with the v2.0 layer which uses the moving mean and variance to normalise
    # the current batch when the layer is frozen (`trainable = False`). This is
    # the expected behaviour when fine-tuning! See here for more information:
    # https://github.com/tensorflow/tensorflow/blob/eda53c63dab8b364872ede8e423e4fed5d1686f7/tensorflow/python/keras/layers/normalization_v2.py#L26-L65
    # as well as here:
    # https://github.com/keras-team/keras/pull/9965#issuecomment-549126009
    return tf.keras.applications.inception_v3.InceptionV3(
        weights=weights, include_top=include_top, input_shape=input_shape,
        layers=tf.keras.layers)


def freeze_weights(inception_model, trainable=None):
    """Freeze weights of Inception v3 model before the `trainable` index.

    `trainable` specifies which layer (and layers above) is trainable, one of
    [None, 'final_inception', 'logits'] or an integer index.
    """
    # freeze all layers
    if trainable is None:
        train_index = len(inception_model.layers)
    # freeze InceptionV3 base layers before final inception module
    # i.e. fine tune only the final inception module (and logits if included)
    elif trainable == "final_inception":
        train_index = 279
    # freeze all InceptionV3 base layers
    # i.e. fine tune only the logits layer (if included)
    elif trainable == "logits":
        train_index = 279
    # freeze layers up to `trainable` index
    elif isinstance(trainable, int):
        train_index = trainable
    # freeze layers
    for layer in inception_model.layers[:train_index]:
        layer.trainable = False
    # make sure layers above this are not frozen
    for layer in inception_model.layers[train_index:]:
        layer.trainable = True
