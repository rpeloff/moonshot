"""TODO(rpeloff)

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: October 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os


import numpy as np
import tensorflow as tf


from moonshot.baselines import fast_dtw
from moonshot.experiments.flickr_speech import flickr_speech
from moonshot.experiments.flickr_vision import flickr_vision


def augment_square_crop(image, size=224, random_scales=None,
                        horizontal_flip=True, colour=False):
    """Augment image (scale, flip, colour) and select random square crop."""
    # get shorter side of image
    image_shape = tf.shape(image)
    h, w = image_shape[0], image_shape[1]
    short_edge = tf.minimum(w, h)

    # scale augmentation
    if random_scales is None:
        # random resize along shorter edge in [256, 480]
        # maxval - minval = power of 2 => unbiased random integers
        rand_resize = tf.random.uniform(
            [], minval=tf.maximum(256, size), maxval=(480+1), dtype=tf.int32)
    else:
        # random resize along shorter edge in `random_scales` if specified
        rand_scale_idx = tf.random.uniform(
            [], maxval=tf.shape(random_scales)[0], dtype=tf.int32)
        rand_resize = tf.convert_to_tensor(rand_resize)[rand_scale_idx]

    resize_hw = (rand_resize * h/short_edge, rand_resize * w/short_edge)
    image = tf.image.resize(image, resize_hw, method="lanczos3")

    # horizontal flip augmentation
    if horizontal_flip:
        image = tf.image.random_flip_left_right(image)

    # colour augmentation (ordering of these ops matters so we shuffle them)
    if colour:
        color_ordering = tf.random.uniform([], maxval=1, dtype=tf.int32)
        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        else:
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)

    # crop augmentation, random sample square (size, size, 3) from resized image
    image = tf.image.random_crop(image, size=(size, size, 3))

    # make sure that we still have an image in range [0, 1]
    image = image - tf.reduce_min(image)
    image = tf.math.divide_no_nan(image, tf.reduce_max(image))
    return image


def resize_square_crop(image, size=224):
    """Resize image along short edge and center square crop."""
    # get shorter side of image
    image_shape = tf.shape(image)
    h, w = image_shape[0], image_shape[1]
    short_edge = tf.minimum(w, h)

    # resize image
    resize_hw = (size * h/short_edge, size * w/short_edge)
    image = tf.image.resize(image, resize_hw, method="lanczos3")

    # center square crop
    image_shape = tf.shape(image)
    h, w = image_shape[0], image_shape[1]
    h_shift = tf.cast((h - size) / 2, tf.int32)
    w_shift = tf.cast((w - size) / 2, tf.int32)
    image = tf.image.crop_to_bounding_box(
        image, h_shift, w_shift, size, size)

    # make sure that we still have an image in range [0, 1]
    image = image - tf.reduce_min(image)
    image = tf.math.divide_no_nan(image, tf.reduce_max(image))
    return image


def load_and_preprocess_image(image_path, crop_size=224, augment_crop=False,
                              normalise=True, random_scales=None,
                              horizontal_flip=True, colour=False):
    """Load image at path and square crop with optional augmentation."""
    # read and decode image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    # scale image to range [0, 1] expected by tf.image functions
    image = tf.cast(image, tf.float32) / 255.

    # random crop image for testing
    if augment_crop:
        image = augment_square_crop(
            image, size=crop_size, random_scales=random_scales,
            horizontal_flip=horizontal_flip, colour=colour)
    else:
        image = resize_square_crop(image, size=crop_size)

    tf.debugging.assert_greater_equal(image, tf.constant(0.))
    tf.debugging.assert_less_equal(image, tf.constant(1.))

    # normalise image from [0, 1] to range [-1, 1]
    if normalise:
        image *= 2.
        image -= 1.

    tf.debugging.assert_greater_equal(image, tf.constant(-1.))
    tf.debugging.assert_less_equal(image, tf.constant(1.))

    return image


def create_flickr_vision_train_data(data_sets, embed_dir=None):
    """Load train and validation Flickr vision data."""

    flickr8k_image_dir = None
    if "flickr8k" in data_sets:
        flickr8k_image_dir = os.path.join("data", "external", "flickr8k_images")

    flickr30k_image_dir = None
    if "flickr30k" in data_sets:
        flickr30k_image_dir = os.path.join(
            "data", "external", "flickr30k_images")

    mscoco_train_image_dir = None
    mscoco_dev_image_dir = None
    if "mscoco" in data_sets:
        mscoco_train_image_dir = os.path.join(
            "data", "external", "mscoco", "train2017")
        mscoco_dev_image_dir = os.path.join(
            "data", "external", "mscoco", "val2017")

    flickr_train_exp = flickr_vision.FlickrVision(
        keywords_split="background_train",
        flickr8k_image_dir=flickr8k_image_dir,
        flickr30k_image_dir=flickr30k_image_dir,
        mscoco_image_dir=mscoco_train_image_dir, embed_dir=embed_dir)

    flickr_dev_exp = flickr_vision.FlickrVision(
        keywords_split="background_dev",
        flickr8k_image_dir=flickr8k_image_dir,
        flickr30k_image_dir=flickr30k_image_dir,
        mscoco_image_dir=mscoco_dev_image_dir, embed_dir=embed_dir)

    return flickr_train_exp, flickr_dev_exp


def load_and_preprocess_speech(speech_path, features, max_length=130,
                               reinterpolate=None, scaling=None):
    # load speech features from numpy binary file
    if isinstance(speech_path, tf.Tensor):
        speech_path = speech_path.numpy().decode("utf-8")

    speech_features = np.load(speech_path)

    # center pad speech features (or crop if longer than max length)
    if reinterpolate is None:
        # add "height" dim
        speech_features = tf.expand_dims(speech_features, axis=0)
        # crop/pad the speech features "image"
        speech_features = tf.image.resize_with_crop_or_pad(
            speech_features, target_height=1, target_width=max_length)
        # remove "height" dim
        speech_features = tf.squeeze(speech_features, axis=0)

    # re-interpolate speech features to max length
    else:
        speech_features = fast_dtw.dtw_reinterp2d(
            speech_features, max_length, interp=reinterpolate)

    # scale speech features
    if scaling == "global":
        speech_features -= flickr_speech.train_global_mean[features]
        speech_features /= np.sqrt(flickr_speech.train_global_var[features])
    elif scaling == "features":
        speech_features -= flickr_speech.train_features_mean[features]
        speech_features /= np.sqrt(flickr_speech.train_features_var[features])
    elif scaling == "segment":
        speech_features = tf.math.divide_no_nan(
            speech_features - np.mean(speech_features),
            np.sqrt(np.var(speech_features)))
    elif scaling == "segment_mean":
        speech_features = speech_features - np.mean(speech_features)

    # TODO
    # if spec_augment:
    #     features = spec_augment(features)

    return speech_features


def create_flickr_audio_train_data(features, embed_dir=None,
                                   speaker_mode="baseline"):
    """Load train and validation Flickr audio data."""

    flickr_train_exp = flickr_speech.FlickrSpeech(
        features=features, keywords_split="background_train",
        embed_dir=embed_dir, speaker_mode=speaker_mode)

    flickr_dev_exp = flickr_speech.FlickrSpeech(
        features=features, keywords_split="background_dev", embed_dir=embed_dir,
        speaker_mode=speaker_mode)

    return flickr_train_exp, flickr_dev_exp


def embedding_to_example_protobuf(embedding):
    """Create tf.Example message (protobuf) from an embedding array."""
    feature = {
        "embed": tf.train.Feature(
            float_list=tf.train.FloatList(value=embedding))}

    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))

    return example_proto


def parse_embedding_protobuf(example_proto):
    """Parse a serialized tf.Example embedding with variable length."""
    feature_description = {
        "embed": tf.io.FixedLenSequenceFeature(
            [], tf.float32, allow_missing=True)}

    return tf.io.parse_single_example(
        example_proto, feature_description)


def create_balanced_batch_dataset(p, k, label_datasets):
    """Creates a dataset that samples a balanced batch from `label_datasets`.

    `p` is number of classes per batch, `k` is number of samples per class,
    `label_datasets` is list of datasets corresponding to class labels.
    """

    num_labels = len(label_datasets)

    def label_generator():
        # sample labels that will compose the balanced batch
        labels = np.random.choice(range(num_labels), p, replace=False)
        for label in labels:
            for _ in range(k):
                yield label

    choice_dataset = tf.data.Dataset.from_generator(label_generator, tf.int64)

    balanced_dataset = tf.data.experimental.choose_from_datasets(
        label_datasets, choice_dataset)

    return balanced_dataset
