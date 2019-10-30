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
    image = image / tf.reduce_max(image)
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
    image = image / tf.reduce_max(image)
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
    # normalise image from [0, 1] to range [-1, 1]
    if normalise:
        image *= 2.
        image -= 1.
    return image


def create_flickr_train_data(data_sets, embed_dir=None):
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


# def create_flickr_background_dataset(
#         image_paths, label_ids, image_preprocess_func=None, tfrecords=False,
#         batch_size=None, num_repeat=None, shuffle=False, shuffle_buffer=1000,
#         prefetch_buffer=tf.data.experimental.AUTOTUNE,
#         n_parallel_calls=tf.data.experimental.AUTOTUNE):
#     """Create Flickr vision tf.data dataset for background training."""

#     if tfrecords:
#         background_image_ds = tf.data.TFRecordDataset(
#             image_paths, compression_type="ZLIB", num_parallel_reads=4)
#     else:
#         background_image_ds = tf.data.Dataset.from_tensor_slices(image_paths)

#     background_image_ds = background_image_ds.map(
#         image_preprocess_func, num_parallel_calls=n_parallel_calls)

#     background_label_ds = tf.data.Dataset.from_tensor_slices(
#         tf.cast(label_ids, tf.int64))

#     background_ds = tf.data.Dataset.zip(
#         (background_image_ds, background_label_ds))

#     if num_repeat is not None:
#         background_ds = background_ds.repeat(num_repeat)

#     if shuffle:
#         background_ds = background_ds.shuffle(shuffle_buffer)

#     if batch_size is not None:
#         background_ds = background_ds.batch(batch_size)

#     if prefetch_buffer is not None:
#         background_ds = background_ds.prefetch(prefetch_buffer)

#     return background_ds


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


# def load_embedding_records(embedding_paths):
#     """Load embeddings as tf.data.TFRecordDataset from TFRecord paths."""
#     embed_records = tf.data.TFRecordDataset(
#         embedding_paths, compression_type="ZLIB")

#     embed_records = embed_records.map(
#         lambda example_proto: parse_embedding_protobuf(
#             example_proto)["embed"],
#         num_parallel_calls=tf.data.experimental.AUTOTUNE)

#     return embed_records


# def create_embedding_dataset(
#         embed_paths, label_ids, batch_size=None, shuffle=False,
#         shuffle_buffer=1000, prefetch_buffer=tf.data.experimental.AUTOTUNE,
#         n_parallel_calls=tf.data.experimental.AUTOTUNE):
#     """TODO"""
#     embedding_ds = tf.data.TFRecordDataset(
#         embed_paths, compression_type="ZLIB", num_parallel_reads=4)

#     embedding_ds = embedding_ds.map(
#         lambda example_proto: parse_embedding_protobuf(example_proto)["embed"],
#         num_parallel_calls=n_parallel_calls)

#     label_ds = tf.data.Dataset.from_tensor_slices(
#         tf.cast(label_ids, tf.int64))

#     embedding_ds = tf.data.Dataset.zip((embedding_ds, label_ds))

#     if shuffle:
#         embedding_ds = embedding_ds.shuffle(shuffle_buffer)

#     if batch_size is not None:
#         embedding_ds = embedding_ds.batch(batch_size)

#     if prefetch_buffer is not None:
#         embedding_ds = embedding_ds.prefetch(prefetch_buffer)

#     return embedding_ds


# def filter_dataset_label(label):
#     """Predicates whether dataset labels match a specific label."""

#     def filter_func(_, y):
#         tf.print("Filtering label", label)
#         return tf.equal(tf.cast(y, tf.int64), tf.cast(label, tf.int64))

#     return filter_func


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

    # create dummy dataset
    # p_k_choice_dataset = tf.data.Dataset.from_tensor_slices([0.])
    
    # # sample p random class labels and repeat k times
    # p_k_choice_dataset = p_k_choice_dataset.flat_map(
    #     lambda _: tf.data.Dataset.from_tensor_slices(
    #         tf.tile(
    #             tf.gather(
    #                 tf.random.shuffle(
    #                     tf.range(len(label_datasets), dtype=tf.int64)),
    #                 tf.range(p)),
    #             [k])))

    # # use p*k class labels to select samples from label datasets
    # ds_p_k = tf.data.experimental.choose_from_datasets(
    #     label_datasets, p_k_choice_dataset)

    # return ds_p_k.batch(p * k)


# def create_balanced_batch_dataset(p, k, label_datasets):
    # num_classes = len(label_datasets)

    # # create dataset of the unique class labels
    # balanced_dataset = tf.data.Dataset.range(num_classes)

    # # randomly select p labels
    # balanced_dataset = balanced_dataset.shuffle(num_classes)
    # balanced_dataset = balanced_dataset.take(p)

    # # map each of the p labels to a set of k samples
    # balanced_dataset = balanced_dataset.flat_map(
    #     #lambda label: sample_k_from_datasets(label_datasets, label, k))
    #     lambda label: tf.data.experimental.choose_from_datasets(
    #         label_datasets, tf.data.Dataset.from_tensor_slices(
    #             tf.tile([label], [k]))))

    # # return dataset containing a single batch of PK samples
    # return balanced_dataset.batch(p * k)


# def sample_k_from_datasets(label_datasets, choice, k):
#     # label_ds = tf.gather(label_datasets, [choice])
#     # return label_ds.shuffle(k).take(k)
#     return tf.data.experimental.choose_from_datasets(
#         label_datasets, tf.data.Dataset.from_tensor_slices(tf.tile([choice], [k])))


