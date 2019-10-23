"""TODO(rpeloff)

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: October 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import functools


import tensorflow as tf


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


def create_flickr_background_dataset(
        image_paths, label_ids, image_preprocess_func=None, tfrecords=False,
        batch_size=32, num_repeat=None, shuffle=False, shuffle_buffer=1000,
        prefetch_buffer=tf.data.experimental.AUTOTUNE,
        n_parallel_calls=tf.data.experimental.AUTOTUNE):
    """Create Flickr vision tf.data dataset for background training."""

    if tfrecords:
        background_image_ds = tf.data.TFRecordDataset(
            image_paths, compression_type="ZLIB", num_parallel_reads=4)
    else:
        background_image_ds = tf.data.Dataset.from_tensor_slices(image_paths)

    background_image_ds = background_image_ds.map(
        image_preprocess_func, num_parallel_calls=n_parallel_calls)

    background_label_ds = tf.data.Dataset.from_tensor_slices(
        tf.cast(label_ids, tf.int64))

    background_ds = tf.data.Dataset.zip(
        (background_image_ds, background_label_ds))

    if num_repeat is not None:
        background_ds = background_ds.repeat(num_repeat)

    if shuffle:
        background_ds = background_ds.shuffle(shuffle_buffer)

    background_ds = background_ds.batch(batch_size)
    background_ds = background_ds.prefetch(prefetch_buffer)

    return background_ds


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


def load_embedding_records(embedding_paths):
    """Load embeddings as tf.data.TFRecordDataset from TFRecord paths."""
    embed_records = tf.data.TFRecordDataset(
        embedding_paths, compression_type="ZLIB")

    embed_records = embed_records.map(
        lambda example_proto: parse_embedding_protobuf(
            example_proto)["embed"],
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return embed_records


# TODO: update old code
# def batch_k_examples_for_p_concepts(
#         x_data,
#         y_labels,
#         p_batch,
#         k_batch, #, unique_concepts=None
#         seed=42
#         ):
#     """Create dataset with batches of P concept classes and K examples per class.   
    
#     Used to produce balanced mini-batches of PK examples, by randomly sampling 
#     P concept labels, and then K examples per concept [1]_. 
#     Parameters
#     ----------
#     x_data : array-like or tf.Tensor
#         Dataset of concept features.
#     y_labels : array-like or tf.Tensor
#         Dataset of concept labels associated with features. 
#     p_batch : int
#         Number of P unique concept labels to sample per batch.
#     k_batch : int
#         Number of K examples to sample per unique concept in a batch.
#     unique_concepts : array-like, optional
#         List of unique concept classes from which P concepts are sampled.
#     Returns
#     -------
#     balanced_dataset : tf.data.Dataset
#         Balanced dataset containing batches (x_batch, y_batch) of PK examples.
#     n_batch : int
#         Number of batches per "epoch".
#     Notes
#     -----
#     Based on code for sampling batches of PK images for triplet loss [1]_: 
#     - https://github.com/VisualComputingInstitute/triplet-reid/blob/f3aed745964d81d7410e1ebe32eb4329af886d2d/train.py#L234-L250.
#     If unique_concepts is not specified then y_labels can only be of type 
#     array-like and not tf.Tensor.
#     References
#     ----------
#     .. [1] A. Hermans, L. Beyer, B. Leibe (2017):
#             In Defense of the Triplet Loss for Person Re-Identification.
#             https://arxiv.org/abs/1703.07737
    
#     Examples
#     --------
#     Create an iterator and get tensors for the batches of data and labels:
#     >>> balanced_dataset, n_batches = ml.data.batch_k_examples_for_p_concepts(...)
#     >>> balanced_dataset = balanced_dataset.prefetch(1)  # Parallel CPU/GPU processing
#     >>> ...
#     >>> with tf.Session() as sess:
#     ...     ...
#     ...     for epoch in range(n_epochs):
#     ...         # Create new iterator and loop over balanced P.K dataset:
#     ...         x_batch, y_batch = balanced_dataset.make_one_shot_iterator().get_next()
#     ...         for i in range(n_batches):
#     ...             sess.run([...], feed_dict={x_in: x_batch, y_in: y_batch})
#     ...         # End of epoch
#     """
#     # # Get the unique concept labels (if None, y_labels can't be tensor)
#     # if unique_concepts is None:
#     #     if isinstance(y_labels, tf.Tensor):
#     #         raise TypeError("Input for y_labels cannot be of type tf.Tensor if "
#     #                         "unique_concept is not specified.")
#     #     unique_concepts = np.unique(y_labels)   
#     # n_concepts = np.shape(unique_concepts)[0]
#     # n_batches = n_concepts // p_batch
#     # n_dataset = n_batches * p_batch  # Multiple of P batch size

#     # Get unique concept labels and count
#     unique_concepts = tf.unique(y_labels)[0]
#     n_concepts = tf.shape(unique_concepts, out_type=tf.int64)[0]
#     # Create shuffled dataset of the unique concept labels
#     balanced_dataset = tf.data.Dataset.from_tensor_slices(unique_concepts)
#     balanced_dataset = balanced_dataset.shuffle(n_concepts, seed=seed)
#     # Select p_batch labels from the shuffled concepts for one batch/episode
#     balanced_dataset = balanced_dataset.take(p_batch) 
#     # Map each of the selected concepts to a set of K exemplars
#     balanced_dataset = balanced_dataset.flat_map(
#         lambda concept: tf.data.Dataset.from_tensor_slices(
#             _sample_k_examples_for_labels(labels=concept,
#                                           x_data=x_data,
#                                           y_labels=y_labels,
#                                           k_size=k_batch)))
#     # Group flattened dataset into batches of P.K exemplars
#     balanced_dataset = balanced_dataset.batch(p_batch * k_batch)
#     # Repeat dataset indefinitely (should be controlled by n_episodes)
#     balanced_dataset = balanced_dataset.repeat(count=-1) 
#     return balanced_dataset



def filter_p_classes(p, n_classes):

    def filter_func(ds):
        p_classes = tf.random.shuffle(tf.range(n_classes, dtype=tf.int64))[:p]
        return ds.filter(
            lambda x, y: tf.reduce_sum(
                tf.cast(tf.equal(tf.cast(y, tf.int64), p_classes),
                        tf.int64)) == 1)

    return filter_func
