"""TODO(rpeloff)

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: September 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl import app
from absl import flags
from absl import logging


import numpy as np
import tensorflow as tf


from moonshot.baselines import base
from moonshot.baselines.pixel_match import pixel_match
from moonshot.experiments.flickr_vision import flickr_vision


# training
# - initialise model
# - train (and validate) model on background data
# (compare to model validated on one-shot background experiment)

# one-shot evaluation
# - initialise trained model
# - load an experiment
# - evaluate model on experiment


FLAGS = flags.FLAGS
flags.DEFINE_integer("episodes", 400, "number of one-shot learning episodes")
flags.DEFINE_integer("L", 5, "number of classes to sample in a task episode (L-way)")
flags.DEFINE_integer("K", 1, "number of task learning samples per class (K-shot)")
flags.DEFINE_integer("N", 15, "number of task evaluation samples")
flags.DEFINE_integer("k_neighbours", 1, "number of nearest neighbours to consider")
flags.DEFINE_boolean("augment_crops", False, "train and test on random scale augmented crops")
flags.DEFINE_integer("num_crops", 10, "number of random augmented crops per image")
flags.DEFINE_list("test_scales", [224, 256, 384, 480, 640],
                  "test images will be random resized to have short edge in this set")

# required flags
# flags.mark_flag_as_required("...")


def augment_square_crop(image, size=224, random_scales=None):
    """Augment image (scale and flip) then select a random (224, 224) square crop."""
    # get shorter side of image
    image_shape = tf.shape(image)
    h, w = image_shape[0], image_shape[1]
    short_edge = tf.minimum(w, h)
    # scale augmentation
    if random_scales is None:  # random resize along shorter edge in [256, 480]
        rand_resize = tf.random.uniform(  # maxval - minval = power of 2 => unbiased random integers
            [1], minval=256, maxval=(480+1), dtype=tf.int32)[0]
    else:  # random resize along shorter edge in random_scales
        rand_scale_idx = np.random.choice(np.arange(len(random_scales)), 1)[0]
        rand_resize = np.asarray(random_scales)[rand_scale_idx]
    resize_hw = (rand_resize * h/short_edge, rand_resize * w/short_edge)
    image = tf.image.resize(image, resize_hw, method="lanczos3")
    # horizontal flip augmentation
    image = tf.image.random_flip_left_right(image)
    # crop augmentation, random sample square (size, size, 3) from resized image
    image = tf.image.random_crop(image, size=(size, size, 3))
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
    h_shift = int((h - 224) / 2)
    w_shift = int((w - 224) / 2)
    image = tf.image.crop_to_bounding_box(image, h_shift, w_shift, 224, 224)
    return image


def load_and_preprocess_image(image_path, augment_crops=True, num_crops=10,
                              normalise=True, **crop_kwargs):
    """Load and preprocess a image at a specified path."""
    # read and decode image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    # random crop images for testing
    if augment_crops:
        crop_images = []
        for _ in range(num_crops):
            crop = augment_square_crop(image, **crop_kwargs)
            crop_images.append(crop)
        image = tf.stack(crop_images)
    else:
        image = resize_square_crop(image, **crop_kwargs)
    # scale pixels between -1 and 1 sample-wise
    if normalise:
        image /= 127.5
        image -= 1.
    return image


def process_image_batch(image_paths, *args, **kwargs):
    """Apply `load_and_preprocess_image` to a batch of image paths."""
    return list(
        map(lambda img: load_and_preprocess_image(img, *args, **kwargs).numpy(),
            image_paths))


def main(argv):
    del argv  # unused

    logging.log(logging.INFO, "Logging application {}".format(__file__))

    pixel_model = pixel_match.PixelMatch(metric="cosine", preprocess=None)

    flickr_exp = flickr_vision.FlickrVision("data/external/flickr8k_images")

    total_correct = 0
    total_tests = 0

    if FLAGS.augment_crops:  # learn and test on a multiple random square crops
        for _ in range(FLAGS.episodes):

            flickr_exp.sample_episode(FLAGS.L, FLAGS.K, FLAGS.N)

            train_paths, y_train = flickr_exp.learning_samples
            test_paths, y_test = flickr_exp.evaluation_samples

            x_train_images = process_image_batch(
                train_paths, augment_crops=True, num_crops=FLAGS.num_crops,
                normalise=True, random_scales=None)

            x_test_images = process_image_batch(
                test_paths, augment_crops=True, num_crops=FLAGS.num_crops,
                normalise=True, random_scales=FLAGS.test_scales)

            y_train = np.concatenate(
                list([label] * FLAGS.num_crops for label in y_train))

            x_train_images = np.concatenate(x_train_images)
            x_train_images = np.reshape(x_train_images, (x_train_images.shape[0], -1))

            x_test_images = np.stack(x_test_images)
            x_test_images = np.reshape(
                x_test_images, (x_test_images.shape[0], x_test_images.shape[1], -1))

            adapted_model = pixel_model.adapt_model(x_train_images, y_train)

            test_predict = list(map(
                lambda image_crops: adapted_model.predict(
                    image_crops, k_neighbours=FLAGS.k_neighbours),
                x_test_images))

            # average over predictions for test image square crops
            test_predict = np.apply_along_axis(
                base.majority_vote, 1, test_predict)

            num_correct = np.sum(test_predict == y_test)

            total_correct += num_correct
            total_tests += len(y_test)

    else:  # learn and test on a single square crop and resize along short edges
        for _ in range(FLAGS.episodes):

            flickr_exp.sample_episode(FLAGS.L, FLAGS.K, FLAGS.N)

            train_paths, y_train = flickr_exp.learning_samples
            test_paths, y_test = flickr_exp.evaluation_samples

            x_train_images = process_image_batch(
                train_paths, augment_crops=False, normalise=True)

            x_test_images = process_image_batch(
                test_paths, augment_crops=False, normalise=True)

            x_train_images = np.stack(x_train_images)
            x_train_images = np.reshape(x_train_images, (x_train_images.shape[0], -1))

            x_test_images = np.stack(x_test_images)
            x_test_images = np.reshape(x_test_images, (x_test_images.shape[0], -1))

            adapted_model = pixel_model.adapt_model(x_train_images, y_train)

            test_predict = adapted_model.predict(x_test_images, FLAGS.k_neighbours)

            num_correct = np.sum(test_predict == y_test)

            total_correct += num_correct
            total_tests += len(y_test)

    logging.log(
        logging.INFO,
        "{}-way {}-shot accuracy after {} episodes : {:.6f}".format(
            FLAGS.L, FLAGS.K, FLAGS.episodes, total_correct/total_tests))


if __name__ == "__main__":
    app.run(main)
