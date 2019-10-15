"""TODO(rpeloff)

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: September 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import datetime


from absl import app
from absl import flags
from absl import logging


import numpy as np
import tensorflow as tf


from moonshot.baselines import base
from moonshot.baselines.dtw import dtw
from moonshot.baselines.pixel_match import pixel_match
from moonshot.experiments.flickr_audio import flickr_audio
from moonshot.experiments.flickr_multimodal import flickr_multimodal


# training
# - initialise model
# - train (and validate) model on background data
# (compare to model validated on one-shot background experiment)

# one-shot evaluation
# - initialise trained model
# - load an experiment
# - evaluate model on experiment


FLAGS = flags.FLAGS
flags.DEFINE_integer("episodes", 400, "number of L-way K-shot learning episodes")
flags.DEFINE_integer("L", 5, "number of classes to sample in a task episode (L-way)")
flags.DEFINE_integer("K", 1, "number of task learning samples per class (K-shot)")
flags.DEFINE_integer("N", 15, "number of task evaluation samples")
flags.DEFINE_integer("k_neighbours", 1, "number of nearest neighbours to consider")
flags.DEFINE_enum("speaker_mode", "baseline", ["baseline", "difficult", "distractor"],
                  "type of speakers selected in a task episode")
flags.DEFINE_integer("max_length", 130, "length ro reinterpolate/crop segments")
flags.DEFINE_enum("scaling", None, ["global", "features", "segment", "segment_mean"],
                  "type of feature scaling applied to speech segments")
flags.DEFINE_enum("features", "mfcc", ["mfcc", "fbank"], "type of processed speech features")
flags.DEFINE_boolean("random", False, "random action baseline")
flags.DEFINE_boolean("debug", False, "debug mode")

# flags.DEFINE_boolean("augment_crops", False, "train and test on random scale augmented crops")
# flags.DEFINE_integer("num_crops", 10, "number of random augmented crops per image")
# flags.DEFINE_list("test_scales", [224, 256, 384, 480, 640],
#                   "test images will be random resized to have short edge in this set")


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
    image = image - tf.reduce_min(image)
    image = image / tf.reduce_max(image)
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
    # scale image to range [0, 1] expected by tf.image functions
    image = tf.cast(image, tf.float32) / 255.
    # random crop images for testing
    if augment_crops:
        crop_images = []
        for _ in range(num_crops):
            crop = augment_square_crop(image, **crop_kwargs)
            crop_images.append(crop)
        image = tf.stack(crop_images)
    else:
        image = resize_square_crop(image, **crop_kwargs)
    # scale image from [0, 1] to range [-1, 1]
    if normalise:
        image *= 2.
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
    if FLAGS.debug:
        logging.set_verbosity(logging.DEBUG)
        logging.log(logging.DEBUG, "Running in debug mode")

    tf_logdir = os.path.join(
        "logs", __file__, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tf_writer = tf.summary.create_file_writer(tf_logdir)

    np.random.seed(42)
    tf.random.set_seed(42)

    import io
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    def figure_to_image(figure):
        """Convert the matplotlib figure to a PNG image.

        From https://www.tensorflow.org/tensorboard/image_summaries#logging_arbitrary_image_data.
        """
        # save the plot to a PNG in memory
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        plt.close(figure)
        buffer.seek(0)
        # convert PNG buffer to TensorFlow image and add batch dimension
        image = tf.image.decode_png(buffer.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        return image

    def normalise_image(image):
        image_new = image - np.min(image)
        image_new /= np.max(image_new)
        return image_new

    def plot_images(images, labels, image_preprocess=None, cols=5,
                    interpolation="lanczos", highlight=None):
        rows = np.ceil(len(images) / cols)
        figure = plt.figure(figsize=(cols * 2, rows * 2))

        for idx, image in enumerate(images):
            if image_preprocess is not None:
                image = image_preprocess(image)

            plt.subplot(rows, cols, idx + 1,
                        title="{}: {}".format(idx, labels[idx]))
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(image, interpolation=interpolation)

            if highlight is not None and idx in highlight:
                rect = patches.Rectangle(
                    (-2.5, -2.5), image.shape[1] + 2.5, image.shape[0] + 2.5,
                    linewidth=5, edgecolor="m", facecolor="none")
                plt.gca().add_patch(rect)

        plt.tight_layout()
        return figure

    dtw_model = dtw.DynamicTimeWarping(metric="cosine", preprocess=None)
    pixel_model = pixel_match.PixelMatch(metric="cosine", preprocess=None)

    def preprocess_segment(x):
        x_p = x.astype(np.float32)
        if FLAGS.scaling == "global":
            x_p -= flickr_audio.train_global_mean[FLAGS.features]
            x_p /= np.sqrt(flickr_audio.train_global_var[FLAGS.features])
        elif FLAGS.scaling == "features":
            x_p -= flickr_audio.train_features_mean[FLAGS.features]
            x_p /= np.sqrt(flickr_audio.train_features_var[FLAGS.features])
        elif FLAGS.scaling == "segment":
            x_p = (x_p - np.mean(x_p))/np.sqrt(np.var(x_p))
        elif FLAGS.scaling == "segment_mean":
            x_p = x_p - np.mean(x_p)
        return dtw.dtw_reinterp2d(x_p, FLAGS.max_length)

    flickr_exp = flickr_multimodal.FlickrMultimodal(
        os.path.join("data", "processed", "flickr_audio", FLAGS.features),
        os.path.join("data", "external", "flickr8k_images"),
        preprocess_audio=preprocess_segment)

    total_correct = 0
    total_tests = 0

    total_audio_correct = 0
    total_audio_tests = 0

    total_vision_correct = 0
    total_vision_tests = 0

    for episode in range(FLAGS.episodes):

        flickr_exp.sample_episode(
            FLAGS.L, FLAGS.K, FLAGS.N, speaker_mode=FLAGS.speaker_mode)

        train_uids, train_audio, train_paths, y_train = flickr_exp.learning_samples
        query_uids, query_audio, query_paths, y_query = flickr_exp.evaluation_samples[:4]
        match_uids, match_audio, match_paths, y_match = flickr_exp.evaluation_samples[4:]

        train_images = process_image_batch(
            train_paths, augment_crops=False, normalise=True)
        query_images = process_image_batch(
            query_paths, augment_crops=False, normalise=True)

        train_images_flat = np.stack(train_images)
        train_images_flat = np.reshape(
            train_images_flat, (train_images_flat.shape[0], -1))

        adapted_dtw_model = dtw_model.adapt_model(
            train_audio, np.arange(len(train_audio)))

        query_predict_idx = adapted_dtw_model.predict(query_audio)
        if FLAGS.random:
            query_predict_idx = flickr_exp.rng.choice(
                len(train_audio), size=len(query_audio), replace=True)
        query_predict_images = train_images_flat[query_predict_idx]

        total_audio_correct += np.sum(
            y_train[query_predict_idx] == y_query)
        total_audio_tests += len(y_query)

        for test_idx, query_predict_image in enumerate(query_predict_images):

            match_images = process_image_batch(
                match_paths[test_idx], augment_crops=False, normalise=True)

            match_images_flat = np.stack(match_images)
            match_images_flat = np.reshape(
                match_images_flat, (match_images_flat.shape[0], -1))

            adapted_pixel_model = pixel_model.adapt_model(
                match_images_flat, np.arange(len(match_images_flat)))

            match_predict_idx = adapted_pixel_model.predict(
                np.array([query_predict_image]))
            if FLAGS.random:
                match_predict_idx = flickr_exp.rng.choice(
                    len(match_images_flat), size=1)

            total_correct += int(
                y_match[test_idx][match_predict_idx] == y_query[test_idx])
            total_tests += 1

            total_vision_correct += int(
                y_match[test_idx][match_predict_idx] == y_train[query_predict_idx][test_idx])
            total_vision_tests += 1

            with tf_writer.as_default():

                tf.summary.scalar(
                    "{}-way {}-shot accuracy".format(FLAGS.L, FLAGS.K),
                    total_correct/total_tests, step=episode)
                tf.summary.scalar(
                    "Auxiliary audio accuracy",
                    total_audio_correct/total_audio_tests, step=episode)
                tf.summary.scalar(
                    "Auxiliary vision accuracy",
                    total_vision_correct/total_vision_tests, step=episode)

                if episode % 25 == 0 and test_idx == 0:

                    train_image_fig = plot_images(
                        train_images, flickr_exp.one_shot_keywords[y_train],
                        image_preprocess=normalise_image,
                        highlight=query_predict_idx[:1])
                    tf.summary.image(
                        "train images", figure_to_image(train_image_fig), step=episode)

                    train_audio_fig = plot_images(
                        train_audio, flickr_exp.one_shot_keywords[y_train],
                        image_preprocess=lambda img: normalise_image(img).T,
                        highlight=query_predict_idx[:1])
                    tf.summary.image(
                        "train audio", figure_to_image(train_audio_fig), step=episode)

                    query_audio_fig = plot_images(
                        query_audio[:1], flickr_exp.one_shot_keywords[y_query][:1],
                        image_preprocess=lambda img: normalise_image(img).T)
                    tf.summary.image(
                        "query audio", figure_to_image(query_audio_fig), step=episode)

                    match_image_fig = plot_images(
                        match_images, flickr_exp.one_shot_keywords[y_match[test_idx]],
                        image_preprocess=normalise_image,
                        highlight=match_predict_idx)
                    tf.summary.image(
                        "match images", figure_to_image(match_image_fig), step=episode)

    assert total_tests == FLAGS.episodes * FLAGS.N

    logging.log(
        logging.INFO,
        "{}-way {}-shot accuracy after {} episodes : {:.6f}".format(
            FLAGS.L, FLAGS.K, FLAGS.episodes, total_correct/total_tests))

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    app.run(main)
