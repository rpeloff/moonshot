"""Test DTW and pixel matching baseline on Flickr one-shot multimodal task.

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


from moonshot.baselines import dataset
from moonshot.baselines import experiment

from moonshot.experiments.flickr_multimodal import flickr_multimodal

from moonshot.utils import file_io
from moonshot.utils import logging as logging_utils


FLAGS = flags.FLAGS


# one-shot evaluation options
flags.DEFINE_integer("episodes", 400, "number of L-way K-shot learning episodes")
flags.DEFINE_integer("L", 10, "number of classes to sample in a task episode (L-way)")
flags.DEFINE_integer("K", 1, "number of task learning samples per class (K-shot)")
flags.DEFINE_integer("N", 15, "number of task evaluation samples")
flags.DEFINE_integer("k_neighbours", 1, "number of nearest neighbours to consider")
flags.DEFINE_string("metric", "cosine", "distance metric to use for nearest neighbours matching")
flags.DEFINE_boolean("random", False, "random action baseline")
flags.DEFINE_enum("speaker_mode", "baseline", ["baseline", "difficult", "distractor"],
                  "type of speakers selected in a task episode")
flags.DEFINE_bool("unseen_match_set", False, "match set contains classes unseen in K-shot learning")
flags.DEFINE_integer("seed", 42, "that magic number")

# image preprocessing
flags.DEFINE_integer("crop_size", 256, "size to resize image square crop taken along shortest edge")

# speech features
flags.DEFINE_enum("features", "mfcc", ["mfcc", "fbank"], "type of processed speech features")
flags.DEFINE_integer("max_length", 140, "length to re-interpolate segments")
flags.DEFINE_enum("scaling", None, ["global", "features", "segment", "segment_mean"],
                  "type of feature scaling applied to speech segments")
flags.DEFINE_enum("reinterpolate", "linear", ["linear", "cubic", "quintic"],
                  "type of re-interpolation to perform in making sequences same length")

# logging options
flags.DEFINE_string("output_dir", None, "directory where logs will be stored"
                    "(defaults to logs/<unique run id>)")
flags.DEFINE_bool("debug", False, "log with debug verbosity level")


def speech_data_preprocess_func(speech_paths):
    """Data batch preprocessing function for input to the baseline model.

    Takes a batch of file paths, loads the speech features and preprocesses the
    features.
    """
    speech_features = []
    for speech_path in speech_paths:
        speech_features.append(
            dataset.load_and_preprocess_speech(
                speech_path, features=FLAGS.features,
                max_length=FLAGS.max_length,
                reinterpolate=FLAGS.reinterpolate,
                scaling=FLAGS.scaling))

    return np.stack(speech_features)


def image_data_preprocess_func(image_paths):
    """Data batch preprocessing function for input to the baseline model.

    Takes a batch of file paths, loads image data and preprocesses the images.
    """
    images = []
    for image_path in image_paths:
        images.append(
            dataset.load_and_preprocess_image(
                image_path, crop_size=FLAGS.crop_size))

    # stack and flatten image arrays
    images = np.stack(images)
    images = np.reshape(images, (images.shape[0], -1))

    return images


def test():
    """Test baseline image and speech matching model for one-shot learning."""

    # load Flickr images and speech one-shot experiment
    one_shot_exp = flickr_multimodal.FlickrMultimodal(
        features=FLAGS.features, keywords_split="one_shot_evaluation",
        flickr8k_image_dir=os.path.join("data", "external", "flickr8k_images"),
        speech_preprocess_func=speech_data_preprocess_func,
        image_preprocess_func=image_data_preprocess_func,
        speaker_mode=FLAGS.speaker_mode,
        unseen_match_set=FLAGS.unseen_match_set)

    # test model on L-way K-shot task
    task_accuracy, _, conf_interval_95 = experiment.test_multimodal_l_way_k_shot(
        one_shot_exp, FLAGS.K, FLAGS.L, n=FLAGS.N, num_episodes=FLAGS.episodes,
        k_neighbours=FLAGS.k_neighbours, metric=FLAGS.metric, speech_dtw=True,
        random=FLAGS.random)

    logging.log(
        logging.INFO,
        f"{FLAGS.L}-way {FLAGS.K}-shot accuracy after {FLAGS.episodes} "
        f"episodes: {task_accuracy:.3%} +- {conf_interval_95*100:.4f}")


def main(argv):
    del argv  # unused

    logging.log(logging.INFO, "Logging application {}".format(__file__))
    if FLAGS.debug:
        logging.set_verbosity(logging.DEBUG)
        logging.log(logging.DEBUG, "Running in debug mode")

    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # create output directory if none specified
    if FLAGS.output_dir is None:
        output_dir = os.path.join(
            "logs", __file__, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        file_io.check_create_dir(output_dir)

    # output directory specified, load model options if found
    else:
        output_dir = FLAGS.output_dir

    # print flag options
    flag_options = {}
    for flag in FLAGS.get_key_flags_for_module(__file__):
        flag_options[flag.name] = flag.value

    # logging
    logging_utils.absl_file_logger(output_dir, f"log.test")

    logging.log(logging.INFO, f"Model directory: {output_dir}")
    logging.log(logging.INFO, f"Flag options: {flag_options}")

    # set seeds for reproducibility
    np.random.seed(FLAGS.seed)
    tf.random.set_seed(FLAGS.seed)

    # test baseline matching model (no background training step)
    test()


if __name__ == "__main__":
    app.run(main)






# def main(argv):
#     del argv  # unused

#     logging.log(logging.INFO, "Logging application {}".format(__file__))
#     if FLAGS.debug:
#         logging.set_verbosity(logging.DEBUG)
#         logging.log(logging.DEBUG, "Running in debug mode")

#     tf_logdir = os.path.join(
#         "logs", __file__, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#     tf_writer = tf.summary.create_file_writer(tf_logdir)

#     np.random.seed(42)
#     tf.random.set_seed(42)

#     import io
#     import matplotlib.pyplot as plt
#     import matplotlib.patches as patches
#     def figure_to_image(figure):
#         """Convert the matplotlib figure to a PNG image.

#         From https://www.tensorflow.org/tensorboard/image_summaries#logging_arbitrary_image_data.
#         """
#         # save the plot to a PNG in memory
#         buffer = io.BytesIO()
#         plt.savefig(buffer, format="png")
#         plt.close(figure)
#         buffer.seek(0)
#         # convert PNG buffer to TensorFlow image and add batch dimension
#         image = tf.image.decode_png(buffer.getvalue(), channels=4)
#         image = tf.expand_dims(image, 0)
#         return image

#     def normalise_image(image):
#         image_new = image - np.min(image)
#         image_new /= np.max(image_new)
#         return image_new

#     def plot_images(images, labels, image_preprocess=None, cols=5,
#                     interpolation="lanczos", highlight=None):
#         rows = np.ceil(len(images) / cols)
#         figure = plt.figure(figsize=(cols * 2, rows * 2))

#         for idx, image in enumerate(images):
#             if image_preprocess is not None:
#                 image = image_preprocess(image)

#             plt.subplot(rows, cols, idx + 1,
#                         title="{}: {}".format(idx, labels[idx]))
#             plt.xticks([])
#             plt.yticks([])
#             plt.grid(False)
#             plt.imshow(image, interpolation=interpolation)

#             if highlight is not None and idx in highlight:
#                 rect = patches.Rectangle(
#                     (-2.5, -2.5), image.shape[1] + 2.5, image.shape[0] + 2.5,
#                     linewidth=5, edgecolor="m", facecolor="none")
#                 plt.gca().add_patch(rect)

#         plt.tight_layout()
#         return figure

#     dtw_model = dtw.DynamicTimeWarping(metric="cosine", preprocess=None)
#     pixel_model = pixel_match.PixelMatch(metric="cosine", preprocess=None)

#     def preprocess_segment(x):
#         x_p = x.astype(np.float32)
#         if FLAGS.scaling == "global":
#             x_p -= flickr_audio.train_global_mean[FLAGS.features]
#             x_p /= np.sqrt(flickr_audio.train_global_var[FLAGS.features])
#         elif FLAGS.scaling == "features":
#             x_p -= flickr_audio.train_features_mean[FLAGS.features]
#             x_p /= np.sqrt(flickr_audio.train_features_var[FLAGS.features])
#         elif FLAGS.scaling == "segment":
#             x_p = (x_p - np.mean(x_p))/np.sqrt(np.var(x_p))
#         elif FLAGS.scaling == "segment_mean":
#             x_p = x_p - np.mean(x_p)
#         return dtw.dtw_reinterp2d(x_p, FLAGS.max_length)

#     flickr_exp = flickr_multimodal.FlickrMultimodal(
#         os.path.join("data", "processed", "flickr_audio", FLAGS.features),
#         os.path.join("data", "external", "flickr8k_images"),
#         preprocess_audio=preprocess_segment)

#     total_correct = 0
#     total_tests = 0

#     total_audio_correct = 0
#     total_audio_tests = 0

#     total_vision_correct = 0
#     total_vision_tests = 0

#     for episode in range(FLAGS.episodes):

#         flickr_exp.sample_episode(
#             FLAGS.L, FLAGS.K, FLAGS.N, speaker_mode=FLAGS.speaker_mode)

#         train_uids, train_audio, train_paths, y_train = flickr_exp.learning_samples
#         query_uids, query_audio, query_paths, y_query = flickr_exp.evaluation_samples[:4]
#         match_uids, match_audio, match_paths, y_match = flickr_exp.evaluation_samples[4:]

#         train_images = process_image_batch(
#             train_paths, augment_crops=False, normalise=True)
#         query_images = process_image_batch(
#             query_paths, augment_crops=False, normalise=True)

#         train_images_flat = np.stack(train_images)
#         train_images_flat = np.reshape(
#             train_images_flat, (train_images_flat.shape[0], -1))

#         adapted_dtw_model = dtw_model.adapt_model(
#             train_audio, np.arange(len(train_audio)))

#         query_predict_idx = adapted_dtw_model.predict(query_audio)
#         if FLAGS.random:
#             query_predict_idx = flickr_exp.rng.choice(
#                 len(train_audio), size=len(query_audio), replace=True)
#         query_predict_images = train_images_flat[query_predict_idx]

#         total_audio_correct += np.sum(
#             y_train[query_predict_idx] == y_query)
#         total_audio_tests += len(y_query)

#         for test_idx, query_predict_image in enumerate(query_predict_images):

#             match_images = process_image_batch(
#                 match_paths[test_idx], augment_crops=False, normalise=True)

#             match_images_flat = np.stack(match_images)
#             match_images_flat = np.reshape(
#                 match_images_flat, (match_images_flat.shape[0], -1))

#             adapted_pixel_model = pixel_model.adapt_model(
#                 match_images_flat, np.arange(len(match_images_flat)))

#             match_predict_idx = adapted_pixel_model.predict(
#                 np.array([query_predict_image]))
#             if FLAGS.random:
#                 match_predict_idx = flickr_exp.rng.choice(
#                     len(match_images_flat), size=1)

#             total_correct += int(
#                 y_match[test_idx][match_predict_idx] == y_query[test_idx])
#             total_tests += 1

#             total_vision_correct += int(
#                 y_match[test_idx][match_predict_idx] == y_train[query_predict_idx][test_idx])
#             total_vision_tests += 1

#             with tf_writer.as_default():

#                 tf.summary.scalar(
#                     "{}-way {}-shot accuracy".format(FLAGS.L, FLAGS.K),
#                     total_correct/total_tests, step=episode)
#                 tf.summary.scalar(
#                     "Auxiliary audio accuracy",
#                     total_audio_correct/total_audio_tests, step=episode)
#                 tf.summary.scalar(
#                     "Auxiliary vision accuracy",
#                     total_vision_correct/total_vision_tests, step=episode)

#                 if episode % 25 == 0 and test_idx == 0:

#                     train_image_fig = plot_images(
#                         train_images, flickr_exp.one_shot_keywords[y_train],
#                         image_preprocess=normalise_image,
#                         highlight=query_predict_idx[:1])
#                     tf.summary.image(
#                         "train images", figure_to_image(train_image_fig), step=episode)

#                     train_audio_fig = plot_images(
#                         train_audio, flickr_exp.one_shot_keywords[y_train],
#                         image_preprocess=lambda img: normalise_image(img).T,
#                         highlight=query_predict_idx[:1])
#                     tf.summary.image(
#                         "train audio", figure_to_image(train_audio_fig), step=episode)

#                     query_audio_fig = plot_images(
#                         query_audio[:1], flickr_exp.one_shot_keywords[y_query][:1],
#                         image_preprocess=lambda img: normalise_image(img).T)
#                     tf.summary.image(
#                         "query audio", figure_to_image(query_audio_fig), step=episode)

#                     match_image_fig = plot_images(
#                         match_images, flickr_exp.one_shot_keywords[y_match[test_idx]],
#                         image_preprocess=normalise_image,
#                         highlight=match_predict_idx)
#                     tf.summary.image(
#                         "match images", figure_to_image(match_image_fig), step=episode)

#     assert total_tests == FLAGS.episodes * FLAGS.N

#     logging.log(
#         logging.INFO,
#         "{}-way {}-shot accuracy after {} episodes : {:.6f}".format(
#             FLAGS.L, FLAGS.K, FLAGS.episodes, total_correct/total_tests))

#     import pdb; pdb.set_trace()


# if __name__ == "__main__":
#     app.run(main)
