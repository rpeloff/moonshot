"""Test simple speech dynamic time warping baseline on Flickr one-shot speech task.

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: September 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import datetime
import os


from absl import app
from absl import flags
from absl import logging


import numpy as np
import tensorflow as tf


from moonshot.baselines import dataset
from moonshot.baselines import experiment

from moonshot.experiments.flickr_speech import flickr_speech

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

# model testing options
flags.DEFINE_integer("seed", 42, "that magic number")

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


def data_preprocess_func(speech_paths):
    """Data batch preprocessing function for input to the speech network.

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


def test():
    """Test baseline speech DTW matching model for one-shot learning."""

    # load Flickr Audio one-shot experiment
    one_shot_exp = flickr_speech.FlickrSpeech(
        features=FLAGS.features,
        keywords_split="one_shot_evaluation", embed_dir=None,
        speaker_mode=FLAGS.speaker_mode)

    # test model on L-way K-shot task
    task_accuracy, _, conf_interval_95 = experiment.test_l_way_k_shot(
        one_shot_exp, FLAGS.K, FLAGS.L, n=FLAGS.N, num_episodes=FLAGS.episodes,
        k_neighbours=FLAGS.k_neighbours, metric=FLAGS.metric, dtw=True,
        random=FLAGS.random, data_preprocess_func=data_preprocess_func)

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
