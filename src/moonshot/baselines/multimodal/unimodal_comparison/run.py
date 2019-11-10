"""Test unimodal vision and speech models on Flickr one-shot multimodal task.

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: October 2019
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
flags.DEFINE_enum("speaker_mode", "baseline", ["baseline", "difficult", "distractor"],
                  "type of speakers selected in a task episode")
flags.DEFINE_bool("unseen_match_set", False, "match set contains classes unseen in K-shot learning")
flags.DEFINE_integer("seed", 42, "that magic number")

# speech features
flags.DEFINE_string("vision_base_dir", None, "directory containing base vision network model")
flags.DEFINE_string("audio_base_dir", None, "directory containing base audio network model")

# logging options
flags.DEFINE_string("output_dir", None, "directory where logs will be stored"
                    "(defaults to logs/<unique run id>)")
flags.DEFINE_bool("debug", False, "log with debug verbosity level")


def data_preprocess_func(embed_paths):
    """Data batch preprocessing function for input to the baseline model.

    Takes a batch of file paths, loads image data and preprocesses the images.
    """
    embed_ds = tf.data.TFRecordDataset(
        embed_paths, compression_type="ZLIB")
    # map sequential to prevent optimization overhead
    preprocess_func = lambda example: dataset.parse_embedding_protobuf(
        example)["embed"]
    embed_ds = embed_ds.map(preprocess_func, num_parallel_calls=8)

    return np.stack(list(embed_ds))


def test():
    """Test extracted image and speech model embeddings for one-shot learning."""

    # load embeddings from (linear) dense layer of base speech and vision models
    speech_embed_dir = os.path.join(FLAGS.audio_base_dir, "embed", "dense")
    image_embed_dir = os.path.join(FLAGS.vision_base_dir, "embed", "dense")

    # load Flickr Audio one-shot experiment
    one_shot_exp = flickr_multimodal.FlickrMultimodal(
        features="mfcc", keywords_split="one_shot_evaluation",
        flickr8k_image_dir=os.path.join("data", "external", "flickr8k_images"),
        speech_embed_dir=speech_embed_dir, image_embed_dir=image_embed_dir,
        speech_preprocess_func=data_preprocess_func,
        image_preprocess_func=data_preprocess_func,
        speaker_mode=FLAGS.speaker_mode,
        unseen_match_set=FLAGS.unseen_match_set)

    # test model on L-way K-shot task
    task_accuracy, _, conf_interval_95 = experiment.test_multimodal_l_way_k_shot(
        one_shot_exp, FLAGS.K, FLAGS.L, n=FLAGS.N, num_episodes=FLAGS.episodes,
        k_neighbours=FLAGS.k_neighbours, metric=FLAGS.metric)

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
