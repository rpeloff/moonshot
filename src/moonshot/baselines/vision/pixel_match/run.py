"""Test simple vision pixel matching baseline on Flickr one-shot image task.

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

from moonshot.experiments.flickr_vision import flickr_vision

from moonshot.utils import file_io
from moonshot.utils import logging as logging_utils


FLAGS = flags.FLAGS


# model options
DEFAULT_OPTIONS = {
    # preprocessing
    "crop_size": 299,  # 32,
    # that magic number
    "seed": 42
}


# one-shot evaluation options
flags.DEFINE_integer("episodes", 400, "number of L-way K-shot learning episodes")
flags.DEFINE_integer("L", 10, "number of classes to sample in a task episode (L-way)")
flags.DEFINE_integer("K", 1, "number of task learning samples per class (K-shot)")
flags.DEFINE_integer("N", 15, "number of task evaluation samples")
flags.DEFINE_integer("k_neighbours", 1, "number of nearest neighbours to consider")
flags.DEFINE_string("metric", "cosine", "distance metric to use for nearest neighbours matching")
flags.DEFINE_boolean("random", False, "random action baseline")

# logging options
flags.DEFINE_string("output_dir", None, "directory where logs will be stored"
                    "(defaults to logs/<unique run id>)")
flags.DEFINE_bool("debug", False, "log with debug verbosity level")


def get_data_preprocess_func(model_options):
    """Create data batch preprocessing function for input to the baseline model.

    Returns function `data_preprocess_func` that takes a batch of file paths,
    loads image data and preprocesses the images.
    """

    def data_preprocess_func(image_paths):
        images = []
        for image_path in image_paths:
            images.append(
                dataset.load_and_preprocess_image(
                    image_path, crop_size=model_options["crop_size"]))

        # stack and flatten image arrays
        images = np.stack(images)
        images = np.reshape(images, (images.shape[0], -1))

        return images

    return data_preprocess_func


def test(model_options):
    """Test baseline image matching model for one-shot learning."""

    # load Flickr 8k one-shot experiment
    one_shot_exp = flickr_vision.FlickrVision(
        keywords_split="one_shot_evaluation",
        flickr8k_image_dir=os.path.join("data", "external", "flickr8k_images"),
        embed_dir=None)

    # get data preprocessing function
    data_preprocess_func = get_data_preprocess_func(model_options)

    # test model on L-way K-shot task
    task_accuracy, _, conf_interval_95 = experiment.test_l_way_k_shot(
        one_shot_exp, FLAGS.K, FLAGS.L, n=FLAGS.N, num_episodes=FLAGS.episodes,
        k_neighbours=FLAGS.k_neighbours, metric=FLAGS.metric,
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

    # add flag options to model options
    model_options = DEFAULT_OPTIONS

    for flag in FLAGS.get_key_flags_for_module(__file__):
        model_options[flag.name] = flag.value

    # logging
    logging_utils.absl_file_logger(output_dir, f"log.test")

    logging.log(logging.INFO, f"Model directory: {output_dir}")
    logging.log(logging.INFO, f"Model options: {model_options}")

    # set seeds for reproducibility
    np.random.seed(model_options["seed"])
    tf.random.set_seed(model_options["seed"])

    # test baseline matching model (no background training step)
    test(model_options)


if __name__ == "__main__":
    app.run(main)
