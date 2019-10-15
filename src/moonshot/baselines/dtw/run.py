"""TODO(rpeloff)

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: September 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os


from absl import app
from absl import flags
from absl import logging


import numpy as np
import tensorflow as tf


from moonshot.baselines import base
from moonshot.baselines.dtw import dtw
from moonshot.experiments.flickr_audio import flickr_audio


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


# required flags
# flags.mark_flag_as_required("...")


def main(argv):
    del argv  # unused

    logging.log(logging.INFO, "Logging application {}".format(__file__))

    dtw_model = dtw.DynamicTimeWarping(metric="cosine", preprocess=None)

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

    flickr_exp = flickr_audio.FlickrAudio(
        os.path.join("data", "processed", "flickr_audio", FLAGS.features),
        preprocess=preprocess_segment)

    total_correct = 0
    total_tests = 0

    for _ in range(FLAGS.episodes):

        flickr_exp.sample_episode(
            FLAGS.L, FLAGS.K, FLAGS.N, speaker_mode=FLAGS.speaker_mode)

        uids_train, x_train, y_train = flickr_exp.learning_samples
        uids_test, x_test, y_test = flickr_exp.evaluation_samples

        adapted_model = dtw_model.adapt_model(x_train, y_train)

        test_predict = adapted_model.predict(x_test, FLAGS.k_neighbours)

        num_correct = np.sum(test_predict == y_test)

        total_correct += num_correct
        total_tests += len(y_test)

    logging.log(
        logging.INFO,
        "{}-way {}-shot accuracy after {} episodes : {:.6f}".format(
            FLAGS.L, FLAGS.K, FLAGS.episodes, total_correct/total_tests))


if __name__ == "__main__":
    app.run(main)
