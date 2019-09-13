"""

Debug: python3 -m pdb src/moonshot/features/filter.py --debug

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


from moonshot.data.datasets import flickr8k
from moonshot.data.datasets import flickraudio
from moonshot.features import keywords


FLAGS = flags.FLAGS
flags.DEFINE_string("spacy_model", "en_core_web_lg", "spaCy language model")
flags.DEFINE_bool("debug", False, "debug mode")


# required flags
# flags.mark_flag_as_required("spacy_model")


def main(argv):
    del argv  # unused

    logging.log(logging.INFO, "Logging application {}".format(__file__))
    if FLAGS.debug:
        logging.set_verbosity(logging.DEBUG)
        logging.log(logging.DEBUG, "Running in debug mode")

    train_captions, _, _ = flickr8k.load_flickr8k_captions(
        os.path.join("data", "external", "flickr8k_text"))

    word_list_dict = flickraudio.fetch_isolated_word_lists(
        os.path.join("data", "processed", "flickr_audio", "mfcc"))

    train_word_data = list(map(
        flickraudio.extract_uid_metadata, word_list_dict["train"]))

    train_keywords = keywords.filter_caption_keywords(
        train_captions, spacy_model=FLAGS.spacy_model)

    keywords.filter_flickr_audio_by_keywords(train_word_data, train_keywords)

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    app.run(main)
