"""Select Flickr 8k keyword-image pairs for one-shot learning and background training.

Write one-shot data and splits to data/splits/flickr_one_shot:
`python3 src/moonshot/features/make_flickr8k_one_shot.py --mode write`

Debug:
`python3 -m pdb src/moonshot/features/make_flickr8k_one_shot.py --debug --mode write`

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


from moonshot.data.datasets import flickr8k
from moonshot.data.datasets import flickraudio
from moonshot.features import keywords
from moonshot.utils import file_io


FLAGS = flags.FLAGS
flags.DEFINE_string("spacy_model", "en_core_web_lg", "spaCy language model")
flags.DEFINE_integer("min_captions", 2, "minimum number of image captions for keyword quality",
                     lower_bound=1, upper_bound=5)
flags.DEFINE_integer("min_occurence", 20, "minimum number of unique images per keyword",
                     lower_bound=1)
flags.DEFINE_integer("one_shot_classes", 50, "number of one-shot keyword classes to random sample",
                     lower_bound=1)
flags.DEFINE_enum("save_images", None, ["one_shot", "background", "both"],
                  "save a number of example images per keyword from specified sets")
flags.DEFINE_integer("num_keywords", None, "number of keywords to sample for saving example images",
                     lower_bound=1)
flags.DEFINE_enum("mode", None, ["write", "statistics", "both"],
                  "target: write keyword sets and/or display keyword statistics")
flags.DEFINE_bool("debug", False, "debug mode")


# required flags
flags.mark_flag_as_required("mode")


def main(argv):
    del argv  # unused

    logging.log(logging.INFO, "Logging application {}".format(__file__))
    if FLAGS.debug:
        logging.set_verbosity(logging.DEBUG)
        logging.log(logging.DEBUG, "Running in debug mode")

    # get flickr 8k train captions
    caption_corpus = flickr8k.load_flickr8k_captions(
        os.path.join("data", "external", "flickr8k_text"),
        splits_dir=os.path.join("data", "splits", "flickr8k"))
    train_captions = caption_corpus[0]

    # get flickr-audio train data
    faudio_uid_dict = flickraudio.fetch_isolated_word_lists(
        os.path.join("data", "processed", "flickr_audio", "mfcc"))
    train_faudio = flickraudio.extract_all_uid_metadata(
        faudio_uid_dict["train"])

    # flickr 8k caption keyword filtering
    # ===================================

    # 1. identify and lemmatize keywords with a language model
    train_caption_keywords = keywords.process_caption_keywords(
        train_captions, spacy_model=FLAGS.spacy_model)

    # 2. filter quality of keyword-image pairs
    train_caption_keywords = keywords.filter_keyword_quality(
        train_caption_keywords, min_caption_occurence=FLAGS.min_captions)

    # 3. remove long tail, keeping minimum required keyword-images for one-shot
    keep_keywords = keywords.get_count_limited_keywords(
        train_caption_keywords, min_occurence=FLAGS.min_occurence, use_lemma=True)
    train_caption_keywords = keywords.filter_keep_keywords(
        train_caption_keywords, keep_keywords)

    # select random classes for one-shot learning
    np.random.seed(42)
    unique_keywords, keyword_counts = keywords.get_unique_keywords_counts(
        train_caption_keywords)
    rand_idx = np.random.choice(
        np.arange(len(unique_keywords)), FLAGS.one_shot_classes, replace=False)

    one_shot_keyword_set = unique_keywords[rand_idx]

    train_keywords_one_shot = keywords.filter_keep_keywords(
        train_caption_keywords, one_shot_keyword_set)

    # remove one-shot keywords for background learning
    train_keywords_background = keywords.filter_remove_keywords(
        train_caption_keywords, one_shot_keyword_set)

    # remove one-shot images for background learning
    train_keywords_background = keywords.filter_remove_images(
        train_keywords_background,
        np.unique(train_keywords_one_shot[0]))

    # align flickr audio spoken word image pairs for one-shot/background learning
    # by removing pairs that do not correspond to a keyword-image pair
    train_faudio_one_shot, train_keywords_one_shot = keywords.filter_flickr_audio_by_keywords(
        train_faudio, train_keywords_one_shot)

    train_faudio_background, train_keywords_background = keywords.filter_flickr_audio_by_keywords(
        train_faudio, train_keywords_background)

    # write keyword set splits to data directory
    if FLAGS.mode == "write" or FLAGS.mode == "both":
        file_io.write_csv(
            os.path.join("data", "splits", "flickr_one_shot", "test_one_shot.csv"),
            *train_keywords_one_shot,
            column_names=["image_uid", "caption_number", "keyword", "lemma"])
        file_io.write_csv(
            os.path.join("data", "splits", "flickr_one_shot", "train_background.csv"),
            *train_keywords_background,
            column_names=["image_uid", "caption_number", "keyword", "lemma"])
        file_io.write_csv(
            os.path.join("data", "splits", "flickr_one_shot", "one_shot_keywords.txt"),
            one_shot_keyword_set)

    # output keywords stats and distribution plots
    if FLAGS.mode == "statistics" or FLAGS.mode == "both":
        pass  # TODO(rpeloff)

    # save example one-shot test images if specified
    if FLAGS.save_images == "one_shot" or FLAGS.save_images == "both":
        save_keywords = np.asarray(one_shot_keyword_set)
        save_keyword_idx = np.random.choice(
            np.arange(len(save_keywords)), FLAGS.num_keywords, replace=False)
        save_keywords = save_keywords[save_keyword_idx]

        keywords.save_keyword_images(
            train_keywords_one_shot,
            os.path.join("data", "external", "flickr8k_images"), save_keywords,
            os.path.join("figures", "flickr8k", "one_shot_keywords"),
            max_per_row=5, max_images=20)

    # save example one-shot background images if specified
    if FLAGS.save_images == "background" or FLAGS.save_images == "both":
        save_keywords = np.delete(unique_keywords, rand_idx)
        save_keyword_idx = np.random.choice(
            np.arange(len(save_keywords)), FLAGS.num_keywords, replace=False)
        save_keywords = save_keywords[save_keyword_idx]

        keywords.save_keyword_images(
            train_keywords_background,
            os.path.join("data", "external", "flickr8k_images"), save_keywords,
            os.path.join("figures", "flickr8k", "background_keywords"),
            max_per_row=5, max_images=20)


if __name__ == "__main__":
    app.run(main)
