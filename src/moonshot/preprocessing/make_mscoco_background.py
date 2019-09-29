"""Select MSCOCO keyword-image pairs for one-shot background training.

Write background split to data/splits/mscoco_one_shot:
`python3 src/moonshot/features/make_mscoco_background.py --mode write`

Debug:
`python3 -m pdb src/moonshot/features/make_mscoco_background.py --debug --mode write`

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: September 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import copy
import os


from absl import app
from absl import flags
from absl import logging


import numpy as np


from moonshot.data import mscoco
from moonshot.preprocessing import keywords
from moonshot.utils import file_io


FLAGS = flags.FLAGS
flags.DEFINE_string("spacy_model", "en_core_web_lg", "spaCy language model")
flags.DEFINE_integer("min_captions", 2, "minimum number of image captions for keyword quality",
                     lower_bound=1, upper_bound=5)
flags.DEFINE_integer("min_occurence", 20, "minimum number of unique images per keyword",
                     lower_bound=1)
flags.DEFINE_float("similarity", 0.85, "threshold for finding semantically similar keywords")
flags.DEFINE_enum("save_images", None, ["one_shot", "background", "both"],
                  "save a number of example images per keyword for specified sets")
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

    # get mscoco text captions with flickr 30k removed
    subsets = ["train", "dev"]  # test has no captions (AFAIK)
    caption_files = [
        "captions_train2017.json",
        "captions_val2017.json"]

    caption_corpus = {}
    for subset, caption_file in zip(subsets, caption_files):
        caption_corpus[subset] = mscoco.load_mscoco_captions(
            os.path.join("data", "external", "mscoco", "annotations"),
            caption_file=caption_file,
            remove_flickr_path=os.path.join(
                "data", "splits", "mscoco", "remove_flickr30k.txt"))

    # mscoco caption keyword filtering
    # ================================

    caption_keywords = {}
    for subset in subsets:
        # 1. identify and lemmatize keywords with a language model
        caption_keywords[subset] = keywords.process_caption_keywords(
            caption_corpus[subset], spacy_model=FLAGS.spacy_model)

        # 2. filter quality of keyword-image pairs
        caption_keywords[subset] = keywords.filter_keyword_quality(
            caption_keywords[subset], min_caption_occurence=FLAGS.min_captions)

    # mscoco one-shot benchmark evaluation data selection
    # ===================================================

    # load one-shot keywords selected from flickr 8k keyword-image pairs
    one_shot_keyword_set = file_io.read_csv(
        os.path.join("data", "splits", "flickr8k", "one_shot_keywords.txt"))[0]

    one_shot_caption_keywords = keywords.filter_keep_keywords(
        caption_keywords["train"], one_shot_keyword_set)

    missing_set = set(one_shot_keyword_set) - set(one_shot_caption_keywords[3])
    if len(missing_set) > 0:
        logging.log(
            logging.INFO, "MSCOCO is missing one-shot keywords: {}".format(
                missing_set))

    # mscoco background data selection and filtering
    # ==============================================

    background_caption_keywords = {}
    background_caption_keywords_full = {}
    for subset in subsets:
        # get list of one-shot keywords and semantically similar keywords
        one_shot_remove_words = keywords.find_similar_keywords(
            caption_keywords[subset], one_shot_keyword_set,
            threshold=FLAGS.similarity, use_lemma=True, spacy_model=FLAGS.spacy_model)

        # remove one-shot keywords and associated images from filtered keywords
        background_caption_keywords[subset] = keywords.filter_remove_keyword_images(
            caption_keywords[subset], one_shot_remove_words)

        # store full version of background data before removing long tail
        background_caption_keywords_full[subset] = copy.deepcopy(
            background_caption_keywords[subset])

        # remove long tail of infrequent keywords by keeping only minimum required
        # keyword-images for one-shot background training (e.g. with meta-learning)
        keep_keywords = keywords.get_count_limited_keywords(
            background_caption_keywords[subset],
            min_occurence=FLAGS.min_occurence, use_lemma=True)
        background_caption_keywords[subset] = keywords.filter_keep_keywords(
            background_caption_keywords[subset], keep_keywords)

    # write one-shot evaluation and background keyword-image set splits
    # =================================================================

    # write keyword set splits to data directory
    if FLAGS.mode == "write" or FLAGS.mode == "both":
        file_io.write_csv(  # one-shot evaluation benchmark split
            os.path.join("data", "splits", "mscoco", "one_shot_evaluation.csv"),
            *one_shot_caption_keywords,
            column_names=["image_uid", "caption_number", "keyword", "lemma"])

        for subset in subsets:
            file_io.write_csv(  # background subset split, one-shot data removed
                os.path.join(
                    "data", "splits", "mscoco", "background_{}.csv".format(subset)),
                *background_caption_keywords[subset],
                column_names=["image_uid", "caption_number", "keyword", "lemma"])

            file_io.write_csv(  # background subset split (with tail), one-shot data removed
                os.path.join(
                    "data", "splits", "mscoco", "background_full_{}.csv".format(subset)),
                *background_caption_keywords_full[subset],
                column_names=["image_uid", "caption_number", "keyword", "lemma"])

    # output keywords stats and .. TODO(rpeloff) distribution plots
    if FLAGS.mode == "statistics" or FLAGS.mode == "both":
        keywords.log_keyword_stats(one_shot_caption_keywords, "one_shot_evaluation")
        for subset in subsets:
            keywords.log_keyword_stats(
                background_caption_keywords[subset], "background_{}".format(subset))
            keywords.log_keyword_stats(
                background_caption_keywords_full[subset], "background_full_{}".format(subset))

    # save example one-shot evaluation images if specified
    if FLAGS.save_images == "one_shot" or FLAGS.save_images == "both":
        save_keywords = np.asarray(one_shot_keyword_set)

        if FLAGS.num_keywords is not None:
            save_keyword_idx = np.random.choice(
                np.arange(len(save_keywords)), FLAGS.num_keywords, replace=False)
            save_keywords = save_keywords[save_keyword_idx]

        keywords.save_keyword_images(
            one_shot_caption_keywords,
            os.path.join("data", "external", "mscoco", "train2017"), save_keywords,
            os.path.join("figures", "mscoco", "one_shot_keywords"),
            max_per_row=5, max_images=20)

    # save example one-shot background images if specified
    if FLAGS.save_images == "background" or FLAGS.save_images == "both":
        save_keywords = np.unique(background_caption_keywords["train"][3])

        if FLAGS.num_keywords is not None:
            save_keyword_idx = np.random.choice(
                np.arange(len(save_keywords)), FLAGS.num_keywords, replace=False)
            save_keywords = save_keywords[save_keyword_idx]

        keywords.save_keyword_images(
            background_caption_keywords["train"],
            os.path.join("data", "external", "mscoco", "train2017"), save_keywords,
            os.path.join("figures", "mscoco", "background_keywords"),
            max_per_row=5, max_images=20)


if __name__ == "__main__":
    app.run(main)
