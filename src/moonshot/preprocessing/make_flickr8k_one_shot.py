"""Select Flickr 8k keyword-image pairs for one-shot learning and background training.

Write one-shot data and splits to data/splits/flickr_one_shot:
`python3 src/moonshot/preprocessing/make_flickr8k_one_shot.py --mode write`

Debug:
`python3 -m pdb src/moonshot/preprocessing/make_flickr8k_one_shot.py --debug --mode write`

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


from moonshot.data import flickr8k
from moonshot.data import flickraudio
from moonshot.preprocessing import keywords
from moonshot.utils import file_io


FLAGS = flags.FLAGS
flags.DEFINE_string("spacy_model", "en_core_web_lg", "spaCy language model")
flags.DEFINE_integer("min_captions", 2, "minimum number of image captions for keyword quality",
                     lower_bound=1, upper_bound=5)
flags.DEFINE_integer("min_occurence", 20, "minimum number of unique images per keyword",
                     lower_bound=1)
flags.DEFINE_integer("one_shot_classes", 50, "number of one-shot keyword classes to random sample",
                     lower_bound=1)
flags.DEFINE_float("similarity", 0.85, "threshold for finding semantically similar keywords")
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

    # get flickr 8k text captions
    caption_corpus = flickr8k.load_flickr8k_captions(
        os.path.join("data", "external", "flickr8k_text"),
        splits_dir=os.path.join("data", "splits", "flickr8k"))

    subsets = ["train", "dev", "test"]
    caption_corpus = {
        subset: caption_set for (subset, caption_set)
        in zip(subsets, caption_corpus)}

    # get flickr-audio data
    faudio_uid_dict = flickraudio.fetch_isolated_word_lists(
        os.path.join("data", "processed", "flickr_audio", "mfcc"))

    faudio_data = {}
    for subset in subsets:
        faudio_data[subset] = flickraudio.extract_all_uid_metadata(
            faudio_uid_dict[subset])

    # flickr 8k caption keyword filtering
    # ===================================

    caption_keywords = {}
    for subset in subsets:
        # 1. identify and lemmatize keywords with a language model
        caption_keywords[subset] = keywords.process_caption_keywords(
            caption_corpus[subset], spacy_model=FLAGS.spacy_model)

        # 2. filter quality of keyword-image pairs
        caption_keywords[subset] = keywords.filter_keyword_quality(
            caption_keywords[subset], min_caption_occurence=FLAGS.min_captions)

    # flickr 8k one-shot benchmark evaluation data selection
    # ======================================================

    # select random classes for one-shot learning evaluation benchmark
    # from keywords paired with at least 20 and no more than 100 unique images

    # minimum required keyword-images for one-shot learning and evaluation
    keep_min_keywords = keywords.get_count_limited_keywords(
        caption_keywords["train"], min_occurence=20, use_lemma=True)

    # limit the effect on background data, specifically flickr-audio words
    remove_max_keywords = keywords.get_count_limited_keywords(
        caption_keywords["train"], min_occurence=100, use_lemma=True)

    one_shot_keyword_range = np.asarray(  # sort because of undefined set order
        list(sorted(set(keep_min_keywords) - set(remove_max_keywords))))

    np.random.seed(42)
    rand_idx = np.random.choice(
        np.arange(len(one_shot_keyword_range)), FLAGS.one_shot_classes, replace=False)

    one_shot_keyword_set = one_shot_keyword_range[rand_idx]

    # select 30 of the random keywords throwing away ambigious terms
    manual_keywords = np.array([
        "asian", "basketball", "bench", "bird", "blonde", "boat", "car",
        "cliff", "climber", "dance", "fire", "floor", "ground", "guitar",
        "hair", "hill", "horse", "obstacle", "paddle", "path", "purple", "rope",
        "sand", "sled", "snowboard", "splash", "suit", "surfboard", "throw",
        "vest"])

    for keyword in manual_keywords:
        assert keyword in one_shot_keyword_set

    one_shot_keyword_set = manual_keywords

    # fetch keyword-image pairs for one-shot evaluation benchmark
    one_shot_caption_keywords = keywords.filter_keep_keywords(
        caption_keywords["train"], one_shot_keyword_set)

    # flickr 8k background data selection and filtering
    # =================================================

    background_caption_keywords = {}
    background_caption_keywords_full = {}
    for subset in subsets:
        # get list of one-shot keywords and semantically similar keywords
        one_shot_remove_words = keywords.find_similar_keywords(
            caption_keywords[subset], one_shot_keyword_set,
            threshold=FLAGS.similarity, use_lemma=True, spacy_model=FLAGS.spacy_model)
        one_shot_remove_words += one_shot_keyword_set.tolist()

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

    # flickr-audio alignment with background and one-shot evaluation data
    # ===================================================================

    # align flickr-audio spoken word image pairs for one-shot/background learning
    # by removing pairs that do not correspond to a keyword-image pair
    faudio_one_shot_data, one_shot_caption_keywords = keywords.filter_flickr_audio_by_keywords(
        faudio_data["train"], one_shot_caption_keywords)

    faudio_background_data = {}
    faudio_background_data_full = {}
    for subset in subsets:
        faudio_background_data[subset], background_caption_keywords[subset] = (
            keywords.filter_flickr_audio_by_keywords(
                faudio_data[subset], background_caption_keywords[subset]))

        faudio_background_data_full[subset], background_caption_keywords_full[subset] = (
            keywords.filter_flickr_audio_by_keywords(
                faudio_data[subset], background_caption_keywords_full[subset]))

    # write one-shot evaluation and background keyword-image set splits
    # =================================================================

    # write keyword set splits to data directory
    if FLAGS.mode == "write" or FLAGS.mode == "both":
        file_io.write_csv(  # keyword list
            os.path.join("data", "splits", "flickr8k", "one_shot_keywords.txt"),
            one_shot_keyword_set)

        file_io.write_csv(  # one-shot evaluation benchmark split
            os.path.join("data", "splits", "flickr8k", "one_shot_evaluation.csv"),
            *one_shot_caption_keywords,
            column_names=["image_uid", "caption_number", "keyword", "lemma"])

        file_io.write_csv(  # aligned flickr-audio uids for one-shot evaluation
            os.path.join("data", "splits", "flickr8k",
                         "faudio_one_shot_evaluation.txt"),
            faudio_one_shot_data[0])

        for subset in subsets:
            file_io.write_csv(  # background subset split, one-shot data removed
                os.path.join("data", "splits", "flickr8k",
                             "background_{}.csv".format(subset)),
                *background_caption_keywords[subset],
                column_names=["image_uid", "caption_number", "keyword", "lemma"])

            file_io.write_csv(  # background subset split (with tail), one-shot data removed
                os.path.join("data", "splits", "flickr8k",
                             "background_full_{}.csv".format(subset)),
                *background_caption_keywords_full[subset],
                column_names=["image_uid", "caption_number", "keyword", "lemma"])

            file_io.write_csv(  # aligned flickr-audio uids for background split
                os.path.join("data", "splits", "flickr8k",
                             "faudio_background_{}.txt".format(subset)),
                faudio_background_data[subset][0])

            file_io.write_csv(  # aligned flickr-audio uids for background split
                os.path.join("data", "splits", "flickr8k",
                             "faudio_background_full_{}.txt".format(subset)),
                faudio_background_data_full[subset][0])

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
            os.path.join("data", "external", "flickr8k_images"), save_keywords,
            os.path.join("figures", "flickr8k", "one_shot_keywords"),
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
            os.path.join("data", "external", "flickr8k_images"), save_keywords,
            os.path.join("figures", "flickr8k", "background_keywords"),
            max_per_row=5, max_images=20)


if __name__ == "__main__":
    app.run(main)
