"""Functions to identify and filter keyword-image pairs from a text caption corpus.

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: July 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import random


from absl import flags
from absl import logging


import spacy
import numpy as np
import pandas as pd


from moonshot.utils import file_io
from moonshot.utils import multiprocess as mp


FLAGS = flags.FLAGS


# tidigits homophones identified with https://www.homophone.com ... :)
TIDIGITS_STOP_WORDS = [
    "one", "won",
    "two", "to", "too",
    "three",
    "four", "for", "fore", "fourth", "forth",
    "five",
    "six", "sics",
    "seven",
    "eight", "ait", "ate",
    "nine",
    "oh", "owe", "ohs", "owes",
    "zero", "-xero-",
]


def process_caption_keywords(caption_set, spacy_model="en_core_web_lg"):
    """Process and lemmatise text captions to select keywords paired with images.

    `caption_set`: Flickr8k, Flickr30k or MSCOCO caption set.

    `spacy_model`: one of "en_core_web_sm", "en_core_web_md", "en_core_web_lg".

    Return filtered keywords with format
    (paired_images, caption_numbers, keywords, lemmatised_keywords).
    """
    logging.log(logging.INFO, "Loading spacy model: {}".format(spacy_model))

    random.seed(0)  # reproducible spacy results
    nlp = spacy.load(spacy_model)

    logging.log(logging.INFO, "Processing captions to select keywords ...")

    image_uids, keywords, keywords_lemma, caption_numbers = [], [], [], []
    texts = caption_set[1].tolist()
    for idx, doc in enumerate(nlp.pipe(texts)):  # process captions in parallel
        image_uid = caption_set[0][idx]
        caption = caption_set[1][idx]
        caption_number = caption_set[2][idx]
        _keep, _throw = [], []  # temp lists for debug
        for token in doc:
            valid_token = (
                not token.is_stop  # remove stopwords
                and not token.is_punct  # remove punctuation
                and not token.is_digit  # remove digits
                and not token.is_oov  # remove out of vocabulary (e.g. spelling errors)
                and nlp.vocab.has_vector(token.lemma_)  # make sure lemma is in vocabulary
                and not bool(sum(stop_word in token.text for stop_word in TIDIGITS_STOP_WORDS))  # remove tidigits
                and not bool(sum(stop_word in token.lemma_ for stop_word in TIDIGITS_STOP_WORDS))  # remove tidigits
            )
            if valid_token:
                keywords.append(token.text)
                keywords_lemma.append(token.lemma_)
                image_uids.append(image_uid)
                caption_numbers.append(caption_number)
                _keep.append((token.text, token.dep_, token.tag_))
            else:
                _throw.append(token.text)

        if "debug" in FLAGS and FLAGS.debug:
            logging.log_every_n(
                logging.DEBUG, "Processing caption: '{}'".format(str(caption)), 1000)
            logging.log_every_n(
                logging.DEBUG, "\tKeep words: {}".format(set(_keep)), 1000)
            logging.log_every_n(
                logging.DEBUG, "\tThrow words: {}".format(set(_throw)), 1000)

    keywords_set = (
        np.asarray(image_uids), np.asarray(caption_numbers),
        np.asarray(keywords), np.asarray(keywords_lemma))

    return keywords_set


def filter_keyword_quality(keywords_set, min_caption_occurence=3):
    """Filter keyword quality selecting those that occur in a minimum number of image captions.

    `min_caption_occurence`: integer 1 to 5; an image keyword must occur in at
    least this many corresponding captions.

    Return filtered keywords, see `process_caption_keywords` for format.
    """
    logging.log(
        logging.INFO,
        "Filtering keyword quality with caption occurence >= {}".format(min_caption_occurence))

    keywords_set_df = pd.DataFrame(
        zip(*keywords_set), columns=["image_uid", "caption_number", "keyword", "lemma"])

    keyword_image_groups = keywords_set_df.groupby("image_uid").apply(pd.Series.tolist)

    filtered_keyword_image_groups = mp.multiprocess_map(
        _filter_keyword_group_quality,
        keyword_image_groups,
        min_caption_occurence,
        n_cores=mp.num_cpus()-1, mode="map")

    return tuple(np.concatenate(x) for x in zip(*filtered_keyword_image_groups))


def _filter_keyword_group_quality(keyword_group_set):
    """Check if keywords for image uid group match quality requirement."""
    keyword_group_set_tuple = tuple(
        np.asarray(x) for x in zip(*keyword_group_set))

    valid_idx = []
    for index, next_keyword in enumerate(keyword_group_set):
        uid, _, keyword, lemma = next_keyword

        keyword_idx = np.union1d(
            np.where(keyword_group_set_tuple[2] == keyword)[0],
            np.where(keyword_group_set_tuple[3] == lemma)[0])
        keyword_idx = np.intersect1d(
            keyword_idx, np.where(keyword_group_set_tuple[0] == uid)[0])

        num_unique_captions = len(set(keyword_group_set_tuple[1][keyword_idx]))

        if num_unique_captions >= mp.SHARED_ARGS[0]:
            if "debug" in FLAGS and FLAGS.debug:
                logging.log_every_n(
                    logging.DEBUG,
                    "Keeping image keyword '{}' which occurs in {} (>= {}) captions".format(
                        keyword, num_unique_captions, mp.SHARED_ARGS[0]), 1000)
            valid_idx.append(index)
        else:
            if "debug" in FLAGS and FLAGS.debug:
                logging.log_every_n(
                    logging.DEBUG,
                    "Throwing image keyword '{}' which occurs in {} (< {}) captions".format(
                        keyword, num_unique_captions, mp.SHARED_ARGS[0]), 1000)

    return tuple(x[valid_idx] for x in keyword_group_set_tuple)


def filter_keep_keywords(keywords_set, keyword_list, use_lemma=True):
    """Filter keywords keeping those that occur in the keyword list.

    `use_lemma`: indicates whether to compare keywords or baseform lemma.

    Return filtered keywords, see `process_caption_keywords` for format.
    """
    logging.log(logging.INFO, "Filtering keywords (by keep list) ...")

    keyword_data = keywords_set[3] if use_lemma else keywords_set[2]
    valid_idx = np.where(np.isin(keyword_data, keyword_list))[0]

    return tuple(x[valid_idx] for x in keywords_set)


def filter_remove_keywords(keywords_set, keyword_list, use_lemma=True):
    """Filter keywords removing those that occur in the keyword list.

    `use_lemma`: indicates whether to compare keywords or baseform lemma.

    Return filtered keywords, see `process_caption_keywords` for format.
    """
    logging.log(logging.INFO, "Filtering keywords (by keyword remove list) ...")

    keyword_data = keywords_set[3] if use_lemma else keywords_set[2]
    valid_idx = np.where(np.invert(np.isin(keyword_data, keyword_list)))[0]

    return tuple(x[valid_idx] for x in keywords_set)


def filter_remove_images(keywords_set, image_uid_list):
    """Filter keyword-image pairs removing images in the specified list.
    """
    logging.log(
        logging.INFO, "Filtering keyword-image pairs (by image remove list)")

    valid_idx = np.where(np.invert(np.isin(keywords_set[0], image_uid_list)))[0]

    return tuple(x[valid_idx] for x in keywords_set)


def filter_remove_keyword_images(keywords_set, keyword_list, use_lemma=True):
    """Filter keyword-image pairs removing all images with keywords in the specified list.

    NOTE: this should be used instead of `filter_remove_images` to remove all
    image instances associated with remove keyword, including pairs with
    keywords not in the remove keyword list.
    """
    logging.log(
        logging.INFO, "Filtering keyword-image pairs (by image remove list)")

    keyword_data = keywords_set[3] if use_lemma else keywords_set[2]
    image_idx = np.where(np.isin(keyword_data, keyword_list))[0]

    remove_uids = np.unique(keywords_set[0][image_idx])

    return filter_remove_images(keywords_set, remove_uids)


def find_similar_keywords(keywords_set, keyword_list, threshold=0.9,
                          use_lemma=True, spacy_model="en_core_web_lg"):
    """Find keywords semantically similar to those in the keyword list."""
    logging.log(logging.INFO, "Loading spacy model: {}".format(spacy_model))

    random.seed(0)  # reproducible spacy results
    nlp = spacy.load(spacy_model)

    logging.log(logging.INFO, "Finding semantically similar keywords ...")

    keyword_tokens = list(doc[0] for doc in nlp.pipe(list(keyword_list)))
    for token in keyword_tokens:
        assert token.has_vector, "Token {} has no vector!".format(token)

    similar_keywords = []
    keyword_data = np.unique(keywords_set[3] if use_lemma else keywords_set[2])
    for doc in nlp.pipe(keyword_data.tolist()):  # process captions in parallel
        for token in doc:
            if token.has_vector:
                for keyword_token in keyword_tokens:
                    similarity = token.similarity(keyword_token)

                    if similarity >= threshold:
                        similar_keywords.append(token.text)

                        if similarity < 1.0 and "debug" in FLAGS and FLAGS.debug:
                            logging.log(
                                logging.DEBUG,
                                "Found token '{}' similar to keyword '{}': score={}".format(
                                    token.text, keyword_token.text, similarity))

    similar_keywords = np.unique(similar_keywords)
    logging.log(
        logging.INFO, "Found {} semantically similar keywords".format(
            len(similar_keywords) - len(keyword_list)))

    return similar_keywords


def filter_flickr_audio_by_keywords(faudio_set, keywords_set):
    """Use keywords to filter (and lemmatise) isolated spoken words.

    See `process_caption_keywords` for `keywords_set`, and
    `flickraudio.extract_uid_metadata` for `faudio_set`.

    Return filtered word data and corresponding keywords,
    (filtered_faudio_set, filtered_keywords_set).
    """
    logging.log(logging.INFO, "Filtering isolated words (by keywords set) ...")

    valid_idx, valid_keywords_idx = [], []
    for idx, (_, label, _, paired_image, production, _) in enumerate(zip(*faudio_set)):
        keyword_idx = np.where(keywords_set[0] == paired_image)[0]
        keyword_idx = np.intersect1d(
            keyword_idx, np.where(keywords_set[1] == production)[0])

        if "debug" in FLAGS and FLAGS.debug:
            logging.log_every_n(
                logging.DEBUG,
                "Comparing word '{}' to valid keywords {} and lemma {}".format(
                    label, keywords_set[2][keyword_idx], keywords_set[3][keyword_idx]), 1000)

        label_keyword_matches = list(map(
            lambda keyword, lemma, label=label: label == keyword or label == lemma,
            keywords_set[2][keyword_idx],
            keywords_set[3][keyword_idx]))
        keyword_idx = keyword_idx[np.where(label_keyword_matches)[0]]

        if keyword_idx.shape[0] > 0:
            valid_idx.append(idx)
            valid_keywords_idx.append(keyword_idx[0])  # get first match

    filtered_faudio_set = tuple(x[valid_idx] for x in faudio_set)
    filtered_keywords_set = tuple(x[valid_keywords_idx] for x in keywords_set)

    return (filtered_faudio_set, filtered_keywords_set)


def get_unique_keywords_counts(keywords_set, use_lemma=True):
    """Get lists of unique keywords and their counts.

    NOTE:
    Keyword counts are the number of unique image IDs per keyword, disregarding
    duplicate keywords per unique image.

    Return as (unique_keywords, keyword_counts).
    """
    keyword_data = keywords_set[3] if use_lemma else keywords_set[2]
    unique_keywords, keyword_idx = np.unique(keyword_data, return_inverse=True)

    keyword_counts = []
    for index in range(len(unique_keywords)):
        current_keyword_idx = np.where(keyword_idx == index)[0]
        keyword_image_uids = keywords_set[0][current_keyword_idx]
        unique_image_uids = np.unique(keyword_image_uids)
        keyword_counts.append(len(unique_image_uids))

    keyword_counts = np.asarray(keyword_counts)
    return (unique_keywords, keyword_counts)


def get_count_limited_keywords(keywords_set, min_occurence=10, **kwargs):
    """Get unique keywords which have at least a minimum number of occurences.

    See `get_unique_keywords_counts` for keyword counts and **kwargs.

    `min_occurence`: integer specifies minimum number of keyword occurences.
    """
    logging.log(
        logging.INFO,
        "Computing count limited keywords with minimum occurence >= {}".format(min_occurence))

    unique_keywords, keyword_counts = get_unique_keywords_counts(
        keywords_set, **kwargs)

    valid_idx = np.where(keyword_counts >= min_occurence)[0]
    limited_keyword_list = unique_keywords[valid_idx]

    return limited_keyword_list


def log_keyword_stats(keywords_set, subset="train"):
    """Log some statitics on the specified keyword set."""
    logging.log(
        logging.INFO, "Logging keyword set statistics: {}".format(subset))

    unique_keywords, keyword_counts = get_unique_keywords_counts(keywords_set)
    argsort_idx = np.argsort(keyword_counts)

    logging.log(
        logging.INFO,
        "\tTotal keyword-image pairs (redundant; only useful for Flickr-Audio): {}".format(
            len(keywords_set[0])))

    logging.log(
        logging.INFO,
        "\tNumber of unique keyword-image pairs: {}".format(keyword_counts.sum()))

    logging.log(
        logging.INFO,
        "\tNumber of unique images: {}".format(len(np.unique(keywords_set[0]))))

    logging.log(
        logging.INFO,
        ("\tUnique keyword occurence statistics: "
         "count={:d} min={:.0f} max={:.0f} mean={:.3f} std={:.3f} "
         "25%={:.1f} 50%={:.1f} 75%={:.1f} 95%={:.1f}").format(
             len(unique_keywords), keyword_counts.min(), keyword_counts.max(),
             keyword_counts.mean(), keyword_counts.std(),
             np.percentile(keyword_counts, 25.0),
             np.percentile(keyword_counts, 50.0),
             np.percentile(keyword_counts, 75.0),
             np.percentile(keyword_counts, 95.0)))

    logging.log(logging.INFO, "\tBottom 15 occuring keywords:")
    for i in range(15):
        logging.log(
            logging.INFO, "\t{}: {} ({})".format(
                i, unique_keywords[argsort_idx[i]],
                keyword_counts[argsort_idx[i]]))

    logging.log(logging.INFO, "\tTop 15 occuring keywords:")
    for i in range(15):
        index = len(argsort_idx) - 15 + i
        logging.log(
            logging.INFO, "\t{}: {} ({})".format(
                index, unique_keywords[argsort_idx[index]],
                keyword_counts[argsort_idx[index]]))


def plot_keyword_count_distribution(keywords_set, output_dir, filename):
    """TODO(rpeloff) document and move to plotting (?)"""
    import matplotlib.pyplot as plt
    import os

    unique_keywords, keyword_counts = get_unique_keywords_counts(keywords_set)

    plt.figure(figsize=(12, len(keyword_counts) * 0.2))
    plt.barh(
        unique_keywords[np.argsort(keyword_counts)], keyword_counts[np.argsort(keyword_counts)])
    plt.title("Unique Keyword Occurences")
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "{}.png".format(filename)))


def save_keyword_images(keywords_set, images_dir, keyword_list, output_dir,
                        max_per_row=5, max_images=20):
    """TODO(rpeloff) document and move to plotting (?)"""
    import matplotlib.pyplot as plt
    from moonshot.utils import image_utils
    import os

    file_io.check_create_dir(output_dir)

    for keyword in keyword_list:
        if keyword not in keywords_set[3]:
            logging.log(
                logging.INFO, "Keyword not found in set: {}".format(keyword))
            continue  # skip to next keyword

        image_uids = np.unique(keywords_set[0][np.where(keywords_set[3] == keyword)[0]])
        n_cols = min(len(image_uids), max_per_row)
        n_rows = min(int(np.ceil(len(image_uids) / max_per_row)), int(max_images / max_per_row))

        plt.figure(figsize=(n_cols * 3, n_rows * 3))
        plt.suptitle(keyword, fontsize=14)

        for image_index, uid in enumerate(image_uids):
            if image_index + 1 > max_images:
                break
            plt.subplot(n_rows, n_cols, image_index + 1)
            plt.imshow(
                image_utils.load_image_array(
                    os.path.join(images_dir, "{}.jpg".format(uid))),
                interpolation="lanczos")
            plt.title(uid)
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "{}_filtered_images.png".format(keyword)))
