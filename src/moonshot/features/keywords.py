"""Functions to identify keywords in Flickr 8k text caption corpus.

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: July 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl import flags
from absl import logging


import spacy
import numpy as np


FLAGS = flags.FLAGS


# homophones identified with https://www.homophone.com .. :)
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

FLICKR_STOP_WORDS = [
    "baby",
    "basketball",
    "bicycle",
    "canoe",
    "jeep",
    "mountain",
    "snowboard",
    "sunglass",
    "surfboard",
    "trampoline",
    "bikini",
    "hat",
    "horse",
    "ocean",
    "river",
]


def process_caption_keywords(caption_set, spacy_model="en_core_web_lg"):
    """Process and lemmatise text captions to select keywords paired with images.

    `caption_set`: Flickr8k, Flickr30k or MSCOCO caption set.

    `spacy_model`: one of "en_core_web_sm", "en_core_web_md", "en_core_web_lg".

    Return filtered keywords with format
    (paired_images, caption_numbers, keywords, lemmatised_keywords).
    """
    logging.log(logging.INFO, "Loading spacy model: {}".format(spacy_model))

    nlp = spacy.load(spacy_model)

    logging.log(logging.INFO, "Processing captions to select keywords ...")

    image_uids, keywords, keywords_lemma, caption_numbers = ([], [], [], [])
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
    logging.log(logging.INFO, "Filtering keyword quality ...")

    valid_idx = []
    for idx, (uid, _, keyword, lemma) in enumerate(zip(*keywords_set)):
        keyword_idx = np.union1d(
            np.where(keywords_set[2] == keyword)[0],
            np.where(keywords_set[3] == lemma)[0])
        keyword_idx = np.intersect1d(
            keyword_idx, np.where(keywords_set[0] == uid)[0])

        num_unique_captions = len(set(keywords_set[1][keyword_idx]))

        if num_unique_captions >= min_caption_occurence:
            valid_idx.append(idx)
            if "debug" in FLAGS and FLAGS.debug:
                logging.log_every_n(
                    logging.DEBUG,
                    "Keeping image keyword '{}' which occurs in {} (>= {}) captions".format(
                        keyword, num_unique_captions, min_caption_occurence), 1000)
        else:
            if "debug" in FLAGS and FLAGS.debug:
                logging.log_every_n(
                    logging.DEBUG,
                    "Throwing image keyword '{}' which occurs in {} (< {}) captions".format(
                        keyword, num_unique_captions, min_caption_occurence), 1000)

    return tuple(x[valid_idx] for x in keywords_set)


def filter_keep_keywords(keywords_set, keyword_list, use_lemma=True):
    """Filter keywords keeping those that occur in the keyword list.

    Return filtered keywords, see `process_caption_keywords` for format.
    """
    logging.log(logging.INFO, "Filtering keywords test (by keep list) ...")

    keyword_data = keywords_set[3] if use_lemma else keywords_set[2]
    valid_idx = np.where(np.isin(keyword_data, keyword_list))[0]

    return tuple(x[valid_idx] for x in keywords_set)


def filter_remove_keywords(keywords_set, keyword_list, use_lemma=True):
    """Filter keywords removing those that occur in the keyword list.

    Return filtered keywords, see `process_caption_keywords` for format.
    """
    logging.log(logging.INFO, "Filtering keywords (by remove list) ...")

    keyword_data = keywords_set[3] if use_lemma else keywords_set[2]
    valid_idx = np.where(np.invert(np.isin(keyword_data, keyword_list)))[0]

    return tuple(x[valid_idx] for x in keywords_set)


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
    logging.log(logging.INFO, "Computing count limited keywords with minimum occurence >= {}".format(min_occurence))

    unique_keywords, keyword_counts = get_unique_keywords_counts(
        keywords_set, **kwargs)

    valid_idx = np.where(keyword_counts >= min_occurence)[0]
    limited_keyword_list = unique_keywords[valid_idx]

    return limited_keyword_list


def log_keyword_stats(keywords_set):
    """TODO(rpeloff)"""
    unique_keywords, keyword_counts = get_unique_keywords_counts(keywords_set)
    logging.log(
        logging.INFO,
        ("Unique keyword occurence statistics: "
         "count={:d} min={:.0f} max={:.0f} mean={:.3f} std={:.3f} "
         "25%={:.1f} 50%={:.1f} 75%={:.1f} 95%={:.1f}").format(
             len(unique_keywords), keyword_counts.min(), keyword_counts.max(),
             keyword_counts.mean(), keyword_counts.std(),
             np.percentile(keyword_counts, 25.0),
             np.percentile(keyword_counts, 50.0),
             np.percentile(keyword_counts, 75.0),
             np.percentile(keyword_counts, 95.0)))


def plot_keyword_count_distribution(keywords_set, output_dir, filename):
    """TODO(rpeloff) document and move to plotting (?)"""
    import matplotlib.pyplot as plt
    import os

    unique_keywords, keyword_counts = get_unique_keywords_counts(keywords_set)

    plt.figure(figsize=(12, len(keyword_counts) * 0.2))
    plt.barh(unique_keywords[np.argsort(keyword_counts)], keyword_counts[np.argsort(keyword_counts)])
    plt.title("Unique Keyword Occurences")
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "{}.png".format(filename)))


def save_keyword_images(faudio_set, keywords_set, images_dir, keyword_list, output_dir, max_per_row=5):
    """TODO(rpeloff) document and move to plotting (?)"""
    import matplotlib.pyplot as plt
    from moonshot.utils import image_utils
    import os

    for keyword in keyword_list:
        image_uids = np.unique(faudio_set[3][np.where(keywords_set[3] == keyword)[0]])
        n_cols = min(len(image_uids), max_per_row)
        n_rows = int(np.ceil(len(image_uids) / max_per_row))
        
        plt.figure(figsize=(n_cols * 3, n_rows * 3))
        plt.suptitle(keyword, fontsize=14)
        
        for image_index, uid in enumerate(image_uids):
            plt.subplot(n_rows, n_cols, image_index + 1)
            plt.imshow(
                image_utils.load_image_array(
                    os.path.join(images_dir, "{}.jpg".format(uid))),
                interpolation="lanczos")
            plt.title(uid)
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "{}_filtered_images.png".format(keyword)))


def filter_flickr_audio_captions(faudio_keyword_set, remove_words, spacy_model="en_core_web_md"):
    """Filter Flickr Audio data removing images with captions containing the specified words."""
    nlp = spacy.load(spacy_model)
    remove_uid = []
    for idx, utterance in enumerate(zip(*faudio_keyword_set)):
        valid_utterance = True
        doc = nlp(str(utterance[8]).lower())
        for token in doc:
            if token.text in remove_words or token.lemma_ in remove_words:
                valid_utterance = False

        if not valid_utterance:
            remove_uid.append(utterance[4])

    remove_uid = np.unique(remove_uid)
    valid_idx = []
    for idx, utterance in enumerate(zip(*faudio_keyword_set)):
        if utterance[4] not in remove_uid:
            valid_idx.append(idx)

    faudio_set_filtered = tuple(x[valid_idx] for x in faudio_keyword_set)
    return faudio_set_filtered


def filter_flickr_audio_semantic_keywords(faudio_keyword_set, remove_words, threshold=0.7, spacy_model="en_core_web_md"):
    nlp = spacy.load(spacy_model)
    faudio_words = list(set(faudio_keyword_set[2]) | set(faudio_keyword_set[7]))
    similar_keywords = {}
    for remove_word in remove_words:
        similar = []
        remove_token = nlp(str(remove_word))
        if remove_token:
            if remove_token.vector_norm:
                for word in faudio_words:
                    token = nlp(str(word))
                    if token:
                        if token.vector_norm:
                            similarity = remove_token.similarity(token)
                            if similarity >= threshold:
                                similar.append((word, similarity))
                                similar.append((token[0].lemma_, similarity))

        similar = np.asarray(list(set(similar)))
        similar_keywords[remove_word] = similar

    return similar_keywords


def compare_keywords_files(keyword_files):
    """Compare the different keyword list files.
    
    Returns the common keywords set and a boolean dict of the unique keyword
    occurences per keyword file.
    """
    keywords_dict = {}
    for keyword_file in keyword_files:
        keywords = []
        with open(keyword_file, "r") as (f):
            for line in f:
                keywords.append(line.strip())

        keywords_dict[keyword_file] = keywords

    common = set()
    all_keywords = set()
    for keyword_file, keyword_list in keywords_dict.items():
        if common == set():
            common = set(keyword_list)
        else:
            common = common & set(keyword_list)
        all_keywords = all_keywords | set(keyword_list)

    unique_dict = {}
    for keyword_file, keyword_list in keywords_dict.items():
        unique_dict[keyword_file] = set(keyword_list) - common

    all_unique = set()
    for unique_set in unique_dict.values():
        all_unique = all_unique | unique_set

    unique_occur_dict = {}
    for keyword_file in keyword_files:
        keyword_occur_dict = {}
        for keyword in all_unique:
            keyword_occur_dict[keyword] = keyword in unique_dict[keyword_file]

        unique_occur_dict[keyword_file] = keyword_occur_dict

    return (common, unique_occur_dict)
