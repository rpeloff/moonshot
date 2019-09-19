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

]


def filter_caption_keywords(caption_set, spacy_model="en_core_web_lg"):
    """Filter and lemmatise text captions to select keywords paired with images.

    `caption_set`: Flickr8k, Flickr30k or MSCOCO caption set.
    `spacy_model`: one of "en_core_web_sm", "en_core_web_md", "en_core_web_lg".
    """
    nlp = spacy.load(spacy_model)

    image_uids, keywords, keywords_lemma, caption_numbers = [], [], [], []
    for image_uid, caption, caption_number in zip(*caption_set):
        doc = nlp(str(caption).lower())
        _keep, _throw = [], []  # temp lists for debug
        for token in doc:
            valid_token = (
                not token.is_stop  # remove stopwords
                and not token.is_punct  # remove punctuation
                and not token.is_digit  # remove digits
                and not bool(sum((stop_word in token.text) for stop_word in TIDIGITS_STOP_WORDS))  # remove tidigits
                and not bool(sum((stop_word in token.lemma_) for stop_word in TIDIGITS_STOP_WORDS))  # remove tidigits
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
            logging.log_first_n(
                logging.DEBUG, "Filtering caption keywords: '{}'".format(str(caption)), 5)
            logging.log_first_n(
                logging.DEBUG, "Keep words: {}".format(set(_keep)), 5)
            logging.log_first_n(
                logging.DEBUG, "Throw words: {}".format(set(_throw)), 5)

    return (np.asarray(image_uids), np.asarray(caption_numbers),
            np.asarray(keywords), np.asarray(keywords_lemma))


# @profile
def filter_flickr_audio_by_keywords(faudio_set, keywords_set):
    """Use filtered keywords to filter (and lemmatise) isolated spoken words.

    See `filter_caption_keywords` for `keywords_set`.
    """
    # # create valid keywords from original and lemmatised keywords
    # valid_keywords = set(keywords_set[2]) | set(keywords_set[3])
    #

    # compare isolated words to the keywords selected for the corresponding image caption
    valid_idx, keywords = [], []
    for idx, (uid, data, label, speaker, paired_image, prod, frames) in enumerate(zip(*faudio_set)):
        keyword_idx = np.where(keywords_set[0] == paired_image)[0]
        keyword_idx = np.intersect1d(
            keyword_idx, np.where(keywords_set[1] == prod)[0])
        keyword_idx = np.intersect1d(
            keyword_idx, np.where(keywords_set[2] == prod)[0])

        logging.log_first_n(
            logging.DEBUG,
            "Filtering word '{}', valid keywords {}, valid lemma {}".format(
                label, keywords_set[2][keyword_idx], keywords_set[3][keyword_idx]),
            5)

        label_keyword_matches = list(map(
            lambda keyword, lemma, label=label: label in keyword or label in lemma,
            keywords_set[2],
            keywords_set[3]))
        keyword_idx = np.intersect1d(
            keyword_idx, np.where(label_keyword_matches)[0])

        if keyword_idx.shape[0] > 0:
            valid_idx.append(idx)
            keywords.append(keywords_set[2][keyword_idx[0]])  # get first match

            if "debug" in FLAGS and FLAGS.debug:
            logging.log_first_n(
                logging.DEBUG, "Isolated word: {} Keywords found (selecting first): keywords: '{}'".format(str(caption)), 5)
            logging.log_first_n(
                logging.DEBUG, "Keep words: {}".format(set(_keep)), 5)
            logging.log_first_n(
                logging.DEBUG, "Throw words: {}".format(set(_throw)), 5)

        np.where(keywords_set[2][keyword_idx] == paired_image)[0]
        if label in keywords_set[2][keyword_idx] or label in keywords_set[3][keyword_idx]:
            valid_idx.append(idx)
            keywords.append()
        logging.log(logging.DEBUG, "Label {} Keywords {}".format(label, keywords_set[3][keyword_idx]))

    # for image_uid, caption_number, keyword, keyword_lemma in zip(*keywords_set):
    #     audio_idx = np.where(faudio_set[4] == image_uid)[0]
    #     audio_idx = np.intersect1d(
    #         audio_idx, np.where(faudio_set[5] == caption_number)[0])
    #     audio_idx = np.intersect1d(
    #         audio_idx, list(set(np.where(faudio_set[2] == text)[0]) | set(np.where(faudio_set[2] == keyword)[0])))


    # valid_idx = []
    # audio_keywords = []
    # audio_captions = []
    # for keyword, text, pos, image_uid, caption, caption_number in zip(*keywords_set):
    #     if keyword in valid_keywords or text in valid_keywords:
    #         # get isolated spoken words for given image ID, caption number and keyword (including pre-lemma word form)
    #         audio_idx = np.where(faudio_set[4] == image_uid)[0]
    #         audio_idx = np.intersect1d(audio_idx, np.where(faudio_set[5] == caption_number)[0])
    #         audio_idx = np.intersect1d(
    #             audio_idx,
    #             list(set(np.where(faudio_set[2] == text)[0]) | set(np.where(faudio_set[2] == keyword)[0])))
    #         for index in audio_idx:
    #             valid_idx.append(index)
    #             audio_keywords.append(keyword)  # lemmatised label
    #             audio_captions.append(caption)  # original caption
    # valid_idx, unique_idx = np.unique(valid_idx, return_index=True)  # remove duplicates
    # audio_keywords = np.asarray(audio_keywords)
    # audio_keywords = audio_keywords[unique_idx]  # remove duplicates
    # audio_captions = np.asarray(audio_captions)
    # audio_captions = audio_captions[unique_idx]  # remove duplicates
    # faudio_set_filtered = tuple((x[valid_idx] for x in faudio_set)) + (audio_keywords, audio_captions)
    # return faudio_set_filtered


# def create_keywords_data(caption_set, remove_verbs=True, spacy_model="en_core_web_lg"):
#     "Apply lemmatisation and filtering to Flickr 8k text captions to select keywords."
#     nlp = spacy.load(spacy_model)  # one of "en_core_web_sm", "en_core_web_md", "en_core_web_lg"

#     # note: not considering compound POS
#     valid_dependency = ["nsubj", "pobj", "dobj", "ROOT"]  # accept nominal subject, objects, and the root
#     valid_tags = ["NN", "NNP", "NNPS", "NNS"]  # accept nouns and proper nouns
#     if not remove_verbs:
#         valid_tags += ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]  # accept all forms of verbs (TODO: try only VBG gerunds?)

#     text, keywords, pos = [], [], []
#     image_uids, captions, caption_numbers = [], [], []
#     for image_uid, caption, caption_number in zip(*caption_set):
#         doc = nlp(str(caption).lower())
#         for token in doc:
#             valid_token = (
#                 token.dep_ in valid_dependency
#                 and token.tag_ in valid_tags
#                 and not token.is_stop)  # remove stopwords
#             if valid_token:
#                 keywords.append(token.lemma_)  # get lemmatised baseform token
#                 text.append(token.text)  # get original textual label
#                 pos.append(token.tag_)  # get fine-grain part of speech
#                 image_uids.append(image_uid)
#                 captions.append(caption)
#                 caption_numbers.append(caption_number)
#     image_uids = np.asarray(image_uids)
#     text = np.asarray(text)
#     keywords = np.asarray(keywords)
#     pos = np.asarray(pos)
#     captions = np.asarray(captions)
#     caption_numbers = np.asarray(caption_numbers)
#     return (keywords, text, pos, image_uids, captions, caption_numbers)


# def create_flickr_audio_keywords_data(faudio_set, keywords_set):
#     """Create Flickr Audio data with keywords, filtered with keywords set.

#     See `filter_caption_keywords` for `keywords_set`.
#     """
#     # create valid keywords from original and lemmatised keywords
#     valid_keywords = set(np.unique(keywords_set[0])) | set(np.unique(keywords_set[1]))
#     valid_idx = []
#     audio_keywords = []
#     audio_captions = []
#     for keyword, text, pos, image_uid, caption, caption_number in zip(*keywords_set):
#         if keyword in valid_keywords or text in valid_keywords:
#             # get isolated spoken words for given image ID, caption number and keyword (including pre-lemma word form)
#             audio_idx = np.where(faudio_set[4] == image_uid)[0]
#             audio_idx = np.intersect1d(audio_idx, np.where(faudio_set[5] == caption_number)[0])
#             audio_idx = np.intersect1d(
#                 audio_idx,
#                 list(set(np.where(faudio_set[2] == text)[0]) | set(np.where(faudio_set[2] == keyword)[0])))
#             for index in audio_idx:
#                 valid_idx.append(index)
#                 audio_keywords.append(keyword)  # lemmatised label
#                 audio_captions.append(caption)  # original caption
#     valid_idx, unique_idx = np.unique(valid_idx, return_index=True)  # remove duplicates
#     audio_keywords = np.asarray(audio_keywords)
#     audio_keywords = audio_keywords[unique_idx]  # remove duplicates
#     audio_captions = np.asarray(audio_captions)
#     audio_captions = audio_captions[unique_idx]  # remove duplicates
#     faudio_set_filtered = tuple((x[valid_idx] for x in faudio_set)) + (audio_keywords, audio_captions)
#     return faudio_set_filtered


def filter_flickr_audio_keyword_images(faudio_keyword_set, caption_set, min_occurence=3, spacy_model="en_core_web_md"):
    """Filter Flickr Audio data keeping images paired with a keyword if majority of its captions contain the keyword.

    Used to improve quality of the keywords and their paired images.

    See `create_flickr_audio_keywords_data` for `faudio_keyword_set`.
    Argument `caption_set` is the corresponding Flickr 8k text caption corpus.

    'Majority' is define as keyword occurs in at least `m` paired image captions.
    """
    nlp = spacy.load(spacy_model)  # one of "en_core_web_sm", "en_core_web_md", "en_core_web_lg"
    keyword_list = np.unique(faudio_keyword_set[7])
    keyword_image_uids = {}
    for keyword in keyword_list:
        keyword_image_uids[keyword] = []  # valid image UIDs for the keyword
        image_uids = np.unique(  # get images from indices where label or lemmatised label match the chosen keyword
            faudio_keyword_set[4][np.union1d(np.where(faudio_keyword_set[2] == keyword)[0],
                                             np.where(faudio_keyword_set[7] == keyword)[0])])
        for image_uid in image_uids:
            image_captions = caption_set[1][np.where(caption_set[0] == image_uid)]  # get image captions from caption data
            assert len(image_captions) == 5  # every image has 5 captions
            keyword_count = 0
            for caption in image_captions:
                doc = nlp(str(caption))
                keyword_found = False
                for token in doc:
                    if keyword == token.text or keyword == token.lemma_:
                        keyword_found = True
                if keyword_found:
                    keyword_count += 1
            if keyword_count >= min_occurence:
                keyword_image_uids[keyword].append(image_uid)
    valid_idx = []
    for idx, utterance in enumerate(zip(*faudio_keyword_set)):
        if utterance[4] in keyword_image_uids[utterance[7]]:
            valid_idx.append(idx)
    faudio_set_filtered = tuple((x[valid_idx] for x in faudio_keyword_set))
    return faudio_set_filtered


def get_unique_keywords_counts(faudio_keyword_set, use_lemma=True):
    """Compute the unique keywords and keyword bin counts over unique keyword-image pairs.

    Note that keyword counts are the number of unique image IDs per keyword,
    disregarding duplicate keywords per unique image.

    See `create_flickr_audio_keywords_data` for `faudio_keyword_set`.
    """
    keyword_data = faudio_keyword_set[7] if use_lemma else faudio_keyword_set[2]  # use lemmatised keywords if specified
    unique_keywords, keyword_idx = np.unique(keyword_data, return_inverse=True)
    # get keyword counts as number of unique image IDs per keyword (disregarding duplicate keywords per unique image)
    keyword_counts = []
    for index in range(len(unique_keywords)):
        current_keyword_idx = np.where(keyword_idx == index)[0]
        keyword_image_uids = faudio_keyword_set[4][current_keyword_idx]
        unique_image_uids = np.unique(keyword_image_uids)  # ignore duplicate image occurences
        keyword_counts.append(len(unique_image_uids))
    keyword_counts = np.asarray(keyword_counts)
    return unique_keywords, keyword_counts


def filter_flickr_audio_keyword_counts(unique_keywords, keyword_counts, min_occurence, bottom_k=None):
    """Filter keyword-image pairs by minimum bincount and selecting bottom-k least occuring.

    Choosing less frequent (bottom-k) keywords for one-shot learning is useful
    to minimise impact on training data where these keywords are removed.

    See `get_unique_keywords_counts` for `unique_keywords` and `keyword_counts`.
    """
    filter_keyword_occurence_idx = np.where(keyword_counts >= min_occurence)[0]
    limited_keyword_list = unique_keywords[filter_keyword_occurence_idx]
    print("Choose keywords with bin count >= {}: unique keyword-image pairs reduced from {} to {}.".format(
        min_occurence, len(unique_keywords), len(filter_keyword_occurence_idx)))
    if bottom_k is not None:
        filter_keyword_bottom_idx = np.argsort(keyword_counts[filter_keyword_occurence_idx])[:bottom_k]
        limited_keyword_list = limited_keyword_list[filter_keyword_bottom_idx]
        print("Choose bottom {} least occuring keywords: unique keyword-image pairs reduced from {} to {}.".format(
            bottom_k, len(filter_keyword_occurence_idx), len(filter_keyword_bottom_idx)))
    return limited_keyword_list


def filter_flickr_audio_keep_keywords(faudio_keyword_set, keep_words):
    """Filter Flickr Audio data keeping keywords if they are in keyword list."""
    keep_filter_idx = np.where(np.isin(faudio_keyword_set[2], keep_words))[0]
    keep_filter_idx = np.union1d(
        keep_filter_idx, np.where(np.isin(faudio_keyword_set[7], keep_words))[0])

    faudio_set_filtered = tuple((x[keep_filter_idx] for x in faudio_keyword_set))
    return faudio_set_filtered


def filter_flickr_audio_remove_keywords(faudio_keyword_set, remove_words):
    """Filter Flickr Audio data removing keywords if they are in keyword list."""
    remove_filter_idx = np.where(np.invert(np.isin(faudio_keyword_set[2], remove_words)))[0]
    remove_filter_idx = np.intersect1d(
        remove_filter_idx, np.where(np.invert(np.isin(faudio_keyword_set[7], remove_words)))[0])

    faudio_set_filtered = tuple((x[remove_filter_idx] for x in faudio_keyword_set))
    return faudio_set_filtered


def filter_flickr_audio_captions(faudio_keyword_set, remove_words, spacy_model="en_core_web_md"):
    """Filter Flickr Audio data removing images with captions containing the specified words."""
    nlp = spacy.load(spacy_model)  # one of "en_core_web_sm", "en_core_web_md", "en_core_web_lg"
    remove_uid = []
    for idx, utterance in enumerate(zip(*faudio_keyword_set)):
        valid_utterance = True
        doc = nlp(str(utterance[8]).lower())
        for token in doc:
            if token.text in remove_words or token.lemma_ in remove_words:
                valid_utterance = False
        if not valid_utterance:
            remove_uid.append(utterance[4])  # add image uid to be removed
    remove_uid = np.unique(remove_uid)
    valid_idx = []
    for idx, utterance in enumerate(zip(*faudio_keyword_set)):
        if utterance[4] not in remove_uid:
            valid_idx.append(idx)
    faudio_set_filtered = tuple((x[valid_idx] for x in faudio_keyword_set))
    return faudio_set_filtered


def filter_flickr_audio_semantic_keywords(faudio_keyword_set, remove_words, threshold=0.7, spacy_model="en_core_web_md"):

    nlp = spacy.load(spacy_model)  # one of "en_core_web_sm", "en_core_web_md", "en_core_web_lg"
    faudio_words = list(set(faudio_keyword_set[2]) | set(faudio_keyword_set[7]))

    similar_keywords = {}

    for remove_word in remove_words:
        similar = []
        remove_token = nlp(str(remove_word))
        if remove_token and remove_token.vector_norm:
            for word in faudio_words:
                token = nlp(str(word))
                if token and token.vector_norm:
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
        with open(keyword_file, "r") as f:
            for line in f:
                keywords.append(line.strip())
        keywords_dict[keyword_file] = keywords
    # compute common keywords set
    common = set()
    all_keywords = set()
    for keyword_file, keyword_list in keywords_dict.items():
        if common == set():
            common = set(keyword_list)
        else:
            common = common & set(keyword_list)
        all_keywords = all_keywords | set(keyword_list)
    # compute the unique keywords per file
    unique_dict = {}
    for keyword_file, keyword_list in keywords_dict.items():
        unique_dict[keyword_file] = set(keyword_list) - common
    # compute the overall set of unique keywords
    all_unique = set()
    for unique_set in unique_dict.values():
        all_unique = all_unique | unique_set
    # compute whether keywords in the overall unique set occurs in each file
    unique_occur_dict = {}
    for keyword_file in keyword_files:
        keyword_occur_dict = {}
        for keyword in all_unique:
            keyword_occur_dict[keyword] = keyword in unique_dict[keyword_file]
        unique_occur_dict[keyword_file] = keyword_occur_dict
    return common, unique_occur_dic
