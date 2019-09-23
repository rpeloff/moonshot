"""File IO for Flickr 8K audio features.

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: July 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import csv


from absl import logging


import numpy as np


def fetch_isolated_word_lists(feats_dir):
    """Fetch train-dev-test UIDs of isolated spoken words.

    `feats_dir` should contain 'train_words.npz', 'dev_words.npz' and
    'test_words.npz' (either mfcc or filterbank speech features).

    Extract word list metadata with `extract_all_uid_metadata`.

    Returns a dict of word lists for train, dev and test.
    """
    logging.log(
        logging.INFO,
        "Fetching Flickr-Audio isolated word lists: {}".format(feats_dir))
    set_dict = {}
    for subset in ["train", "dev", "test"]:
        set_path = os.path.join(feats_dir, "{}_words.npz".format(subset))
        with np.load(set_path) as npz:
            set_dict[subset] = npz.files
    return set_dict


def load_isolated_word_npz(feats_dir):
    """Load train-dev-test .npz archives of isolated spoken words.

    `feats_dir` should contain 'train_words.npz', 'dev_words.npz' and
    'test_words.npz' (either mfcc or filterbank speech features).

    NOTE:
    Returns a dict of NpzFile objects for train, dev and test, each of which
    contains a file handle and should be closed during cleanup.
    """
    logging.log(
        logging.INFO,
        "Loading Flickr-Audio isolated word archives: {}".format(feats_dir))
    set_dict = {}
    for subset in ["train", "dev", "test"]:
        set_path = os.path.join(feats_dir, "{}_words.npz".format(subset))
        set_dict[subset] = np.load(set_path)
    return set_dict


def extract_uid_metadata(uid):
    """Extract metadata for a given UID.

    Apply to all UIDs: see `extract_all_uid_metadata`.

    Return extracted metadata with format
    [uid, label, speaker, paired_image, production, frames].
    """
    uid_parts = uid.split("_")
    extracted = [
        uid, uid_parts[0], uid_parts[1],
        "{}_{}".format(uid_parts[2], uid_parts[3]), uid_parts[4], uid_parts[5]]
    return extracted


def extract_all_uid_metadata(uid_list):
    """Extract metadata for a given list of UID.

    Return extracted metadata arrays with format
    (uids, labels, speakers, paired_images, productions, frames).
    """
    return tuple(
        np.array(metadata) for metadata in
        zip(*map(extract_uid_metadata, uid_list)))


def load_keyword_splits_csv(csv_path):
    """Load (UID, keyword) pairs from a specified set split csv.

    `csv_path` should reference 'background_train.csv',
    'background_validation.csv' or 'one_shot_test.csv'.
    """
    with open(csv_path, "r", newline="") as csvfile:
        csv_reader = csv.DictReader(csvfile)
        faudio_uids, faudio_keywords = [], []
        for row in csv_reader:
            faudio_uids.append(row["flickr_audio_uid"])
            faudio_keywords.append(row["keyword"])
    return faudio_uids, faudio_keywords


# TODO(rpeloff) old code, remove if sure not using this

# def write_flickr_audio_filtered_sets(
#         feats_dir="speech_features/flickr_audio/extracted/features",
#         keyword_dir="data/flickr_audio"):
#     """Write the filtered Flickr-Audio keywords data sets to NumPy archives."""
    
#     background_train_csv = os.path.join(keyword_dir, "background_train.csv")
#     background_val_csv = os.path.join(keyword_dir, "background_validation.csv")
#     one_shot_test_csv = os.path.join(keyword_dir, "one_shot_test.csv")
    
#     background_train_ids_keywords = load_flickr_audio_uid_keyword_csv(background_train_csv)
#     background_val_ids_keywords = load_flickr_audio_uid_keyword_csv(background_val_csv)
#     one_shot_test_ids_keywords = load_flickr_audio_uid_keyword_csv(one_shot_test_csv)

#     for speech_feats in ["mfcc", "fbank"]:
#         print("Reading Flickr-Audio '{}' features".format(speech_feats))
#         speech_feats_dir = os.path.join(feats_dir, speech_feats)
#         faudio_train, faudio_val, faudio_test = load_isolated_words(
#             speech_feats_dir, filtered=False)
        
#         train_filter_idx = np.where(
#             np.isin(faudio_train[0], background_train_ids_keywords[0]))[0]
#         val_filter_idx = np.where(
#             np.isin(faudio_val[0], background_val_ids_keywords[0]))[0]
#         test_filter_idx = np.where(
#             np.isin(faudio_test[0], one_shot_test_ids_keywords[0]))[0]
#         faudio_background_train = {
#             "{}_{}".format(uid, keyword): data
#             for uid, data, keyword in zip(faudio_train[0][train_filter_idx],
#                                           faudio_train[1][train_filter_idx],
#                                           background_train_ids_keywords[1])}
#         faudio_background_val = {
#             "{}_{}".format(uid, keyword): data
#             for uid, data, keyword in zip(faudio_val[0][val_filter_idx],
#                                           faudio_val[1][val_filter_idx],
#                                           background_val_ids_keywords[1])}
#         faudio_one_shot_test = {
#             "{}_{}".format(uid, keyword): data
#             for uid, data, keyword in zip(faudio_test[0][test_filter_idx],
#                                           faudio_test[1][test_filter_idx],
#                                           one_shot_test_ids_keywords[1])}

#         np.savez(
#             os.path.join(speech_feats_dir, "train_words_filtered.npz"),
#             **faudio_background_train)
#         np.savez(
#             os.path.join(speech_feats_dir, "dev_words_filtered.npz"),
#             **faudio_background_val)
#         np.savez(
#             os.path.join(speech_feats_dir, "test_words_filtered.npz"),
#             **faudio_one_shot_test)
