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


def fetch_audio_paths(features_dir, uid_list):
    """Fetch Flickr Audio extracted speech feature paths."""
    audio_paths = []
    for uid in uid_list:
        audio_paths.append(os.path.join(features_dir, f"{uid}.npy"))
        assert os.path.exists(audio_paths[-1])  # lazy check :)

    return audio_paths
