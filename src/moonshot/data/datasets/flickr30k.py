"""File IO for Flickr 30K images and text captions.

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: September 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os


from absl import logging 


import numpy as np


def load_flickr30k_splits(splits_dir="data/splits/flickr30k",
                          flickr8k_splits=None):
    """Load train-dev-test splits from Flicker 30k text caption corpus."""
    set_dict = {}
    for subset in ["train", "dev", "test"]:
        if subset not in set_dict:
            set_dict[subset] = []

        subset_path = os.path.join(
            splits_dir, "{}.txt".format(subset))
        assert os.path.exists(subset_path)

        logging.log(logging.INFO, "Loading Flickr 30k {} split: {}".format(
            subset, subset_path))

        with open(subset_path) as f:
            for line in f:
                set_dict[subset].append(os.path.splitext(line.strip())[0])

    if flickr8k_splits is not None:  # remove flickr 8k images from 30k splits
        set_dict = remove_flickr8k_splits(set_dict, flickr8k_splits)

    return set_dict


def remove_flickr8k_splits(flickr30k_splits, flickr8k_splits):
    """Remove Flickr 8k images from Flickr 30k train-dev-test splits."""
    flickr8k_all = []
    for _, uids in flickr8k_splits.items():
        flickr8k_all.extend(list(map(lambda uid: uid.split("_")[0], uids)))

    flickr30_removed = {}
    for subset, uids in flickr30k_splits.items():
        uids_removed = []
        for uid in uids:
            if uid not in flickr8k_all:
                uids_removed.append(uid)
        flickr30_removed[subset] = uids_removed

    return flickr30_removed


def _load_flickr30k_unrelated_captions(splits_dir="data/splits/flickr30k"):
    """Load unrelated image captions from the Flickr 30k text caption corpus."""
    path = os.path.join(splits_dir, "UNRELATED_CAPTIONS")
    assert os.path.exists(path)

    image_uids, caption_numbers = [], []
    with open(path, "rb") as f:
        next(f)  # skip header line

        for line in f:
            image_uid, caption_number = line.decode("utf8").strip().split(" ")
            image_uids.append(image_uid)
            caption_numbers.append(str(int(caption_number) - 1))

    image_uids = np.asarray(image_uids)
    caption_numbers = np.asarray(caption_numbers)

    return image_uids, caption_numbers


def load_flickr30k_captions(captions_dir, splits_dir="data/splits/flickr30k",
                            flickr8k_splits=None):
    """Load Flickr 30k text caption corpus."""
    train, val, test = None, None, None

    split_dict = load_flickr30k_splits(splits_dir, flickr8k_splits)

    captions_path = os.path.join(
        captions_dir, "results_20130124.token")
    assert os.path.exists(captions_path)

    logging.log(logging.INFO, "Loading Flickr 30k text caption corpus: {}".format(
        captions_path))

    image_uids, captions, caption_numbers = [], [], []
    with open(captions_path, "rb") as f:
        for line in f:
            caption_image, caption = line.decode("utf8").split("\t")
            image_uid, caption_number = caption_image.split("#")
            image_uid = image_uid.split(".jpg")[0]
            image_uids.append(image_uid)
            captions.append(str(caption).strip().lower())
            caption_numbers.append(caption_number)

    # remove unrelated captions
    flickr30k_unrelated = _load_flickr30k_unrelated_captions(splits_dir)

    def filter_remove_unrelated(index):
        unrelated_idx = np.where(flickr30k_unrelated[0] == image_uids[index])[0]
        return caption_numbers[index] not in flickr30k_unrelated[1][unrelated_idx]

    filter_idx = list(filter(filter_remove_unrelated, range(len(image_uids))))

    image_uids = np.asarray(image_uids)[filter_idx]
    captions = np.asarray(captions)[filter_idx]
    caption_numbers = np.asarray(caption_numbers)[filter_idx]

    # split into train-dev-test
    train_idx = np.isin(image_uids, split_dict["train"])
    val_idx = np.isin(image_uids, split_dict["dev"])
    test_idx = np.isin(image_uids, split_dict["test"])

    train = (image_uids[train_idx], captions[train_idx], caption_numbers[train_idx])
    val = (image_uids[val_idx], captions[val_idx], caption_numbers[val_idx])
    test = (image_uids[test_idx], captions[test_idx], caption_numbers[test_idx])

    return train, val, test


def fetch_flickr30k_image_paths(images_dir, splits_dir="data/splits/flickr30k",
                                flickr8k_splits=None):
    """Fetch Flickr 30k image paths corresponding to the caption corpus splits."""
    train, val, test = None, None, None

    split_dict = load_flickr30k_splits(splits_dir, flickr8k_splits)

    image_paths = np.asarray([
        os.path.join(images_dir, name) for name in os.listdir(images_dir)])
    image_uids = np.asarray([
        os.path.splitext(os.path.split(path)[-1])[0] for path in image_paths])

    train_idx = np.isin(image_uids, split_dict["train"])
    val_idx = np.isin(image_uids, split_dict["dev"])
    test_idx = np.isin(image_uids, split_dict["test"])

    train = (image_uids[train_idx], image_paths[train_idx])
    val = (image_uids[val_idx], image_paths[val_idx])
    test = (image_uids[test_idx], image_paths[test_idx])

    return train, val, test
