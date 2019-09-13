"""File IO for Flickr 8K images and text captions.

Note:
The official site for the Flickr 8k data appears to have been taken down.
Thankfully Jason Brownlee has created direct download links to the original data
on his [datasets repo](https://github.com/jbrownlee/Datasets/releases).

I have also placed a copy [here](https://github.com/rpeloff/Flickr8k/releases).

Author: Ryan Eloff and Herman Kamper
Contact: ryan.peter.eloff@gmail.com
Date: July 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os


from absl import logging


import numpy as np


# TODO(rpeloff) old code, remove if sure not using this
# def extract_flickr8k_images(images_dir, captions_dir, size=(224, 224)):
#     """Load Flickr 8K images according to the caption corpus splits.
    
#     The Flicker 8k text caption corpus is used only for dataset splits.
#     """
#     train, val, test = None, None, None

#     split_dict = flickr8k_split.get_flickr8k_train_test_dev(captions_dir)
    
#     assert os.path.exists(os.path.join(images_dir, "{}.jpg".format(split_dict["train"][0])))
    
#     image_files = [os.path.join(images_dir, name) for name in os.listdir(images_dir)]
    
#     image = Image.open(image_files[0])
#     print("Example image info:", image_files[0])
#     print("Format: {} Mode: {} Size: {}".format(image.format, image.mode, image.size))

#     image_data = []
#     image_uids = []
#     for img_file in sorted(image_files):
#         image_arr = image_utils.load_image_array(img_file)
#         image_data.append(image_utils.resize_square_crop(image_arr, size=size))
#         uid = os.path.splitext(os.path.split(img_file)[-1])[0]
#         image_uids.append(uid)

#     image_data = np.asarray(image_data)
#     image_uids = np.asarray(image_uids)

#     train_idx = np.isin(image_uids, split_dict["train"])
#     val_idx = np.isin(image_uids, split_dict["dev"])
#     test_idx = np.isin(image_uids, split_dict["test"])

#     train = (image_uids[train_idx], image_data[train_idx])
#     val = (image_uids[val_idx], image_data[val_idx])
#     test = (image_uids[test_idx], image_data[test_idx])

#     return train, val, test


def load_flickr8k_splits(captions_dir):
    """Load train-dev-test splits from Flicker 8k text caption corpus."""
    set_dict = {}
    for subset in ["train", "dev", "test"]:
        if subset not in set_dict:
            set_dict[subset] = []

        subset_path = os.path.join(
            captions_dir, "Flickr_8k.{}Images.txt".format(subset))
        assert os.path.exists(subset_path)

        logging.log(logging.INFO, "Loading Flickr 8k {} split: {}".format(
            subset, subset_path))

        with open(subset_path) as f:
            for line in f:
                set_dict[subset].append(os.path.splitext(line.strip())[0])
    return set_dict


def load_flickr8k_captions(captions_dir, load_lemma=False):
    """Load Flickr 8k text caption corpus."""
    train, val, test = None, None, None

    split_dict = load_flickr8k_splits(captions_dir)

    captions_path = os.path.join(
        captions_dir,
        "Flickr8k{}.token.txt".format(".lemma" if load_lemma else ""))
    assert os.path.exists(captions_path)

    logging.log(logging.INFO, "Loading Flickr 8k text caption corpus: {}".format(
        captions_path))

    image_uids, captions, caption_numbers = [], [], []
    with open(captions_path, "r") as f:
        for line in f:
            caption_image, caption = line.split("\t")
            image_uid, caption_number = caption_image.split("#")
            image_uid = image_uid.split(".jpg")[0]
            image_uids.append(image_uid)
            captions.append(str(caption).strip().lower())
            caption_numbers.append(caption_number)

    image_uids = np.asarray(image_uids)
    captions = np.asarray(captions)
    caption_numbers = np.asarray(caption_numbers)

    train_idx = np.isin(image_uids, split_dict["train"])
    val_idx = np.isin(image_uids, split_dict["dev"])
    test_idx = np.isin(image_uids, split_dict["test"])

    train = (image_uids[train_idx], captions[train_idx], caption_numbers[train_idx])
    val = (image_uids[val_idx], captions[val_idx], caption_numbers[val_idx])
    test = (image_uids[test_idx], captions[test_idx], caption_numbers[test_idx])

    return train, val, test


def fetch_flickr8k_image_paths(images_dir, captions_dir):
    """Fetch Flickr 8K image paths corresponding to the caption corpus splits."""
    train, val, test = None, None, None

    split_dict = load_flickr8k_splits(captions_dir)

    image_paths = [
        os.path.join(images_dir, name) for name in os.listdir(images_dir)]
    image_uids = [
        os.path.splitext(os.path.split(path)[-1])[0] for path in image_paths]

    train_idx = np.isin(image_uids, split_dict["train"])
    val_idx = np.isin(image_uids, split_dict["dev"])
    test_idx = np.isin(image_uids, split_dict["test"])

    train = (image_uids[train_idx], image_paths[train_idx])
    val = (image_uids[val_idx], image_paths[val_idx])
    test = (image_uids[test_idx], image_paths[test_idx])

    return train, val, test


def fetch_flickr_audio_image_paths(images_dir, faudio_uids):
    """Fetch Flickr 8K image paths corresponding to a list of Flickr-Audio UIDs."""
    image_uids, image_paths = [], []
    for uid in faudio_uids:
        uid_parts = uid.split("_")
        image_uids.append("{}_{}".format(uid_parts[2], uid_parts[3]))
        image_paths.append(
            os.path.join(images_dir, "{}.jpg".format(image_uids[-1])))
        assert os.path.exists(image_paths[-1])  # lazy check :)

    return image_uids, image_paths
