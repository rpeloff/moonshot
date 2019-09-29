"""File IO for Microsoft Common Objects in Context (MSCOCO) images and text captions.

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: September 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import json


from absl import logging


import numpy as np


from moonshot.utils import file_io


def load_mscoco_captions(captions_dir, caption_file="captions_train2017.json",
                         remove_flickr_path="data/splits/mscoco/remove_flickr30k.txt"):
    """Load MSCOCO text captions from the specified subset file."""
    captions_path = os.path.join(
        captions_dir, caption_file)
    assert os.path.exists(captions_path)

    logging.log(logging.INFO, "Loading MSCOCO text caption subset: {}".format(
        captions_path))

    image_uids, captions, caption_uids = [], [], []
    with open(captions_path) as f:
        caption_set = json.load(f)

    for annotation in caption_set["annotations"]:
        image_uids.append("{:012d}".format(annotation["image_id"]))
        captions.append(annotation["caption"].strip().lower())
        caption_uids.append(str(annotation["id"]))

    remove_flickr_image_uids = file_io.read_csv(remove_flickr_path)[0]
    remove_flickr_image_uids = list(
        map(lambda uid: os.path.splitext(uid)[0], remove_flickr_image_uids))

    def filter_remove_flickr(index):
        return image_uids[index] not in remove_flickr_image_uids

    filter_idx = list(filter(filter_remove_flickr, range(len(image_uids))))

    image_uids = np.asarray(image_uids)[filter_idx]
    captions = np.asarray(captions)[filter_idx]
    caption_uids = np.asarray(caption_uids)[filter_idx]

    return image_uids, captions, caption_uids


def fetch_mscoco_image_paths(images_dir, subset="train2017",
                             remove_flickr_path="data/splits/mscoco/remove_flickr30k.txt"):
    """Fetch MSCOCO image paths from the specified subset directory."""
    images_dir = os.path.join(images_dir, subset)

    logging.log(logging.INFO, "Loading MSCOCO image paths (subset '{}'): {}".format(
        subset, images_dir))

    image_paths = [
        os.path.join(images_dir, name) for name in os.listdir(images_dir)]
    image_uids = [
        os.path.splitext(os.path.split(path)[-1])[0] for path in image_paths]

    remove_flickr_image_uids = file_io.read_csv(remove_flickr_path)[0]
    remove_flickr_image_uids = list(
        map(lambda uid: os.path.splitext(uid)[0], remove_flickr_image_uids))

    def filter_remove_flickr(index):
        return image_uids[index] not in remove_flickr_image_uids

    filter_idx = list(filter(filter_remove_flickr, range(len(image_uids))))

    image_uids = np.asarray(image_uids)[filter_idx]
    image_paths = np.asarray(image_paths)[filter_idx]

    return image_uids, image_paths
