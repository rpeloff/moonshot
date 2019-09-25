"""Find Flickr 30k images in the MSCOCO dataset (also sourced from Flickr).

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: September 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import hashlib
import os


from absl import app
from absl import logging


import numpy as np


from moonshot.utils import file_io


def main(argv):
    del argv  # unused

    logging.log(logging.INFO, "Logging application {}".format(__file__))

    # Compute md5 hash of Flickr 30k images
    flickr30k_path = os.path.join("data", "external", "flickr30k_images")
    flickr30k_files = os.listdir(flickr30k_path)

    logging.log(logging.INFO, "Computing Flickr 30k image hashes ...")
    flickr30k_hash = []
    for filename in flickr30k_files:
        with open(os.path.join(flickr30k_path, filename), "rb") as f:
            image_bytes = f.read()
        flickr30k_hash.append(hashlib.md5(image_bytes).hexdigest())

    flickr30k_hash = np.asarray(flickr30k_hash)

    # Compute md5 hash of MSCOCO images
    mscoco_path = os.path.join("data", "external", "mscoco", "train2017")
    mscoco_files = os.listdir(mscoco_path)

    logging.log(logging.INFO, "Computing MSCOCO image hashes ...")
    mscoco_hash = []
    for filename in mscoco_files:
        with open(os.path.join(mscoco_path, filename), "rb") as f:
            image_bytes = f.read()
        mscoco_hash.append(hashlib.md5(image_bytes).hexdigest())

    mscoco_hash = np.asarray(mscoco_hash)

    # Find Flickr 30k images with hashes in MSOCO hashes
    match_idx = np.where(np.isin(flickr30k_hash, mscoco_hash))[0]

    mscoco_remove = []
    for index in match_idx:
        mscoco_index = np.where(mscoco_hash == flickr30k_hash[index])[0][0]
        logging.log(
            logging.INFO,
            "Found Flickr30k image {} matching MSCOCO (train 2017) image {}".format(
                flickr30k_files[index],
                mscoco_files[mscoco_index]))
        mscoco_remove.append(mscoco_files[mscoco_index])

    # Write matches to file
    output_path = os.path.join("data", "splits", "mscoco", "remove_flickr30k.txt")
    logging.log(
        logging.INFO,
        "Writing list of Flickr 30k images in MSCOCO dataset: {}".format(output_path))

    file_io.write_csv(
        output_path,
        mscoco_remove)


if __name__ == "__main__":
    app.run(main)
