"""Function to load TIDigits features.

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: June 2019
"""



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import numpy as np


def extract_tidigits(feats_dir):
    """Load TIDigits speech features (MFCC or Filterbank) and extract metadata.
    """
    # load tidigits mfcc or filterbank archives (specified by feats_dir)
    train = np.load(os.path.join(feats_dir, "train.npz"))
    test = np.load(os.path.join(feats_dir, "test.npz"))

    # function to extract data and metadata
    def extract_data(dataset):
        ids, data, labels, speakers, src_seqs, productions, fa_frames = [], [], [], [], [], [], []
        for uid in dataset:
            ids.append(uid)
            data.append(dataset[uid])
            # split uid into delimitted values
            uid_parts = uid.split("_")
            labels.append(uid_parts[0])
            speakers.append(uid_parts[1])
            src_seqs.append(uid_parts[2])
            productions.append(uid_parts[3])
            fa_frames.append(uid_parts[4])
        return (
            np.asarray(ids), np.asarray(data), np.asarray(labels),
            np.asarray(speakers), np.asarray(src_seqs),
            np.asarray(productions), np.asarray(fa_frames))

    # extract train and test data
    train_extracted = extract_data(train)
    test_extracted = extract_data(test)

    # return as (train, test) where each set contains (ids, data, labels, speakers, src_seqs, productions, fa_frames)
    return train_extracted, test_extracted