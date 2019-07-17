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


# define class names
class_names = ["o", "z", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# get class label lookup
class_labels = {label: idx for idx, label in enumerate(class_names)}


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


if __name__ == "__main__":
    train, test = extract_tidigits(
        os.path.join("speech_features", "tidigits", "extracted", "features", "mfcc"))
    assert train[0][0] == "o_mr_o3o5351a_a_000017-000045"
    assert test[0][0] == "3_ct_9443a_a_000139-000192"
    assert len(train[0]) == 28329
    assert len(test[0]) == 28583
