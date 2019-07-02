"""Function to sample multimodal paired data.

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: June 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import csv
import numpy as np


import datasets


mnsit_tidigits_image_to_speech_labels = {
    0: ["o", "z"],
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9"
}


def sample_multimodal_pairs(image_labels, speech_labels, image_to_speech_labels):
    """Random sample multimodal pair indices from image and speech label sets.
    
    Multimodal pairs are chosen as follows:
    - Match images to speech according to `image_to_speech_labels` mapping
    - Random choose pair indices without replacement
    - Repeat the smaller of the two sets
    - Return multimodal pair indices with same size as larger of the two sets
    - Shuffle multimodal pair indices
    """
    image_classes = np.unique(image_labels)
    assert np.array_equal(sorted(image_to_speech_labels.keys()), sorted(image_classes))
    paired_image_idx = []
    paired_speech_idx = []
    for image_cls in image_classes:
        image_speech_cls = image_to_speech_labels[image_cls]
        if type(image_speech_cls) is not list:
            image_speech_cls = [image_speech_cls]  # pretend its a list
        for speech_cls in image_speech_cls:
            image_cls_idx = np.where(image_labels == image_cls)[0]
            speech_cls_idx = np.where(speech_labels == speech_cls)[0]
            max_length = max(len(image_cls_idx), len(speech_cls_idx))
            # random sample paired indices without replacement (modulo operator repeats smaller of two sets to max_length)
            paired_image_cls_idx = np.random.choice(max_length, size=max_length, replace=False) % len(image_cls_idx)
            paired_speech_cls_idx = np.random.choice(max_length, size=max_length, replace=False) % len(speech_cls_idx)
            paired_image_idx.extend(image_cls_idx[paired_image_cls_idx])
            paired_speech_idx.extend(speech_cls_idx[paired_speech_cls_idx])
    # double check that we have sampled same number of speech and images and shuffle
    assert len(paired_image_idx) == len(paired_speech_idx)
    shuffle_idx = np.random.permutation(len(paired_image_idx))
    paired_image_idx = np.array(paired_image_idx)[shuffle_idx]
    paired_speech_idx = np.array(paired_speech_idx)[shuffle_idx]
    return paired_image_idx, paired_speech_idx


def write_mnist_tidigits_csv(
        mnist_fp="data/id_mnist.npz",
        tidigits_fp="speech_features/tidigits/extracted/features/mfcc",
        output_dir="data/mnist_tidigits"):
    """Write MNIST-TIDigits paired dataset IDs to csv."""
    # load mnist and tidigits data
    mnist_trainval, mnist_test = datasets.id_mnist.read_id_mnist_arch(mnist_fp)
    tidigits_trainval, tidigits_test = datasets.tidigits.extract_tidigits(tidigits_fp)
    # split into train and validation
    mnist_val = tuple((x[:5000] for x in mnist_trainval))
    mnist_train = tuple((x[5000:] for x in mnist_trainval))
    tidigits_val = tuple((x[:5000] for x in tidigits_trainval))
    tidigits_train = tuple((x[5000:] for x in tidigits_trainval))
    # random sample multimodal pair indices for train and test sets
    mnist_tidigits_train = sample_multimodal_pairs(mnist_train[2], tidigits_train[2], mnsit_tidigits_image_to_speech_labels)
    mnist_tidigits_val = sample_multimodal_pairs(mnist_val[2], tidigits_val[2], mnsit_tidigits_image_to_speech_labels)
    mnist_tidigits_test = sample_multimodal_pairs(mnist_test[2], tidigits_test[2], mnsit_tidigits_image_to_speech_labels)
    # get multimodal pair IDs
    mnist_train_uids = mnist_train[0][mnist_tidigits_train[0]]
    tidigits_train_uids = tidigits_train[0][mnist_tidigits_train[1]]
    mnist_val_uids = mnist_val[0][mnist_tidigits_val[0]]
    tidigits_val_uids = tidigits_val[0][mnist_tidigits_val[1]]
    mnist_test_uids = mnist_test[0][mnist_tidigits_test[0]]
    tidigits_test_uids = tidigits_test[0][mnist_tidigits_test[1]]
    # map tidigits ID to include paired mnist ID
    def paired_mnist_tidigits_id(mnist_uid, tidigits_uid):
        label, speaker, src_seq, production, fa_frames = tidigits_uid.split("_")
        return "{}_{}_{}_{}_{}_{}".format(label, speaker, src_seq, mnist_uid, production, fa_frames)
    paired_tidigits_train_uids = np.asarray(list(map(paired_mnist_tidigits_id, mnist_train_uids, tidigits_train_uids)))
    paired_tidigits_val_uids = np.asarray(list(map(paired_mnist_tidigits_id, mnist_val_uids, tidigits_val_uids)))
    paired_tidigits_test_uids = np.asarray(list(map(paired_mnist_tidigits_id, mnist_test_uids, tidigits_test_uids)))
    # write multimodal pair IDs data to csv
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, "train.csv"), "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=",")
        csv_writer.writerow(["mnist_id", "tidigits_id"])
        csv_writer.writerows(zip(
            mnist_train_uids,
            paired_tidigits_train_uids))
    with open(os.path.join(output_dir, "val.csv"), "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=",")
        csv_writer.writerow(["mnist_id", "tidigits_id"])
        csv_writer.writerows(zip(
            mnist_val_uids,
            paired_tidigits_val_uids))
    with open(os.path.join(output_dir, "test.csv"), "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=",")
        csv_writer.writerow(["mnist_id", "tidigits_id"])
        csv_writer.writerows(zip(
            mnist_test_uids,
            paired_tidigits_test_uids))


def read_mnist_tidigits_csv(path="data/mnist_tidigits/train.csv"):
    with open(path, "r", newline="") as csvfile:
        csv_reader = csv.DictReader(csvfile)
        mnist_train_uids = []
        tidigits_train_uids = []
        for row in csv_reader:
            mnist_train_uids.append(row["mnist_id"])
            tidigits_train_uids.append(row["tidigits_id"])
    return np.asarray(mnist_train_uids), np.asarray(tidigits_train_uids)
