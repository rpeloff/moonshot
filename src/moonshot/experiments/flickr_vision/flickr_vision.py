"""TODO(rpeloff)

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
import pandas as pd


from moonshot.data import flickr8k
from moonshot.experiments import base
from moonshot.utils import file_io


class FlickrVision(base.Experiment):
    """TODO(rpeloff).
    """

    def __init__(self, images_dir, keywords_split="one_shot_evaluation.csv",
                 splits_dir=os.path.join("data", "splits", "flickr8k"),
                 embed_dir=None, **kwargs):
        super().__init__(**kwargs)

        # load specified Flickr 8k (or Flickr30k/MSCOCO) keywords set
        keywords_path = os.path.join(splits_dir, keywords_split)
        logging.log(
            logging.INFO, f"Creating vision experiment from: {keywords_path}")

        keywords_set = file_io.read_csv(keywords_path, skip_first=True)
        self.keywords_set = tuple(np.asarray(x) for x in keywords_set)

        # load paths of Flickr 8k (or Flickr30k/MSCOCO) images
        image_paths = flickr8k.fetch_image_paths(
            images_dir, self.keywords_set[0])
        self.image_paths = np.asarray(image_paths)

        # load paths of Flickr 8k image embeddings if specified
        self.embed_paths = None
        if embed_dir is not None:
            embed_paths = []
            for image_path in image_paths:
                embed_paths.append(
                    os.path.join(
                        embed_dir, f"{os.path.split(image_path)[1]}.tfrecord"))
                assert os.path.exists(embed_paths[-1])
            self.embed_paths = np.asarray(embed_paths)

        # get unique keywords # and class label lookup dict
        self.keywords = np.unique(self.keywords_set[3])

        # get lookup for unique image indices per keyword class
        self.class_unique_indices = {}
        for keyword_cls in self.keywords:
            cls_idx = np.where(self.keywords_set[3] == keyword_cls)[0]
            cls_imgs = self.keywords_set[0][cls_idx]

            _, unique_image_idx = np.unique(cls_imgs, return_index=True)  # only need first index

            self.class_unique_indices[keyword_cls] = cls_idx[unique_image_idx]

        # get lookup for valid keyword labels and indices per unique image uid
        unique_image_uids, unique_image_paths, unique_image_keywords = [], [], []

        keywords_set_df = pd.DataFrame(
            zip(keywords_set[0], keywords_set[3], self.image_paths),
            columns=["image_uid", "keyword", "paths"])

        keyword_image_groups = keywords_set_df.groupby(
            "image_uid").apply(pd.Series.tolist)

        for group in keyword_image_groups:
            group_uid, group_keywords, group_paths = list(zip(*group))

            unique_image_uids.append(group_uid[0])
            unique_image_paths.append(group_paths[0])
            unique_image_keywords.append(np.unique(group_keywords))

        self.unique_image_uids = unique_image_uids
        self.unique_image_paths = unique_image_paths
        self.unique_image_keywords = unique_image_keywords

    def _sample_episode(self, L, K, N, episode_labels=None):

        # sample episode learning task (defined by L-way classes)
        if episode_labels is None:
            episode_labels = self.rng.choice(self.keywords, L, replace=False)

        # sample learning examples from episode task
        x_train_idx, y_train = [], []
        for ep_label in episode_labels:
            rand_cls_idx = self.rng.choice(
                self.class_unique_indices[ep_label], K, replace=False)
            x_train_idx.extend(rand_cls_idx)
            y_train.extend([ep_label] * K)

        # sample evaluation examples from episode task
        ep_test_labels = self.rng.choice(episode_labels, N, replace=True)

        x_test_idx, y_test = [], []
        for ep_label in ep_test_labels:
            rand_cls_idx = self.rng.choice(
                self.class_unique_indices[ep_label], 1, replace=False)
            x_test_idx.extend(rand_cls_idx)
            y_test.append(ep_label)

        self.curr_episode_train = np.asarray(x_train_idx), np.asarray(y_train)
        self.curr_episode_test = np.asarray(x_test_idx), np.asarray(y_test)

    @property
    def _learning_samples(self):
        if self.embed_paths is None:
            return (
                self.image_paths[self.curr_episode_train[0]],
                self.curr_episode_train[1])
        else:
            return (
                self.embed_paths[self.curr_episode_train[0]],
                self.curr_episode_train[1])

    @property
    def _evaluation_samples(self):
        if self.embed_paths is None:
            return (
                self.image_paths[self.curr_episode_test[0]],
                self.curr_episode_test[1])
        else:
            return (
                self.embed_paths[self.curr_episode_test[0]],
                self.curr_episode_test[1])
