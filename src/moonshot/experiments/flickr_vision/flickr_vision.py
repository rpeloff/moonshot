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


from moonshot.data import flickr8k
from moonshot.experiments import base
from moonshot.utils import file_io


class FlickrVision(base.Experiment):
    """TODO(rpeloff).
    """

    def __init__(self, images_dir, splits_dir=os.path.join("data", "splits", "flickr8k"), **kwargs):
        super(FlickrVision, self).__init__(**kwargs)

        # load Flickr 8k one-shot evaluation keywords set
        one_shot_keywords_set = file_io.read_csv(
            os.path.join(splits_dir, "one_shot_evaluation.csv"),
            skip_first=True)
        self.one_shot_keywords_set = tuple(  # convert to numpy arrays
            np.asarray(x) for x in one_shot_keywords_set)

        # load paths of Flickr 8k images
        one_shot_image_paths = flickr8k.fetch_image_paths(
            images_dir, self.one_shot_keywords_set[0])
        self.one_shot_image_paths = np.asarray(one_shot_image_paths)

        # get unique one-shot keywords and class label lookup dict
        self.one_shot_keywords = np.unique(self.one_shot_keywords_set[3])

        self.keyword_id_lookup = {
            keyword: idx for idx, keyword in enumerate(self.one_shot_keywords)}

        # get lookup for unique image indices per class label
        self.class_unique_indices = {}
        for os_cls in self.one_shot_keywords:
            os_cls_label = self.keyword_id_lookup[os_cls]

            cls_idx = np.where(self.one_shot_keywords_set[3] == os_cls)[0]
            cls_imgs = self.one_shot_keywords_set[0][cls_idx]

            _, unique_image_idx = np.unique(  # only need the first index
                cls_imgs, return_index=True)

            self.class_unique_indices[os_cls_label] = cls_idx[unique_image_idx]
        
        # get lookup for valid keywords per unique image uid
        self.image_keywords = {}
        for image_uid in np.unique(self.one_shot_keywords_set[0]):
            image_idx = np.where(self.one_shot_keywords_set[0] == image_uid)[0]
            self.image_keywords[image_uid] = np.unique(
                self.one_shot_keywords_set[3][image_idx])

    def _sample_episode(self, L, K, N, episode_labels=None):

        # sample episode learning task (defined by L-way classes)
        if episode_labels is None:
            episode_labels = self.rng.choice(
                np.arange(len(self.one_shot_keywords)), L, replace=False)

        # sample learning examples from episode task
        x_train_idx, y_train = [], []
        for ep_label in episode_labels:
            rand_cls_idx = self.rng.choice(
                self.class_unique_indices[ep_label], K, replace=False)
            x_train_idx.extend(rand_cls_idx)
            y_train.extend([ep_label] * K)

        # sample evaluation examples from episode task
        ep_test_labels_idx = self.rng.choice(
            np.arange(len(episode_labels)), N, replace=True)
        ep_test_labels = episode_labels[ep_test_labels_idx]

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
        return (
            self.one_shot_image_paths[self.curr_episode_train[0]],
            self.curr_episode_train[1])

    @property
    def _evaluation_samples(self):
        return (
            self.one_shot_image_paths[self.curr_episode_test[0]],
            self.curr_episode_test[1])
