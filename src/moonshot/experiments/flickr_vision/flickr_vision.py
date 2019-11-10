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

    def __init__(self, keywords_split="one_shot_evaluation",
                 flickr8k_image_dir=None, flickr30k_image_dir=None,
                 mscoco_image_dir=None, embed_dir=None, preprocess_func=None,
                 **kwargs):
        """TODO

        `keywords_split` one of `['one_shot_evaluation', 'one_shot_development',
        'background_train', 'background_dev', 'background_test']`.
        """
        super().__init__(**kwargs)

        assert (flickr8k_image_dir is not None or
                flickr30k_image_dir is not None or
                mscoco_image_dir is not None), "Specify at least one image dir"

        assert keywords_split in [
            "one_shot_evaluation", "one_shot_development", "background_train",
            "background_dev", "background_test"]

        logging.log(logging.INFO, f"Creating Flickr vision experiment")

        # load Flickr 8k and/or Flickr30k and/or MSCOCO keywords set(s)
        image_dirs, splits_dirs = [], []

        embed_dirs = None
        if embed_dir is not None:
            embed_dirs = []

        if flickr8k_image_dir is not None:
            image_dirs.append(flickr8k_image_dir)
            splits_dirs.append(os.path.join("data", "splits", "flickr8k"))
            if embed_dir is not None:
                embed_dirs.append(os.path.join(embed_dir, "flickr8k"))

        if flickr30k_image_dir is not None:
            image_dirs.append(flickr30k_image_dir)
            splits_dirs.append(os.path.join("data", "splits", "flickr30k"))
            if embed_dir is not None:
                embed_dirs.append(os.path.join(embed_dir, "flickr30k"))

        if mscoco_image_dir is not None:
            image_dirs.append(mscoco_image_dir)
            splits_dirs.append(os.path.join("data", "splits", "mscoco"))
            if embed_dir is not None:
                embed_dirs.append(os.path.join(embed_dir, "mscoco"))

        # load each keyword set and corresponding image paths
        image_paths = []
        keywords_set = None
        embed_paths = None if embed_dir is None else []

        for i, (image_dir, splits_dir) in enumerate(zip(image_dirs, splits_dirs)):
            keywords_path = os.path.join(splits_dir, f"{keywords_split}.csv")
            _keywords_set = file_io.read_csv(keywords_path, skip_first=True)

            if keywords_set is None:
                keywords_set = _keywords_set
            else:
                keywords_set = tuple(
                    (x + y) for x, y in zip(keywords_set, _keywords_set))

            # load image paths
            _image_paths = flickr8k.fetch_image_paths(image_dir, _keywords_set[0])
            image_paths.extend(_image_paths)

            # load image embedding paths if specified
            if embed_dir is not None:
                for image_path in _image_paths:
                    embed_paths.append(
                        os.path.join(
                            embed_dirs[i], f"{keywords_split}",
                            f"{os.path.split(image_path)[1]}.tfrecord"))
                    assert os.path.exists(embed_paths[-1])

        self.keywords_set = tuple(np.asarray(x) for x in keywords_set)
        self.image_paths = np.asarray(image_paths)
        self.embed_paths = None
        if embed_dir is not None:
            self.embed_paths = np.asarray(embed_paths)

        # get unique keywords and keyword class label lookup dict
        self.keywords = sorted(np.unique(self.keywords_set[3]).tolist())
        self.keyword_labels = {
            keyword: idx for idx, keyword in enumerate(self.keywords)}

        # get lookup for unique image indices per keyword class
        self.class_unique_indices = {}
        for keyword in self.keywords:
            cls_idx = np.where(self.keywords_set[3] == keyword)[0]
            cls_imgs = self.keywords_set[0][cls_idx]

            _, unique_image_idx = np.unique(cls_imgs, return_index=True)  # only need first index

            self.class_unique_indices[keyword] = cls_idx[unique_image_idx]

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

        # set image data as raw paths or extracted embedding paths
        if self.embed_paths is None:
            self.image_data = self.image_paths
        else:
            self.image_data = self.embed_paths

        # preprocess image data paths if specified
        if preprocess_func is not None:
            self.image_data = preprocess_func(self.image_data)

    @property
    def data(self):
        return self.image_data

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

        curr_episode_train = np.asarray(x_train_idx), np.asarray(y_train)
        self.curr_episode_train = curr_episode_train

        curr_episode_test = np.asarray(x_test_idx), np.asarray(y_test)
        self.curr_episode_test = curr_episode_test

        return curr_episode_train, curr_episode_test

    @property
    def _learning_samples(self):
        return (
            self.data[self.curr_episode_train[0]], self.curr_episode_train[1])

    @property
    def _evaluation_samples(self):
        return (
            self.data[self.curr_episode_test[0]], self.curr_episode_test[1])
