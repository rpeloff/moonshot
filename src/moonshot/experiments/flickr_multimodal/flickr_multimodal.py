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


from moonshot.experiments.flickr_audio import flickr_audio
from moonshot.experiments.flickr_vision import flickr_vision


class FlickrMultimodal(flickr_audio.FlickrAudio):
    """TODO(rpeloff).
    """

    def __init__(self, audio_dir, images_dir,
                 splits_dir=os.path.join("data", "splits", "flickr8k"),
                 preprocess_audio=None, **kwargs):
        super().__init__(audio_dir, splits_dir=splits_dir,
                         preprocess=preprocess_audio, **kwargs)

        self.flickr_vision_exp = flickr_vision.FlickrVision(
            images_dir, splits_dir=splits_dir, **kwargs)

    def _sample_episode(self, L, K, N, speaker_mode="baseline",
                        episode_labels=None, unseen_match_set=False):
        # sample a Flickr-Audio experiment episode to get train and query sets
        super()._sample_episode(
            L=L, K=K, N=N, speaker_mode=speaker_mode,
            episode_labels=episode_labels)

        x_train_idx, y_train = self.curr_episode_train
        x_query_idx, y_query = self.curr_episode_test

        # get valid image indices for the match set not seen in train & query
        sampled_image_uids = []
        sampled_image_uids.extend(
            self.faudio_one_shot_metadata[3][x_train_idx])
        sampled_image_uids.extend(
            self.faudio_one_shot_metadata[3][x_query_idx])

        valid_match_image_idx = np.where(np.invert(np.isin(
            self.faudio_one_shot_metadata[3], sampled_image_uids)))

        # sample a new L-way match set for each query
        x_match_idx, y_match = [], []
        for query_label in y_query:
            if unseen_match_set:  # match set contains labels not seen in train
                y_query_match = self.rng.choice(
                    self.one_shot_keywords, L - 1, replace=False)
                y_query_match = np.concatenate(([query_label], y_query_match))
            else:  # match set contains only labels seen in train
                y_query_match = np.unique(y_train)
            self.rng.shuffle(y_query_match)

            x_query_match_idx = []
            for match_label in y_query_match:
                valid_idx = np.intersect1d(  # sample unique unseen images
                    valid_match_image_idx,
                    self.flickr_vision_exp.class_unique_indices[match_label])

                rand_cls_idx = self.rng.choice(
                    valid_idx, 1,
                    replace=False)

                x_query_match_idx.extend(rand_cls_idx)

            x_match_idx.append(x_query_match_idx)
            y_match.append(y_query_match)

        self.curr_episode_train = np.asarray(x_train_idx), np.asarray(y_train)
        self.curr_episode_test = (
            np.asarray(x_query_idx), np.asarray(y_query),
            np.asarray(x_match_idx), np.asarray(y_match))

    @property
    def _learning_samples(self):
        x_train_idx, y_train = self.curr_episode_train
        faudio_uids = self.faudio_one_shot_uids[x_train_idx]
        faudio_feats = self.faudio_one_shot_feats[x_train_idx]
        images = self.flickr_vision_exp.one_shot_image_paths[x_train_idx]

        return (faudio_uids, faudio_feats, images, y_train)

    @property
    def _evaluation_samples(self):
        x_query_idx, y_query = self.curr_episode_test[:2]
        faudio_query_uids = self.faudio_one_shot_uids[x_query_idx]
        faudio_query_feats = self.faudio_one_shot_feats[x_query_idx]
        query_images = self.flickr_vision_exp.one_shot_image_paths[x_query_idx]

        x_match_idx, y_match = self.curr_episode_test[2:]
        faudio_match_uids = self.faudio_one_shot_uids[x_match_idx]
        faudio_match_feats = self.faudio_one_shot_feats[x_match_idx]
        match_images = self.flickr_vision_exp.one_shot_image_paths[x_match_idx]

        return (
            faudio_query_uids, faudio_query_feats, query_images, y_query,
            faudio_match_uids, faudio_match_feats, match_images, y_match)
