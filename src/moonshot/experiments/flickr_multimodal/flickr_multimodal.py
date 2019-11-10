"""TODO(rpeloff)

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: September 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl import logging


import numpy as np


from moonshot.experiments.flickr_speech import flickr_speech
from moonshot.experiments.flickr_vision import flickr_vision


class FlickrMultimodal(flickr_speech.FlickrSpeech):

    def __init__(self, features="mfcc", keywords_split="one_shot_evaluation",
                 flickr8k_image_dir=None, speech_embed_dir=None,
                 image_embed_dir=None, image_preprocess_func=None,
                 speech_preprocess_func=None, speaker_mode="baseline",
                 unseen_match_set=False, **kwargs):
        """TODO"""
        logging.log(logging.INFO, f"Creating Flickr multimodal experiment")

        super().__init__(
            features=features, keywords_split=keywords_split,
            embed_dir=speech_embed_dir, preprocess_func=speech_preprocess_func,
            speaker_mode=speaker_mode, **kwargs)

        # TODO: add flickr30k/mscoco and sample paired audio with same keywords?
        self.flickr_vision_exp = flickr_vision.FlickrVision(
            keywords_split=keywords_split,
            flickr8k_image_dir=flickr8k_image_dir, flickr30k_image_dir=None,
            mscoco_image_dir=None, embed_dir=image_embed_dir,
            preprocess_func=image_preprocess_func, **kwargs)

        self.unseen_match_set = unseen_match_set

    @property
    def data(self):
        return (self.speech_experiment.data, self.vision_experiment.data)

    def _sample_episode(self, L, K, N, episode_labels=None):

        # sample a Flickr-Audio experiment episode to get train and query sets
        curr_episode_train, curr_episode_test = super()._sample_episode(
            L=L, K=K, N=N, episode_labels=episode_labels)

        x_train_idx, y_train = curr_episode_train
        x_query_idx, y_query = curr_episode_test

        # get valid image indices for the match set not seen in train & query
        sampled_image_uids = []
        sampled_image_uids.extend(
            self.faudio_metadata[3][x_train_idx])
        sampled_image_uids.extend(
            self.faudio_metadata[3][x_query_idx])

        valid_match_image_idx = np.where(np.invert(np.isin(
            self.faudio_metadata[3], sampled_image_uids)))

        # sample a new L-way match set FOR EACH query
        x_match_idx, y_match = [], []
        for query_label in y_query:

            # match set can contain labels not seen in train samples
            if self.unseen_match_set:
                y_query_match = self.rng.choice(
                    self.keywords, L - 1, replace=False)
                y_query_match = np.concatenate(([query_label], y_query_match))

            # match set contains only labels seen in train samples
            else:
                y_query_match = np.unique(y_train)

            self.rng.shuffle(y_query_match)

            # sample data indices for each match set label
            x_query_match_idx = []
            for match_label in y_query_match:
                # sample only unique and unseen images
                valid_idx = np.intersect1d(
                    valid_match_image_idx,
                    self.flickr_vision_exp.class_unique_indices[match_label])

                rand_cls_idx = self.rng.choice(
                    valid_idx, 1,
                    replace=False)

                x_query_match_idx.extend(rand_cls_idx)

            x_match_idx.append(x_query_match_idx)
            y_match.append(y_query_match)

        # store episode multimodal train set
        curr_episode_train = np.asarray(x_train_idx), np.asarray(y_train)
        self.curr_episode_train = curr_episode_train

        # store episode cross-modal test queries and corresponding matching sets
        curr_episode_test = (
            np.asarray(x_query_idx), np.asarray(y_query),
            np.asarray(x_match_idx), np.asarray(y_match))
        self.curr_episode_test = curr_episode_test

        return curr_episode_train, curr_episode_test

    @property
    def _learning_samples(self):
        return (
            self.speech_experiment.data[self.curr_episode_train[0]],
            self.vision_experiment.data[self.curr_episode_train[0]],
            self.curr_episode_train[1])

    @property
    def _evaluation_samples(self):
        return (
            # queries
            self.speech_experiment.data[self.curr_episode_test[0]],
            self.vision_experiment.data[self.curr_episode_test[0]],
            self.curr_episode_test[1],
            # matching sets
            self.speech_experiment.data[self.curr_episode_test[2]],
            self.vision_experiment.data[self.curr_episode_test[2]],
            self.curr_episode_test[3])

    @property
    def speech_experiment(self):
        return super()

    @property
    def vision_experiment(self):
        return self.flickr_vision_exp
