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


from moonshot.data import flickraudio
from moonshot.experiments import base
from moonshot.utils import file_io


# standard scaling mean and variance computed from background one-shot train set
# global statistics across all dimensions
train_global_mean = {}  # pylint: disable=invalid-name
train_global_var = {}  # pylint: disable=invalid-name

train_global_mean["mfcc"] = -0.0012316286453624911
train_global_var["mfcc"] = 0.357552599879396
train_global_mean["fbank"] = 17.536853868827944
train_global_var["fbank"] = 14.451713587029394

# statistics per feature dimension
train_features_mean = {}  # pylint: disable=invalid-name
train_features_var = {}  # pylint: disable=invalid-name

train_features_mean["mfcc"] = np.array([
    1.45404133e-01, 7.05124218e-02, -2.25155954e-02, -5.27207242e-02,
    -1.23470517e-01, 1.41696135e-02, 9.70206899e-02, 1.00961395e-02,
    -4.04510896e-02, -3.70518759e-02, -1.43785842e-02, -2.31842051e-02,
    -5.13556296e-02, 2.30975992e-03, -3.59762721e-03, -6.86087161e-03,
    -2.91658676e-03, -3.84707507e-03, -3.33740578e-03, -3.44501179e-03,
    1.12572620e-03, 2.17261901e-03, -3.00884583e-04, 1.30207316e-03,
    -1.14561179e-03, -2.95034939e-06, 1.83845552e-03, 1.01052150e-03,
    -1.93420298e-03, 7.50036265e-05, -9.49022869e-04, -1.54850954e-03,
    -1.14620760e-03, -5.92800875e-04, 7.44965619e-04, 4.86446171e-04,
    3.07715411e-04, 2.39644397e-05, 1.19223641e-04])

train_features_var["mfcc"] = np.array([
    0.90495510, 1.00278979, 1.05603216, 0.97615661, 1.10716870, 0.97989772,
    1.03440061, 0.99740001, 0.99416635, 0.99008145, 1.01287372, 1.02063596,
    1.01535574, 0.04315907, 0.03666656, 0.04706265, 0.03750768, 0.04708205,
    0.04918479, 0.05042579, 0.05484876, 0.05810199, 0.05975362, 0.06352696,
    0.06517387, 0.06725218, 0.00593720, 0.00503791, 0.00705450, 0.00539502,
    0.00699791, 0.00805341, 0.00829784, 0.00932410, 0.01005616, 0.01042311,
    0.01144392, 0.01171824, 0.01232431])

train_features_mean["fbank"] = np.array([
    15.0010193, 16.52221203, 17.21956436, 17.28917779, 17.52829303,
    17.9476045, 18.20919269, 18.39880114, 18.27553108, 18.15116583,
    17.96383801, 17.85039428, 17.74765765, 17.62426303, 17.48809714,
    17.46712867, 17.62177132, 17.79543066, 17.92447276, 17.94926686,
    17.99152793, 18.06130667, 18.13777556, 18.22327715, 18.19228442,
    18.13901125, 18.16512201, 18.20195995, 18.19529178, 18.04613382,
    17.807884, 17.58705372, 17.35348645, 17.00204111, 16.6604676,
    16.56610464, 16.60924063, 16.58555385, 16.32581397, 15.64793613])

train_features_var["fbank"] = np.array([
    9.70291673, 10.86865288, 11.91190246, 11.77119552, 12.12582745,
    13.06724347, 14.00404348, 15.06750012, 15.74459199, 15.70382931,
    15.35208757, 15.10357258, 14.57729958, 14.32143048, 14.28153334,
    14.1243881, 13.91507576, 13.95722025, 14.14770297, 13.99625825,
    13.64077315, 13.4780904, 13.5526731, 13.73474199, 13.90657549,
    13.63569483, 13.53585278, 13.83639229, 13.96122663, 13.5603344,
    13.40241177, 13.78019927, 14.15885987, 14.40267803, 14.45279511,
    14.72349228, 14.74706251, 14.97020874, 15.00968267, 14.6490656])


class FlickrAudio(base.Experiment):
    """TODO(rpeloff).
    """

    def __init__(self, audio_dir,
                 splits_dir=os.path.join("data", "splits", "flickr8k"),
                 preprocess=None, **kwargs):
        super(FlickrAudio, self).__init__(**kwargs)

        # load Flickr 8k one-shot evaluation keywords set
        one_shot_keywords_set = file_io.read_csv(
            os.path.join(splits_dir, "one_shot_evaluation.csv"),
            skip_first=True)
        self.one_shot_keywords_set = tuple(  # convert to numpy arrays
            np.asarray(x) for x in one_shot_keywords_set)

        # load aligned Flickr-Audio one-shot evaluation UIDs and metadata
        self.faudio_one_shot_uids = np.asarray(file_io.read_csv(
            os.path.join(splits_dir, "faudio_one_shot_evaluation.txt"))[0])

        self.faudio_one_shot_metadata = flickraudio.extract_all_uid_metadata(
            self.faudio_one_shot_uids)

        # load Flickr-Audio NumPy archive file handles
        faudio_npz = flickraudio.load_isolated_word_npz(audio_dir)

        # load Flickr-Audio speech features
        faudio_one_shot_feats = []
        for uid in self.faudio_one_shot_uids:
            faudio_one_shot_feats.append(faudio_npz["train"][uid])

        for _, npz in faudio_npz.items():
            npz.close()

        # apply preprocess function to each speech segment if specified
        if preprocess is not None:
            faudio_one_shot_feats = [
                preprocess(x) for x in faudio_one_shot_feats]

        # stack speech feature arrays
        self.faudio_one_shot_feats = np.asarray(faudio_one_shot_feats)

        # get unique one-shot keywords and class label lookup dict
        self.one_shot_keywords = np.unique(self.one_shot_keywords_set[3])

        self.keyword_id_lookup = {
            keyword: idx for idx, keyword in enumerate(self.one_shot_keywords)}

        # get unique one-shot speakers and valid distractor speakers and labels
        self.one_shot_speakers = np.unique(self.faudio_one_shot_metadata[2])

        distractor_speaker_labels = {}
        for speaker in self.one_shot_speakers:
            speaker_idx = np.where(
                self.faudio_one_shot_metadata[2] == speaker)[0]
            unique_keywords, counts = np.unique(
                self.one_shot_keywords_set[3][speaker_idx], return_counts=True)

            speaker_labels = []
            for keyword, count in zip(unique_keywords, counts):
                if count > 5:  # constrain min. training samples per keyword
                    speaker_labels.append(self.keyword_id_lookup[keyword])

            if len(speaker_labels) < 10:  # constrain min. keywords per speaker
                continue
            else:
                distractor_speaker_labels[speaker] = speaker_labels

        # results in 7 distinct speakers, 42 unique classes and 1577 samples
        self.distractor_speaker_labels = distractor_speaker_labels

        # get lookup for unique indices per class label
        self.class_unique_indices = {}
        for os_cls in self.one_shot_keywords:
            os_cls_label = self.keyword_id_lookup[os_cls]

            cls_idx = np.where(self.one_shot_keywords_set[3] == os_cls)[0]

            self.class_unique_indices[os_cls_label] = cls_idx

    def _sample_episode(self, L, K, N, speaker_mode="baseline",
                        episode_labels=None):
        """TODO

        `speaker_mode` options:
            "baseline"
            - randomly choose learning and evaluation samples in episode labels

            "difficult":
            - choose learning samples as usual
            - choose evaluation samples from speakers not seen during learning

            "distractor":
            - choose a random speaker for evaluation
            - choose a random label for evaluation
            - choose learning samples from same speaker as evaluation
            - except for the evaluation label, sample from different speaker(s)
        """
        assert speaker_mode in ["baseline", "difficult", "distractor"]

        # sample episode learning task (defined by L-way classes)
        if episode_labels is None:
            episode_labels = self.rng.choice(
                len(self.one_shot_keywords), L, replace=False)

        if speaker_mode == "distractor":
            # choose a random speaker & label (from valid set) for evaluation
            query_speaker = self.rng.choice(
                list(self.distractor_speaker_labels.keys()), 1)[0]

            episode_labels = self.rng.choice(
                self.distractor_speaker_labels[query_speaker], L, replace=False)

            query_label = self.rng.choice(episode_labels, 1)[0]

        # sample learning examples from episode task
        x_train_idx, y_train = [], []
        for ep_label in episode_labels:

            valid_class_indices = self.class_unique_indices[ep_label]

            if speaker_mode == "distractor":
                if ep_label == query_label:
                    # choose different speakers
                    valid_speaker_indices = np.where(np.isin(
                        self.faudio_one_shot_metadata[2],
                        list(self.distractor_speaker_labels.keys())))[0]
                    valid_speaker_indices = np.intersect1d(
                        valid_speaker_indices,
                        np.where(
                            self.faudio_one_shot_metadata[2] != query_speaker)[0])
                else:
                    # choose same speaker
                    valid_speaker_indices = np.where(
                        self.faudio_one_shot_metadata[2] == query_speaker)[0]

                valid_class_indices = np.intersect1d(
                    valid_class_indices, valid_speaker_indices)

            rand_cls_idx = self.rng.choice(
                valid_class_indices, K, replace=False)
            x_train_idx.extend(rand_cls_idx)
            y_train.extend([ep_label] * K)

        # sample evaluation examples from episode task
        ep_test_labels_idx = self.rng.choice(
            np.arange(len(episode_labels)), N, replace=True)
        y_test = episode_labels[ep_test_labels_idx]

        if speaker_mode == "difficult":  # choose different speakers
            train_speakers = self.faudio_one_shot_metadata[2][x_train_idx]
            valid_speaker_indices = np.where(np.invert(np.isin(
                self.faudio_one_shot_metadata[2], train_speakers)))[0]
        if speaker_mode == "distractor":  # all evaluation samples same label
            y_test = [query_label] * N
            # choose same speaker
            valid_speaker_indices = np.where(
                self.faudio_one_shot_metadata[2] == query_speaker)[0]

        x_test_idx = []
        for test_label in y_test:

            valid_class_indices = self.class_unique_indices[test_label]

            if speaker_mode == "difficult" or speaker_mode == "distractor":
                valid_class_indices = np.intersect1d(
                    valid_class_indices, valid_speaker_indices)

            rand_cls_idx = self.rng.choice(
                valid_class_indices, 1, replace=False)
            x_test_idx.extend(rand_cls_idx)

        self.curr_episode_train = np.asarray(x_train_idx), np.asarray(y_train)
        self.curr_episode_test = np.asarray(x_test_idx), np.asarray(y_test)

    @property
    def _learning_samples(self):
        faudio_uids = self.faudio_one_shot_uids[self.curr_episode_train[0]]
        faudio_feats = self.faudio_one_shot_feats[self.curr_episode_train[0]]

        return (faudio_uids, faudio_feats, self.curr_episode_train[1])

    @property
    def _evaluation_samples(self):
        faudio_uids = self.faudio_one_shot_uids[self.curr_episode_test[0]]
        faudio_feats = self.faudio_one_shot_feats[self.curr_episode_test[0]]

        return (faudio_uids, faudio_feats, self.curr_episode_test[1])
