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


from moonshot.data import flickr_audio
from moonshot.experiments import base
from moonshot.utils import file_io


# standard scaling mean and variance computed from training data (background_train.csv)

# global statistics across all dimensions
train_global_mean = {}  # pylint: disable=invalid-name
train_global_var = {}  # pylint: disable=invalid-name

train_global_mean["mfcc"] = -0.0010950847199735192
train_global_var["mfcc"] = 0.3567110590621045
train_global_mean["fbank"] = 17.509969510358815
train_global_var["fbank"] = 14.452916650749247

# statistics per feature dimension
train_features_mean = {}  # pylint: disable=invalid-name
train_features_var = {}  # pylint: disable=invalid-name

train_features_mean["mfcc"] = np.array([
    +1.32153148e-01, +5.34269911e-02, -2.15415947e-02, -4.94970795e-02,
    -1.23090949e-01, +1.40466060e-02, +9.75989624e-02, +1.47936022e-02,
    -4.13564958e-02, -2.91437051e-02, -5.80961896e-03, -1.96842286e-02,
    -4.55897921e-02, +2.44488166e-03, -3.23168103e-03, -6.59671831e-03,
    -2.67478198e-03, -3.82002095e-03, -3.47037479e-03, -3.26529530e-03,
    +1.05246510e-03, +1.89005922e-03, -2.15055509e-04, +1.25535136e-03,
    -1.09850116e-03, +8.23052806e-05, +1.83033459e-03, +1.21922030e-03,
    -1.84041176e-03, +6.59660647e-05, -9.16896159e-04, -1.48896116e-03,
    -1.10792069e-03, -6.03055868e-04, +7.30406810e-04, +4.05984679e-04,
    +2.76476503e-04, -1.50802754e-05, +7.71538868e-05])

train_features_var["mfcc"] = np.array([
    0.91296818, 1.00177075, 1.04598884, 0.97943741, 1.10167072, 0.98497306,
    1.03107042, 0.99571673, 0.99097156, 0.98902959, 1.00651163, 1.01512197,
    1.01092469, 0.04358867, 0.03717848, 0.04696080, 0.03748896, 0.04700193,
    0.04922860, 0.05028348, 0.05473835, 0.05797368, 0.05976523, 0.06351620,
    0.06509738, 0.06707076, 0.00604690, 0.00509468, 0.00710361, 0.00539015,
    0.00699638, 0.00804983, 0.00827325, 0.00931896, 0.01004667, 0.01042689,
    0.01145421, 0.01172219, 0.01229539])

train_features_mean["fbank"] = np.array([
    14.94743563, 16.45409437, 17.13997318, 17.21678179, 17.46014044,
    17.88050357, 18.14726162, 18.33959425, 18.21709028, 18.09417002,
    17.90817997, 17.79310673, 17.69135520, 17.57243688, 17.44124024,
    17.42636888, 17.58523693, 17.76074110, 17.89012335, 17.92002578,
    17.96491240, 18.03654056, 18.12163615, 18.21561456, 18.19172980,
    18.14125008, 18.16387385, 18.19773438, 18.19301181, 18.05073942,
    17.81676462, 17.59843241, 17.36506128, 17.01731537, 16.67868699,
    16.57665983, 16.61765614, 16.59339746, 16.32898486, 15.64291821])

train_features_var["fbank"] = np.array([
    +9.80457338, 11.02937828, 12.10755132, 11.95119139, 12.28248128,
    13.20537295, 14.11372048, 15.14463048, 15.78990233, 15.72936093,
    15.36991563, 15.08655164, 14.52818211, 14.26114153, 14.22071484,
    14.07992226, 13.87976692, 13.91862834, 14.09054846, 13.93680669,
    13.58307387, 13.43652145, 13.53714681, 13.73358586, 13.91080542,
    13.64155872, 13.51368075, 13.79055016, 13.91272786, 13.51251354,
    13.36768644, 13.75936469, 14.12302948, 14.35858512, 14.42953534,
    14.69363512, 14.73356627, 14.96371951, 14.97985992, 14.632985])


class FlickrSpeech(base.Experiment):

    def __init__(self, features="mfcc", keywords_split="one_shot_evaluation",
                 embed_dir=None, **kwargs):
        """TODO

        `features` one of `["mfcc", "fbank"]`.

        `keywords_split` one of `['one_shot_evaluation', 'one_shot_development',
        'background_train', 'background_dev', 'background_test']`.
        """
        super().__init__(**kwargs)

        logging.log(logging.INFO, f"Creating Flickr audio experiment")

        assert features in ["mfcc", "fbank"]

        assert keywords_split in [
            "one_shot_evaluation", "one_shot_development", "background_train",
            "background_dev", "background_test"]

        if keywords_split == "background_test":
            subset = "test"
        elif keywords_split == "background_dev":
            subset = "dev"
        else:  # rest fall under train subset
            subset = "train"

        # load Flickr 8k keywords set
        keywords_path = os.path.join(
            "data", "splits", "flickr8k", f"{keywords_split}.csv")
        keywords_set = file_io.read_csv(keywords_path, skip_first=True)

        # load aligned Flickr Audio UIDs and metadata
        faudio_path = os.path.join(
            "data", "splits", "flickr8k", f"faudio_{keywords_split}.txt")
        faudio_uids = file_io.read_csv(faudio_path)[0]
        self.faudio_uids = np.asarray(faudio_uids)

        self.faudio_metadata = flickr_audio.extract_all_uid_metadata(
            self.faudio_uids)

        # load audio paths
        audio_paths = flickr_audio.fetch_audio_paths(
            os.path.join("data", "processed", "flickr_audio", features, subset),
            self.faudio_uids)

        # load audio embedding paths if specified
        if embed_dir is not None:
            embed_paths = []
            for audio_uid in keywords_set[0]:
                embed_paths.append(
                    os.path.join(
                        embed_dir, "flickr_audio", f"{keywords_split}",
                        f"{audio_uid}.tfrecord"))
                assert os.path.exists(embed_paths[-1])

        self.keywords_set = tuple(np.asarray(x) for x in keywords_set)
        self.audio_paths = np.asarray(audio_paths)
        self.embed_paths = None
        if embed_dir is not None:
            self.embed_paths = np.asarray(embed_paths)

        # get unique keywords and keyword class label lookup dict
        self.keywords = sorted(np.unique(self.keywords_set[3]).tolist())
        self.keyword_labels = {
            keyword: idx for idx, keyword in enumerate(self.keywords)}

        # get unique speakers and valid distractor speakers and labels
        self.speakers = np.unique(self.faudio_metadata[2])

        distractor_speaker_labels = {}
        for speaker in self.speakers:
            speaker_idx = np.where(
                self.faudio_metadata[2] == speaker)[0]
            unique_keywords, counts = np.unique(
                self.keywords_set[3][speaker_idx], return_counts=True)

            speaker_labels = []
            for keyword, count in zip(unique_keywords, counts):
                if count > 5:  # constrain min. training samples per keyword
                    speaker_labels.append(self.keyword_labels[keyword])

            if len(speaker_labels) < 10:  # constrain min. keywords per speaker
                continue
            else:
                distractor_speaker_labels[speaker] = speaker_labels

        self.distractor_speaker_labels = distractor_speaker_labels

        # get lookup for unique indices per class label
        self.class_unique_indices = {}
        for keyword in self.keywords:
            label = self.keyword_labels[keyword]
            cls_idx = np.where(self.keywords_set[3] == keyword)[0]

            self.class_unique_indices[label] = cls_idx

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
                len(self.keywords), L, replace=False)

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
                        self.faudio_metadata[2],
                        list(self.distractor_speaker_labels.keys())))[0]
                    valid_speaker_indices = np.intersect1d(
                        valid_speaker_indices,
                        np.where(
                            self.faudio_metadata[2] != query_speaker)[0])
                else:
                    # choose same speaker
                    valid_speaker_indices = np.where(
                        self.faudio_metadata[2] == query_speaker)[0]

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
            train_speakers = self.faudio_metadata[2][x_train_idx]
            valid_speaker_indices = np.where(np.invert(np.isin(
                self.faudio_metadata[2], train_speakers)))[0]
        if speaker_mode == "distractor":  # all evaluation samples same label
            y_test = [query_label] * N
            # choose same speaker
            valid_speaker_indices = np.where(
                self.faudio_metadata[2] == query_speaker)[0]

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
        if self.embed_paths is None:
            return (
                self.audio_paths[self.curr_episode_train[0]],
                self.curr_episode_train[1])
        else:
            return (
                self.embed_paths[self.curr_episode_train[0]],
                self.curr_episode_train[1])

    @property
    def _evaluation_samples(self):
        if self.embed_paths is None:
            return (
                self.audio_paths[self.curr_episode_test[0]],
                self.curr_episode_test[1])
        else:
            return (
                self.embed_paths[self.curr_episode_test[0]],
                self.curr_episode_test[1])
