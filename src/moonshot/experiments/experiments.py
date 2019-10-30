"""TODO(rpeloff)

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: October 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tqdm import tqdm
import numpy as np


from moonshot.baselines.pixel_match import pixel_match


def test_l_way_k_shot(experiment, k, l, n=15, num_episodes=400, metric="cosine",
                      num_neighbours=1, data_preprocess_func=None):

    keyword_classes = list(sorted(set(experiment.keywords)))
    keyword_id_lookup = {
        keyword: idx for idx, keyword in enumerate(keyword_classes)}

    # TODO: switch to base nearest neighbour comparator
    pixel_model = pixel_match.PixelMatch(metric=metric)

    task_accuracies = []

    experiment.reset()
    for _ in tqdm(range(num_episodes)):

        experiment.sample_episode(l, k, n)

        x_train, y_train = experiment.learning_samples
        x_test, y_test = experiment.evaluation_samples

        if data_preprocess_func is not None:
            x_train = data_preprocess_func(x_train)
            x_test = data_preprocess_func(x_test)

        # TODO: does this cover other cases? or y_train = label_preprocess_func(y_train)?
        y_train_labels = map(lambda keyword: keyword_id_lookup[keyword], y_train)
        y_train_labels = np.stack(list(y_train_labels))

        y_test_labels = map(lambda keyword: keyword_id_lookup[keyword], y_test)
        y_test_labels = np.stack(list(y_test_labels))

        adapted_model = pixel_model.adapt_model(x_train, y_train_labels)

        test_predict = adapted_model.predict(x_test, num_neighbours)

        num_correct = np.sum(test_predict == y_test_labels)

        task_accuracies.append(num_correct / len(y_test))

    mean_accuracy = np.mean(task_accuracies, axis=0)
    std_dev = np.std(task_accuracies, axis=0)
    conf_interval_95 = 1.96 * std_dev / np.sqrt(num_episodes)

    return mean_accuracy, std_dev, conf_interval_95
