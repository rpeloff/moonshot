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
import tensorflow as tf


from moonshot.baselines import base


def test_l_way_k_shot(experiment, k, l, n=15, num_episodes=400,
                      k_neighbours=1, metric="cosine", dtw=False,
                      classification=False, model=None,
                      data_preprocess_func=None, embedding_model_func=None,
                      fine_tune_steps=None, fine_tune_lr=1e-2,
                      reset_experiment=True):
    """Perform L-way K-shot experiment with specified metric and model.

    If `model` is not specified, baseline metric results are computed.
    """

    #
    if fine_tune_steps is not None:
        base_weights = model.weights
        # gd_optimizer = base.gradient_descent_optimizer(fine_tune_lr)
        gd_optimizer = tf.keras.optimizers.Adam(fine_tune_lr)  # TODO

    # make sure experiment is reproducible in the case of multiple tests
    if reset_experiment:
        experiment.reset()

    task_accuracies = []
    for _ in tqdm(range(num_episodes)):

        experiment.sample_episode(l, k, n)

        x_train, y_train = experiment.learning_samples
        x_test, y_test = experiment.evaluation_samples

        if data_preprocess_func is not None:
            x_train = data_preprocess_func(x_train)
            x_test = data_preprocess_func(x_test)

        # TODO: does this cover all cases? or `y_train = label_preprocess_func(y_train)` instead?
        keyword_classes = list(sorted(set(y_train)))
        keyword_id_lookup = {
            keyword: idx for idx, keyword in enumerate(keyword_classes)}

        y_train = map(lambda keyword: keyword_id_lookup[keyword], y_train)
        y_train = np.stack(list(y_train))

        y_test = map(lambda keyword: keyword_id_lookup[keyword], y_test)
        y_test = np.stack(list(y_test))

        # embed preprocessed data with model if specified
        if model is not None:
            task_model = model

            # adapt model on learning samples if adapt function specified
            if fine_tune_steps is not None:
                if len(gd_optimizer.weights) != 0:  # TODO reset here so no state leakage!
                    zeros_init = [tf.zeros_like(var) for var in gd_optimizer.weights[1:]]
                    gd_optimizer.set_weights([np.int64(0)] + zeros_init)

                task_model.weights = base_weights
                for _ in range(fine_tune_steps):
                    task_model.train_step(
                        x_train, y_train, optimizer=gd_optimizer, training=True)

            # create embedding model from task model if specified
            if embedding_model_func is not None:
                task_model = embedding_model_func(task_model.model)

            # compute embeddings of task data
            x_train = task_model.predict(x_train)
            x_test = task_model.predict(x_test)

        # get nearest neighbour predictions
        if not classification:
            test_predict = base.knn(
                x_query=x_test, x_memory=x_train, y_memory=y_train,
                k_neighbours=k_neighbours, metric=metric, dtw=dtw)
        # get class predictions from logits (or softmax) test outputs
        else:
            test_predict = tf.argmax(x_test, axis=1)

        # compute task accuracy and store result
        num_correct = np.sum(test_predict == y_test)
        task_accuracies.append(num_correct / len(y_test))

    # restore model base weights in case of fine tuning
    if fine_tune_steps is not None:
        model.weights = base_weights

    # compute and return test result statistics
    mean_accuracy = np.mean(task_accuracies, axis=0)
    std_dev = np.std(task_accuracies, axis=0)
    conf_interval_95 = 1.96 * std_dev / np.sqrt(num_episodes)

    return mean_accuracy, std_dev, conf_interval_95
