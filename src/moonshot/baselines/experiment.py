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
                      classification=False, random=False, model=None,
                      embedding_model_func=None,
                      fine_tune_steps=None, fine_tune_lr=1e-2,
                      reset_experiment=True):
    """Perform L-way K-shot experiment with specified metric.

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

        # TODO: does this cover all cases? or `y_train = label_preprocess_func(y_train)` instead?
        keyword_classes = list(sorted(set(y_train)))
        keyword_id_lookup = {
            keyword: idx for idx, keyword in enumerate(keyword_classes)}
        missing_label = len(keyword_classes)

        y_train = map(lambda keyword: keyword_id_lookup.get(keyword, missing_label), y_train)
        y_train = np.stack(list(y_train))

        y_test = map(lambda keyword: keyword_id_lookup.get(keyword, missing_label), y_test)
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

        # get random action predictions (random sample train labels)
        if random:
            test_predict = experiment.rng.choice(
                y_train, size=len(y_test), replace=True)

        # get class predictions from logits (or softmax) test outputs
        elif classification:
            test_predict = tf.argmax(x_test, axis=1)

        # get nearest neighbour predictions
        else:
            test_predict = base.knn(
                x_query=x_test, x_memory=x_train, y_memory=y_train,
                k_neighbours=k_neighbours, metric=metric, dtw=dtw)

        # compute task accuracy and store result
        num_correct = np.sum(test_predict == y_test)
        task_accuracies.append(num_correct / len(y_test))

    # restore model base weights in case of fine tuning
    if fine_tune_steps is not None and model is not None:
        model.weights = base_weights

    # compute and return test result statistics
    mean_accuracy = np.mean(task_accuracies, axis=0)
    std_dev = np.std(task_accuracies, axis=0)
    conf_interval_95 = 1.96 * std_dev / np.sqrt(num_episodes)

    return mean_accuracy, std_dev, conf_interval_95


def test_multimodal_l_way_k_shot(
        experiment, k, l, n=15, num_episodes=400, k_neighbours=1,
        metric="cosine", speech_dtw=False, random=False, direct_match=False,
        multimodal_model=None, multimodal_embedding_func=None, optimizer=None,
        fine_tune_steps=None, fine_tune_lr=1e-2, reset_experiment=True):
    """Perform multimodal L-way K-shot experiment with specified metric.

    If `multimodal_model` or both `speech_model` and `vision_model` are not
    specified, baseline metric results are computed.
    """

    #
    if fine_tune_steps is not None:
        base_a_weights = multimodal_model.speech_model.weights
        base_b_weights = multimodal_model.vision_model.weights

        # gd_optimizer = base.gradient_descent_optimizer(fine_tune_lr)

        init_adam = False
        gd_optimizer = optimizer

        if gd_optimizer is None:
            init_adam = True
            gd_optimizer = tf.keras.optimizers.Adam(fine_tune_lr)  # TODO

    # make sure experiment is reproducible in the case of multiple tests
    if reset_experiment:
        experiment.reset()

    l2_norm_layer = tf.keras.layers.Lambda(
            lambda x: tf.nn.l2_normalize(x, axis=1))

    task_accuracies = []
    aux_1_accuracies = []
    aux_2_accuracies = []
    for _ in tqdm(range(num_episodes)):

        experiment.sample_episode(l, k, n)

        x_train_s, x_train_i, y_train = experiment.learning_samples
        x_query_s, x_query_i, y_query, x_match_s, x_match_i, y_match = (
            experiment.evaluation_samples)

        # TODO: does this cover all cases? or `y_train = label_preprocess_func(y_train)` instead?
        keyword_classes = list(sorted(set(y_train)))
        keyword_id_lookup = {
            keyword: idx for idx, keyword in enumerate(keyword_classes)}
        missing_label = len(keyword_classes)

        y_train = map(lambda keyword: keyword_id_lookup.get(keyword, missing_label), y_train)
        y_train = np.stack(list(y_train))

        y_query = map(lambda keyword: keyword_id_lookup.get(keyword, missing_label), y_query)
        y_query = np.stack(list(y_query))

        y_match = np.apply_along_axis(
            lambda labels: np.stack(
                list(map(lambda keyword: keyword_id_lookup.get(keyword, missing_label), labels))),
            1, y_match)

        # embed preprocessed data with model if specified
        if multimodal_model is not None:
            task_model = multimodal_model

            # adapt model on learning samples if adapt function specified
            if fine_tune_steps is not None:
                if init_adam and len(gd_optimizer.weights) != 0:  # TODO reset here so no state leakage!
                    zeros_init = [tf.zeros_like(var) for var in gd_optimizer.weights[1:]]
                    gd_optimizer.set_weights([np.int64(0)] + zeros_init)

                task_model.speech_model.weights = base_a_weights
                task_model.vision_model.weights = base_b_weights
                for _ in range(fine_tune_steps):
                    task_model.train_step(
                        x_train_s, x_train_i, optimizer=gd_optimizer, training=True)

            # create embedding model from multimodal task model if specified
            if multimodal_embedding_func is not None: # TODO:  creates new graphs -> slow!
                task_model.speech_model = multimodal_embedding_func(task_model.speech_model.model)
                task_model.vision_model = multimodal_embedding_func(task_model.vision_model.model)

            # compute embeddings of task data TODO remove l2 norm
            x_train_s = np.asarray(l2_norm_layer(task_model.speech_model.predict(x_train_s)))
            x_query_s = np.asarray(l2_norm_layer(task_model.speech_model.predict(x_query_s)))
            match_s = []
            for x_s in x_match_s:
                match_s.append(l2_norm_layer(task_model.speech_model.predict(x_s)))
            x_match_s = np.stack(match_s)

            x_train_i = np.asarray(l2_norm_layer(task_model.vision_model.predict(x_train_i)))
            x_query_i = np.asarray(l2_norm_layer(task_model.vision_model.predict(x_query_i)))
            match_i = []
            for x_i in x_match_i:
                match_i.append(l2_norm_layer(task_model.vision_model.predict(x_i)))
            x_match_i = np.stack(match_i)

        # TODO make optional for reverse task or within modality matching?
        x_query = x_query_s
        x_match = x_match_i

        x_aux_1 = x_train_s
        x_aux_2 = x_train_i
        aux_dtw_1 = True if speech_dtw else False
        aux_dtw_2 = False

        # get random action predictions (random sample from match sets)
        if random:
            num_queries, num_match = np.shape(y_match)
            test_predict = experiment.rng.choice(
                num_match, size=num_queries, replace=True)

        # get nearest neighbour predictions
        else:

            # predict auxiliary query in target modality through the train set
            if not direct_match:

                train_predict_idx = base.knn(
                    x_query=x_query, x_memory=x_aux_1,
                    y_memory=np.arange(len(x_aux_1)), k_neighbours=k_neighbours,
                    metric=metric, dtw=aux_dtw_1)

                x_query = x_aux_2[train_predict_idx]

                aux_1_predict = y_train[train_predict_idx]
                aux_1_accuracies.append(
                    np.sum(aux_1_predict == y_query) / len(y_query))

            # retrieve match set samples most similar to each query 
            test_predict = []
            for query, match_set, match_labels in zip(x_query, x_match, y_match):
                query = np.stack([query])

                test_predict.extend(
                    base.knn(x_query=query, x_memory=match_set,
                             y_memory=match_labels, k_neighbours=k_neighbours,
                             metric=metric, dtw=aux_dtw_2))

            test_predict = np.stack(test_predict)

            if not direct_match:
                aux_2_accuracies.append(
                    np.sum(aux_1_predict == test_predict) / len(test_predict))

        # compute task accuracy and store result
        task_accuracies.append(
            np.sum(test_predict == y_query) / len(y_query))

    # restore model base weights in case of fine tuning
    if fine_tune_steps is not None and multimodal_model is not None:
        multimodal_model.speech_model.weights = base_a_weights
        multimodal_model.vision_model.weights = base_b_weights

    # compute and return test result statistics
    mean_accuracy = np.mean(task_accuracies, axis=0)
    std_dev = np.std(task_accuracies, axis=0)
    conf_interval_95 = 1.96 * std_dev / np.sqrt(num_episodes)

    return mean_accuracy, std_dev, conf_interval_95
