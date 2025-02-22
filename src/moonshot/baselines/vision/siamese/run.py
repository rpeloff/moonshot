"""Train vision similarity network and test on Flickr one-shot image task.

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: October 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import datetime
import functools
import os
import time


from absl import app
from absl import flags
from absl import logging


import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm


from moonshot.baselines import base
from moonshot.baselines import dataset
from moonshot.baselines import experiment
from moonshot.baselines import inceptionv3
from moonshot.baselines import losses
from moonshot.baselines import model_utils

from moonshot.experiments.flickr_vision import flickr_vision

from moonshot.utils import file_io
from moonshot.utils import logging as logging_utils


FLAGS = flags.FLAGS


# model options (default if not loaded)
DEFAULT_OPTIONS = {
    # training data
    "data": ["flickr8k", "flickr30k", "mscoco"],
    "one_shot_validation": True,
    # preprocessing (if training from scratch)
    "crop_size": 299,
    "augment_train": True,
    "random_scales": None,
    "horizontal_flip": True,
    "colour": True,
    # data pipeline
    "batch_size": 32*4,  # used if "balanced": False
    "balanced": True,
    "p": 32,
    "k": 4,
    "num_batches": 2500,
    # siamese model
    "oracle": False,  # if using inceptionv3 network with ImageNet weights
    "pretrained": False,  # train base from scratch or ImageNet (if not using base embeddings/model)
    "dense_units": [1024],  # hidden layers on top of base network (last layer is linear)
    "dropout_rate": 0.2,
    # triplet objective
    "margin": 0.2,
    "metric": "squared_euclidean",  # or "euclidean", "cosine",
    # training
    "learning_rate": 3e-4,
    "decay_steps": 5000,
    "decay_rate": 0.96,
    "gradient_clip_norm": 5.,
    "epochs": 100,
    # that magic number
    "seed": 42
}

# one-shot evaluation (and validation) options
flags.DEFINE_integer("episodes", 400, "number of L-way K-shot learning episodes")
flags.DEFINE_integer("L", 5, "number of classes to sample in a task episode (L-way)")
flags.DEFINE_integer("K", 1, "number of task learning samples per class (K-shot)")
flags.DEFINE_integer("N", 15, "number of task evaluation samples")
flags.DEFINE_integer("k_neighbours", 1, "number of nearest neighbours to consider")
flags.DEFINE_string("metric", "cosine", "distance metric to use for nearest neighbours matching")
flags.DEFINE_integer("fine_tune_steps", None, "number of fine-tune gradient steps on one-shot data")
flags.DEFINE_float("fine_tune_lr", 1e-3, "learning rate for gradient descent fine-tune")
flags.DEFINE_bool("classification", False, "whether to use softmax predictions as match function"
                  "(requires fine-tuning of new logits layer)")

# model train/test options
flags.DEFINE_string("base_dir", None, "directory containing base network model")
flags.DEFINE_bool("l2_norm", True, "L2-normalise embedding predictions (as done in training)")
flags.DEFINE_bool("load_best", False, "load previous best model for resumed training or testing")
flags.DEFINE_bool("load_initial", False, "load initial model (epoch 0) for training or testing")
flags.DEFINE_bool("mc_dropout", False, "make embedding predictions with MC Dropout")
flags.DEFINE_bool("use_embeddings", False, "train on extracted embeddings from base model"
                  "(default loads base network ('<best_>model.h5') and fine-tunes siamese loss)")

# logging and target options
flags.DEFINE_enum("target", "train", ["train", "validate", "embed", "test"],
                  "train or load and test a model")
flags.DEFINE_string("output_dir", None, "directory where logs and checkpoints will be stored"
                    "(default is 'logs/<unique run id>')")
flags.DEFINE_bool("resume", True, "resume training if a checkpoint is found at output directory")
flags.DEFINE_bool("tensorboard", True, "log train and test summaries to TensorBoard")
flags.DEFINE_bool("debug", False, "log with debug verbosity level")


def get_training_objective(model_options):
    """Get training loss for ranking image similarity with siamese triplets."""

    triplet_loss = losses.triplet_semihard_loss(
        margin=model_options["margin"], metric=model_options["metric"])

    return triplet_loss


def get_data_preprocess_func(model_options):
    """Create data batch preprocessing function for input to the speech network.

    Returns function `data_preprocess_func` that takes a batch of file paths,
    loads image or base model embedding data and preprocesses the data.
    """

    # load images
    if model_options["base_dir"] is None or not model_options["use_embeddings"]:
        def data_preprocess_func(image_paths):
            images = []
            for image_path in image_paths:
                images.append(
                    dataset.load_and_preprocess_image(
                        image_path, crop_size=model_options["crop_size"]))

            return np.stack(images)

    # load base model embeddings
    else:
        def data_preprocess_func(embed_paths):
            embed_ds = tf.data.TFRecordDataset(
                embed_paths, compression_type="ZLIB", num_parallel_reads=8)
            # map sequential to prevent optimization overhead
            embed_ds = embed_ds.map(
                lambda example: dataset.parse_embedding_protobuf(
                    example)["embed"])

            return np.stack(list(embed_ds))

    return data_preprocess_func


def create_vision_network(model_options, build_model=True):
    """Create vision similarity network from model options."""

    # get input shape
    input_shape = None
    if build_model:
        input_shape = model_options["input_shape"]

    # train entire similarity network from scratch or with imagenet weights
    if model_options["base_dir"] is None:

        # oracle model with imagenet weights and class logits as embedding layer
        if model_options["oracle"]:
            if build_model:
                logging.log(
                    logging.INFO, "Fine-tuning oracle with imagenet weights")

            inception_network = inceptionv3.create_inceptionv3_network(
                input_shape=input_shape, pretrained=True, include_top=True)

            # train only the logits embeddings layer
            inceptionv3.freeze_weights(
                inception_network, trainable="logits")

            # set output layer to be linear (instead of softmax)
            inception_network.layers[-1].activation = None

            model_layers = [inception_network]

            # no additional dense layers if training oracle model
            model_options["dense_units"] = None

        # train network from scratch (or with imagenet weights for debugging)
        else:
            inception_network = inceptionv3.create_inceptionv3_network(
                input_shape=input_shape, pretrained=model_options["pretrained"],
                include_top=False)

            # inception model with imagenet weights and our own top dense layers
            # ... alternative "oracle"?
            if model_options["pretrained"]:
                if build_model:
                    logging.log(
                        logging.INFO,
                        "Debugging with imagenet weights and custom top layer")

                # train final inception module and the top dense layers
                inceptionv3.freeze_weights(
                    inception_network, trainable="final_inception")

            # inception model with random weights and our own top dense layers
            elif build_model:
                logging.log(logging.INFO, "Training entire model from scratch")

            model_layers = [
                inception_network,
                tf.keras.layers.GlobalAveragePooling2D()]

            if model_options["dropout_rate"] is not None:
                model_layers.append(
                    tf.keras.layers.Dropout(model_options["dropout_rate"]))

            model_layers.append(
                tf.keras.layers.Dense(model_options["dense_units"][0]))

    # train similarity ranking layers on base model or its extracted embeddings
    else:

        # train dense layers on extracted base embeddings
        if model_options["use_embeddings"]:
            if build_model:
                logging.log(logging.INFO, "Training model on base embeddings")

            if model_options["dropout_rate"] is not None:
                model_layers = [
                    tf.keras.layers.Dropout(
                        model_options["dropout_rate"], input_shape=input_shape),
                    tf.keras.layers.Dense(model_options["dense_units"][0])]
            else:
                model_layers = [
                    tf.keras.layers.Dense(
                        model_options["dense_units"][0],
                        input_shape=input_shape)]

        # load base model and fine-tune final layer
        else:
            if build_model:
                logging.log(
                    logging.INFO, "Warm start training from pretrained network")

            base_model_file = os.path.join(
                model_options["base_dir"], f"{model_options['base_model']}.h5")
            base_step_file = os.path.join(
                model_options["base_dir"], f"{model_options['base_model']}.step")

            base_network, _ = model_utils.load_model(
                model_file=base_model_file, model_step_file=base_step_file,
                loss=lambda y_t, y_p: y_p)  # dummy loss to load model ..

            # set output layer to be linear (in case of softmax, sigmoid, etc.)
            base_network.layers[-1].activation = None

            # fine-tune only the final base embedding layer
            for layer in base_network.layers[:-1]:
                layer.trainable = False
            base_network.layers[-1].trainable = True

            model_layers = [base_network]

            # no additional dense layers
            model_options["dense_units"] = None

    # add top layer hidden units (final layer linear)
    if model_options["dense_units"] is not None:
        for dense_units in model_options["dense_units"][1:]:

            model_layers.append(tf.keras.layers.ReLU())

            if model_options["dropout_rate"] is not None:
                model_layers.append(
                    tf.keras.layers.Dropout(model_options["dropout_rate"]))

            model_layers.append(tf.keras.layers.Dense(dense_units))

    vision_network = tf.keras.Sequential(model_layers)

    if build_model:
        vision_network.summary()

    return vision_network


def create_embedding_model(vision_network):
    """Create embedding model from vision similarity network."""

    # classification on fine-tuned class logits
    if FLAGS.classification:
        embedding_network = vision_network

    # l2-normalise network output as done in training of siamese objective
    elif FLAGS.l2_norm:
        l2_norm_layer = tf.keras.layers.Lambda(
            lambda x: tf.nn.l2_normalize(x, axis=1))

        embedding_network = tf.keras.Model(
            inputs=vision_network.input,
            outputs=l2_norm_layer(vision_network.output))

    else:
        embedding_network = vision_network

    embedding_model = base.BaseModel(
        embedding_network, None, mc_dropout=FLAGS.mc_dropout)

    return embedding_model


def create_fine_tune_model(model_options, vision_network, num_classes):
    """Create siamese similarity model for fine-tuning on unseen triplets."""

    # create clone of the vision network (so that it remains unchanged)
    vision_network_clone = model_utils.create_and_copy_model(
        vision_network, create_vision_network, model_options=model_options,
        build_model=True)  # TODO: figure out how to get this working without build (for MAML inner loop)

    # freeze all model layers for transfer learning (except final dense layer)
    freeze_index = -1

    for layer in vision_network_clone.layers[:freeze_index]:
        layer.trainable = False

     # fine-tune image similarity network on siamese objective
    if not FLAGS.classification:
        fine_tune_network = vision_network_clone

        # use same objective as during training for fine-tuning
        fine_tune_loss = get_training_objective(model_options)

    # add a categorical logits layer for fine-tuning on unseen classes
    else:
        vision_network_clone.layers[-1].trainable = False

        model_outputs = vision_network_clone.output
        model_outputs = tf.keras.layers.Dense(num_classes)(model_outputs)

        fine_tune_network = tf.keras.Model(
            inputs=vision_network_clone.input, outputs=model_outputs)

        # use categorical cross entropy objective to fine-tune logits
        fine_tune_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)

    few_shot_model = base.BaseModel(
        fine_tune_network, fine_tune_loss, mc_dropout=FLAGS.mc_dropout)

    return few_shot_model


def train(model_options, output_dir, model_file=None, model_step_file=None,
          tf_writer=None):
    """Create and train image similarity model for one-shot learning."""

    # load embeddings from dense layer of base model
    embed_dir = None
    if model_options["use_embeddings"]:
        embed_dir = os.path.join(
            model_options["base_dir"], "embed", "dense")

    # load training data
    train_exp, dev_exp = dataset.create_flickr_vision_train_data(
        model_options["data"], embed_dir=embed_dir)

    train_labels = []
    for keyword in train_exp.keywords_set[3]:
        label = train_exp.keyword_labels[keyword]
        train_labels.append(label)
    train_labels = np.asarray(train_labels)

    dev_labels = []
    for keyword in dev_exp.keywords_set[3]:
        label = train_exp.keyword_labels[keyword]
        dev_labels.append(label)
    dev_labels = np.asarray(dev_labels)

    # define preprocessing for images if no base model embeddings specified
    if model_options["base_dir"] is None or not model_options["use_embeddings"]:
        train_paths = train_exp.image_paths

        preprocess_data_func = functools.partial(
            dataset.load_and_preprocess_image,
            crop_size=model_options["crop_size"],
            augment_crop=model_options["augment_train"],
            random_scales=model_options["random_scales"],
            horizontal_flip=model_options["horizontal_flip"],
            colour=model_options["colour"])

    # otherwise define preprocessing for base model embeddings
    else:
        train_paths = train_exp.embed_paths

        preprocess_data_func = lambda example: dataset.parse_embedding_protobuf(
            example)["embed"]

    # create balanced batch training dataset pipeline
    if model_options["balanced"]:
        assert model_options["p"] is not None
        assert model_options["k"] is not None

        shuffle_train = False
        prefetch_train = False
        num_repeat = model_options["num_batches"]
        model_options["batch_size"] = model_options["p"] * model_options["k"]

        # get unique path train indices per unique label
        train_labels_series = pd.Series(train_labels)
        train_label_idx = {
            label: idx.values[
                np.unique(train_paths[idx.values], return_index=True)[1]]
            for label, idx in train_labels_series.groupby(
                train_labels_series).groups.items()}

        # cache paths to speed things up a little ...
        file_io.check_create_dir(os.path.join(output_dir, "cache"))

        # create a dataset for each unique keyword label (shuffled and cached)
        train_label_datasets = [
            tf.data.Dataset.zip((
                tf.data.Dataset.from_tensor_slices(train_paths[idx]),
                tf.data.Dataset.from_tensor_slices(train_labels[idx]))).cache(
                    os.path.join(
                        output_dir, "cache", str(label))).shuffle(20)  # len(idx)
            for label, idx in train_label_idx.items()]

        # create a dataset that samples balanced batches from the label datasets
        background_train_ds = dataset.create_balanced_batch_dataset(
            model_options["p"], model_options["k"], train_label_datasets)

    # create standard training dataset pipeline (shuffle and load training set)
    else:
        shuffle_train = True
        prefetch_train = True
        num_repeat = model_options["num_augment"]

        background_train_ds = tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(train_paths),
            tf.data.Dataset.from_tensor_slices(train_labels)))

    # load embedding TFRecords (faster here than before balanced sampling)
    if model_options["use_embeddings"]:
        # batch to read files in parallel
        background_train_ds = background_train_ds.batch(
            model_options["batch_size"])

        background_train_ds = background_train_ds.flat_map(
            lambda paths, labels: tf.data.Dataset.zip((
                tf.data.TFRecordDataset(
                    paths, compression_type="ZLIB", num_parallel_reads=8),
                tf.data.Dataset.from_tensor_slices(labels))))

    # map data preprocessing function across training data
    background_train_ds = background_train_ds.map(
        lambda data, label: (preprocess_data_func(data), label),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # repeat augmentation, shuffle and batch train data
    if num_repeat is not None:
        background_train_ds = background_train_ds.repeat(num_repeat)

    if shuffle_train:
        background_train_ds = background_train_ds.shuffle(1000)

    background_train_ds = background_train_ds.batch(
        model_options["batch_size"])

    if prefetch_train:
        background_train_ds = background_train_ds.prefetch(
            tf.data.experimental.AUTOTUNE)

    # create dev set pipeline for siamese validation
    if model_options["use_embeddings"]:
        background_dev_ds = tf.data.Dataset.zip((
            tf.data.TFRecordDataset(
                dev_exp.embed_paths, compression_type="ZLIB",
                num_parallel_reads=8).map(preprocess_data_func),
            tf.data.Dataset.from_tensor_slices(dev_labels)))
    else:
        background_dev_ds = tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(
                dev_exp.image_paths).map(preprocess_data_func),
            tf.data.Dataset.from_tensor_slices(dev_labels)))

    background_dev_ds = background_dev_ds.batch(
        batch_size=model_options["batch_size"])

    # write example batch to TensorBoard
    if tf_writer is not None and not model_options["use_embeddings"]:
        logging.log(logging.INFO, "Writing example images to TensorBoard")
        with tf_writer.as_default():
            for x_batch, y_batch in background_train_ds.take(1):
                tf.summary.image("Example train images", (x_batch+1)/2,
                                 max_outputs=30, step=0)
                labels = ""
                for i, label in enumerate(y_batch[:30]):
                    labels += f"{i}: {np.asarray(train_exp.keywords)[label]} "
                tf.summary.text("Example train labels", labels, step=0)

    # get training objective
    triplet_loss = get_training_objective(model_options)

    # get model input shape
    if model_options["use_embeddings"]:
        for x_batch, _ in background_train_ds.take(1):
            model_options["base_embed_size"] = int(
                tf.shape(x_batch)[1].numpy())

        model_options["input_shape"] = [model_options["base_embed_size"]]
    else:
        model_options["input_shape"] = [
            model_options["crop_size"], model_options["crop_size"], 3]

    # load or create model
    if model_file is not None:
        vision_network, train_state = model_utils.load_model(
            model_file=os.path.join(output_dir, model_file),
            model_step_file=os.path.join(output_dir, model_step_file),
            loss=triplet_loss)

        # get previous tracking variables
        initial_model = False
        global_step, model_epochs, _, best_val_score = train_state
    else:
        vision_network = create_vision_network(model_options)

        # create tracking variables
        initial_model = True
        global_step = 0
        model_epochs = 0

        if model_options["one_shot_validation"]:
            best_val_score = -np.inf
        else:
            best_val_score = np.inf

    # load or create Adam optimizer with decayed learning rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        model_options["learning_rate"], decay_rate=model_options["decay_rate"],
        decay_steps=model_options["decay_steps"], staircase=True)

    if model_file is not None:
        logging.log(logging.INFO, "Restoring optimizer state")
        optimizer = vision_network.optimizer
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # compile model to store optimizer with model when saving
    vision_network.compile(optimizer=optimizer, loss=triplet_loss)

    # create few-shot model from vision network for background training
    vision_few_shot_model = base.BaseModel(vision_network, triplet_loss)

    # test model on one-shot validation task prior to training
    if model_options["one_shot_validation"]:

        one_shot_dev_exp = flickr_vision.FlickrVision(
            keywords_split="background_dev",
            flickr8k_image_dir=os.path.join(
                "data", "external", "flickr8k_images"),
            flickr30k_image_dir=os.path.join(
                "data", "external", "flickr30k_images"),
            mscoco_image_dir=os.path.join(
                "data", "external", "mscoco", "val2017"),
            preprocess_func=get_data_preprocess_func(model_options),
            embed_dir=embed_dir)

        embedding_model_func = create_embedding_model

        classification = False
        if FLAGS.classification:
            assert FLAGS.fine_tune_steps is not None
            classification = True

        # create few-shot model from vision network for one-shot validation
        if FLAGS.fine_tune_steps is not None:
            test_few_shot_model = create_fine_tune_model(
                model_options, vision_few_shot_model.model, num_classes=FLAGS.L)
        else:
            test_few_shot_model = base.BaseModel(
                vision_few_shot_model.model, None, mc_dropout=FLAGS.mc_dropout)

        val_task_accuracy, _, conf_interval_95 = experiment.test_l_way_k_shot(
            one_shot_dev_exp, FLAGS.K, FLAGS.L, n=FLAGS.N, num_episodes=FLAGS.episodes,
            k_neighbours=FLAGS.k_neighbours, metric=FLAGS.metric,
            classification=classification, model=test_few_shot_model,
            embedding_model_func=embedding_model_func,
            fine_tune_steps=FLAGS.fine_tune_steps,
            fine_tune_lr=FLAGS.fine_tune_lr)

        logging.log(
            logging.INFO,
            f"Base model: {FLAGS.L}-way {FLAGS.K}-shot accuracy after "
            f"{FLAGS.episodes} episodes: {val_task_accuracy:.3%} +- "
            f"{conf_interval_95*100:.4f}")

    # create training metrics
    loss_metric = tf.keras.metrics.Mean()
    best_model = False

    # store model options on first run
    if initial_model:
        file_io.write_json(
            os.path.join(output_dir, "model_options.json"), model_options)

        # also store initial model for probing and things
        model_utils.save_model(
            vision_few_shot_model.model, output_dir, 0, 0, "not tested", 0., 0.,
            name="initial_model")

    # train model
    for epoch in range(model_epochs, model_options["epochs"]):
        logging.log(logging.INFO, f"Epoch {epoch:03d}")

        loss_metric.reset_states()

        # train on epoch of training data
        step_pbar = tqdm(background_train_ds,
                         bar_format="{desc} [{elapsed},{rate_fmt}{postfix}]")
        for step, (x_batch, y_batch) in enumerate(step_pbar):

            loss_value, y_predict = vision_few_shot_model.train_step(
                x_batch, y_batch, optimizer,
                clip_norm=model_options["gradient_clip_norm"])

            loss_metric.update_state(loss_value)

            step_loss = tf.reduce_mean(loss_value)
            train_loss = loss_metric.result().numpy()

            step_pbar.set_description_str(
                f"\tStep {step:03d}: "
                f"Step loss: {step_loss:.6f}, "
                f"Loss: {train_loss:.6f}")

            if tf_writer is not None:
                with tf_writer.as_default():
                    tf.summary.scalar(
                        "Train step loss", step_loss, step=global_step)
            global_step += 1

        # validate siamese model
        loss_metric.reset_states()

        for x_batch, y_batch in background_dev_ds:
            y_predict = vision_few_shot_model.predict(x_batch, training=False)
            loss_value = vision_few_shot_model.loss(y_batch, y_predict)

            loss_metric.update_state(loss_value)

        dev_loss = loss_metric.result().numpy()

        # validate model on one-shot dev task if specified
        if model_options["one_shot_validation"]:

            if FLAGS.fine_tune_steps is not None:
                test_few_shot_model = create_fine_tune_model(
                    model_options, vision_few_shot_model.model,
                    num_classes=FLAGS.L)
            else:
                test_few_shot_model = base.BaseModel(
                    vision_few_shot_model.model, None,
                    mc_dropout=FLAGS.mc_dropout)

            val_task_accuracy, _, conf_interval_95 = experiment.test_l_way_k_shot(
                one_shot_dev_exp, FLAGS.K, FLAGS.L, n=FLAGS.N,
                num_episodes=FLAGS.episodes, k_neighbours=FLAGS.k_neighbours,
                metric=FLAGS.metric, classification=classification,
                model=test_few_shot_model,
                embedding_model_func=embedding_model_func,
                fine_tune_steps=FLAGS.fine_tune_steps,
                fine_tune_lr=FLAGS.fine_tune_lr)

            val_score = val_task_accuracy
            val_metric = f"{FLAGS.L}-way {FLAGS.K}-shot accuracy"

            if val_score >= best_val_score:
                best_val_score = val_score
                best_model = True

        # otherwise, validate on siamese task
        else:
            val_score = dev_loss
            val_metric = "loss"

            if val_score <= best_val_score:
                best_val_score = val_score
                best_model = True

        # log results
        logging.log(logging.INFO, f"Train: Loss: {train_loss:.6f}")

        logging.log(
            logging.INFO,
            f"Validation: Loss: {dev_loss:.6f} {'*' if best_model else ''}")

        if model_options["one_shot_validation"]:
            logging.log(
                logging.INFO,
                f"Validation: {FLAGS.L}-way {FLAGS.K}-shot accuracy after "
                f"{FLAGS.episodes} episodes: {val_task_accuracy:.3%} +- "
                f"{conf_interval_95*100:.4f} {'*' if best_model else ''}")

        if tf_writer is not None:
            with tf_writer.as_default():
                tf.summary.scalar(
                    "Train step loss", train_loss, step=global_step)
                tf.summary.scalar(
                    f"Validation loss", dev_loss, step=global_step)
                if model_options["one_shot_validation"]:
                    tf.summary.scalar(
                        f"Validation {FLAGS.L}-way {FLAGS.K}-shot accuracy",
                        val_task_accuracy, step=global_step)

        # store model and results
        model_utils.save_model(
            vision_few_shot_model.model, output_dir, epoch + 1, global_step,
            val_metric, val_score, best_val_score, name="model")

        if best_model:
            best_model = False
            model_utils.save_model(
                vision_few_shot_model.model, output_dir, epoch + 1, global_step,
                val_metric, val_score, best_val_score, name="best_model")


def embed(model_options, output_dir, model_file, model_step_file):
    """Load siamese image similarity model and extract embeddings."""

    # get base embeddings directory if specified, otherwise embed images
    embed_dir = None
    if model_options["use_embeddings"]:
        # load embeddings from dense layer of base model
        embed_dir = os.path.join(
            model_options["base_dir"], "embed", "dense")

    # load model
    vision_network, _ = model_utils.load_model(
        model_file=os.path.join(output_dir, model_file),
        model_step_file=os.path.join(output_dir, model_step_file),
        loss=get_training_objective(model_options))

    # get model embedding model and data preprocessing
    embedding_model = create_embedding_model(vision_network)
    data_preprocess_func = get_data_preprocess_func(model_options)

    # load image datasets and compute embeddings
    for data in ["flickr8k", "flickr30k", "mscoco"]:

        train_image_dir_dict = {}
        dev_image_dir_dict = {}

        if data == "flickr8k":
            train_image_dir_dict["flickr8k_image_dir"] = os.path.join(
                "data", "external", "flickr8k_images")
            dev_image_dir_dict = train_image_dir_dict

        if data == "flickr30k":
            train_image_dir_dict["flickr30k_image_dir"] = os.path.join(
                "data", "external", "flickr30k_images")
            dev_image_dir_dict = train_image_dir_dict

        if data == "mscoco":
            train_image_dir_dict["mscoco_image_dir"] = os.path.join(
                "data", "external", "mscoco", "train2017")
            dev_image_dir_dict["mscoco_image_dir"] = os.path.join(
                "data", "external", "mscoco", "val2017")

        one_shot_exp = flickr_vision.FlickrVision(
            keywords_split="one_shot_evaluation", **train_image_dir_dict,
            embed_dir=embed_dir)

        background_train_exp = flickr_vision.FlickrVision(
            keywords_split="background_train", **train_image_dir_dict,
            embed_dir=embed_dir)

        background_dev_exp = flickr_vision.FlickrVision(
            keywords_split="background_dev", **dev_image_dir_dict,
            embed_dir=embed_dir)

        subset_exp = {
            "one_shot_evaluation": one_shot_exp,
            "background_train": background_train_exp,
            "background_dev": background_dev_exp,
        }

        for subset, exp in subset_exp.items():
            output_embed_dir = os.path.join(
                output_dir, "embed", "dense", data, subset)
            file_io.check_create_dir(output_embed_dir)

            if model_options["use_embeddings"]:
                subset_paths = exp.embed_paths
            else:
                subset_paths = exp.image_paths

            unique_paths = np.unique(subset_paths)

            # batch images/base embeddings for faster embedding inference
            path_ds = tf.data.Dataset.from_tensor_slices(unique_paths)
            path_ds = path_ds.batch(model_options["batch_size"])
            path_ds = path_ds.prefetch(tf.data.experimental.AUTOTUNE)

            num_samples = int(
                np.ceil(len(unique_paths) / model_options["batch_size"]))

            start_time = time.time()
            paths, embeddings = [], []
            for path_batch in tqdm(path_ds, total=num_samples):
                path_embeddings = embedding_model.predict(
                    data_preprocess_func(path_batch))

                paths.extend(path_batch.numpy())
                embeddings.extend(path_embeddings.numpy())
            end_time = time.time()

            logging.log(
                logging.INFO,
                f"Computed embeddings for {data} {subset} in "
                f"{end_time - start_time:.4f} seconds")

            # serialize and write embeddings to TFRecord files
            for path, embedding in zip(paths, embeddings):
                example_proto = dataset.embedding_to_example_protobuf(embedding)

                path = path.decode("utf-8")
                path = path.split(".tfrecord")[0]  # remove any ".tfrecord" ext
                path = os.path.join(
                    output_embed_dir, f"{os.path.split(path)[1]}.tfrecord")

                with tf.io.TFRecordWriter(path, options="ZLIB") as writer:
                    writer.write(example_proto.SerializeToString())

            logging.log(logging.INFO, f"Embeddings stored at: {output_embed_dir}")


def test(model_options, output_dir, model_file, model_step_file):
    """Load and test siamese image similarity model for one-shot learning."""

    # get base embeddings directory if specified, otherwise embed images
    embed_dir = None
    if model_options["use_embeddings"]:
        # load embeddings from dense layer of base model
        embed_dir = os.path.join(
            model_options["base_dir"], "embed", "dense")

    # load Flickr 8k one-shot experiment
    one_shot_exp = flickr_vision.FlickrVision(
        keywords_split="one_shot_evaluation",
        flickr8k_image_dir=os.path.join("data", "external", "flickr8k_images"),
        preprocess_func=get_data_preprocess_func(model_options),
        embed_dir=embed_dir)

    # load model
    vision_network, _ = model_utils.load_model(
        model_file=os.path.join(output_dir, model_file),
        model_step_file=os.path.join(output_dir, model_step_file),
        loss=get_training_objective(model_options))

    embedding_model_func = create_embedding_model

    # create few-shot model from vision network for one-shot testing
    if FLAGS.fine_tune_steps is not None:
        test_few_shot_model = create_fine_tune_model(
            model_options, vision_network, num_classes=FLAGS.L)
    else:
        test_few_shot_model = base.BaseModel(
            vision_network, None, mc_dropout=FLAGS.mc_dropout)

    classification = False
    if FLAGS.classification:
        assert FLAGS.fine_tune_steps is not None
        classification = True

    logging.log(logging.INFO, "Created few-shot model from vision network")
    test_few_shot_model.model.summary()

    # test model on L-way K-shot task
    task_accuracy, _, conf_interval_95 = experiment.test_l_way_k_shot(
        one_shot_exp, FLAGS.K, FLAGS.L, n=FLAGS.N, num_episodes=FLAGS.episodes,
        k_neighbours=FLAGS.k_neighbours, metric=FLAGS.metric,
        classification=classification, model=test_few_shot_model,
        embedding_model_func=embedding_model_func,
        fine_tune_steps=FLAGS.fine_tune_steps, fine_tune_lr=FLAGS.fine_tune_lr)

    logging.log(
        logging.INFO,
        f"{FLAGS.L}-way {FLAGS.K}-shot accuracy after {FLAGS.episodes} "
        f"episodes: {task_accuracy:.3%} +- {conf_interval_95*100:.4f}")


def main(argv):
    """Main program logic."""
    del argv  # unused

    logging.log(logging.INFO, "Logging application {}".format(__file__))
    if FLAGS.debug:
        logging.set_verbosity(logging.DEBUG)
        logging.log(logging.DEBUG, "Running in debug mode")

    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    model_found = False
    # no prior run specified, train model
    if FLAGS.output_dir is None:
        if FLAGS.target != "train":
            raise ValueError(
                f"Target `{FLAGS.target}` requires --output_dir to be specified.")

        output_dir = os.path.join(
            "logs", __file__, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        file_io.check_create_dir(output_dir)

        model_options = DEFAULT_OPTIONS

        # add flag options to model options
        model_options["base_dir"] = FLAGS.base_dir

        if model_options["base_dir"] is not None:
            model_options["use_embeddings"] = FLAGS.use_embeddings
        else:
            model_options["use_embeddings"] = False

        model_options["base_model"] = (
            "best_model" if FLAGS.load_best else "model")

    # prior run specified, resume training or test model
    else:
        output_dir = FLAGS.output_dir

        # load current, initial or best model
        if FLAGS.load_best:
            model_file = "best_model.h5"
            model_step_file = "best_model.step"
        elif FLAGS.load_initial:
            model_file = "initial_model.h5"
            model_step_file = "initial_model.step"
        else:
            model_file = "model.h5"
            model_step_file = "model.step"

        if FLAGS.base_dir is not None:
            raise ValueError(
                f"Flag --base_dir should not be set for target `{FLAGS.target}`.")

        if os.path.exists(os.path.join(output_dir, model_file)):

            model_found = True
            model_options = file_io.read_json(
                os.path.join(output_dir, "model_options.json"))

        elif FLAGS.target != "train":
            raise ValueError(
                f"Target `{FLAGS.target}` specified but `{model_file}` not "
                f"found in {output_dir}.")

    # gather flag options
    flag_options = {}
    for flag in FLAGS.get_key_flags_for_module(__file__):
        flag_options[flag.name] = flag.value

    # logging
    logging_utils.absl_file_logger(output_dir, f"log.{FLAGS.target}")

    logging.log(logging.INFO, f"Model directory: {output_dir}")
    logging.log(logging.INFO, f"Model options: {model_options}")
    logging.log(logging.INFO, f"Flag options: {flag_options}")

    tf_writer = None
    if FLAGS.tensorboard and FLAGS.target == "train":
        tf_writer = tf.summary.create_file_writer(output_dir)

    # set seeds for reproducibility
    np.random.seed(model_options["seed"])
    tf.random.set_seed(model_options["seed"])

    # run target
    if FLAGS.target == "train":
        if model_found and FLAGS.resume:
            train(model_options, output_dir, model_file, model_step_file,
                  tf_writer=tf_writer)
        else:
            train(model_options, output_dir, tf_writer=tf_writer)
    elif FLAGS.target == "validate":  # TODO
        raise NotImplementedError
    elif FLAGS.target == "embed":
        embed(model_options, output_dir, model_file, model_step_file)
    else:
        test(model_options, output_dir, model_file, model_step_file)


if __name__ == "__main__":
    app.run(main)
