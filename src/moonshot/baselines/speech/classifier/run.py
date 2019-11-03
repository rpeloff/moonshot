"""Train spoken word classifier and test on Flickr-Audio one-shot speech task.

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


import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm


from moonshot.baselines import base
from moonshot.baselines import dataset
from moonshot.baselines import davenet
from moonshot.baselines import experiment
from moonshot.baselines import losses
from moonshot.baselines import model_utils

from moonshot.experiments.flickr_speech import flickr_speech

from moonshot.utils import file_io
from moonshot.utils import logging as logging_utils


FLAGS = flags.FLAGS


# model options (default if not loaded)
DEFAULT_OPTIONS = {
    # training data
    "one_shot_validation": True,
    # data pipeline
    "batch_size": 32,
    # DAVEnet spoken word classifier
    "batch_norm_spectrogram": True,
    "batch_norm_conv": True,
    "downsample": True,
    "embedding_dim": 1024,
    "padding": "same",
    "dense_units": [2048],  # hidden layers on top of DAVEnet base network (followed by logits)
    "dropout_rate": 0.2,
    # objective
    "cross_entropy_label_smoothing": 0.1,
    # training
    "learning_rate": 3e-4,
    "decay_steps": 4000,  # TODO  # slightly less than two epochs
    "decay_rate": 0.96,
    "gradient_clip_norm": 5.,
    "epochs": 100,
    # that magic number
    "seed": 42
}

# one-shot evaluation options
flags.DEFINE_integer("episodes", 400, "number of L-way K-shot learning episodes")
flags.DEFINE_integer("L", 10, "number of classes to sample in a task episode (L-way)")
flags.DEFINE_integer("K", 1, "number of task learning samples per class (K-shot)")
flags.DEFINE_integer("N", 15, "number of task evaluation samples")
flags.DEFINE_integer("k_neighbours", 1, "number of nearest neighbours to consider")
flags.DEFINE_string("metric", "cosine", "distance metric to use for DTW nearest neighbours")
flags.DEFINE_integer("fine_tune_steps", None, "number of fine-tune gradient steps on one-shot data")
flags.DEFINE_float("fine_tune_lr", 1e-3, "learning rate for gradient descent fine-tune")
flags.DEFINE_bool("classification", False, "whether to use softmax predictions as match function")
flags.DEFINE_enum("speaker_mode", "baseline", ["baseline", "difficult", "distractor"],
                  "type of speakers selected in a task episode")

# speech features (for train target)
flags.DEFINE_enum("features", "mfcc", ["mfcc", "fbank"], "type of processed speech features")
flags.DEFINE_integer("max_length", 140, "length to re-interpolate or crop segments")
flags.DEFINE_enum("scaling", None, ["global", "features", "segment", "segment_mean"],
                  "type of feature scaling applied to speech segments")

# model train/test options
flags.DEFINE_enum("embed_layer", "dense", ["avg_pool", "dense", "logits", "softmax"],
                  "model layer to extract embeddings from")
flags.DEFINE_bool("load_best", False, "load previous best model for resumed training or testing")
flags.DEFINE_bool("mc_dropout", False, "make embedding predictions with MC Dropout")

# logging and target options
flags.DEFINE_enum("target", "train", ["train", "validate", "embed", "test"],
                  "train or load and test a model")
flags.DEFINE_string("output_dir", None, "directory where logs and checkpoints will be stored"
                    "(defaults to logs/<unique run id>)")
flags.DEFINE_bool("resume", True, "resume training if a checkpoint is found at output directory")
flags.DEFINE_bool("tensorboard", True, "log train and test summaries to TensorBoard")
flags.DEFINE_bool("debug", False, "log with debug verbosity level")


def get_training_objective(model_options):
    """Get training loss for spoken word classification."""

    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True,
        label_smoothing=model_options["cross_entropy_label_smoothing"])

    return loss


def get_data_preprocess_func(model_options):
    """Create data batch preprocessing function for input to the speech network.

    Returns function `data_preprocess_func` that takes a batch of file paths,
    loads speech features and preprocesses the features.
    """

    def data_preprocess_func(speech_paths):
        speech_features = []
        for speech_path in speech_paths:
            speech_features.append(
                dataset.load_and_preprocess_speech(
                    speech_path, features=model_options["features"],
                    max_length=model_options["max_length"],
                    scaling=model_options["scaling"]))

        return np.stack(speech_features)

    return data_preprocess_func


def create_speech_network(model_options, build_model=True):
    """Create spoken word classification model from model options."""

    # get input shape
    input_shape = None
    if build_model:
        input_shape = model_options["input_shape"]

    # train DAVEnet audio base network from scratch
    davenet_audio_network = davenet.create_davenet_audio_network(
        input_shape=input_shape,
        batch_norm_spectrogram=model_options["batch_norm_spectrogram"],
        batch_norm_conv=model_options["batch_norm_conv"],
        downsample=model_options["downsample"],
        embedding_dim=model_options["embedding_dim"],
        padding=model_options["padding"])

    model_layers = [
        davenet_audio_network,
        tf.keras.layers.GlobalAveragePooling1D()
    ]

    if model_options["dropout_rate"] is not None:
        model_layers.append(
            tf.keras.layers.Dropout(model_options["dropout_rate"]))

    # add top layer hidden units
    if model_options["dense_units"] is not None:
        for dense_units in model_options["dense_units"]:

            model_layers.append(tf.keras.layers.Dense(dense_units))

            model_layers.append(tf.keras.layers.ReLU())

            if model_options["dropout_rate"] is not None:
                model_layers.append(
                    tf.keras.layers.Dropout(model_options["dropout_rate"]))

    # add final class logits layer
    model_layers.append(tf.keras.layers.Dense(model_options["n_classes"]))

    speech_network = tf.keras.Sequential(model_layers)

    if build_model:
        speech_network.summary()

    return speech_network


def create_embedding_model(model_options, speech_network):
    """Create embedding model from speech network."""

    # slice embedding model from specified layer
    if FLAGS.embed_layer == "avg_pool":  # global average pool layer
        slice_index = 1
    elif FLAGS.embed_layer == "dense":  # dense layer before relu & logits layer
        slice_index = -3 if model_options["dropout_rate"] is None else -4
    elif FLAGS.embed_layer == "logits":  # unnormalised log probabilities
        slice_index = -1
    elif FLAGS.embed_layer == "softmax":
        slice_index = -1

    model_input = (
        speech_network.layers[0].input if slice_index == 0 else
        speech_network.input)

    model_output = speech_network.layers[slice_index].output

    if FLAGS.embed_layer == "softmax":  # softmax class probabilities
        model_output = tf.nn.softmax(model_output)

    embedding_network = tf.keras.Model(inputs=model_input, outputs=model_output)

    embedding_model = base.FewShotModel(
        embedding_network, None, mc_dropout=FLAGS.mc_dropout)

    return embedding_model


def create_fine_tune_model(model_options, speech_network, num_classes):
    """Create classification model for fine-tuning on unseen classes."""

    # create clone of the speech network (so that it remains unchanged)
    speech_network_clone = model_utils.create_and_copy_model(
        speech_network, create_speech_network, model_options=model_options,
        build_model=True)  # TODO: figure out how to get this working without build (for MAML inner loop)

    # freeze model layers up to dense layer before relu & logits layer
    if FLAGS.embed_layer == "dense":
        freeze_index = -3 if model_options["dropout_rate"] is None else -4
    # freeze all model layers for transfer learning (except final logits)
    else:
        freeze_index = -1

    for layer in speech_network_clone.layers[:freeze_index]:
        layer.trainable = False

    # replace the logits layer with categorical logits layer for unseen classes
    model_outputs = speech_network_clone.layers[-2].output
    model_outputs = tf.keras.layers.Dense(num_classes)(model_outputs)

    fine_tune_network = tf.keras.Model(
        inputs=speech_network_clone.input, outputs=model_outputs)

    fine_tune_loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)

    few_shot_model = base.FewShotModel(
        fine_tune_network, fine_tune_loss, mc_dropout=FLAGS.mc_dropout)

    return few_shot_model


def train(model_options, output_dir, model_file=None, model_step_file=None,
          tf_writer=None):
    """Create and train spoken word classification model for one-shot learning."""

    # load training data
    train_exp, dev_exp = dataset.create_flickr_audio_train_data(
        model_options["features"])

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

    train_paths = train_exp.audio_paths
    dev_paths = dev_exp.audio_paths

    lb = LabelBinarizer()
    train_labels_one_hot = lb.fit_transform(train_labels)
    dev_labels_one_hot = lb.transform(dev_labels)

    # define preprocessing for speech features
    preprocess_speech_func = functools.partial(
        dataset.load_and_preprocess_speech, features=model_options["features"],
        max_length=model_options["max_length"],
        scaling=model_options["scaling"])

    preprocess_speech_ds_func = lambda path: tf.py_function(
        func=preprocess_speech_func, inp=[path], Tout=tf.float32)

    # create standard training dataset pipeline
    background_train_ds = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(train_paths),
        tf.data.Dataset.from_tensor_slices(train_labels_one_hot)))

    # map data preprocessing function across training data
    background_train_ds = background_train_ds.map(
        lambda path, label: (preprocess_speech_ds_func(path), label),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # shuffle and batch train data
    background_train_ds = background_train_ds.shuffle(1000)

    background_train_ds = background_train_ds.batch(
        model_options["batch_size"])

    background_train_ds = background_train_ds.prefetch(
        tf.data.experimental.AUTOTUNE)

    # create dev set pipeline for classification validation
    background_dev_ds = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(
            dev_paths).map(preprocess_speech_ds_func),
        tf.data.Dataset.from_tensor_slices(dev_labels_one_hot)))

    background_dev_ds = background_dev_ds.batch(
        batch_size=model_options["batch_size"])

    # write example batch to TensorBoard
    if tf_writer is not None:
        logging.log(logging.INFO, "Writing example features to TensorBoard")
        with tf_writer.as_default():
            for x_batch, y_batch in background_train_ds.take(1):

                speech_feats = []
                for feats in x_batch[:30]:
                    feats = np.transpose(feats)
                    speech_feats.append(
                        (feats - np.min(feats)) / np.max(feats))

                tf.summary.image(
                    f"Example train speech {model_options['features']}",
                    np.expand_dims(speech_feats, axis=-1), max_outputs=30, step=0)

                labels = ""
                for i, label in enumerate(y_batch[:30]):
                    labels += f"{i}: {np.asarray(train_exp.keywords)[label]} "

                tf.summary.text("Example train labels", labels, step=0)

    # get training objective
    loss = get_training_objective(model_options)

    # get model input shape
    if model_options["features"] == "mfcc":
        model_options["input_shape"] = [model_options["max_length"], 39]
    else:
        model_options["input_shape"] = [model_options["max_length"], 40]

    # load or create model
    if model_file is not None:
        assert model_options["n_classes"] == len(train_exp.keywords)

        speech_network, train_state = model_utils.load_model(
            model_file=os.path.join(output_dir, model_file),
            model_step_file=os.path.join(output_dir, model_step_file),
            loss=loss)

        # get previous tracking variables
        initial_model = False
        global_step, model_epochs, _, best_val_score = train_state
    else:
        model_options["n_classes"] = len(train_exp.keywords)

        speech_network = create_speech_network(model_options)

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
        optimizer = speech_network.optimizer
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # compile model to store optimizer with model when saving
    speech_network.compile(optimizer=optimizer, loss=loss)

    # create few-shot model from speech network for background training
    speech_few_shot_model = base.FewShotModel(speech_network, loss)

    # test model on one-shot validation task prior to training
    if model_options["one_shot_validation"]:
        data_preprocess_func = get_data_preprocess_func(model_options)
        embedding_model_func = lambda speech_network: create_embedding_model(
            model_options, speech_network)

        classification = False
        if FLAGS.classification:
            assert FLAGS.embed_layer in ["logits", "softmax"]
            classification = True

        # create few-shot model from speech network for one-shot validation
        if FLAGS.fine_tune_steps is not None:
            test_few_shot_model = create_fine_tune_model(
                model_options, speech_few_shot_model.model, num_classes=FLAGS.L)
        else:
            test_few_shot_model = base.FewShotModel(
                speech_few_shot_model.model, None, mc_dropout=FLAGS.mc_dropout)

        val_task_accuracy, _, conf_interval_95 = experiment.test_l_way_k_shot(
            dev_exp, FLAGS.K, FLAGS.L, n=FLAGS.N, num_episodes=FLAGS.episodes,
            k_neighbours=FLAGS.k_neighbours, metric=FLAGS.metric,
            classification=classification, model=test_few_shot_model,
            data_preprocess_func=data_preprocess_func,
            embedding_model_func=embedding_model_func,
            fine_tune_steps=FLAGS.fine_tune_steps,
            fine_tune_lr=FLAGS.fine_tune_lr)

        logging.log(
            logging.INFO,
            f"Base model: {FLAGS.L}-way {FLAGS.K}-shot accuracy after "
            f"{FLAGS.episodes} episodes: {val_task_accuracy:.3%} +- "
            f"{conf_interval_95*100:.4f}")

    # create training metrics
    accuracy_metric = tf.keras.metrics.CategoricalAccuracy()
    loss_metric = tf.keras.metrics.Mean()
    best_model = False

    # store model options on first run
    if initial_model:
        # store model options
        file_io.write_json(
            os.path.join(output_dir, "model_options.json"), model_options)

    # train model
    for epoch in range(model_epochs, model_options["epochs"]):
        logging.log(logging.INFO, f"Epoch {epoch:03d}")

        accuracy_metric.reset_states()
        loss_metric.reset_states()

        # train on epoch of training data
        step_pbar = tqdm(background_train_ds,
                         bar_format="{desc} [{elapsed},{rate_fmt}{postfix}]")
        for step, (x_batch, y_batch) in enumerate(step_pbar):

            loss_value, y_predict = speech_few_shot_model.train_step(
                x_batch, y_batch, optimizer,
                clip_norm=model_options["gradient_clip_norm"])

            accuracy_metric.update_state(y_batch, y_predict)
            loss_metric.update_state(loss_value)

            step_loss = tf.reduce_mean(loss_value)
            train_loss = loss_metric.result().numpy()
            train_accuracy = accuracy_metric.result().numpy()

            step_pbar.set_description_str(
                f"\tStep {step:03d}: "
                f"Step loss: {step_loss:.6f}, "
                f"Loss: {train_loss:.6f}, "
                f"Categorical accuracy: {train_accuracy:.3%}")

            if tf_writer is not None:
                with tf_writer.as_default():
                    tf.summary.scalar(
                        "Train step loss", step_loss, step=global_step)
            global_step += 1

        # validate classification model
        accuracy_metric.reset_states()
        loss_metric.reset_states()

        for x_batch, y_batch in background_dev_ds:
            y_predict = speech_few_shot_model.predict(x_batch, training=False)
            loss_value = speech_few_shot_model.loss(y_batch, y_predict)

            accuracy_metric.update_state(y_batch, y_predict)
            loss_metric.update_state(loss_value)

        dev_loss = loss_metric.result().numpy()
        dev_accuracy = accuracy_metric.result().numpy()

        # validate model on one-shot dev task if specified
        if model_options["one_shot_validation"]:

            if FLAGS.fine_tune_steps is not None:
                test_few_shot_model = create_fine_tune_model(
                    model_options, speech_few_shot_model.model,
                    num_classes=FLAGS.L)
            else:
                test_few_shot_model = base.FewShotModel(
                    speech_few_shot_model.model, None,
                    mc_dropout=FLAGS.mc_dropout)

            val_task_accuracy, _, conf_interval_95 = experiment.test_l_way_k_shot(
                dev_exp, FLAGS.K, FLAGS.L, n=FLAGS.N,
                num_episodes=FLAGS.episodes, k_neighbours=FLAGS.k_neighbours,
                metric=FLAGS.metric, classification=classification,
                model=test_few_shot_model,
                data_preprocess_func=data_preprocess_func,
                embedding_model_func=embedding_model_func,
                fine_tune_steps=FLAGS.fine_tune_steps,
                fine_tune_lr=FLAGS.fine_tune_lr)

            val_score = val_task_accuracy
            val_metric = f"{FLAGS.L}-way {FLAGS.K}-shot accuracy"

        # otherwise, validate on classification task
        else:
            val_score = dev_accuracy
            val_metric = "categorical accuracy"

        if val_score >= best_val_score:
            best_val_score = val_score
            best_model = True

        # log results
        logging.log(
            logging.INFO,
            f"Train: Loss: {train_loss:.6f}, Categorical accuracy: "
            f"{train_accuracy:.3%}")

        logging.log(
            logging.INFO,
            f"Validation: Loss: {dev_loss:.6f}, Categorical accuracy: "
            f"{dev_accuracy:.3%} {'*' if best_model else ''}")

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
                    "Train categorical accuracy", train_accuracy, step=global_step)
                tf.summary.scalar(
                    "Validation loss", dev_loss, step=global_step)
                tf.summary.scalar(
                    "Validation categorical accuracy", dev_accuracy, step=global_step)
                if model_options["one_shot_validation"]:
                    tf.summary.scalar(
                        f"Validation {FLAGS.L}-way {FLAGS.K}-shot accuracy",
                        val_task_accuracy, step=global_step)

        # store model and results
        model_utils.save_model(
            speech_few_shot_model.model, output_dir, epoch + 1, global_step,
            val_metric, val_score, best_val_score, name="model")

        if best_model:
            best_model = False

            model_utils.save_model(
                speech_few_shot_model.model, output_dir, epoch + 1, global_step,
                val_metric, val_score, best_val_score, name="best_model")


def embed(model_options, output_dir, model_file, model_step_file):
    """Load spoken word classification model and extract embeddings."""

    # load model
    speech_network, _ = model_utils.load_model(
        model_file=os.path.join(output_dir, model_file),
        model_step_file=os.path.join(output_dir, model_step_file),
        loss=get_training_objective(model_options))

    # get embedding model and data preprocessing
    embedding_model = create_embedding_model(model_options, speech_network)
    data_preprocess_func = get_data_preprocess_func(model_options)

    # load Flickr Audio dataset and compute embeddings
    one_shot_exp = flickr_speech.FlickrSpeech(
        features=model_options["features"],
        keywords_split="one_shot_evaluation")

    background_train_exp = flickr_speech.FlickrSpeech(
        features=model_options["features"], keywords_split="background_train")

    background_dev_exp = flickr_speech.FlickrSpeech(
        features=model_options["features"], keywords_split="background_dev")

    subset_exp = {
        "one_shot_evaluation": one_shot_exp,
        "background_train": background_train_exp,
        "background_dev": background_dev_exp,
    }

    for subset, exp in subset_exp.items():
        embed_dir = os.path.join(
            output_dir, "embed", FLAGS.embed_layer, "flickr_audio", subset)
        file_io.check_create_dir(embed_dir)

        unique_paths = np.unique(exp.audio_paths)

        # batch audio paths for faster embedding inference
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
            f"Computed embeddings ({FLAGS.embed_layer}) for Flickr Audio "
            f"{subset} in {end_time - start_time:.4f} seconds")

        # serialize and write embeddings to TFRecord files
        for path, embedding in zip(paths, embeddings):
            example_proto = dataset.embedding_to_example_protobuf(embedding)

            path = path.decode("utf-8")
            path = os.path.join(
                embed_dir, f"{os.path.split(path)[1]}.tfrecord")

            with tf.io.TFRecordWriter(path, options="ZLIB") as writer:
                writer.write(example_proto.SerializeToString())

        logging.log(logging.INFO, f"Embeddings stored at: {embed_dir}")


def test(model_options, output_dir, model_file, model_step_file):
    """Load and test spoken word classification model for one-shot learning."""

    # load Flickr Audio one-shot experiment
    one_shot_exp = flickr_speech.FlickrSpeech(
        features=model_options["features"],
        keywords_split="one_shot_evaluation", embed_dir=None)

    # load model
    speech_network, _ = model_utils.load_model(
        model_file=os.path.join(output_dir, model_file),
        model_step_file=os.path.join(output_dir, model_step_file),
        loss=get_training_objective(model_options))

    data_preprocess_func = get_data_preprocess_func(model_options)
    embedding_model_func = lambda speech_network: create_embedding_model(
        model_options, speech_network)

    # create few-shot model from speech network for one-shot testing
    if FLAGS.fine_tune_steps is not None:
        test_few_shot_model = create_fine_tune_model(
            model_options, speech_network, num_classes=FLAGS.L)
    else:
        test_few_shot_model = base.FewShotModel(
            speech_network, None, mc_dropout=FLAGS.mc_dropout)

    classification = False
    if FLAGS.classification:
        assert FLAGS.embed_layer in ["logits", "softmax"]
        classification = True

    logging.log(logging.INFO, "Created few-shot model from speech network")
    test_few_shot_model.model.summary()

    # test model on L-way K-shot task
    task_accuracy, _, conf_interval_95 = experiment.test_l_way_k_shot(
        one_shot_exp, FLAGS.K, FLAGS.L, n=FLAGS.N, num_episodes=FLAGS.episodes,
        k_neighbours=FLAGS.k_neighbours, metric=FLAGS.metric,
        classification=classification, model=test_few_shot_model,
        data_preprocess_func=data_preprocess_func,
        embedding_model_func=embedding_model_func,
        fine_tune_steps=FLAGS.fine_tune_steps, fine_tune_lr=FLAGS.fine_tune_lr)

    logging.log(
        logging.INFO,
        f"{FLAGS.L}-way {FLAGS.K}-shot accuracy after {FLAGS.episodes} "
        f"episodes: {task_accuracy:.3%} +- {conf_interval_95*100:.4f}")


def main(argv):
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
                "Target `{FLAGS.target}` requires --output_dir to be specified.")

        output_dir = os.path.join(
            "logs", __file__, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        file_io.check_create_dir(output_dir)

        model_options = DEFAULT_OPTIONS

        # add flag options to model options
        model_options["features"] = FLAGS.features
        model_options["max_length"] = FLAGS.max_length
        model_options["scaling"] = FLAGS.scaling

    # prior run specified, resume training or test model
    else:
        output_dir = FLAGS.output_dir

        # load current or best model
        model_file = "best_model.h5" if FLAGS.load_best else "model.h5"
        model_step_file = "best_model.step" if FLAGS.load_best else "model.step"

        if os.path.exists(os.path.join(output_dir, model_file)):

            model_found = True
            model_options = file_io.read_json(
                os.path.join(output_dir, "model_options.json"))

        elif FLAGS.target != "train":
            raise ValueError(
                f"Target `{FLAGS.target}` specified but `{model_file}` not "
                f"found in {output_dir}.")

    logging_utils.absl_file_logger(output_dir, f"log.{FLAGS.target}")

    logging.log(logging.INFO, f"Model directory: {output_dir}")
    logging.log(logging.INFO, f"Model options: {model_options}")

    tf_writer = None
    if FLAGS.tensorboard and FLAGS.target == "train":
        tf_writer = tf.summary.create_file_writer(output_dir)

    np.random.seed(model_options["seed"])
    tf.random.set_seed(model_options["seed"])

    if FLAGS.target == "train":
        if model_found and FLAGS.resume:
            train(model_options, output_dir, model_file=model_file,
                  model_step_file=model_step_file, tf_writer=tf_writer)
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
