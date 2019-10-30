"""TODO(rpeloff)

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: September 2019
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
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm


from moonshot.baselines import base
from moonshot.baselines import dataset
from moonshot.baselines import inceptionv3
from moonshot.baselines import losses
from moonshot.baselines import vision_cnn

from moonshot.experiments import experiments
from moonshot.experiments.flickr_vision import flickr_vision

from moonshot.utils import file_io
from moonshot.utils import logging as logging_utils


FLAGS = flags.FLAGS


# model options (default if not loaded)
DEFAULT_OPTIONS = {
    # training data
    "data": ["flickr8k", "flickr30k", "mscoco"],
    "one_shot_validation": True,
    # preprocessing
    "crop_size": 299,
    "augment_train": True,
    "num_augment": 4,
    "random_scales": None,
    "horizontal_flip": True,
    "colour": True,
    # data pipeline
    "batch_size": 32,
    # inceptionv3 classifier
    "pretrained": False,  # train base from scratch or ImageNet
    "dense_units": [2048],  # hidden layers on top of base network (followed by logits)
    "dropout_rate": 0.2,
    # objective
    "loss": "focal",  # "cross_entropy",
    "cross_entropy_pos_weight": 2.0,
    "cross_entropy_label_smoothing": 0.0,  # causes NaN when set to .1?
    "focal_alpha": None,
    "focal_gamma": 2.0,
    # training
    "learning_rate": 1e-4,
    "decay_steps": 25000,  # slightly more than two epochs (all data x4 augment)
    "decay_rate": 0.96,
    "gradient_clip_norm": 5.,
    "epochs": 100,
    # that magic number
    "seed": 42
}

# one-shot evaluation options
flags.DEFINE_integer("episodes", 400, "number of L-way K-shot learning episodes")
flags.DEFINE_integer("L", 5, "number of classes to sample in a task episode (L-way)")
flags.DEFINE_integer("K", 1, "number of task learning samples per class (K-shot)")
flags.DEFINE_integer("N", 15, "number of task evaluation samples")
flags.DEFINE_integer("k_neighbours", 1, "number of nearest neighbours to consider")

# model train/test options
flags.DEFINE_enum("embed_layer", "dense", ["avg_pool", "dense", "logits"],
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
    """Get custom training loss for multi-label classification."""

    if model_options["loss"] == "cross_entropy":
        multi_label_loss = losses.weighted_cross_entropy_with_logits(
            pos_weight=model_options["cross_entropy_pos_weight"],
            label_smoothing=model_options["cross_entropy_label_smoothing"])

    elif model_options["loss"] == "focal":
        multi_label_loss = losses.focal_loss_with_logits(
            alpha=model_options["focal_alpha"],
            gamma=model_options["focal_gamma"])

    else:
        raise ValueError(f"Invalid loss function: {model_options['loss']}")

    return multi_label_loss


def create_model(model_options):
    """Create multi-label classification model from model options."""

    # train network from scratch or with imagenet weights
    input_shape = (model_options["crop_size"], model_options["crop_size"], 3)
    inception_network = inceptionv3.create_inceptionv3_network(
        input_shape=input_shape, pretrained=model_options["pretrained"],
        include_top=False)

    # inception model with imagenet weights and our own top dense layers (NOTE: debug oracle only)
    if model_options["pretrained"]:
        logging.log(
            logging.INFO,
            "Training model with imagenet weights and custom top layer")

        # train final inception module and the top dense layers
        inceptionv3.freeze_weights(
            inception_network, trainable="final_inception")

    # inception model with random weights and our own top dense layers
    else:
        logging.log(logging.INFO, "Training model from scratch")

    model_layers = [
        inception_network,
        tf.keras.layers.GlobalAveragePooling2D()]

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

    vision_network = tf.keras.Sequential(model_layers)
    vision_network.summary()

    return vision_network


def get_embedding_function(vision_network, model_options):
    """Create embedding function from vision network.

    Returns function `embedding_func` that takes a batch of file paths, loads
    and preprocesses the files and computes their embeddings.
    """

    # slice embedding model from specified layer
    if FLAGS.embed_layer == "avg_pool":  # global average pool layer
        slice_index = 1
    elif FLAGS.embed_layer == "dense":  # dense layer before relu & logits layer
        slice_index = -3 if model_options["dropout_rate"] is None else -4
    elif FLAGS.embed_layer == "logits":  # unnormalised log probabilities
        slice_index = -1

    model_input = (vision_network.layers[0].input if slice_index == 0 else
                   vision_network.input)

    embed_model = tf.keras.Model(
        inputs=model_input,
        outputs=vision_network.layers[slice_index].output)

    embed_few_shot_model = vision_cnn.FewShotModel(
        embed_model, None, mc_dropout=FLAGS.mc_dropout)

    # get embedding function for image inputs
    def embedding_func(image_paths):
        image_ds = tf.data.Dataset.from_tensor_slices(image_paths)
        image_ds = image_ds.map(
            lambda img_path: dataset.load_and_preprocess_image(
                img_path, crop_size=model_options["crop_size"]),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        images = np.stack(list(image_ds))
        return embed_few_shot_model.predict(images)

    return embedding_func


def train(model_options, output_dir, model_file=None, model_step_file=None,
          tf_writer=None):
    """Create and train classification model for one-shot learning."""

    # load training data
    train_exp, dev_exp = dataset.create_flickr_train_data(model_options["data"])

    train_labels = []
    for image_keywords in train_exp.unique_image_keywords:
        labels = map(
            lambda keyword: train_exp.keyword_labels[keyword], image_keywords)
        train_labels.append(np.array(list(labels)))
    train_labels = np.asarray(train_labels)

    dev_labels = []
    for image_keywords in dev_exp.unique_image_keywords:
        labels = map(
            lambda keyword: train_exp.keyword_labels[keyword], image_keywords)
        dev_labels.append(np.array(list(labels)))
    dev_labels = np.asarray(dev_labels)

    train_paths = train_exp.unique_image_paths
    dev_paths = dev_exp.unique_image_paths

    mlb = MultiLabelBinarizer()
    train_labels_multi_hot = mlb.fit_transform(train_labels)
    dev_labels_multi_hot = mlb.transform(dev_labels)

    # define preprocessing for images
    preprocess_images_func = functools.partial(
        dataset.load_and_preprocess_image,
        crop_size=model_options["crop_size"],
        augment_crop=model_options["augment_train"],
        random_scales=model_options["random_scales"],
        horizontal_flip=model_options["horizontal_flip"],
        colour=model_options["colour"])

    # create standard training dataset pipeline
    background_train_ds = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(train_paths),
        tf.data.Dataset.from_tensor_slices(train_labels_multi_hot)))

    # map data preprocessing function across training data
    background_train_ds = background_train_ds.map(
        lambda path, label: (preprocess_images_func(path), label),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # repeat augmentation, shuffle and batch train data
    if model_options["num_augment"] is not None:
        background_train_ds = background_train_ds.repeat(
            model_options["num_augment"])

    background_train_ds = background_train_ds.shuffle(1000)

    background_train_ds = background_train_ds.batch(
        model_options["batch_size"])

    background_train_ds = background_train_ds.prefetch(
        tf.data.experimental.AUTOTUNE)

    # create dev set pipeline for classification validation
    background_dev_ds = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(
            dev_paths).map(preprocess_images_func),
        tf.data.Dataset.from_tensor_slices(dev_labels_multi_hot)))

    background_dev_ds = background_dev_ds.batch(
        batch_size=model_options["batch_size"])

    # write example batch to TensorBoard
    if tf_writer is not None:
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
    multi_label_loss = get_training_objective(model_options)

    # load or create model
    if model_file is not None:
        assert model_options["n_classes"] == len(train_exp.keywords)

        vision_network, train_state = base.load_model(
            model_file=os.path.join(output_dir, model_file),
            model_step_file=os.path.join(output_dir, model_step_file),
            loss=multi_label_loss)

        # get previous tracking variables
        initial_model = False
        global_step, model_epochs, _, best_val_score = train_state
    else:
        model_options["n_classes"] = len(train_exp.keywords)

        vision_network = create_model(model_options)

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
    vision_network.compile(optimizer=optimizer, loss=multi_label_loss)

    # wrap few-shot model for background training
    vision_few_shot_model = vision_cnn.FewShotModel(
        vision_network, multi_label_loss)

    # get embedding function for one-shot validation and early stopping
    if model_options["one_shot_validation"]:
        embedding_func = get_embedding_function(vision_network, model_options)

        # test model on one-shot validation task prior to any training
        val_task_accuracy, _, conf_interval_95 = experiments.test_l_way_k_shot(
            dev_exp, FLAGS.K, FLAGS.L, n=FLAGS.N,
            num_episodes=FLAGS.episodes, metric="cosine",
            data_preprocess_func=embedding_func)

        logging.log(
            logging.INFO,
            f"Base model: {FLAGS.L}-way {FLAGS.K}-shot accuracy after "
            f"{FLAGS.episodes} episodes: {val_task_accuracy:.3%} +- "
            f"{conf_interval_95*100:.4f}")

    # create training metrics
    precision_metric = tf.keras.metrics.Precision()
    recall_metric = tf.keras.metrics.Recall()
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

        precision_metric.reset_states()
        recall_metric.reset_states()
        loss_metric.reset_states()

        # train on epoch of training data
        step_pbar = tqdm(background_train_ds,
                         bar_format="{desc} [{elapsed},{rate_fmt}{postfix}]")
        for step, (x_batch, y_batch) in enumerate(step_pbar):

            loss_value, y_predict = vision_few_shot_model.train_step(
                x_batch, y_batch, optimizer,
                clip_norm=model_options["gradient_clip_norm"])

            y_one_hot_predict = tf.round(tf.nn.sigmoid(y_predict))

            precision_metric.update_state(y_batch, y_one_hot_predict)
            recall_metric.update_state(y_batch, y_one_hot_predict)
            loss_metric.update_state(loss_value)

            step_loss = tf.reduce_mean(loss_value)
            train_loss = loss_metric.result().numpy()
            train_precision = precision_metric.result().numpy()
            train_recall = recall_metric.result().numpy()
            train_f1 = 2 / ((1/train_precision) + (1/train_recall))

            step_pbar.set_description_str(
                f"\tStep {step:03d}: "
                f"Step loss: {step_loss:.6f}, "
                f"Loss: {train_loss:.6f}, "
                f"Precision: {train_precision:.3%}, "
                f"Recall: {train_recall:.3%}, "
                f"F-1: {train_f1:.3%}")

            if tf_writer is not None:
                with tf_writer.as_default():
                    tf.summary.scalar(
                        "Train step loss", step_loss, step=global_step)
            global_step += 1

        # validate classification model
        precision_metric.reset_states()
        recall_metric.reset_states()
        loss_metric.reset_states()

        for x_batch, y_batch in background_dev_ds:
            y_predict = vision_few_shot_model.predict(x_batch, training=False)
            loss_value = vision_few_shot_model.loss(y_batch, y_predict)

            y_one_hot_predict = tf.round(tf.nn.sigmoid(y_predict))

            precision_metric.update_state(y_batch, y_one_hot_predict)
            recall_metric.update_state(y_batch, y_one_hot_predict)
            loss_metric.update_state(loss_value)

        dev_loss = loss_metric.result().numpy()
        dev_precision = precision_metric.result().numpy()
        dev_recall = recall_metric.result().numpy()
        dev_f1 = 2 / ((1/dev_precision) + (1/dev_recall))

        # validate model on one-shot dev task if specified
        if model_options["one_shot_validation"]:
            val_task_accuracy, _, conf_interval_95 = experiments.test_l_way_k_shot(
                dev_exp, FLAGS.K, FLAGS.L, n=FLAGS.N,
                num_episodes=FLAGS.episodes, metric="cosine",
                data_preprocess_func=embedding_func)

            val_score = val_task_accuracy
            val_metric = f"{FLAGS.L}-way {FLAGS.K}-shot accuracy"

        # otherwise, validate on classification task
        else:
            val_score = dev_f1
            val_metric = "F-1"

        if val_score >= best_val_score:
            best_val_score = val_score
            best_model = True

        # log results
        logging.log(
            logging.INFO,
            f"Train: Loss: {train_loss:.6f}, Precision: {train_precision:.3%}, "
            f"Recall: {train_recall:.3%}, F-1: {train_f1:.3%}")

        logging.log(
            logging.INFO,
            f"Validation: Loss: {dev_loss:.6f}, Precision: "
            f"{dev_precision:.3%}, Recall: {dev_recall:.3%}, F-1: "
            f"{dev_f1:.3%} {'*' if best_model else ''}")

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
                    "Train precision", train_precision, step=global_step)
                tf.summary.scalar(
                    "Train recall", train_recall, step=global_step)
                tf.summary.scalar(
                    "Train F-1", train_f1, step=global_step)
                tf.summary.scalar(
                    "Validation loss", dev_loss, step=global_step)
                tf.summary.scalar(
                    "Validation precision", dev_precision, step=global_step)
                tf.summary.scalar(
                    "Validation recall", dev_recall, step=global_step)
                tf.summary.scalar(
                    "Validation F-1", dev_f1, step=global_step)
                if model_options["one_shot_validation"]:
                    tf.summary.scalar(
                        f"Validation {FLAGS.L}-way {FLAGS.K}-shot accuracy",
                        val_task_accuracy, step=global_step)

        # store model and results
        base.save_model(
            vision_few_shot_model.model, output_dir, epoch + 1, global_step,
            val_metric, val_score, best_val_score, name="model")

        if best_model:
            best_model = False

            base.save_model(
                vision_few_shot_model.model, output_dir, epoch + 1, global_step,
                val_metric, val_score, best_val_score, name="best_model")


def embed(model_options, output_dir, model_file, model_step_file):
    """Load classification model and extract embeddings."""

    # load model
    vision_network, _ = base.load_model(
        model_file=os.path.join(output_dir, model_file),
        model_step_file=os.path.join(output_dir, model_step_file),
        loss=get_training_objective(model_options))

    # get model embedding function
    embedding_func = get_embedding_function(vision_network, model_options)

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
            keywords_split="one_shot_evaluation", **train_image_dir_dict)

        background_train_exp = flickr_vision.FlickrVision(
            keywords_split="background_train", **train_image_dir_dict)

        background_dev_exp = flickr_vision.FlickrVision(
            keywords_split="background_dev", **dev_image_dir_dict)

        subset_exp = {
            "one_shot_evaluation": one_shot_exp,
            "background_train": background_train_exp,
            "background_dev": background_dev_exp,
        }

        for subset, exp in subset_exp.items():
            embed_dir = os.path.join(
                output_dir, "embed", FLAGS.embed_layer, data, subset)
            file_io.check_create_dir(embed_dir)

            unique_paths = np.unique(exp.image_paths)

            # batch images paths for faster embedding inference
            path_ds = tf.data.Dataset.from_tensor_slices(unique_paths)
            path_ds = path_ds.batch(model_options["batch_size"])
            path_ds = path_ds.prefetch(tf.data.experimental.AUTOTUNE)

            num_samples = int(
                np.ceil(len(unique_paths) / model_options["batch_size"]))

            start_time = time.time()
            paths, embeddings = [], []
            for path_batch in tqdm(path_ds, total=num_samples):
                path_embeddings = embedding_func(path_batch)

                paths.extend(path_batch.numpy())
                embeddings.extend(path_embeddings.numpy())
            end_time = time.time()

            logging.log(
                logging.INFO,
                f"Computed embeddings ({FLAGS.embed_layer}) for {data} {subset}"
                f" in {end_time - start_time:.4f} seconds")

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
    """Load and test classification model for one-shot learning."""

    # load Flickr 8k one-shot experiment
    one_shot_exp = flickr_vision.FlickrVision(
        keywords_split="one_shot_evaluation",
        flickr8k_image_dir=os.path.join("data", "external", "flickr8k_images"),
        embed_dir=None)

    # load model
    triplet_loss = losses.triplet_semihard_loss(
        margin=model_options["margin"], metric=model_options["metric"])

    vision_network, _ = base.load_model(
        model_file=os.path.join(output_dir, model_file),
        model_step_file=os.path.join(output_dir, model_step_file),
        loss=triplet_loss)

    # get model embedding function
    embedding_func = get_embedding_function(vision_network, model_options)

    # test model on L-way K-shot task
    task_accuracy, _, conf_interval_95 = experiments.test_l_way_k_shot(
        one_shot_exp, FLAGS.K, FLAGS.L, n=FLAGS.N, num_episodes=FLAGS.episodes,
        metric="cosine", num_neighbours=FLAGS.k_neighbours,
        data_preprocess_func=embedding_func)

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
        if FLAGS.target == "test" or FLAGS.target == "validate":
            raise ValueError(
                "Target `{FLAGS.target}` requires --output_dir to be specified.")

        output_dir = os.path.join(
            "logs", __file__, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        file_io.check_create_dir(output_dir)

        model_options = DEFAULT_OPTIONS

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

        elif FLAGS.target == "test" or FLAGS.target == "validate":
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
