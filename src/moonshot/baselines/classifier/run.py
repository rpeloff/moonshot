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
from moonshot.baselines import losses
from moonshot.baselines.classifier import vision_cnn
from moonshot.baselines.pixel_match import pixel_match
from moonshot.experiments.flickr_vision import flickr_vision
from moonshot.models import inceptionv3
from moonshot.utils import file_io
from moonshot.utils import logging as logging_utils


FLAGS = flags.FLAGS


# model options (default if not loaded)
DEFAULT_OPTIONS = {
    # training data
    "data": ["flickr8k", "flickr30k", "mscoco"],
    # preprocessing
    "crop_size": 299,
    "batch_size": 32,
    "augment_train": True,
    "num_augment": 4,
    "random_scales": None,
    "horizontal_flip": True,
    "colour": True,
    # inceptionv3 classifier
    "pretrained": False,
    "dense_units": [2048],
    "dropout_rate": 0.25,
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

# logging and target options
flags.DEFINE_enum("target", "train", ["train", "validate", "embed", "test"],
                  "train or load and test a model")
flags.DEFINE_string("output_dir", None, "directory where logs and checkpoints will be stored"
                    "(defaults to logs/<unique run id>)")
flags.DEFINE_bool("load_best", False, "load previous best model for resumed training or testing")
flags.DEFINE_enum("embed_layer", "dense", ["conv", "avg_pool", "dense", "logits"],
                  "model layer to slice embeddings from")
flags.DEFINE_bool("mc_dropout", False, "Make embedding predictions with MC Dropout")
flags.DEFINE_bool("resume", True, "resume training if a checkpoint is found at output directory")
flags.DEFINE_bool("tensorboard", True, "log train and test summaries to TensorBoard")
flags.DEFINE_bool("debug", False, "log with debug verbosity level")


def create_train_data(data_sets):

    train_paths = []
    train_keywords = []
    dev_paths = []
    dev_keywords = []
    keyword_classes = set()

    if "flickr8k" in data_sets:
        flickr8k_train_exp = flickr_vision.FlickrVision(
            os.path.join("data", "external", "flickr8k_images"),
            keywords_split="background_train.csv",
            splits_dir=os.path.join("data", "splits", "flickr8k"))

        flickr8k_dev_exp = flickr_vision.FlickrVision(
            os.path.join("data", "external", "flickr8k_images"),
            keywords_split="background_dev.csv",
            splits_dir=os.path.join("data", "splits", "flickr8k"))

        train_paths += flickr8k_train_exp.unique_image_paths
        train_keywords += flickr8k_train_exp.unique_image_keywords
        dev_paths += flickr8k_dev_exp.unique_image_paths
        dev_keywords += flickr8k_dev_exp.unique_image_keywords
        keyword_classes |= set(flickr8k_train_exp.keywords)

    if "flickr30k" in data_sets:
        flickr30k_train_exp = flickr_vision.FlickrVision(
            os.path.join("data", "external", "flickr30k_images"),
            keywords_split="background_train.csv",
            splits_dir=os.path.join("data", "splits", "flickr30k"))

        flickr30k_dev_exp = flickr_vision.FlickrVision(
            os.path.join("data", "external", "flickr30k_images"),
            keywords_split="background_dev.csv",
            splits_dir=os.path.join("data", "splits", "flickr30k"))

        train_paths += flickr30k_train_exp.unique_image_paths
        train_keywords += flickr30k_train_exp.unique_image_keywords
        dev_paths += flickr30k_dev_exp.unique_image_paths
        dev_keywords += flickr30k_dev_exp.unique_image_keywords
        keyword_classes |= set(flickr30k_train_exp.keywords)

    if "mscoco" in data_sets:
        mscoco_train_exp = flickr_vision.FlickrVision(
            os.path.join("data", "external", "mscoco", "train2017"),
            keywords_split="background_train.csv",
            splits_dir=os.path.join("data", "splits", "mscoco"))

        mscoco_dev_exp = flickr_vision.FlickrVision(
            os.path.join("data", "external", "mscoco", "val2017"),
            keywords_split="background_dev.csv",
            splits_dir=os.path.join("data", "splits", "mscoco"))

        train_paths += mscoco_train_exp.unique_image_paths
        train_keywords += mscoco_train_exp.unique_image_keywords
        dev_paths += mscoco_dev_exp.unique_image_paths
        dev_keywords += mscoco_dev_exp.unique_image_keywords
        keyword_classes |= set(mscoco_train_exp.keywords)

    keyword_classes = list(sorted(keyword_classes))

    keyword_id_lookup = {
        keyword: idx for idx, keyword in enumerate(keyword_classes)}

    train_labels = []
    for image_keywords in train_keywords:
        labels = map(lambda keyword: keyword_id_lookup[keyword], image_keywords)
        train_labels.append(np.array(list(labels)))

    dev_labels = []
    for image_keywords in dev_keywords:
        labels = map(lambda keyword: keyword_id_lookup[keyword], image_keywords)
        dev_labels.append(np.array(list(labels)))

    return (keyword_classes,
            (train_paths, train_keywords, train_labels),
            (dev_paths, dev_keywords, dev_labels))


def get_training_objective(model_options):
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

    inception_base = inceptionv3.inceptionv3_base(
        input_shape=(
            model_options["crop_size"], model_options["crop_size"], 3),
        pretrained=model_options["pretrained"])

    if model_options["pretrained"]:
        inceptionv3.freeze_base_model(inception_base, trainable=None)

    model_layers = [
        inception_base,
        tf.keras.layers.GlobalAveragePooling2D(),
    ]

    if model_options["dropout_rate"] is not None:
        model_layers.append(
            tf.keras.layers.Dropout(model_options["dropout_rate"]))

    if model_options["dense_units"] is not None:
        for dense_units in model_options["dense_units"]:
            model_layers.append(
                tf.keras.layers.Dense(dense_units))

            model_layers.append(tf.keras.layers.ReLU())

            if model_options["dropout_rate"] is not None:
                model_layers.append(
                    tf.keras.layers.Dropout(model_options["dropout_rate"]))

    model_layers.append(tf.keras.layers.Dense(model_options["n_classes"]))

    vision_classifier = tf.keras.Sequential(model_layers)

    return vision_classifier


def train(model_options, output_dir, model_file=None, model_step_file=None,
          tf_writer=None):

    # laod and create datasets
    keyword_classes, train_data, dev_data = create_train_data(
        model_options["data"])

    train_paths, train_keywords, train_labels = train_data
    dev_paths, dev_keywords, dev_labels = dev_data

    mlb = MultiLabelBinarizer()
    train_labels_multi_hot = mlb.fit_transform(train_labels)
    dev_labels_multi_hot = mlb.transform(dev_labels)

    preprocess_images_func = functools.partial(
        dataset.load_and_preprocess_image, crop_size=model_options["crop_size"],
        augment_crop=model_options["augment_train"],
        random_scales=model_options["random_scales"],
        horizontal_flip=model_options["horizontal_flip"],
        colour=model_options["colour"])

    background_train_ds = dataset.create_flickr_background_dataset(
        train_paths, train_labels_multi_hot,
        image_preprocess_func=preprocess_images_func,
        batch_size=model_options["batch_size"],
        num_repeat=model_options["num_augment"], shuffle=True)

    background_dev_ds = dataset.create_flickr_background_dataset(
        dev_paths, dev_labels_multi_hot,
        image_preprocess_func=preprocess_images_func,
        batch_size=model_options["batch_size"], shuffle=False)

    if tf_writer is not None:
        with tf_writer.as_default():
            for x_batch, y_batch in background_train_ds.take(1):
                tf.summary.image("Example train images", (x_batch+1)/2,
                                 max_outputs=25, step=0)

    # get training objective
    multi_label_loss = get_training_objective(model_options)

    # load or create model
    if model_file is not None:
        assert model_options["n_classes"] == len(keyword_classes)

        vision_classifier, train_state = base.load_model(
            model_file=os.path.join(output_dir, model_file),
            model_step_file=os.path.join(output_dir, model_step_file),
            loss=multi_label_loss)

        global_step, model_epochs, _, best_val_score = train_state
    else:
        model_options["n_classes"] = len(keyword_classes)

        vision_classifier = create_model(model_options)

        global_step = 0
        model_epochs = 0
        best_val_score = -np.inf

    best_model = False

    # load or create Adam optimizer with decayed learning rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        model_options["learning_rate"], decay_rate=model_options["decay_rate"],
        decay_steps=model_options["decay_steps"], staircase=True)

    if model_file is not None:
        logging.log(logging.INFO, "Restoring optimizer state")
        optimizer = vision_classifier.optimizer
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # compile model to store optimizer with model when saving
    vision_classifier.compile(optimizer=optimizer, loss=multi_label_loss)

    # wrap few-shot model for background training
    vision_few_shot_model = vision_cnn.FewShotModel(
        vision_classifier, multi_label_loss)

    # create training metrics
    precision_metric = tf.keras.metrics.Precision()
    recall_metric = tf.keras.metrics.Recall()
    loss_metric = tf.keras.metrics.Mean()

    # store model options
    file_io.write_json(
        os.path.join(output_dir, "model_options.json"), model_options)

    # train model
    for epoch in range(model_epochs, model_options["epochs"]):
        logging.log(logging.INFO, f"Epoch {epoch:03d}")

        precision_metric.reset_states()
        recall_metric.reset_states()
        loss_metric.reset_states()

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

        if dev_f1 >= best_val_score:
            best_val_score = dev_f1
            best_model = True

        logging.log(
            logging.INFO,
            f"Train: Loss: {train_loss:.6f}, Precision: {train_precision:.3%}, "
            f"Recall: {train_recall:.3%}, F-1: {train_f1:.3%}")
        logging.log(
            logging.INFO,
            f"Validation: Loss: {dev_loss:.6f}, Precision: "
            f"{dev_precision:.3%}, Recall: {dev_recall:.3%}, F-1: "
            f"{dev_f1:.3%} {'*' if best_model else ''}")

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

        base.save_model(
            vision_few_shot_model.model, output_dir, epoch + 1, global_step,
            "F-1", dev_f1, best_val_score, name="model")

        if best_model:
            best_model = False

            base.save_model(
                vision_few_shot_model.model, output_dir, epoch + 1, global_step,
                "F-1", dev_f1, best_val_score, name="best_model")


def validate(model_options, output_dir, model_file=None, model_step_file=None):

    # laod and create datasets
    keyword_classes, train_data, dev_data = create_train_data(
        model_options["data"])

    train_paths, train_keywords, train_labels = train_data
    dev_paths, dev_keywords, dev_labels = dev_data

    mlb = MultiLabelBinarizer()
    train_labels_multi_hot = mlb.fit_transform(train_labels)
    dev_labels_multi_hot = mlb.transform(dev_labels)

    preprocess_images_func = functools.partial(
        dataset.load_and_preprocess_image, crop_size=model_options["crop_size"],
        augment_crop=False)

    background_train_ds = dataset.create_flickr_background_dataset(
        train_paths, train_labels_multi_hot,
        image_preprocess_func=preprocess_images_func,
        batch_size=model_options["batch_size"], shuffle=True)

    background_dev_ds = dataset.create_flickr_background_dataset(
        dev_paths, dev_labels_multi_hot,
        image_preprocess_func=preprocess_images_func,
        batch_size=model_options["batch_size"], shuffle=False)

    # get training objective
    multi_label_loss = get_training_objective(model_options)

    # load model
    vision_classifier, _ = base.load_model(
        model_file=os.path.join(output_dir, model_file),
        model_step_file=os.path.join(output_dir, model_step_file),
        loss=get_training_objective(model_options))

    # wrap few-shot model for background training
    vision_few_shot_model = vision_cnn.FewShotModel(
        vision_classifier, multi_label_loss)

    # create training metrics
    precision_metric = tf.keras.metrics.Precision()
    recall_metric = tf.keras.metrics.Recall()
    loss_metric = tf.keras.metrics.Mean()

    for x_batch, y_batch in background_train_ds:
        y_predict = vision_few_shot_model.predict(x_batch, training=False)
        loss_value = vision_few_shot_model.loss(y_batch, y_predict)

        y_one_hot_predict = tf.round(tf.nn.sigmoid(y_predict))

        precision_metric.update_state(y_batch, y_one_hot_predict)
        recall_metric.update_state(y_batch, y_one_hot_predict)
        loss_metric.update_state(loss_value)

    train_loss = loss_metric.result().numpy()
    train_precision = precision_metric.result().numpy()
    train_recall = recall_metric.result().numpy()
    train_f1 = 2 / ((1/train_precision) + (1/train_recall))

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

    logging.log(
        logging.INFO,
        f"Train: Loss: {train_loss:.6f}, Precision: {train_precision:.3%}, "
        f"Recall: {train_recall:.3%}, F-1: {train_f1:.3%}")
    logging.log(
        logging.INFO,
        f"Validation: Loss: {dev_loss:.6f}, Precision: {dev_precision:.3%}, "
        f"Recall: {dev_recall:.3%}, F-1: {dev_f1:.3%}")


def embed(model_options, output_dir, model_file, model_step_file):

    # load model
    vision_classifier, _ = base.load_model(
        model_file=os.path.join(output_dir, model_file),
        model_step_file=os.path.join(output_dir, model_step_file),
        loss=get_training_objective(model_options))

    # slice embedding model from specified layer
    if FLAGS.embed_layer == "conv":  # conv layer (flattened)
        slice_index = 0
    elif FLAGS.embed_layer == "avg_pool":  # global average pool layer
        slice_index = 1
    elif FLAGS.embed_layer == "dense":  # dense layer before relu & logits layer
        slice_index = -3 if model_options["dropout_rate"] is None else -4
    elif FLAGS.embed_layer == "logits":  # unnormalised log probabilities
        slice_index = -1

    model_input = (vision_classifier.layers[0].input if slice_index == 0 else
                   vision_classifier.input)
    embed_model = tf.keras.Model(
        inputs=model_input,
        outputs=vision_classifier.layers[slice_index].output)

    embed_few_shot_model = vision_cnn.FewShotModel(
        embed_model, None, mc_dropout=FLAGS.mc_dropout)

    # create fast autograph embedding function
    input_shape = [
        None, model_options["crop_size"], model_options["crop_size"], 3]
    @tf.function(
        input_signature=(tf.TensorSpec(shape=input_shape, dtype=tf.float32),))
    def embed_images(image_batch):
        return embed_few_shot_model.predict(image_batch)

    # load datasets seen in training and embed the image data
    for data in model_options["data"]:

        if data == "mscoco":
            train_image_dir = os.path.join("mscoco", "train2017")
            dev_image_dir = os.path.join("mscoco", "val2017")
        else:
            train_image_dir = dev_image_dir = f"{data}_images"

        one_shot_exp = flickr_vision.FlickrVision(
            os.path.join("data", "external", train_image_dir),
            keywords_split="one_shot_evaluation.csv",
            splits_dir=os.path.join("data", "splits", data))

        background_train_exp = flickr_vision.FlickrVision(
            os.path.join("data", "external", train_image_dir),
            keywords_split="background_train.csv",
            splits_dir=os.path.join("data", "splits", data))

        background_dev_exp = flickr_vision.FlickrVision(
            os.path.join("data", "external", dev_image_dir),
            keywords_split="background_dev.csv",
            splits_dir=os.path.join("data", "splits", data))

        subset_exp = {
            "one_shot_evaluation": one_shot_exp,
            "background_train": background_train_exp,
            "background_dev": background_dev_exp,
        }

        for subset, exp in subset_exp.items():
            embed_dir = os.path.join(
                output_dir, "embed", FLAGS.embed_layer, data, subset)
            file_io.check_create_dir(embed_dir)

            unique_image_paths = np.unique(exp.image_paths)

            # load and center square crop images along shortest edges
            embed_ds = tf.data.Dataset.from_tensor_slices(unique_image_paths)
            embed_ds = embed_ds.map(
                lambda img_path: (
                    img_path, dataset.load_and_preprocess_image(
                        img_path, crop_size=model_options["crop_size"])),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
            # batch images for faster embedding inference
            embed_ds = embed_ds.batch(model_options["batch_size"])
            embed_ds = embed_ds.prefetch(tf.data.experimental.AUTOTUNE)

            num_samples = int(
                np.ceil(len(unique_image_paths) / model_options["batch_size"]))

            start_time = time.time()
            image_paths, image_embeddings = [], []
            for path_batch, image_batch in tqdm(embed_ds, total=num_samples):
                embeddings = embed_images(image_batch)
                embeddings = tf.reshape(  # flatten in case of conv output
                    embeddings, [tf.shape(embeddings)[0], -1])

                image_paths.extend(path_batch.numpy())
                image_embeddings.extend(embeddings.numpy())
            end_time = time.time()

            logging.log(
                logging.INFO,
                f"Computed embeddings ({FLAGS.embed_layer}) for {data} {subset}"
                f" in {end_time - start_time:.4f} seconds")

            # serialize and write embeddings to TFRecord files
            for image_path, image_embed in zip(image_paths, image_embeddings):
                example_proto = dataset.embedding_to_example_protobuf(
                    image_embed)

                image_path = image_path.decode("utf-8")
                image_path = os.path.join(
                    embed_dir, f"{os.path.split(image_path)[1]}.tfrecord")

                with tf.io.TFRecordWriter(image_path, options="ZLIB") as writer:
                    writer.write(example_proto.SerializeToString())

            # TODO: remove old code for npz embedding files
            # for image_path, image_embed in zip(image_paths, image_embeddings):
            #     image_path = image_path.decode("utf-8")
            #     np.savez_compressed(
            #         os.path.join(
            #             embed_dir, f"{os.path.split(image_path)[1]}.npz"),
            #         embed=image_embed)

            logging.log(logging.INFO, f"Embeddings stored at: {embed_dir}")


def test(model_options, output_dir, model_file, model_step_file):

    # get embeddings directory
    embed_dir = os.path.join(
        output_dir, "embed", FLAGS.embed_layer, "flickr8k",
        "one_shot_evaluation")

    if not os.path.exists(embed_dir):
        raise ValueError(
            f"Directory '{embed_dir}' does not exist. Compute embeddings with "
            "`--target embed` first.")

    # load and create datasets
    one_shot_exp = flickr_vision.FlickrVision(
        os.path.join("data", "external", "flickr8k_images"),
        keywords_split="one_shot_evaluation.csv",
        splits_dir=os.path.join("data", "splits", "flickr8k"),
        embed_dir=embed_dir)

    keyword_classes = list(sorted(set(one_shot_exp.keywords)))
    keyword_id_lookup = {
        keyword: idx for idx, keyword in enumerate(keyword_classes)}

    pixel_model = pixel_match.PixelMatch(metric="cosine", preprocess=None)

    total_correct = 0
    total_tests = 0

    for episode in tqdm(range(FLAGS.episodes)):

        one_shot_exp.sample_episode(FLAGS.L, FLAGS.K, FLAGS.N)

        train_paths, y_train = one_shot_exp.learning_samples
        test_paths, y_test = one_shot_exp.evaluation_samples

        x_train_embeds = dataset.load_embedding_records(train_paths)
        x_train_embeds = np.stack(list(x_train_embeds))

        y_train_labels = map(lambda keyword: keyword_id_lookup[keyword], y_train)
        y_train_labels = np.stack(list(y_train_labels))

        x_test_embeds = dataset.load_embedding_records(test_paths)
        x_test_embeds = np.stack(list(x_test_embeds))

        y_test_labels = map(lambda keyword: keyword_id_lookup[keyword], y_test)
        y_test_labels = np.stack(list(y_test_labels))

        adapted_model = pixel_model.adapt_model(x_train_embeds, y_train_labels)

        test_predict = adapted_model.predict(x_test_embeds, FLAGS.k_neighbours)

        num_correct = np.sum(test_predict == y_test_labels)

        total_correct += num_correct
        total_tests += len(y_test)

    logging.log(
        logging.INFO,
        "{}-way {}-shot accuracy after {} episodes : {:.6f}".format(
            FLAGS.L, FLAGS.K, FLAGS.episodes, total_correct/total_tests))


def main(argv):
    del argv  # unused

    logging.log(logging.INFO, "Logging application {}".format(__file__))
    if FLAGS.debug:
        logging.set_verbosity(logging.DEBUG)
        logging.log(logging.DEBUG, "Running in debug mode")

    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    model_found = False
    if FLAGS.output_dir is None:
        if FLAGS.target == "test" or FLAGS.target == "validate":
            raise ValueError(
                "Target `{FLAGS.target}` requires --output_dir to be specified.")

        output_dir = os.path.join(
            "logs", __file__, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        file_io.check_create_dir(output_dir)

        model_options = DEFAULT_OPTIONS
    else:
        output_dir = FLAGS.output_dir
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
    elif FLAGS.target == "validate":
        validate(model_options, output_dir, model_file, model_step_file)
    elif FLAGS.target == "embed":
        embed(model_options, output_dir, model_file, model_step_file)
    else:
        test(model_options, output_dir, model_file, model_step_file)


if __name__ == "__main__":
    app.run(main)
