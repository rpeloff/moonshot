"""TODO(rpeloff)

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: October 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import datetime


from absl import app
from absl import flags
from absl import logging


import numpy as np
import tensorflow as tf
from tqdm import tqdm


from moonshot.baselines import base
from moonshot.baselines import dataset
from moonshot.baselines import losses
from moonshot.baselines.classifier import vision_cnn
from moonshot.experiments.flickr_vision import flickr_vision
from moonshot.utils import file_io
from moonshot.utils import logging as logging_utils


FLAGS = flags.FLAGS


# model options (default if not loaded)
DEFAULT_OPTIONS = {
    # training data
    "data": ["flickr8k", "flickr30k", "mscoco"],
    # preprocessing
    "input_size": 2048,
    "batch_size": 256,
    # siamese layers
    "dense_units": [2048, 2048, 2048],
    "dropout_rate": 0.25,
    # triplet objective
    "margin": .2,  # 1.
    # training
    "learning_rate": 1e-4,
    "decay_steps": 25000,
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
flags.DEFINE_string("embed_dir", None, "directory containing base network embeddings (should "
                    "contain 'flickr8k', 'flickr30k' and 'mscoco' directories)")
flags.DEFINE_string("output_dir", None, "directory where logs and checkpoints will be stored"
                    "(defaults to logs/<unique run id>)")
flags.DEFINE_bool("load_best", False, "load previous best model for resumed training or testing")
flags.DEFINE_enum("embed_layer", "dense", ["conv", "avg_pool", "dense", "logits"],
                  "model layer to slice embeddings from")
flags.DEFINE_bool("mc_dropout", False, "Make embedding predictions with MC Dropout")
flags.DEFINE_bool("resume", True, "resume training if a checkpoint is found at output directory")
flags.DEFINE_bool("tensorboard", True, "log train and test summaries to TensorBoard")
flags.DEFINE_bool("debug", False, "log with debug verbosity level")

# required flags
flags.mark_flag_as_required("embed_layer")


def create_train_data(data_sets, embed_dir):

    train_paths = []
    train_keywords = []
    dev_paths = []
    dev_keywords = []
    keyword_classes = set()

    if "flickr8k" in data_sets:
        flickr8k_train_exp = flickr_vision.FlickrVision(
            os.path.join("data", "external", "flickr8k_images"),
            keywords_split="background_train.csv",
            embed_dir=os.path.join(embed_dir, "flickr8k", "background_train"),
            splits_dir=os.path.join("data", "splits", "flickr8k"))

        flickr8k_dev_exp = flickr_vision.FlickrVision(
            os.path.join("data", "external", "flickr8k_images"),
            keywords_split="background_dev.csv",
            embed_dir=os.path.join(embed_dir, "flickr8k", "background_dev"),
            splits_dir=os.path.join("data", "splits", "flickr8k"))

        train_paths += flickr8k_train_exp.embed_paths.tolist()
        train_keywords += flickr8k_train_exp.keywords_set[3].tolist()
        dev_paths += flickr8k_dev_exp.embed_paths.tolist()
        dev_keywords += flickr8k_dev_exp.keywords_set[3].tolist()
        keyword_classes |= set(flickr8k_train_exp.keywords)

    if "flickr30k" in data_sets:
        flickr30k_train_exp = flickr_vision.FlickrVision(
            os.path.join("data", "external", "flickr30k_images"),
            keywords_split="background_train.csv",
            embed_dir=os.path.join(embed_dir, "flickr30k", "background_train"),
            splits_dir=os.path.join("data", "splits", "flickr30k"))

        flickr30k_dev_exp = flickr_vision.FlickrVision(
            os.path.join("data", "external", "flickr30k_images"),
            keywords_split="background_dev.csv",
            embed_dir=os.path.join(embed_dir, "flickr30k", "background_dev"),
            splits_dir=os.path.join("data", "splits", "flickr30k"))

        train_paths += flickr30k_train_exp.embed_paths.tolist()
        train_keywords += flickr30k_train_exp.keywords_set[3].tolist()
        dev_paths += flickr30k_dev_exp.embed_paths.tolist()
        dev_keywords += flickr30k_dev_exp.keywords_set[3].tolist()
        keyword_classes |= set(flickr30k_train_exp.keywords)

    if "mscoco" in data_sets:
        mscoco_train_exp = flickr_vision.FlickrVision(
            os.path.join("data", "external", "mscoco", "train2017"),
            keywords_split="background_train.csv",
            embed_dir=os.path.join(embed_dir, "mscoco", "background_train"),
            splits_dir=os.path.join("data", "splits", "mscoco"))

        mscoco_dev_exp = flickr_vision.FlickrVision(
            os.path.join("data", "external", "mscoco", "val2017"),
            keywords_split="background_dev.csv",
            embed_dir=os.path.join(embed_dir, "mscoco", "background_dev"),
            splits_dir=os.path.join("data", "splits", "mscoco"))

        train_paths += mscoco_train_exp.embed_paths.tolist()
        train_keywords += mscoco_train_exp.keywords_set[3].tolist()
        dev_paths += mscoco_dev_exp.embed_paths.tolist()
        dev_keywords += mscoco_dev_exp.keywords_set[3].tolist()
        keyword_classes |= set(mscoco_train_exp.keywords)

    keyword_classes = list(sorted(keyword_classes))

    keyword_id_lookup = {
        keyword: idx for idx, keyword in enumerate(keyword_classes)}

    train_labels = []
    for keyword in train_keywords:
        label = keyword_id_lookup[keyword]
        train_labels.append(label)

    dev_labels = []
    for keyword in dev_keywords:
        label = keyword_id_lookup[keyword]
        dev_labels.append(label)

    return (keyword_classes,
            (train_paths, train_keywords, train_labels),
            (dev_paths, dev_keywords, dev_labels))


def create_model(model_options, input_shape=None):

    assert model_options["dense_units"] is not None

    model_layers = []

    if model_options["dropout_rate"] is not None:
        model_layers.append(
            tf.keras.layers.Dropout(model_options["dropout_rate"]))

    for dense_units in model_options["dense_units"][:-2]:
        model_layers.append(
            tf.keras.layers.Dense(dense_units))

        model_layers.append(tf.keras.layers.ReLU())

        if model_options["dropout_rate"] is not None:
            model_layers.append(
                tf.keras.layers.Dropout(model_options["dropout_rate"]))

    model_layers.append(tf.keras.layers.Dense(model_options["dense_units"][-1]))

    vision_network = tf.keras.Sequential(model_layers)

    if input_shape is not None:
        vision_network.build(input_shape)

    return vision_network


def train(model_options, output_dir, model_file=None,
          model_step_file=None, tf_writer=None):

    # laod and create datasets
    keyword_classes, train_data, dev_data = create_train_data(
        model_options["data"], model_options["embed_dir"])

    train_paths, train_keywords, train_labels = train_data
    dev_paths, dev_keywords, dev_labels = dev_data

    def preprocess_embeds_func(embedding_proto):
        return dataset.parse_embedding_protobuf(embedding_proto)["embed"]

    background_train_ds = dataset.create_flickr_background_dataset(
        train_paths, train_labels, tfrecords=True,
        image_preprocess_func=preprocess_embeds_func,
        batch_size=model_options["batch_size"], shuffle=True)

    background_dev_ds = dataset.create_flickr_background_dataset(
        dev_paths, dev_labels, tfrecords=True,
        image_preprocess_func=preprocess_embeds_func,
        batch_size=model_options["batch_size"], shuffle=False)

    # get training objective
    triplet_loss = losses.triplet_semihard_loss(margin=model_options["margin"])

    # load or create model
    if model_file is not None:
        vision_network, train_state = base.load_model(
            model_file=os.path.join(output_dir, model_file),
            model_step_file=os.path.join(output_dir, model_step_file),
            loss=triplet_loss)

        global_step, model_epochs, best_val_score = train_state
    else:
        vision_network = create_model(
            model_options, input_shape=[None, model_options["input_size"]])

        global_step = 0
        model_epochs = 0
        best_val_score = np.inf

    best_model = False

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

    # wrap few-shot model for background training
    vision_few_shot_model = vision_cnn.FewShotModel(
        vision_network, triplet_loss)

    # create training metrics
    loss_metric = tf.keras.metrics.Mean()

    # store model options
    file_io.write_json(
        os.path.join(output_dir, "model_options.json"), model_options)

    # train model
    for epoch in range(model_epochs, model_options["epochs"]):
        logging.log(logging.INFO, f"Epoch {epoch:03d}")

        loss_metric.reset_states()

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

        loss_metric.reset_states()

        for x_batch, y_batch in background_dev_ds:
            y_predict = vision_few_shot_model.predict(x_batch, training=False)
            loss_value = vision_few_shot_model.loss(y_batch, y_predict)

            loss_metric.update_state(loss_value)

        dev_loss = loss_metric.result().numpy()

        if dev_loss <= best_val_score:
            best_val_score = dev_loss
            best_model = True

        logging.log(
            logging.INFO,
            f"Train: Loss: {train_loss:.6f}")
        logging.log(
            logging.INFO,
            f"Validation: Loss: {dev_loss:.6f} {'*' if best_model else ''}")

        if tf_writer is not None:
            with tf_writer.as_default():
                tf.summary.scalar(
                    "Train step loss", train_loss, step=global_step)
                tf.summary.scalar(
                    "Validation loss", dev_loss, step=global_step)

        base.save_model(
            vision_few_shot_model.model, output_dir, epoch + 1, global_step,
            "loss", dev_loss, best_val_score, name="model")

        if best_model:
            best_model = False

            base.save_model(
                vision_few_shot_model.model, output_dir, epoch + 1, global_step,
                "loss", dev_loss, best_val_score, name="best_model")


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
        if FLAGS.target in ["test", "embed", "validate"]:
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

        elif FLAGS.target in ["test", "embed", "validate"]:
            raise ValueError(
                f"Target `{FLAGS.target}` specified but `{model_file}` not "
                f"found in {output_dir}.")

    model_options["embed_dir"] = FLAGS.embed_dir

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
    # elif FLAGS.target == "validate":
    #     validate(model_options, output_dir, model_file, model_step_file)
    # elif FLAGS.target == "embed":
    #     embed(model_options, output_dir, model_file, model_step_file)
    # else:
    #     test(model_options, output_dir, model_file, model_step_file)

    import pdb; pdb.set_trace()

if __name__ == "__main__":
    app.run(main)
