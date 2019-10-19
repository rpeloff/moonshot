"""TODO(rpeloff)

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: September 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import datetime
import functools


from absl import app
from absl import flags
from absl import logging


import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm


from moonshot.baselines.classifier import vision_cnn
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
    "dropout_rate": 0.5,
    # objective
    "loss": "cross_entropy",  # "focal"
    "cross_entropy_pos_weight": 3.0,
    "cross_entropy_label_smoothing": 0.0,
    "focal_alpha": 0.9,
    "focal_gamma": 1.0,
    # training
    "learning_rate": 1e-3,
    "decay_steps": 10000,
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
flags.DEFINE_enum("target", "train", ["train", "test"], "train or load and test a model")
flags.DEFINE_string("output_dir", None, "directory where logs and checkpoints will be stored"
                    "(defaults to logs/<unique run id>)")
flags.DEFINE_bool("load_best", False, "load previous best model for resumed training or testing")
flags.DEFINE_bool("resume", True, "resume training if a checkpoint is found at output directory")
flags.DEFINE_bool("tensorboard", True, "log train and test summaries to TensorBoard")
flags.DEFINE_bool("debug", False, "log with debug verbosity level")


def weighted_cross_entropy_with_logits(pos_weight=1., label_smoothing=0):
    """Cross entropy computed from unnormalised log probabilities (logits)."""
    def loss(y_true, y_pred):
        labels = tf.cast(y_true, tf.float32)
        logits = tf.cast(y_pred, tf.float32)

        if label_smoothing > 0:
            # label smoothing between binary classes (Szegedy et al. 2015)
            labels *= 1.0 - label_smoothing
            labels += 0.5 * label_smoothing

        return tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                labels=labels, logits=logits, pos_weight=pos_weight),
            axis=-1)

    return loss


def focal_loss_with_logits(alpha=0.25, gamma=2.0):
    """Modulated cross entropy for imbalanced classes (Lin et al., 2017)."""
    def loss(y_true, y_pred):
        labels = tf.cast(y_true, tf.float32)
        logits = tf.cast(y_pred, tf.float32)

        per_entry_cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits)

        prediction_probabilities = tf.sigmoid(logits)

        p_t = labels * prediction_probabilities
        p_t += (1 - labels) * (1 - prediction_probabilities)

        modulating_factor = 1.0
        if gamma is not None:
            modulating_factor = tf.pow(1.0 - p_t, gamma)

        alpha_weight_factor = 1.0
        if alpha is not None:
            alpha_weight_factor = labels*alpha + (1 - labels)*(1 - alpha)

        focal_cross_entropy_loss = (
            modulating_factor * alpha_weight_factor * per_entry_cross_ent)

        return tf.reduce_mean(focal_cross_entropy_loss, axis=-1)

    return loss


def create_background_dataset(
        image_paths, label_ids, batch_size=32,
        crop_size=224, augment_crop=False, num_augment=4, normalise=True,
        random_scales=None, horizontal_flip=True, colour=False,
        shuffle=False, shuffle_buffer=1000,
        prefetch_buffer=tf.data.experimental.AUTOTUNE,
        n_parallel_calls=tf.data.experimental.AUTOTUNE):

    preprocess_images_func = functools.partial(
        load_and_preprocess_image, crop_size=crop_size,
        augment_crop=augment_crop, normalise=normalise,
        random_scales=random_scales, horizontal_flip=horizontal_flip,
        colour=colour)

    background_image_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    background_image_ds = background_image_ds.map(
        preprocess_images_func, num_parallel_calls=n_parallel_calls)

    background_label_ds = tf.data.Dataset.from_tensor_slices(
        tf.cast(label_ids, tf.int64))

    background_ds = tf.data.Dataset.zip(
        (background_image_ds, background_label_ds))

    if augment_crop:
        background_ds = background_ds.repeat(num_augment)

    if shuffle:
        background_ds = background_ds.shuffle(shuffle_buffer)

    background_ds = background_ds.batch(batch_size)
    background_ds = background_ds.prefetch(prefetch_buffer)

    return background_ds


def augment_square_crop(image, size=224, random_scales=None,
                        horizontal_flip=True, colour=False):
    """Augment image (scale, flip, colour) and select random square crop."""
    # get shorter side of image
    image_shape = tf.shape(image)
    h, w = image_shape[0], image_shape[1]
    short_edge = tf.minimum(w, h)

    # scale augmentation
    if random_scales is None:
        # random resize along shorter edge in [256, 480]
        # maxval - minval = power of 2 => unbiased random integers
        rand_resize = tf.random.uniform(
            [1], minval=tf.maximum(256, size), maxval=(480+1),
            dtype=tf.int32)[0]
    else:
        # random resize along shorter edge in `random_scales` if specified
        rand_scale_idx = np.random.choice(np.arange(len(random_scales)), 1)[0]
        rand_resize = np.asarray(random_scales)[rand_scale_idx]

    resize_hw = (rand_resize * h/short_edge, rand_resize * w/short_edge)
    image = tf.image.resize(image, resize_hw, method="lanczos3")

    # horizontal flip augmentation
    if horizontal_flip:
        image = tf.image.random_flip_left_right(image)

    # colour augmentation (ordering of these ops matters so we shuffle them)
    if colour:
        color_ordering = tf.random.uniform([], maxval=1, dtype=tf.int32)
        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        else:
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)

    # crop augmentation, random sample square (size, size, 3) from resized image
    image = tf.image.random_crop(image, size=(size, size, 3))

    # make sure that we still have an image in range [0, 1]
    image = image - tf.reduce_min(image)
    image = image / tf.reduce_max(image)
    return image


def resize_square_crop(image, size=224):
    """Resize image along short edge and center square crop."""
    # get shorter side of image
    image_shape = tf.shape(image)
    h, w = image_shape[0], image_shape[1]
    short_edge = tf.minimum(w, h)

    # resize image
    resize_hw = (size * h/short_edge, size * w/short_edge)
    image = tf.image.resize(image, resize_hw, method="lanczos3")

    # center square crop
    image_shape = tf.shape(image)
    h, w = image_shape[0], image_shape[1]
    h_shift = tf.cast((h - size) / 2, tf.int32)
    w_shift = tf.cast((w - size) / 2, tf.int32)
    image = tf.image.crop_to_bounding_box(
        image, h_shift, w_shift, size, size)

    # make sure that we still have an image in range [0, 1]
    image = image - tf.reduce_min(image)
    image = image / tf.reduce_max(image)
    return image


def load_and_preprocess_image(image_path, crop_size=224, augment_crop=False,
                              normalise=True, random_scales=None,
                              horizontal_flip=True, colour=False):
    """Load image at path and square crop with optional augmentation."""
    # read and decode image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    # scale image to range [0, 1] expected by tf.image functions
    image = tf.cast(image, tf.float32) / 255.
    # random crop image for testing
    if augment_crop:
        image = augment_square_crop(
            image, size=crop_size, random_scales=random_scales,
            horizontal_flip=horizontal_flip, colour=colour)
    else:
        image = resize_square_crop(image, size=crop_size)
    # normalise image from [0, 1] to range [-1, 1]
    if normalise:
        image *= 2.
        image -= 1.
    return image


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
        multi_label_loss = weighted_cross_entropy_with_logits(
            pos_weight=model_options["cross_entropy_pos_weight"],
            label_smoothing=model_options["cross_entropy_label_smoothing"])
    elif model_options["loss"] == "focal":
        multi_label_loss = focal_loss_with_logits(
            alpha=model_options["focal_alpha"],
            gamma=model_options["focal_gamma"])
    else:
        raise ValueError(f"Invalid loss function: {model_options['loss']}")

    return multi_label_loss


def load_model(model_file, model_step_file, loss):
    logging.log(logging.INFO, f"Loading model: {model_file}")

    vision_classifier = tf.keras.models.load_model(
        model_file, custom_objects={"loss": loss})

    model_epochs, global_step, val_score, best_val_score = file_io.read_csv(
        model_step_file)[0]

    model_epochs = int(model_epochs)
    global_step = int(global_step)
    val_score = float(val_score)
    best_val_score = float(best_val_score)

    logging.log(
        logging.INFO,
        f"Model trained for {model_epochs} epochs ({global_step} steps)")
    logging.log(
        logging.INFO,
        f"Validation: current F-1: {val_score:.5f}, previous best F-1: "
        f"{best_val_score:.5f}")

    return vision_classifier, (global_step, model_epochs, best_val_score)


def create_model(model_options):
    global_step = 0
    model_epochs = 0
    best_val_score = -np.inf

    inception_base = inceptionv3.inceptionv3_base(
        input_shape=(
            model_options["crop_size"], model_options["crop_size"], 3),
        pretrained=model_options["pretrained"])

    # inceptionv3.freeze_base_model(inception_base, trainable=None)

    model_layers = [
        # small model debug
        # tf.keras.layers.Conv2D(128, (3, 3), activation="relu", input_shape=(224,224,3)),
        # tf.keras.layers.MaxPool2D((3, 3)),
        # tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        # tf.keras.layers.MaxPool2D((3, 3)),
        # tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        # vision_cnn.small_vision_cnn(input_shape=(224, 224, 3), batch_norm=True),
        inception_base,
        tf.keras.layers.GlobalAveragePooling2D(),
    ]

    if model_options["dropout_rate"] is not None:
        model_layers.append(
            tf.keras.layers.Dropout(model_options["dropout_rate"]))

    if model_options["dense_units"] is not None:
        for dense_units in model_options["dense_units"]:
            model_layers.append(
                tf.keras.layers.Dense(dense_units, activation="relu"))

            if model_options["dropout_rate"] is not None:
                model_layers.append(
                    tf.keras.layers.Dropout(model_options["dropout_rate"]))

    model_layers.append(tf.keras.layers.Dense(model_options["n_classes"]))

    vision_classifier = tf.keras.Sequential(model_layers)

    return vision_classifier, (global_step, model_epochs, best_val_score)


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

    background_train_ds = create_background_dataset(
        train_paths, train_labels_multi_hot,
        batch_size=model_options["batch_size"],
        crop_size=model_options["crop_size"],
        augment_crop=model_options["augment_train"],
        num_augment=model_options["num_augment"],
        random_scales=model_options["random_scales"],
        horizontal_flip=model_options["horizontal_flip"],
        colour=model_options["colour"], shuffle=True)

    background_dev_ds = create_background_dataset(
        dev_paths, dev_labels_multi_hot,
        batch_size=model_options["batch_size"],
        crop_size=model_options["crop_size"], augment_crop=False, shuffle=False)

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

        vision_classifier, train_state = load_model(
            model_file=os.path.join(output_dir, model_file),
            model_step_file=os.path.join(output_dir, model_step_file),
            loss=multi_label_loss)
    else:
        model_options["n_classes"] = len(keyword_classes)

        vision_classifier, train_state = create_model(model_options)

    global_step, model_epochs, best_val_score = train_state

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
                    if step == 0:
                        tf.summary.image("Example train images", (x_batch+1)/2,
                                         max_outputs=25, step=global_step)
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

        vision_few_shot_model.model.save(os.path.join(output_dir, "model.h5"))
        file_io.write_csv(
            os.path.join(output_dir, "model.step"),
            [epoch + 1, global_step, dev_f1, best_val_score])

        if best_model:
            best_model = False

            vision_few_shot_model.model.save(
                os.path.join(output_dir, "best_model.h5"))
            file_io.write_csv(
                os.path.join(output_dir, "best_model.step"),
                [epoch + 1, global_step, dev_f1, best_val_score])

def test(model_options, output_dir, model_file, model_step_file, tf_writer=None):

    # load and create datasets
    one_shot_exp = flickr_vision.FlickrVision(
        os.path.join("data", "external", "flickr8k_images"),
        keywords_split="one_shot_evaluation.csv",
        splits_dir=os.path.join("data", "splits", "flickr8k"))

    # load model
    vision_classifier, _ = load_model(
        model_file=os.path.join(output_dir, model_file),
        model_step_file=os.path.join(output_dir, model_step_file),
        loss=get_training_objective(model_options))

    import pdb; pdb.set_trace()

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
        if FLAGS.target == "test":
            raise ValueError(
                "Target `test` requires --output_dir to be specified.")

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

        elif FLAGS.target == "test":
            raise ValueError(f"Target `test` specified but `{model_file}` not "
                             f"found in {output_dir}.")

    logging_utils.absl_file_logger(output_dir)

    logging.log(logging.INFO, f"Model directory: {output_dir}")
    logging.log(logging.INFO, f"Model options: {model_options}")

    tf_writer = None
    if FLAGS.tensorboard:
        tf_writer = tf.summary.create_file_writer(output_dir)

    np.random.seed(model_options["seed"])
    tf.random.set_seed(model_options["seed"])

    if FLAGS.target == "train":
        if model_found and FLAGS.resume:
            train(model_options, output_dir, model_file=model_file,
                  model_step_file=model_step_file, tf_writer=tf_writer)
        else:
            train(model_options, output_dir, tf_writer=tf_writer)
    else:
        test(model_options, output_dir, model_file, model_step_file,
             tf_writer=tf_writer)

        # flickr_exp = flickr_vision.FlickrVision(
        #     os.path.join("data", "external", "flickr8k_images"))

        # total_correct = 0
        # total_tests = 0

        # for episode in range(FLAGS.episodes):

        #     total_tests += 1
        #     # ...

        # logging.log(
        #     logging.INFO,
        #     "{}-way {}-shot accuracy after {} episodes : {:.6f}".format(
        #         FLAGS.L, FLAGS.K, FLAGS.episodes, total_correct/total_tests))

    import pdb; pdb.set_trace()

if __name__ == "__main__":
    app.run(main)
