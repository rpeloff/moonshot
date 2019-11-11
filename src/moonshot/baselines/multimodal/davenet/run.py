"""Train deep audio-visual similarity network (DAVEnet) and test on Flickr one-shot multimodal task.

Author: Ryan Eloff
Contact: ryan.peter.eloff@gmail.com
Date: October 2019
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import datetime
import os


from absl import app
from absl import flags
from absl import logging


import numpy as np
import tensorflow as tf
from tqdm import tqdm


from moonshot.baselines import base
from moonshot.baselines import dataset
from moonshot.baselines import experiment
from moonshot.baselines import losses
from moonshot.baselines import model_utils

from moonshot.experiments.flickr_multimodal import flickr_multimodal

from moonshot.utils import file_io
from moonshot.utils import logging as logging_utils


FLAGS = flags.FLAGS


# model options (default if not loaded)
DEFAULT_OPTIONS = {
    # training data
    "one_shot_validation": True,
    # data pipeline
    "batch_size": 256,
    "num_batches": 2500,
    # audio-visual model (layers on top of base networks; final layers linear)
    "dense_units": [1024], #[2048, 2048, 2048, 1024],
    "dropout_rate": 0.2,
    # triplet imposter objective
    "blended": True,
    "margin": 1.,
    "metric": "cosine",  # or "euclidean", "cosine",
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
flags.DEFINE_integer("L", 10, "number of classes to sample in a task episode (L-way)")
flags.DEFINE_integer("K", 1, "number of task learning samples per class (K-shot)")
flags.DEFINE_integer("N", 15, "number of task evaluation samples")
flags.DEFINE_integer("k_neighbours", 1, "number of nearest neighbours to consider")
flags.DEFINE_string("metric", "cosine", "distance metric to use for nearest neighbours matching")
flags.DEFINE_integer("fine_tune_steps", None, "number of fine-tune gradient steps on one-shot data")
flags.DEFINE_float("fine_tune_lr", 1e-3, "learning rate for gradient descent fine-tune")
flags.DEFINE_enum("speaker_mode", "baseline", ["baseline", "difficult", "distractor"],
                  "type of speakers selected in a task episode")
flags.DEFINE_bool("direct_match", True, "directly match speech to images")
flags.DEFINE_bool("unseen_match_set", False, "match set contains classes unseen in K-shot learning")

# model train/test options
flags.DEFINE_string("vision_base_dir", None, "directory containing base vision network model")
flags.DEFINE_string("audio_base_dir", None, "directory containing base audio network model")
flags.DEFINE_bool("l2_norm", True, "L2-normalise embedding predictions (as done in training)")
flags.DEFINE_bool("load_best", False, "load previous best model for resumed training or testing")
flags.DEFINE_bool("mc_dropout", False, "make embedding predictions with MC Dropout")
# TODO optional train from scratch or fine-tune base model?
# flags.DEFINE_bool("use_embeddings", False, "train on extracted embeddings from base model"
#                   "(default loads base network ('<best_>model.h5') and fine-tunes siamese loss)")

# logging and target options
flags.DEFINE_enum("target", "train", ["train", "validate", "embed", "test"],
                  "train or load and test a model")
flags.DEFINE_string("output_dir", None, "directory where logs and checkpoints will be stored"
                    "(default is 'logs/<unique run id>')")
flags.DEFINE_bool("resume", True, "resume training if a checkpoint is found at output directory")
flags.DEFINE_bool("tensorboard", True, "log train and test summaries to TensorBoard")
flags.DEFINE_bool("debug", False, "log with debug verbosity level")


def get_training_objective(model_options):
    """Get training loss for ranking speech-image similarity with siamese triplets."""

    if "blended" in model_options and model_options["blended"]:
        triplet_loss = losses.blended_triplet_imposter_loss(
            margin=model_options["margin"], metric=model_options["metric"])
    else:
        triplet_loss = losses.triplet_imposter_random_sample_loss(
            margin=model_options["margin"], metric=model_options["metric"])

    return triplet_loss


def data_preprocess_func(embed_paths):
    """Data batch preprocessing function for input to the baseline model.

    Takes a batch of file paths, loads image data and preprocesses the images.
    """
    embed_ds = tf.data.TFRecordDataset(
        embed_paths, compression_type="ZLIB")
    # map sequential to prevent optimization overhead
    preprocess_func = lambda example: dataset.parse_embedding_protobuf(
        example)["embed"]
    embed_ds = embed_ds.map(preprocess_func, num_parallel_calls=8)

    return np.stack(list(embed_ds))


def create_speech_network(model_options, build_model=True):
    """Create spoken word network branch from model options."""

    # get input shape
    input_shape = [None]
    if build_model:
        input_shape = model_options["audio_input_shape"]

    # train dense layers on extracted base embeddings
    if build_model:
        logging.log(logging.INFO, "Training audio branch on base embeddings")

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

    # add top layer hidden units (final layer linear)
    if model_options["dense_units"] is not None:
        for dense_units in model_options["dense_units"][1:]:

            model_layers.append(tf.keras.layers.ReLU())

            if model_options["dropout_rate"] is not None:
                model_layers.append(
                    tf.keras.layers.Dropout(model_options["dropout_rate"]))

            model_layers.append(tf.keras.layers.Dense(dense_units))

    speech_network = tf.keras.Sequential(model_layers)

    if build_model:
        speech_network.summary()

    return speech_network


def create_vision_network(model_options, build_model=True):
    """Create vision network branch from model options."""

    # get input shape
    input_shape = [None]
    if build_model:
        input_shape = model_options["vision_input_shape"]

    # train dense layers on extracted base embeddings
    if build_model:
        logging.log(logging.INFO, "Training vision branch on base embeddings")

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


def create_embedding_model(base_network):
    """Create embedding model from vision or speech network branch."""

    # l2-normalise network output as done in training of siamese objective
    if FLAGS.l2_norm:
        l2_norm_layer = tf.keras.layers.Lambda(
            lambda x: tf.nn.l2_normalize(x, axis=1))

        embedding_network = tf.keras.Model(
            inputs=base_network.input,
            outputs=l2_norm_layer(base_network.output))
    else:
        embedding_network = base_network

    embedding_model = base.BaseModel(
        embedding_network, None, mc_dropout=FLAGS.mc_dropout)

    return embedding_model


def create_fine_tune_model(model_options, speech_network, vision_network):
    """Create siamese audio-visual similarity model for fine-tuning on unseen triplets."""
    # TODO
    # create clone of the vision network (so that it remains unchanged)
    speech_network_clone = model_utils.create_and_copy_model(
        speech_network, create_speech_network, model_options=model_options,
        build_model=True)  # TODO: figure out how to get this working without build (for MAML inner loop)

    vision_network_clone = model_utils.create_and_copy_model(
        vision_network, create_vision_network, model_options=model_options,
        build_model=True)  # TODO: figure out how to get this working without build (for MAML inner loop)
    
    # freeze all model layers for transfer learning (except final dense layer)
    freeze_index = -1

    for layer in vision_network_clone.layers[:freeze_index]:
        layer.trainable = False

    for layer in speech_network_clone.layers[:freeze_index]:
        layer.trainable = False

    # use same objective as during training for fine-tuning
    fine_tune_loss = get_training_objective(model_options)

    few_shot_model = base.WeaklySupervisedModel(
        speech_network_clone, vision_network_clone, fine_tune_loss,
        mc_dropout=FLAGS.mc_dropout)

    return few_shot_model


def train(model_options, output_dir, model_file=None, model_step_file=None,
          tf_writer=None):
    """Create and train audio-visual similarity model for one-shot learning."""

    # load embeddings from (linear) dense layer of base speech and vision models
    speech_embed_dir = os.path.join(
        model_options["audio_base_dir"], "embed", "dense")

    image_embed_dir = os.path.join(
        model_options["vision_base_dir"], "embed", "dense")

    # load training data (embed dir determines mfcc/fbank speech features)
    train_exp, dev_exp = dataset.create_flickr_multimodal_train_data(
        "mfcc", speech_embed_dir=speech_embed_dir,
        image_embed_dir=image_embed_dir, speaker_mode=FLAGS.speaker_mode,
        unseen_match_set=FLAGS.unseen_match_set)

    train_speech_paths = train_exp.speech_experiment.data
    train_image_paths = train_exp.vision_experiment.data

    dev_speech_paths = dev_exp.speech_experiment.data
    dev_image_paths = dev_exp.vision_experiment.data

    # define preprocessing for base model embeddings
    preprocess_data_func = lambda example: dataset.parse_embedding_protobuf(
        example)["embed"]

    # create standard training dataset pipeline
    background_train_ds = tf.data.Dataset.zip((
        tf.data.TFRecordDataset(
            train_speech_paths, compression_type="ZLIB"),
        tf.data.TFRecordDataset(
            train_image_paths, compression_type="ZLIB")))

    # map data preprocessing function across training data
    background_train_ds = background_train_ds.map(
        lambda speech_path, image_path: (
            preprocess_data_func(speech_path), preprocess_data_func(image_path)),
        num_parallel_calls=8)

    # shuffle and batch train data
    background_train_ds = background_train_ds.repeat(-1)

    background_train_ds = background_train_ds.shuffle(1000)

    background_train_ds = background_train_ds.batch(
        model_options["batch_size"])

    background_train_ds = background_train_ds.take(
        model_options["num_batches"])

    background_train_ds = background_train_ds.prefetch(
            tf.data.experimental.AUTOTUNE)

    # create dev set pipeline for validation
    background_dev_ds = tf.data.Dataset.zip((
        tf.data.TFRecordDataset(
            dev_speech_paths, compression_type="ZLIB"),
        tf.data.TFRecordDataset(
            dev_image_paths, compression_type="ZLIB")))

    background_dev_ds = background_dev_ds.map(
        lambda speech_path, image_path: (
            preprocess_data_func(speech_path), preprocess_data_func(image_path)),
        num_parallel_calls=8)

    background_dev_ds = background_dev_ds.batch(
        batch_size=model_options["batch_size"])

    # get training objective
    triplet_loss = get_training_objective(model_options)

    # get model input shape
    for speech_batch, image_batch in background_train_ds.take(1):
        model_options["audio_base_embed_size"] = int(
            tf.shape(speech_batch)[1].numpy())
        model_options["vision_base_embed_size"] = int(
            tf.shape(image_batch)[1].numpy())

        model_options["audio_input_shape"] = [
            model_options["audio_base_embed_size"]]
        model_options["vision_input_shape"] = [
            model_options["vision_base_embed_size"]]

    # load or create models
    if model_file is not None:
        join_network_model, train_state = model_utils.load_model(
            model_file=os.path.join(output_dir, model_file),
            model_step_file=os.path.join(output_dir, model_step_file),
            loss=get_training_objective(model_options))

        speech_network = tf.keras.Model(
            inputs=join_network_model.inputs[0],
            outputs=join_network_model.outputs[0])

        vision_network = tf.keras.Model(
            inputs=join_network_model.inputs[1],
            outputs=join_network_model.outputs[1])

        # get previous tracking variables
        initial_model = False
        global_step, model_epochs, _, best_val_score = train_state
    else:
        speech_network = create_speech_network(model_options)
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
        optimizer = join_network_model.optimizer
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # compile models to store optimizer with model when saving
    join_network_model = tf.keras.Model(
        inputs=[speech_network.input, vision_network.input],
        outputs=[speech_network.output, vision_network.output])

    # speech_network.compile(optimizer=optimizer, loss=triplet_loss)
    join_network_model.compile(optimizer=optimizer, loss=triplet_loss)

    # create few-shot model from speech network for background training
    multimodal_model = base.WeaklySupervisedModel(
        speech_network, vision_network, triplet_loss)

    # test model on one-shot validation task prior to training
    if model_options["one_shot_validation"]:

        one_shot_dev_exp = flickr_multimodal.FlickrMultimodal(
            features="mfcc", keywords_split="background_dev",
            flickr8k_image_dir=os.path.join("data", "external", "flickr8k_images"),
            speech_embed_dir=speech_embed_dir, image_embed_dir=image_embed_dir,
            speech_preprocess_func=data_preprocess_func,
            image_preprocess_func=data_preprocess_func,
            speaker_mode=FLAGS.speaker_mode,
            unseen_match_set=FLAGS.unseen_match_set)

        # create few-shot model from speech and vision networks for one-shot validation
        if FLAGS.fine_tune_steps is not None:
            test_few_shot_model = create_fine_tune_model(
                model_options, speech_network, vision_network)
        else:
            test_few_shot_model = base.WeaklySupervisedModel(
                speech_network, vision_network, None,
                mc_dropout=FLAGS.mc_dropout)

        val_task_accuracy, _, conf_interval_95 = experiment.test_multimodal_l_way_k_shot(
            one_shot_dev_exp, FLAGS.K, FLAGS.L, n=FLAGS.N,
            num_episodes=FLAGS.episodes, k_neighbours=FLAGS.k_neighbours,
            metric=FLAGS.metric, direct_match=FLAGS.direct_match,
            multimodal_model=test_few_shot_model,
            multimodal_embedding_func=None, #create_embedding_model,
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

    # train model
    for epoch in range(model_epochs, model_options["epochs"]):
        logging.log(logging.INFO, f"Epoch {epoch:03d}")

        loss_metric.reset_states()

        # train on epoch of training data
        step_pbar = tqdm(background_train_ds,
                         bar_format="{desc} [{elapsed},{rate_fmt}{postfix}]")
        for step, (speech_batch, image_batch) in enumerate(step_pbar):

            loss_value, y_speech, y_image = multimodal_model.train_step(
                speech_batch, image_batch, optimizer,
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

        # validate multimodal model
        loss_metric.reset_states()

        for speech_batch, image_batch in background_dev_ds:
            y_speech = multimodal_model.speech_model.predict(
                speech_batch, training=False)
            y_image = multimodal_model.vision_model.predict(
                image_batch, training=False)
            loss_value = multimodal_model.loss(y_speech, y_image)

            loss_metric.update_state(loss_value)

        dev_loss = loss_metric.result().numpy()

        # validate model on one-shot dev task if specified
        if model_options["one_shot_validation"]:

            if FLAGS.fine_tune_steps is not None:
                test_few_shot_model = create_fine_tune_model(
                    model_options, speech_network, vision_network)
            else:
                test_few_shot_model = base.WeaklySupervisedModel(
                    speech_network, vision_network, None,
                    mc_dropout=FLAGS.mc_dropout)

            val_task_accuracy, _, conf_interval_95 = experiment.test_multimodal_l_way_k_shot(
                one_shot_dev_exp, FLAGS.K, FLAGS.L, n=FLAGS.N,
                num_episodes=FLAGS.episodes, k_neighbours=FLAGS.k_neighbours,
                metric=FLAGS.metric, direct_match=FLAGS.direct_match,
                multimodal_model=test_few_shot_model,
                multimodal_embedding_func=None, #create_embedding_model,
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
        # model_utils.save_model(
        #     multimodal_model.model_a.model, output_dir, epoch + 1, global_step,
        #     val_metric, val_score, best_val_score, name="audio_model")
        # model_utils.save_model(
        #     multimodal_model.model_b.model, output_dir, epoch + 1, global_step,
        #     val_metric, val_score, best_val_score, name="vision_model")
        model_utils.save_model(
            join_network_model, output_dir, epoch + 1, global_step,
            val_metric, val_score, best_val_score, name="model")

        if best_model:
            best_model = False
            # model_utils.save_model(
            #     multimodal_model.model_a.model, output_dir, epoch + 1, global_step,
            #     val_metric, val_score, best_val_score, name="audio_best_model")
            # model_utils.save_model(
            #     multimodal_model.model_b.model, output_dir, epoch + 1, global_step,
            #     val_metric, val_score, best_val_score, name="vision_best_model")
            model_utils.save_model(
                join_network_model, output_dir, epoch + 1, global_step,
                val_metric, val_score, best_val_score, name="best_model")


    import pdb; pdb.set_trace()


def test(model_options, output_dir, model_file, model_step_file):
    """Load and test siamese audio-visual similarity model for one-shot learning."""

    # load embeddings from (linear) dense layer of base speech and vision models
    speech_embed_dir = os.path.join(
        model_options["audio_base_dir"], "embed", "dense")

    image_embed_dir = os.path.join(
        model_options["vision_base_dir"], "embed", "dense")

    # load Flickr Audio one-shot experiment
    one_shot_exp = flickr_multimodal.FlickrMultimodal(
        features="mfcc", keywords_split="one_shot_evaluation",
        flickr8k_image_dir=os.path.join("data", "external", "flickr8k_images"),
        speech_embed_dir=speech_embed_dir, image_embed_dir=image_embed_dir,
        speech_preprocess_func=data_preprocess_func,
        image_preprocess_func=data_preprocess_func,
        speaker_mode=FLAGS.speaker_mode,
        unseen_match_set=FLAGS.unseen_match_set)

    # load joint audio-visual model
    join_network_model, _ = model_utils.load_model(
        model_file=os.path.join(output_dir, model_file),
        model_step_file=os.path.join(output_dir, model_step_file),
        loss=get_training_objective(model_options))

    speech_network = tf.keras.Model(
        inputs=join_network_model.inputs[0],
        outputs=join_network_model.outputs[0])

    vision_network = tf.keras.Model(
        inputs=join_network_model.inputs[1],
        outputs=join_network_model.outputs[1])

    # create few-shot model from speech and vision networks for one-shot validation
    if FLAGS.fine_tune_steps is not None:
        test_few_shot_model = create_fine_tune_model(
            model_options, speech_network, vision_network)
    else:
        test_few_shot_model = base.WeaklySupervisedModel(
            speech_network, vision_network, None,
            mc_dropout=FLAGS.mc_dropout)

    logging.log(logging.INFO, "Created few-shot model from speech network")
    test_few_shot_model.speech_model.model.summary()

    logging.log(logging.INFO, "Created few-shot model from vision network")
    test_few_shot_model.vision_model.model.summary()

    # test model on L-way K-shot multimodal task
    task_accuracy, _, conf_interval_95 = experiment.test_multimodal_l_way_k_shot(
        one_shot_exp, FLAGS.K, FLAGS.L, n=FLAGS.N, num_episodes=FLAGS.episodes,
        k_neighbours=FLAGS.k_neighbours, metric=FLAGS.metric,
        direct_match=FLAGS.direct_match,
        multimodal_model=test_few_shot_model,
        multimodal_embedding_func=None, #create_embedding_model,
        fine_tune_steps=FLAGS.fine_tune_steps,
        fine_tune_lr=FLAGS.fine_tune_lr)

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
        model_options["audio_base_dir"] = FLAGS.audio_base_dir
        model_options["vision_base_dir"] = FLAGS.vision_base_dir

        if FLAGS.audio_base_dir is None:
            raise ValueError(
                f"Target `{FLAGS.target}` requires --audio_base_dir to be "
                "specified.")

        if FLAGS.vision_base_dir is None:
            raise ValueError(
                f"Target `{FLAGS.target}` requires --vision_base_dir to be "
                "specified.")

        model_options["base_model"] = (
            "best_model" if FLAGS.load_best else "model")

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
                f"Target `{FLAGS.target}` specified but "
                f"`<audio/vision>_{model_file}` not found in {output_dir}.")

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
        raise NotImplementedError
    else:
        test(model_options, output_dir, model_file, model_step_file)


if __name__ == "__main__":
    app.run(main)
