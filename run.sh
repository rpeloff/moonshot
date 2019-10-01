#!/bin/bash

# Setup and run the things.
#
# Author: Ryan Eloff
# Contact: ryan.peter.eloff@gmail.com
# Date: September 2019

# some usage notes ...
valid_targets="['extract', 'baselines', 'experiment', 'test' or 'test_debug']"
echo "usage: run.sh RUN_TARGET [options]"
echo ""
echo "** NOTE **"
echo "=========="
echo "(1) Update 'data_paths.sh' according to your data storage locations!"
echo "(2) First argument RUN_TARGET specifies the target application, one of ${valid_targets}."
echo "(3) Optional arguments for 'run-docker-env' may be specified after RUN_TARGET."
echo "    Some examples"
echo "      - experiment in your own research image: 'run.sh experiment --image your_image'."
echo "      - reuse Docker container state by setting container name: 'run.sh experiment --name clever_name'."
echo "      - set Jupyter password with argument '--jupyter-password your_password'."
echo "    See 'run-docker-env -h' for more information."
echo "(4) Set number of CPU cores to speed up speech feature extraction."
echo "    (e.g. 'run.sh extract --env N_CPU_CORES=8')"
echo ""

# source dataset paths (**NOTE TO USER** )
source data_paths.sh

# point external data directories to dataset paths
ln -snf /data/$(basename ${TIDIGITS_DIR}) data/external/tidigits
ln -snf /data/$(basename ${FLICKR_AUDIO_DIR}) data/external/flickr_audio
ln -snf /data/$(basename ${FLICKR_8K_IMAGE_DIR}) data/external/flickr8k_images
ln -snf /data/$(basename ${FLICKR_8K_TEXT_DIR}) data/external/flickr8k_text
ln -snf /data/$(basename ${FLICKR_30K_IMAGE_DIR}) data/external/flickr30k_images
ln -snf /data/$(basename ${FLICKR_30K_TEXT_DIR}) data/external/flickr30k_text
ln -snf /data/$(basename ${MSCOCO_DIR}) data/external/mscoco

# check that docker script exists 
if ! [ -x "$(command -v run-docker-env)" ]; then 
    echo "Docker environment script 'run-docker-env' cannot be found. See installation instructions at https://github.com/rpeloff/research_images."
    exit 1
fi

# get run target and make sure not empty
run_target="$1"
if [ -z "${run_target}" ]; then
    echo "Missing run target which must be specified with first arg as one of ${valid_targets}."
    exit 1
else
    echo "Run target: ${run_target}"
fi

# check if additional args for run-docker-env and print ...
args="${@:2}"
if ! [ -z "${args}" ]; then
    echo "Additional arguments for run-docker-env: ${args}"
fi

echo ""

if [ "${run_target}" == "extract" ]; then
    # extract speech features in Kaldi Docker container
    run-docker-env -c "make extraction && make clean && make features" \
        --env N_CPU_CORES=${N_CPU_CORES} \
        --data-dir ${TIDIGITS_DIR} \
        --data-dir ${FLICKR_AUDIO_DIR} \
        --data-dir ${FLICKR_8K_TEXT_DIR} \
        --image reloff/kaldi:5.4 \
        --sudo \
        ${args}
elif [ "${run_target}" == "baselines" ]; then
    # run baseline experiments in Docker container to reproduce results
    run-docker-env -c "make baselines && pip install jupyter jupyterlab" \
        --env FEATURES_DIR="/research" \
        --data-dir ${TIDIGITS_DIR} \
        --data-dir ${FLICKR_AUDIO_DIR} \
        --data-dir ${FLICKR_8K_IMAGE_DIR} \
        --data-dir ${FLICKR_8K_TEXT_DIR} \
        --data-dir ${FLICKR_30K_IMAGE_DIR} \
        --data-dir ${FLICKR_30K_TEXT_DIR} \
        --data-dir ${MSCOCO_DIR} \
        --image reloff/ml-research:tf-2.0.0-py36-cuda100 \
        --nvidia-gpu \
        --jupyter lab \
        --jupyter-port 8888 \
        --sudo \
        ${args}
elif [ "${run_target}" == "experiment" ]; then
    # run your own experiments with moonshot in a Docker container
    # (defaults to the baseline TensorFlow image; change with --image argument)
    run-docker-env -c "make moonshot && pip install jupyter jupyterlab" \
        --env FEATURES_DIR="/research" \
        --data-dir ${TIDIGITS_DIR} \
        --data-dir ${FLICKR_AUDIO_DIR} \
        --data-dir ${FLICKR_8K_IMAGE_DIR} \
        --data-dir ${FLICKR_8K_TEXT_DIR} \
        --data-dir ${FLICKR_30K_IMAGE_DIR} \
        --data-dir ${FLICKR_30K_TEXT_DIR} \
        --data-dir ${MSCOCO_DIR} \
        --image reloff/ml-research:tf-2.0.0-py36-cuda100 \
        --nvidia-gpu \
        --jupyter lab \
        --jupyter-port 8888 \
        --sudo \
        ${args}
elif [ "${run_target}" == "test" ] || [ "${run_target}" == "test_debug" ]; then
    # run tests (optionally in debug mode with pdb postmortem)
    run-docker-env -c "make moonshot && make ${run_target}" \
        --env FEATURES_DIR="/research" \
        --data-dir ${TIDIGITS_DIR} \
        --data-dir ${FLICKR_AUDIO_DIR} \
        --data-dir ${FLICKR_8K_IMAGE_DIR} \
        --data-dir ${FLICKR_8K_TEXT_DIR} \
        --data-dir ${FLICKR_30K_IMAGE_DIR} \
        --data-dir ${FLICKR_30K_TEXT_DIR} \
        --data-dir ${MSCOCO_DIR} \
        --image reloff/ml-research:tf-2.0.0-py36-cuda100 \
        --nvidia-gpu \
        --sudo \
        ${args}
else
    echo "Invalid run target: '${run_target}'. Must be one of ${valid_targets}"
fi
