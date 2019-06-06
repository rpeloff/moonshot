#!/bin/bash

# set number of CPU cores to use during feature extraction
N_CPU_CORES=8

# source dataset paths (update paths.sh according to your storage locations)
source data_paths.sh

# check that docker script exists 
if ! [ -x "$(command -v run-docker-env)" ]; then 
    echo "Docker environment script 'run-docker-env' cannot be found. See installation instructions at https://github.com/rpeloff/research_images."
fi

# extract tidigits features in Kaldi Docker container
run-docker-env -c "./extract_features.sh" \
    --env N_CPU_CORES=${N_CPU_CORES} \
    --env FEATURES_DIR="/research" \
    --data-dir ${TIDIGITS_DIR} \
    --data-dir ${FLICKR_IMAGE_DIR} \
    --data-dir ${FLICKR_AUDIO_DIR} \
    --data-dir ${FLICKR_TEXT_DIR} \
    --image reloff/kaldi:5.4 \
    --sudo \
    --bash
