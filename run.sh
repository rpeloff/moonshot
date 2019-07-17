#!/bin/bash

# source dataset paths (update paths.sh according to your storage locations)
source speech_features/data_paths.sh

# check that docker script exists 
if ! [ -x "$(command -v run-docker-env)" ]; then 
    echo "Docker environment script 'run-docker-env' cannot be found. See installation instructions at https://github.com/rpeloff/research_images."
fi

# run-docker-env -c "./exp_setup.sh" \
run-docker-env \
    --env N_CPU_CORES=${N_CPU_CORES} \
    --env FEATURES_DIR="/research" \
    --data-dir ${FLICKR_IMAGE_DIR} \
    --data-dir ${FLICKR_TEXT_DIR} \
    --image reloff/ml-research:tf-2.0.0-beta1-py36-cuda100 \
    --nvidia-gpu \
    --jupyter lab \
    --jupyter-port 8888 \
    --sudo
