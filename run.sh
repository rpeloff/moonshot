# run-docker-env -c "./exp_setup.sh" \
run-docker-env \
    --env N_CPU_CORES=${N_CPU_CORES} \
    --env FEATURES_DIR="/research" \
    --data-dir "speech_features" \
    --image reloff/ml-research:tf-2.0.0a0-py36-cuda100 \
    --jupyter lab \
    --jupyter-port 8888 \
    --sudo
