#!/usr/bin/env bash

image_stacks_path="/em/data/stacks"
label_stacks_path="/em/data/stacks"
save_folder="/em/data/predictions/rat"

# make sure we are in project dir
cd "$(dirname "$0")"
cd ..
cd ..
cd ..
projectdir=$(pwd)

echo '> Initiating docker and running training pipeline.'
docker build -t pytorch-organelle-ml .
docker run --gpus all --entrypoint python3 -v "$projectdir/data:/em/data" pytorch-organelle-ml -u scripts/mito/competition/predict_rat.py "$image_stacks_path" "$label_stacks_path" $save_folder

