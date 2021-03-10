#!/usr/bin/env bash

image_stacks_path="/em/data/stacks"
label_stacks_path="/em/data/stacks"

# make sure we are in project dir
cd "$(dirname "$0")"
cd ..
cd ..
cd ..
projectdir=$(pwd)

echo '> Initiating docker and running training pipeline.'
docker build -t pytorch-organelle-ml .
docker run --gpus all --entrypoint python3 -v "$projectdir/data:/em/data" pytorch-organelle-ml -u scripts/mito/competition/train_human_boundary.py "$image_stacks_path" "$label_stacks_path"

