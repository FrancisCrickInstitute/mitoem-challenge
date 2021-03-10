import argparse

import torch

from src.ml.model import HUNet
from src.ml.loss import dice_loss
from src.ml.train import train_model
from src.config import MITO_COMPETITION_HUMAN_TRAIN_STACKS,\
    MITO_COMPETITION_HUMAN_VALIDATION_STACKS, PATCH_SHAPE, MITO_LEARNING_RATE


def main():
    parser = argparse.ArgumentParser(description='Train a mitochondria segmentation model')
    parser.add_argument('images', type=str,
                        help='Folder location of the image stacks for training')
    parser.add_argument('labels', type=str,
                        help='Folder location of the label stacks for training')

    args = parser.parse_args()
    train_stacks_dir = args.images
    label_stacks_dir = args.labels

    print("Starting mitochondria (competition) training script...")
    print(f"Training stack folder: {train_stacks_dir}")
    print(f"Label stack folder: {label_stacks_dir}")
    print()

    model = HUNet(PATCH_SHAPE[0], 32)
    optimizer = torch.optim.Adam(model.parameters(), lr=MITO_LEARNING_RATE)

    train_model(model=model,
                model_name='mito_8nm_human_boundary',
                loss_func=dice_loss,
                optimizer=optimizer,
                image_in_folder=train_stacks_dir,
                label_in_folder=label_stacks_dir,
                train_stacks=MITO_COMPETITION_HUMAN_TRAIN_STACKS,
                validation_stacks=MITO_COMPETITION_HUMAN_VALIDATION_STACKS,
                patch_shape=PATCH_SHAPE,
                batch_size=12,
                epochs=1000,
                label_postfix='_boundary')


if __name__ == '__main__':
    main()

