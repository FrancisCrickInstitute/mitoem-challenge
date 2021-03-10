import argparse


from src.image import read_tiff
from src.ml.model import HUNet
from src.ml.train import restore_latest_model_epoch
from src.ml.predict import segment_save_and_print_metrics
from src.config import PATCH_SHAPE


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('images', type=str)
    parser.add_argument('labels', type=str)
    parser.add_argument('save_folder', type=str)

    args = parser.parse_args()
    image_stacks_folder = args.images
    label_stacks_folder = args.labels
    save_folder = args.save_folder

    for mito_test_stack in ["mito_r_test"]:
        print(f"loading {mito_test_stack} for prediction")
        image_stack = read_tiff(image_stacks_folder, mito_test_stack)

        print("running boundary predictions")
        model = HUNet(PATCH_SHAPE[0], 32)
        restore_latest_model_epoch(model, "mito_8nm_rat_boundary")

        try:
            label_stack = read_tiff(label_stacks_folder, mito_test_stack+'_boundary')
        except FileNotFoundError:
            label_stack = None

        segment_save_and_print_metrics(model=model,
                                       stack_name=mito_test_stack,
                                       save_folder=save_folder,
                                       image_stack=image_stack,
                                       benchmark_segmentations=label_stack,
                                       save_postfix="_boundary")

        print("running area predictions")

        model = HUNet(PATCH_SHAPE[0], 32)
        restore_latest_model_epoch(model, "mito_8nm_rat_area")

        try:
            label_stack = read_tiff(label_stacks_folder, mito_test_stack+'_area')
        except FileNotFoundError:
            label_stack = None

        segment_save_and_print_metrics(model=model,
                                       stack_name=mito_test_stack,
                                       save_folder=save_folder,
                                       image_stack=image_stack,
                                       benchmark_segmentations=label_stack,
                                       save_postfix="_area")


if __name__ == '__main__':
    main()



