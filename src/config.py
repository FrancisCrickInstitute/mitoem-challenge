import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')

MITO_COMPETITION_HUMAN_TRAIN_STACKS = [
    "mito_h_train_z000_x0000_y_0000",
    "mito_h_train_z000_x0000_y_2048",
    "mito_h_train_z000_x2048_y_0000",
    "mito_h_train_z000_x2048_y_2048",
    "mito_h_train_z100_x0000_y_0000",
    "mito_h_train_z100_x0000_y_2048",
    "mito_h_train_z100_x2048_y_0000",
    "mito_h_train_z100_x2048_y_2048",
    "mito_h_train_z200_x0000_y_0000",
    "mito_h_train_z200_x0000_y_2048",
    "mito_h_train_z200_x2048_y_0000",
    "mito_h_train_z200_x2048_y_2048",
    "mito_h_train_z300_x0000_y_0000",
    "mito_h_train_z300_x0000_y_2048",
    "mito_h_train_z300_x2048_y_0000",
    "mito_h_train_z300_x2048_y_2048",
]

MITO_COMPETITION_RAT_TRAIN_STACKS = [
    "mito_r_train_z000_x0000_y_0000",
    "mito_r_train_z000_x0000_y_2048",
    "mito_r_train_z000_x2048_y_0000",
    "mito_r_train_z000_x2048_y_2048",
    "mito_r_train_z100_x0000_y_0000",
    "mito_r_train_z100_x0000_y_2048",
    "mito_r_train_z100_x2048_y_0000",
    "mito_r_train_z100_x2048_y_2048",
    "mito_r_train_z200_x0000_y_0000",
    "mito_r_train_z200_x0000_y_2048",
    "mito_r_train_z200_x2048_y_0000",
    "mito_r_train_z200_x2048_y_2048",
    "mito_r_train_z300_x0000_y_0000",
    "mito_r_train_z300_x0000_y_2048",
    "mito_r_train_z300_x2048_y_0000",
    "mito_r_train_z300_x2048_y_2048",
]

MITO_COMPETITION_HUMAN_VALIDATION_STACKS = [
    "mito_h_val_x0000_y_0000",
    "mito_h_val_x0000_y_2048",
    "mito_h_val_x2048_y_0000",
    "mito_h_val_x2048_y_2048",
]

MITO_COMPETITION_RAT_VALIDATION_STACKS = [
    "mito_r_val_x0000_y_0000",
    "mito_r_val_x0000_y_2048",
    "mito_r_val_x2048_y_0000",
    "mito_r_val_x2048_y_2048",
]

PATCH_SHAPE = (12, 256, 256)

# 0.0005 is tuned well for the nuclear envelope problem
NE_LEARNING_RATE = 0.0005
# 0.0001 is working for mitochondria, but more tuning would be good
MITO_LEARNING_RATE = 0.0001


