import os
import time
from typing import List, Tuple

import torch
import numpy as np
from tqdm import tqdm

from src.image import read_tiff
from src.ml.model import HUNet
from src.ml.loss import dice_loss
from src.config import DATA_DIR


def save_model_checkpoint_and_log(model,
                                  model_name,
                                  train_stack_name,
                                  val_stack_name,
                                  optimizer,
                                  epoch,
                                  training_loss,
                                  validation_loss):
    timestamp = int(time.time())
    lr = 0
    # method to get learning rate from pytorch optimizer
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        break
    model_save_dir = os.path.join(DATA_DIR, 'training')
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    model_checkpoint_filename = os.path.join(model_save_dir, f'{model_name}.{epoch}.pt')
    model_log_filename = os.path.join(model_save_dir, f'{model_name}.csv')
    torch.save(model.state_dict(), model_checkpoint_filename)
    with open(model_log_filename, 'a') as f:
        if os.path.getsize(model_log_filename) == 0:
            f.write('epoch,train stack name,training loss,val stack name,validation loss,learning rate,timestamp,stack name\n')
        f.write(f'{epoch},{train_stack_name},{training_loss:.4f},{val_stack_name},{validation_loss:.4f},{lr},{timestamp}\n')


def restore_latest_model_epoch(model, model_name) -> int:
    model_save_dir = os.path.join(DATA_DIR, 'training')
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    files = os.listdir(model_save_dir)
    latest_epoch = 0
    latest_model_filename = None
    for file in files:
        filename, ext = os.path.splitext(file)
        # check this is a pytorch parameter file
        if ext == '.pt':
            file_model_name, epoch = filename.rsplit('.', 1)
            # check this is the right model
            if model_name == file_model_name:
                epoch = int(epoch)
                if epoch > latest_epoch:
                    latest_epoch = epoch
                    latest_model_filename = file
    if latest_model_filename is not None:
        print(f'Reloading model epoch {latest_epoch} from {latest_model_filename}')
        model.load_state_dict(torch.load(os.path.join(model_save_dir, latest_model_filename)))
        model.eval()
    return latest_epoch


def normalize_batch(train_batch: np.ndarray) -> np.ndarray:
    """
    Patch wise normalization approach Harry settled on. Works well in practice.
    Also reasonably performant is centering and scaling the stacks based on
    standard 0-255 image intensity range.
    """
    mean = np.mean(train_batch, axis=(1, 2, 3), keepdims=True)
    std = np.std(train_batch, axis=(1, 2, 3), keepdims=True)
    return (train_batch - mean) / (std + 0.0001)


def normalize_batch_and_labels(train_batch: np.ndarray,
                               label_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    The label batch normalization is thresholding at 0.5 to get binary labels.
    """
    return normalize_batch(train_batch), label_batch > 0.5


def random_patches_from_stacks(image_stack: np.ndarray,
                               label_stack: np.ndarray,
                               patch_shape: Tuple[int, int, int],
                               batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng()
    # fill training and label patch arrays of shape: (batch_size, z, x, y)
    #       if multiple classes are present this becomes: (batch_size, n_classes, z, x, y)
    image_patches = np.zeros((batch_size, patch_shape[0], patch_shape[1], patch_shape[2]))
    n_classes = 1
    # more axes in label stack means there is a class axis
    if len(label_stack.shape) != len(image_stack.shape):
        n_classes = label_stack.shape[0]
    if n_classes > 1:
        label_patches = np.zeros((batch_size, n_classes, patch_shape[0], patch_shape[1], patch_shape[2]))
    else:
        label_patches = np.zeros((batch_size, patch_shape[0], patch_shape[1], patch_shape[2]))
    for i in range(batch_size):
        # get a random patch from the image/label stack with the correct dimensions for the model
        max_coords_for_patch_start = (
            image_stack.shape[0] - patch_shape[0],
            image_stack.shape[1] - patch_shape[1],
            image_stack.shape[2] - patch_shape[2],
        )
        start_coords = rng.integers((0, 0, 0), max_coords_for_patch_start, endpoint=True)
        image_patches[i, :, :, :] = image_stack[...,
                                                start_coords[0]:start_coords[0] + patch_shape[0],
                                                start_coords[1]:start_coords[1] + patch_shape[1],
                                                start_coords[2]:start_coords[2] + patch_shape[2]]
        label_patches[i, ...] = label_stack[...,
                                            start_coords[0]:start_coords[0] + patch_shape[0],
                                            start_coords[1]:start_coords[1] + patch_shape[1],
                                            start_coords[2]:start_coords[2] + patch_shape[2]]
    return image_patches, label_patches


def evaluate_validation_batches(model,
                                loss_func,
                                stack_name: str,
                                image_stack: np.ndarray,
                                label_stack: np.ndarray,
                                patch_shape: Tuple[int, int, int],
                                batch_size: int,
                                number_of_batches: int,
                                compute_device):
    n_classes = label_stack.shape[0] if len(label_stack.shape) != len(image_stack.shape) else 1
    running_loss = [0]
    if n_classes > 1:
        running_loss += [0]*n_classes
    # no_grad required or otherwise a memory leak ensues because pytorch accumulates gradients
    with torch.no_grad(), tqdm(total=number_of_batches, desc=f'\tValidation on {stack_name}') as progress:
        for i in range(1, number_of_batches+1):
            # get a batch of samples and labels for one training iteration
            x_train, y_label = random_patches_from_stacks(image_stack=image_stack,
                                                          label_stack=label_stack,
                                                          patch_shape=patch_shape,
                                                          batch_size=batch_size)
            # normalize batch
            x_train, y_label = normalize_batch_and_labels(x_train, y_label)
            # convert to pytorch tensors
            x_train = torch.from_numpy(x_train.astype(np.float32)).to(compute_device)
            y_label = torch.from_numpy(y_label.astype(np.float32)).to(compute_device)
            # run forward pass
            y_pred = model(x_train)
            loss = loss_func(y_pred, y_label)

            # progress update
            running_loss[0] += loss.item()
            pupdate = {'loss (all classes)': running_loss[0] / i}
            if n_classes > 1:
                for c in range(n_classes):
                    y_pred_class = y_pred[:, c, ...]
                    y_label_class = y_label[:, c, ...]
                    loss_for_class = loss_func(y_pred_class, y_label_class)
                    running_loss[c+1] += loss_for_class.item()
                    current_loss = running_loss[c+1]/i
                    if c == 0:
                        pupdate['loss (area)'] = current_loss
                    elif c == 1:
                        pupdate['loss (boundary)'] = current_loss
                    else:
                        pupdate[f'loss (class {c+1})'] = current_loss

            progress.set_postfix(pupdate)
            progress.update()
    return running_loss[0]/number_of_batches


def train_one_epoch(model,
                    loss_func,
                    optimizer,
                    stack_name: str,
                    image_stack: np.ndarray,
                    label_stack: np.ndarray,
                    patch_shape: Tuple[int, int, int],
                    batch_size: int,
                    steps: int,
                    epoch_no: int,
                    compute_device) -> float:
    cpu_time = 0
    gpu_time = 0
    running_loss = 0
    with tqdm(total=steps, desc=f'Epoch {epoch_no} on {stack_name}') as progress:
        for i in range(1, steps+1):
            loop_iter_start_time = time.time()
            # get a batch of samples and labels for one training iteration
            x_train, y_label = random_patches_from_stacks(image_stack=image_stack,
                                                          label_stack=label_stack,
                                                          patch_shape=patch_shape,
                                                          batch_size=batch_size)

            # data augmentation for batch (disabled to train in time for competition)
            # x_train, y_label = aug.data_augmentation_transform(x_train, y_label)

            # normalize batch
            x_train, y_label = normalize_batch_and_labels(x_train, y_label)

            # count moving tensors on to gpu as gpu time
            cpu_cycle_done_time = time.time()

            # convert to pytorch tensors
            x_train = torch.from_numpy(x_train.astype(np.float32)).to(compute_device)
            y_label = torch.from_numpy(y_label.astype(np.float32)).to(compute_device)

            # run forward pass
            optimizer.zero_grad()
            y_pred = model(x_train)
            loss = loss_func(y_pred, y_label)
            # run backward pass
            loss.backward()
            optimizer.step()

            if compute_device.type == 'cuda':
                cpu_time += cpu_cycle_done_time - loop_iter_start_time
                gpu_time += time.time() - cpu_cycle_done_time
            else:
                cpu_time += time.time() - loop_iter_start_time

            # progress update
            running_loss += loss.item()
            progress.set_postfix({
                'average loss': running_loss/i,
                'cpu time': f'{cpu_time:.2f}s',
                'gpu time': f'{gpu_time:.2f}s',
            })
            progress.update()
    return running_loss/steps


def train_model(model,
                model_name: str,
                loss_func,
                optimizer,
                image_in_folder: str,
                label_in_folder: str,
                train_stacks: List[str],
                validation_stacks: List[str],
                patch_shape: Tuple[int, int, int],
                batch_size: int = 12,
                epochs: int = 100,
                steps_per_epoch: int = 100,
                label_postfix=""):
    latest_epoch = restore_latest_model_epoch(model, model_name)
    if torch.cuda.is_available():
        compute_device = torch.device('cuda')
        model.cuda()
    else:
        compute_device = torch.device('cpu')
    n_train_stacks = len(train_stacks)
    n_validation_stacks = len(validation_stacks)
    for i in range(latest_epoch+1, latest_epoch+epochs+1):
        # iterate available stacks so that they are used in equal amounts
        target_stack_idx = i % n_train_stacks
        train_stack_name = train_stacks[target_stack_idx]
        # load a stack in to memory for this epoch
        image_stack = read_tiff(image_in_folder, train_stack_name)
        label_stack = read_tiff(label_in_folder, train_stack_name+label_postfix)
        # train the model for one epoch on the stack we loaded
        training_loss = train_one_epoch(model=model,
                                        loss_func=loss_func,
                                        optimizer=optimizer,
                                        stack_name=train_stack_name,
                                        image_stack=image_stack,
                                        label_stack=label_stack,
                                        patch_shape=patch_shape,
                                        batch_size=batch_size,
                                        steps=steps_per_epoch,
                                        epoch_no=i,
                                        compute_device=compute_device)
        # run test on validation stack
        target_stack_idx = i % n_validation_stacks
        val_stack_name = validation_stacks[target_stack_idx]
        image_stack = read_tiff(image_in_folder, val_stack_name)
        label_stack = read_tiff(label_in_folder, val_stack_name+label_postfix)
        validation_loss = evaluate_validation_batches(model=model,
                                                      loss_func=loss_func,
                                                      stack_name=val_stack_name,
                                                      image_stack=image_stack,
                                                      label_stack=label_stack,
                                                      patch_shape=patch_shape,
                                                      batch_size=batch_size,
                                                      number_of_batches=20,
                                                      compute_device=compute_device)
        # save model checkpoint and log
        save_model_checkpoint_and_log(model, model_name, train_stack_name, val_stack_name, optimizer, i, training_loss, validation_loss)


