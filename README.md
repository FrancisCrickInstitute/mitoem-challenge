### Reproducing results

#### Training
1. Open `scripts/mito/competition/make_train_val_stacks.py`
   and modify the paths at the top of the file to point to
   the location of the competition data for human/rat as
   appropriate.
2. Run `scripts/mito/competition/make_train_val_stacks.py`
   which will by default output training and validation
   stacks to `data/stacks/`.
3. Run the Python training file in `scripts/mito/competition/`
   for the model you want to train, passing in the location of
   the training data created in step 2. Shell scripts are
   also available for environments that support building with
   docker. For improved weight initialization/results before
   training a model, copy the previous model parameters (in the
   order described in accompanying report) to `data/training/{model_name}.1.pt`
   Model name can be seen inside the Python file.
4. Terminate training after convergence and a strong epoch
   as described in the accompanying report. Logs are written
   to `data/training/`.

#### Prediction
1. Stack test data in to a `.tiff` file. Either name the file
   `mito_h_test.tiff`/`mito_r_test.tiff`, or open the associated
   Python prediction script in `scripts/mito/competition/` and
   modify the filenames passed to the for loop for prediction
   (no extension).
2. Run the script in `scripts/mito/competition/` for the model you
   want to predict with, passing in the folder location of the image
   stacks from step 1.
3. For better performance, before passing to the instance segmentation
   evaluator, create a combined boundary from predictions over
   three different orientations, as described in the accompanying
   report.
4. Run watershed postprocessing with `scripts/mito/competition/postprocess.py`.




