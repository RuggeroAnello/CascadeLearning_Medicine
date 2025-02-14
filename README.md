# Personalized_ML

## Overview

This repository contains all code for the ADLM Project "Cascade Learning for Personalized Medicine".

### Structure

This repository contains several branches:
- "main"                : Here all the code tested and working code of the development is merged into.
- "cluster_productive"  : This branch contains the code that is ready to be ran on the cluster. It is updated from the main branch.
- "development_-name-"  : The personal branches can be used to develop new features or test new code. They are based on the main branch and should be merged back into the main branch once the code is tested and working.

The folders in the repository are structured as follows:
- "data"                : Contains the different .csv-files for different datasets and the `dataset.py` file that contains the `CheXpertDataset` classs. Each dataset has its own subfolder, which contains the specific .csv-files for the different splits (e.g., training, validation, test for ap, pa, fronal, etc.). The subfolders are structured as follows:
  - **`original_data_unprocessed`**: Contains the original CSV files downloaded from the CheXpert dataset.
  - **`splitted`**: Contains a dataset that uses the validation set as test set. The training and validation set are created from the original training set using a 95/5 split.
  - **`new_90_5_5`**: Contains a dataset that uses the original training, validation and test set. All samples are used and split into 90/5/5.
  - **`90_5_5`**: Contains a dataset that uses the original training and validation set. All samples are used and split into 90/5/5.
    - This dataset additionally contains a subset of the test set where the exact same amount of AP, PA, and Lateral images are used.
    - `fr_lat_test_balanced.csv`: Contains the balanced test set with the same anount of frontal and lateral images for fr/lat split.
    - `fr_test_balanced.csv`: Contains the balanced test set with the same anount of ap and pa images for ap/pa split.
    - `test_balanced.csv`: Contains the balanced test set with the same anount of ap, pa, and lateral images for the baseline and three-stage model.
  - **`original_data`**: Uses the original splits, split into the different views.
  - **`only_valid_samples`**: Contains a dataset that uses the original validation set. From the training set a subset is created as test set that contains no uncertainty labels for the 5 labels: Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion.
- "data_analysis"       : Contains the exploratory data analysis code and the notebook that was used to split the .csv-files
- final_models          : Contains the weights for the final models. Due to size restrictions only the weigths for the best models are stored here.
- logs                  : Contains the logs of the different runs. Each subfolder contains the logs for a family of models. Within these subfolders the weights and parameters of the models are stored.
- "model"               : Contains the code for all models.
  - "multi-stage_model": Contains the code for the multi-stage models.
  - "one_stage_model"  : Contains the code for the one-stage models.
- "results"             : Contains the results of the different runs evaluated using the test set.
- "sbatch_scripts"      : Contains the sbatch scripts that are used to run the code on the cluster. Before running a scipt it has to be moved to the root directory of the repository.
- "training_configs"    : Contains the different training configurations for the models as defined in the sbatch scripts.

### Training and Testing

For training and testing, the following files are used:

- `test.ipynb` : This notebook is used to test the models. It loads the weights of the models and the test data and calculates the metrics for the models.
- `train.ipynb` : This notebook is used for experimenting with the training of the models.
- `train.py` : This script is used to train the models on the cluster. It uses the training configurations defined in the `training_configs` folder. And is started using the sbatch scripts from the `sbatch_scripts` folder.

## Prerequistes

### Dataset Download

In order to work with the CheXpert small dataset is has to be downloaded first:
1. Go to `../image_data/` relative to this repository: `cd ../image_data/`. If it not exists create it: `mkdir image_data`
2. Download the CheXpert-v1.0-small dataset: `curl -L -o ./archive.zip https://www.kaggle.com/api/v1/datasets/download/willarevalo/chexpert-v10-small`
3. Extract the dataset to `data`: `unzip archive.zip`

### Python Setup

1. Create your own conda environment `conda create -n personalized_ml python=3.12`
2. Activate conda environment `conda activate personalized_ml`
3. Install pytorch `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`
4. Install all requirements.txt in the conda environment.

##  Code Style

Use the RUFF formatter (https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) with default settings to keep the code consistent.

## Run jobs with SLURM

Execute a sbatch script:
```sh
sbatch your_script_name.sh config train_configs/config.json
```

Show current jobs running
```sh
squeue
```

Show jobs for current user
```sh
squeue -u $USER
```

# Training Parameters

The training parameters are defined in the `.json` files in the `training_configs` folder. The parameters are structured as follows:

## General Parameters

- model_type: The type of the model that is used. The following models are available:
  - "one_stage_baseline": The baseline model with one stage.
  - "two_stage_first": The first stage of the two-stage model.
  - "two_stage_second_ap": The second stage of the two-stage model classifying AP images.
  - "two_stage_second_pa": The second stage of the two-stage model classifying PA images.
  - "two_stage_second_frontal": The second stage of the two-stage model classifying Frontal images.
  - "two_stage_second_lateral": The second stage of the two-stage model classifying Lateral images.

## Paths 

- data_root_dir: The root directory where the images are stored.
- train_csv_file_path: The path to the training .csv file.
- val_csv_file_path: The path to the validation .csv file.

## Targets

The labels that are used for training. And the corresponding index corresponding to the column in the .csv file. The following labels are available:
  - "no_finding": 5
  - "enlarged_cardiomediastinum": 6
  - "cardiomegaly": 7
  - "lung_opacity": 8
  - "lung_lesion": 9
  - "edema": 10
  - "consolidation": 11
  - "pneumonia": 12
  - "atelectasis": 13
  - "pneumothorax": 14
  - "pleural_effusion": 15
  - "pleural_other": 16
  - "fracture": 17
  - "support devices": 18

## Weights

If provided the defined weights are used in the model. If a pretrained model is used define the path to the weights here.

## Training Parameters

The training parameters have the following options:
- "params_transform": The parameters for the transformation of the images. The following parameters are available:
  - "resize": The size of the images after resizing. Default is `256`.
  - "degree_range": The range of the rotation of the images. Default is `-15, 15`.
  - "translate": The range of the translation of the images. Default is `0.1, 0.2`.
  - "scale": The range of the scaling of the images. Default is `0.2, 1.0`.
  - "ratio": The range of the aspect ratio of the images. Default is `0.75, 1.33`.
  - "gaussian_blut_kernel": The kernel size of the gaussian blur. Default is `3`.
  - "contrast": The range of the contrast of the images. Default is `0.75, 1.25`.
  - "saturation": The range of the saturation of the images. Default is `0.75, 1.25`.
  - "brightness": The range of the brightness of the images. Default is `0.75, 1.25`.
- "lr": The learning rate of the model. Default is `0.00005`.
- "lr_decay": The learning rate decay of the model. Default is `0.9`.
- "lr_decay_step": The learning rate decay step of the model. Default is `2`.
- "save_epoch": Every `n`epochs the model is saved. Default is `1`.
- "batch_size": The batch size of the model. Default is `256`.
- "num_epochs": The number of epochs the model is trained. Default is `100`.
- "use_weighted_sampler": If `True` a weighted sampler is used. Default is `False`.
- "label_smoothing": The label smoothing factor. Default is `0.2`.
- "input_channels": The number of input channels. Default is `1`.
- "optimizer": The optimizer that is used. Default is `adam`.
- "num_workers": The number of workers that are used for the dataloader. Default is `22`.
- "loss_fn": The loss function that is used. Options:
  - "cross_entropy": The cross entropy loss.
  - "mse_loss": The mean squared error loss.
  - "BCEWithLogitsLoss": The binary cross entropy loss.
  - "weighted_bce_loss": The weighted binary cross entropy loss. If this loss is used additional parameters have to be defined:
    - "pos_weights": The positive weights for the focal loss. Used for in-class weighting.
    - "pos_weights_train": The positive weights for the training set. Used for in-class weighting.
    - "pos_weights_val": The positive weights for the validation set. Used for in-class weighting.
  - "multilabel_focal_loss": The multilabel focal loss. If this loss is used additional parameters have to be defined:
    - "alpha": The alpha parameter of the focal loss. Default is `0.25`.
    - "gamma": The gamma parameter of the focal loss. Default is `2.0`.
    - "reduction": The reduction parameter of the focal loss. Default is `mean`.
    - "pos_weights": The positive weights for the focal loss. Used for in-class weighting.
    - "class_weights": The class weights for the focal loss. Used for inter-class weighting.
    - "pos_weights_train": The positive weights for the training set. Used for in-class weighting.
    - "pos_weights_val": The positive weights for the validation set. Used for in-class weighting.
    - "class_weights_train": The class weights for the focal loss. Used for inter-class weighting.
    - "class_weights_val": The class weights for the focal loss. Used for inter-class weighting.
  - "metrics": The metrics that are used for the evaluation. Available:
    - "accuracy",
    - "precision",
    - "recall",
    - "confusion_matrix",
    - "auc",
    - "auroc",
    - "multilabel_accuracy",
    - "multilabel_auprc",
    - "multilabel_precision_recall_curve",
    - "mcc"
  - "confidence_threshold": The confidence threshold for the evaluation. Default is `0.5`.

## Multi-Stage Models

The multi-stage models have additional parameters that can be defined:

- AP/PA Split
  - `"confidence_threshold_first_ap_pa"`: The confidence threshold for the first stage of the AP/PA split. Default is `0.5`.
- Frontal/Lateral Split
  - `"confidence_threshold_first_frontal_lateral"`: The confidence threshold for the first stage of the Frontal/Lateral split. Default is `0.5`.
- Three-Stage Model