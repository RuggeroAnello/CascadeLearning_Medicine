# Personalized_ML

## Overview

This repository contains all code for the ADLM Project "Cascade Learning for Personalized Medicine: Enhancing Diagnostic Accuracy Through Population-Specific Models".

### Structure

This repository contains several branches:
- "main"                : Here all the code tested and working code of the development is merged into.
- "cluster_productive"  : This branch contains the code that is ready to be ran on the cluster. It is updated from the main branch.
- "development_-name-"  : The personal branches can be used to develop new features or test new code. They are based on the main branch and should be merged back into the main branch once the code is tested and working.

The folders in the repository are structured as follows:
- "data"                : Contains the different .csv-files for different datasets. Each dataset has its own subfolder, which contains the specific .csv-files for the different splits (e.g., training, validation, test for ap, pa, fronal, etc.). The subfolders are structured as follows:
  - **`original_data_unprocessed`**: Contains the original CSV files downloaded from the CheXpert dataset.
  - **`splitted`**: Contains a dataset that uses the validation set as test set. The training and validation set are created from the original training set using a 95/5 split.
  - **`new_90_5_5`**: Contains a dataset that uses the original training, validation and test set. All samples are used and split into 90/5/5.
  - **`90_5_5`**: Contains a dataset that uses the original training and validation set. All samples are used and split into 90/5/5.
  - **`original_data`**: Uses the original splits, splitted into the different views.
  - **`only_valid_samples`**: Contains a dataset that uses the original validation set. From the training set a subset is created as test set that contains no uncertainty labels for the 5 labels: Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion.
- "data_analysis"       : Contains the exploratory data analysis code and the notebook that was used to split the .csv-files
- final_models          : Contains the weights for the final models. Due to size restrictions only the weigths for the best models are stored here.
- logs                  : Contains the logs of the different runs. Each subfolder contains the logs for a family of models. Within these subfolders the weights and parameters of the models are stored.
- "model"               : Contains the code for all models.
  - "multi-stage_model": Contains the code for the multi-stage models.
  - "one_stage_model"  : Contains the code for the one-stage models.
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
3. Extract the dataset to `data`:  `unzip archive.zip`

### Python Setup

1. Create your own conda environment `conda create -n personalized_ml python=3.12`
2. Activate conda environment `conda activate personalized_ml`
3. Install pytorch `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`
4. Install all requirements.txt in the conda environment.

##  Code Style

Use the Black formatter (https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter) with default settings to keep the code consistent.

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