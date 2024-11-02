# Personalized_ML

## Overview

This repository contains all code for the ADLM Project "Cascade Learning for Personalized Medicine: Enhancing Diagnostic Accuracy Through Population-Specific Models".

### Structure

This repository contains several branches:
- "main"                : Here all the code tested and working code of the development is merged into.
- "cluster_productive"  : This branch contains the code that is ready to be ran on the cluster. It is updated from the main branch.
- "development_-name-"  : The personal branches can be used as a sandbox to try some ideas. Should regularly be updated from the main branch.
- "development_-issue-" : In these branches predefined tasks defined in notion are worked on. The origin of these branches is the main branch. When the task is finised in the main branch is pushed again.

## Prerequistes

### Dataset Download

In order to work with the CheXpert small dataset is has to be downloaded first:
1. Go to `./data/`: `cd ./data/`
2. Download the CheXpert-v1.0-small dataset: `curl -L -o ./archive.zip https://www.kaggle.com/api/v1/datasets/download/willarevalo/chexpert-v10-small`
3. Extract the dataset to `data`:  `unzip archive.zip`

### Python Setup

1. Create your own conda environment `conda create -n personalized_ml python=3.12`
2. Activate conda environment `conda activate personalized_ml`
3. Install pytorch `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`
4. Install all requirements.txt in the conda environment.

##  Code Style

Use the Black formatter (https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter) with default settings to keep the code consistent.