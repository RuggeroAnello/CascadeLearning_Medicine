# NOTE: the base structure of this script can be used to train:
# - the model of the one stage baseline
# - The first model of the two stage cascading model
# - The two models of the second stage of the cascading model

# Use the argument parser to specify the model type
# run "python train.py --help" to see the options

# It is indicated with "CHANGE HERE FOR DIFFERENT MODEL" in the
# code what can be changed:
# - Select target for the training
# - Specify model that is used
# - Select the train and valid set
# - Define the task that is trained

print("Started training script.")

from model.one_model.one_stage_models import ResNet50OneStage, ResNet18OneStage
from data.dataset import CheXpertDataset

import torch
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import wandb

from datetime import datetime
from pathlib import Path
import argparse
import json
import os

print("\nImported all libraries")

def preprocess_params(params_dict):
    for params_key, params_value in params_dict.items():
        if params_key == 'params_transform':
            for transform_key, transform_value in params_value.items():
                if type(transform_value) == list:
                    params_dict[params_key][transform_key] = tuple(transform_value)
        else:
            if type(params_value) == list:
                params_dict[params_key] = tuple(params_value)
    return params_dict

def read_json_config(file_path):
    with open(file_path, "r") as file:
        json_data = json.load(file)

    model_type = json_data['model_type']
    training_name = json_data['training_name']
    task = json_data['task']
    paths = json_data['paths']
    targets = json_data['targets']
    params = json_data['params']
    params = preprocess_params(params)
    params_transform = params['params_transform']
    
    return model_type, training_name, task, paths, targets, params, params_transform
# Argument parser
parser = argparse.ArgumentParser(description="Train a model with different configurations.")
parser.add_argument(
    "--model_type",
    type=str,
    choices=["one_stage_baseline", "two_stage_first", "two_stage_second_ap", "two_stage_second_pa"],
    required=True,
    help="Specify the type of model to train."
)
args = parser.parse_args()

json_config_path = Path.cwd().joinpath("configs", "config_one_stage_baseline.json")

if args.model_type == "one_stage_baseline":
    json_config_path = Path.cwd().joinpath("train_configs", "config_one_stage_baseline.json")
elif args.model_type == "two_stage_first":
    json_config_path = Path.cwd().joinpath("train_configs", "config_two_stage_first.json")
elif args.model_type == "two_stage_second_ap":
    json_config_path = Path.cwd().joinpath("train_configs", "config_two_stage_second_ap.json")
elif args.model_type == "two_stage_second_pa":
    json_config_path = Path.cwd().joinpath("train_configs", "config_two_stage_second_pa.json")
else:
    raise ValueError("Invalid model type specified.")

model_type, training_name, task, paths, targets, params, params_transform = read_json_config(json_config_path)

# To prevent the kernel from dying.
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

transform = transforms.Compose( 
    [
        transforms.Resize(params_transform["resize"]),
        transforms.ToTensor(),
        transforms.RandomRotation(params_transform["degree_range"]),
        transforms.RandomAffine(
            degrees=params_transform["degree_range"],
            translate=params_transform["translate"],
        ),
        transforms.RandomResizedCrop(
            size=params_transform["resize"],
            scale=params_transform["scale"],
            ratio=params_transform["ratio"],
        ),
        transforms.GaussianBlur(kernel_size=params_transform["gaussian_blur_kernel"]),
        transforms.ColorJitter(
            brightness=params_transform["brightness"],
            contrast=params_transform["contrast"],
            saturation=params_transform["saturation"],
        ),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize(params_transform["resize"]),
        transforms.ToTensor(),
    ]
)

train_dataset = CheXpertDataset(
    csv_file=paths["train_csv_file_path"],
    root_dir=paths["data_root_dir"],
    targets=targets,
    transform=transform,
    uncertainty_mapping=True
)
val_dataset = CheXpertDataset(
    csv_file=paths["val_csv_file_path"],
    root_dir=paths["data_root_dir"],
    targets=targets,
    transform=val_transform,
    uncertainty_mapping=True
)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Valid dataset size: {len(val_dataset)}")

assert len(train_dataset.labels) == len(
    train_dataset
), "Mismatch between targets and dataset size!"

with wandb.init(name=training_name, project=model_type, config=params, dir='./logs/wandb'):
    if model_type == "one_stage_baseline":
        model = ResNet50OneStage(
            params=params,
            targets = targets,
            input_channels=params["input_channels"],
        )
    elif model_type == "two_stage_first" or model_type == "two_stage_second_ap" or model_type == "two_stage_second_pa":
        model = ResNet18OneStage(
            params=params,
            targets = targets,
            input_channels=params["input_channels"],
        )
    else:
        raise ValueError("Invalid model type specified.")
    
    # wandb.watch(model, log="all")

    dirname = os.getcwd()
    path = os.path.join(dirname, "logs", f"{model.name}_{task}")
    if not os.path.exists(path):
        os.makedirs(path)
        num_of_runs = 0
    else:
        num_of_runs = len(os.listdir(path))
    path = os.path.join(
        path, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{model_type}_{training_name}"
    )
    os.makedirs(path)

    # Create tensorboard logger
    # tb_logger = SummaryWriter(path)
    tb_logger = None

    # Save the model parameters
    model.save_hparams(path)

    # Train the model
    model.train(train_dataset, val_dataset, tb_logger, path, log_wandb=True)

print("Finished training script.")