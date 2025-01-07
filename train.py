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

print("Started training script")

# Argument parser
import argparse
parser = argparse.ArgumentParser(description="Train a model with different configurations.")
parser.add_argument(
    "--model_type",
    type=str,
    choices=["one_stage_baseline", "two_stage_first", "two_stage_second_ap", "two_stage_second_pa"],
    required=True,
    help="Specify the type of model to train."
)
args = parser.parse_args()

print("Start importing libraries")
import os
from datetime import datetime

import torch
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from model.one_model.one_stage_models import ResNet50OneStage, ResNet18OneStage
from data.dataset import CheXpertDataset

import wandb

print("\nImported all libaries")

# Set variable based on argument
if args.model_type == "one_stage_baseline":
    model_type = "one_stage_baseline"
elif args.model_type == "two_stage_first":
    model_type = "two_stage_first"
elif args.model_type == "two_stage_second_ap":
    model_type = "two_stage_second_ap"
elif args.model_type == "two_stage_second_pa":
    model_type = "two_stage_second_pa"
else:
    raise ValueError("Invalid model type specified.")

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  # To prevent the kernel from dying.

# Define parameter dictionary
params_transform = {
    "resize": (256, 256),
    "degree_range": (-15, 15),
    "translate": (0.1, 0.2),
    "scale": (0.2, 1.0),
    "ratio": (0.75, 1.3333333333333333),
    "gaussian_blur_kernel": 3,
    "contrast": (0.75, 1.25),
    "saturation": (0.75, 1.25),
    "brightness": (0.75, 1.25),
}

# Set the parameters for the model training to be saved later in a log file
params = {
    "train_transfrom": params_transform,
    "lr": 1e-4,
    "save_epoch": 5,
    "batch_size": 128,
    "num_epochs": 100,
    "num_labels": 1,
    "input_channels": 1,
    "optimizer": "adam",
    # BCE with Sigmoid activation function
    "loss_fn": "torch.nn.BCEWithLogitsLoss()",
    # For multilabel: MultiLabelSoftMarginLoss
    "metrics": ["accuracy", "precision", "recall"],
    "confidence_threshold": 0.5,
}

transform = transforms.Compose(squeue 
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

if model_type == "one_stage_baseline":
    targets = {
    # "sex": 1,
    # "age": 2,
    # "frontal/lateral": 3,
    # "ap/pa": 4,
    # "no_finding": 5,
    # "enlarged_cardiomediastinum": 6,
    # "cardiomegaly": 7,
    # "lung_opacity": 8,
    # "lung_lesion": 9,
    # "edema": 10,
    # "consolidation": 11,
    # "pneumonia": 12,
    # "atelectasis": 13,
    # "pneumothorax": 14,
    # "pleural_effusion": 15,
    # "pleural_other": 16,
    # "fracture": 17,
    "support_devices": 18,
    # "ap/pa map": 22,
}
elif model_type == "two_stage_first":
    targets = {
    # "sex": 1,
    # "age": 2,
    # "frontal/lateral": 3,
    # "ap/pa": 4,
    # "no_finding": 5,
    # "enlarged_cardiomediastinum": 6,
    # "cardiomegaly": 7,
    # "lung_opacity": 8,
    # "lung_lesion": 9,
    # "edema": 10,
    # "consolidation": 11,
    # "pneumonia": 12,
    # "atelectasis": 13,
    # "pneumothorax": 14,
    # "pleural_effusion": 15,
    # "pleural_other": 16,
    # "fracture": 17,
    # "support_devices": 18,
    "ap/pa map": 22,
    }
elif model_type == "two_stage_second_ap":
    targets = {
    # "sex": 1,
    # "age": 2,
    # "frontal/lateral": 3,
    # "ap/pa": 4,
    # "no_finding": 5,
    # "enlarged_cardiomediastinum": 6,
    # "cardiomegaly": 7,
    # "lung_opacity": 8,
    # "lung_lesion": 9,
    # "edema": 10,
    # "consolidation": 11,
    # "pneumonia": 12,
    # "atelectasis": 13,
    # "pneumothorax": 14,
    # "pleural_effusion": 15,
    # "pleural_other": 16,
    # "fracture": 17,
    "support_devices": 18,
    # "ap/pa map": 22,
} 
elif model_type == "two_stage_second_pa":
    targets = {
    # "sex": 1,
    # "age": 2,
    # "frontal/lateral": 3,
    # "ap/pa": 4,
    # "no_finding": 5,
    # "enlarged_cardiomediastinum": 6,
    # "cardiomegaly": 7,
    # "lung_opacity": 8,
    # "lung_lesion": 9,
    # "edema": 10,
    # "consolidation": 11,
    # "pneumonia": 12,
    # "atelectasis": 13,
    # "pneumothorax": 14,
    # "pleural_effusion": 15,
    # "pleural_other": 16,
    # "fracture": 17,
    "support_devices": 18,
    # "ap/pa map": 22,
}
else:
    raise ValueError("Invalid model type specified.")

# 2: CHANGE HERE FOR DIFFERENT MODEL
if model_type == "one_stage_baseline":
    train_csv_file_path = "./data/splitted/train.csv"
    val_csv_file_path = "./data/splitted/valid.csv"
elif model_type == "two_stage_first":
    train_csv_file_path = "./data/splitted/train.csv"
    val_csv_file_path = "./data/splitted/valid.csv"
elif model_type == "two_stage_second_ap":
    train_csv_file_path = "data/splitted/ap_train.csv"
    val_csv_file_path = "data/splitted/ap_valid.csv"
elif model_type == "two_stage_second_pa":
    train_csv_file_path = "data/splitted/pa_train.csv"
    val_csv_file_path = "data/splitted/pa_valid.csv"
else:
    raise ValueError("Invalid model type specified.")

train_dataset = CheXpertDataset(
    csv_file=train_csv_file_path,
    root_dir="../image_data/",
    targets=targets,
    transform=transform,
)
val_dataset = CheXpertDataset(
    csv_file=val_csv_file_path,
    root_dir="../image_data/",
    targets=targets,
    transform=val_transform,
)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Valid dataset size: {len(val_dataset)}")

assert len(train_dataset.labels) == len(
    train_dataset
), "Mismatch between targets and dataset size!"

with wandb.init(project=model_type, config=params, dir='./logs/wandb'):
    # 3: CHANGE HERE FOR DIFFERENT MODEL
    if model_type == "one_stage_baseline":
        model = ResNet50OneStage(
            params=params,
            num_labels=params["num_labels"],
            input_channels=params["input_channels"],
        )
    elif model_type == "two_stage_first":
        model = ResNet18OneStage(
            params=params,
            num_labels=params["num_labels"],
            input_channels=params["input_channels"],
        )
    elif model_type == "two_stage_second_ap":
        model = ResNet18OneStage(
            params=params,
            num_labels=params["num_labels"],
            input_channels=params["input_channels"],
        )
    elif model_type == "two_stage_second_pa":
        model = ResNet18OneStage(
            params=params,
            num_labels=params["num_labels"],
            input_channels=params["input_channels"],
        )
    else:
        raise ValueError("Invalid model type specified.")
    
    wandb.watch(model, log="all")
    
    model.set_labels(train_dataset.labels)

    # TODO: Put set_labels as a parameter of the Model

    # 4: CHANGE HERE FOR DIFFERENT MODEL
    if model_type == "one_stage_baseline":
        task = "one_stage_pred_sup-dev"
    elif model_type == "two_stage_first":
        task = "first_stage_pred_ap-pa"
    elif model_type == "two_stage_second_ap":
        task = "second_stage_ap_pred_sup-dev"
    elif model_type == "two_stage_second_pa":
        task = "second_stage_pa_pred_sup-dev"
    else:
        raise ValueError("Invalid model type specified.")
    
    dirname = os.getcwd()
    path = os.path.join(dirname, "logs", f"{model.name}_{task}")
    if not os.path.exists(path):
        os.makedirs(path)
        num_of_runs = 0
    else:
        num_of_runs = len(os.listdir(path))
    path = os.path.join(
        path, f"run_{num_of_runs:03d}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    os.makedirs(path)

    # Create tensorboard logger
    tb_logger = SummaryWriter(path)

    # Save the model parameters
    model.save_hparams(path)

    # Train the model
    model.train(train_dataset, val_dataset, tb_logger, path)

    torch.save(model, os.path.join(path, "model.pth"))
