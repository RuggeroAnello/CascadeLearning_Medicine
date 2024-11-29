# NOTE: the base structure of this script can be used to train:
# - the model of the one stage baseline
# - The first model of the two stage cascading model
# - The two models of the second stage of the cascading model

# It is indicated with "CHANGE HERE FOR DIFFERENT MODEL" in the
# code what needs to be changes. It is the following:
# - Select target for the training
# - Specify model that is used
# - Select the train and valid set
# - Define the task that is trained

import torch
import os
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from model.one_model.one_stage_models import ResNet50OneStage
from data.dataset import CheXpertDataset


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
# _________________________________________________________________
# 1: CHANGE HERE FOR DIFFERENT MODEL
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
# _________________________________________________________________

# _________________________________________________________________	
# 2: CHANGE HERE FOR DIFFERENT MODEL
train_dataset = CheXpertDataset(
    csv_file="./data/train_new.csv",
    root_dir="../image_data/",
    targets=targets,
    transform=transform,
)
val_dataset = CheXpertDataset(
    csv_file="./data/valid.csv",
    root_dir="../image_data/",
    targets=targets,
    transform=val_transform,
)
# _________________________________________________________________

print(f"Train dataset size: {len(train_dataset)}")
print(f"Valid dataset size: {len(val_dataset)}")


assert len(train_dataset.labels) == len(
    train_dataset
), "Mismatch between targets and dataset size!"

# Set the parameters for the model training to be saved later in a log file
params = {
    "train_transfrom": params_transform,
    "lr": 0.001,
    "save_epoch": 5,
    "batch_size": 32,
    "num_epochs": 100,
    "num_labels": 1,
    "input_channels": 1,
    "optimizer": "adam",
    # BCE with Sigmoid activation function
    "loss_fn": "torch.nn.BCEWithLogitsLoss()",
    # For multilabel: MultiLabelSoftMarginLoss
    "metrics": ["accuracy", "f1_score", "precision", "recall", "confusion_matrix"],
    "confidence_threshold": 0.5,
}

# _________________________________________________________________
# 3: CHANGE HERE FOR DIFFERENT MODEL
model = ResNet50OneStage(
    params=params,
    num_labels=params["num_labels"],
    input_channels=params["input_channels"],
)
# _________________________________________________________________
model.set_labels(train_dataset.labels)

# TODO: Put set_labels as a parameter of the Model

# _________________________________________________________________
# 4: CHANGE HERE FOR DIFFERENT MODEL
# Train the model
task = "one_stage_pred_sup-dev"
# _________________________________________________________________

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
