import torch
import torchvision
import os
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

class AbstractOneStageModel(torch.nn.Module):

    def __init__(
        self,
        params: dict,
        **kwargs,
    ):
        """
        Initialize the model with the given hyperparameters.

        Args:
            params (dict): Dictionary containing the hyperparameters.
            loss_fn (torch.nn.Module): Loss function to use for training
            optimizer (torch.optim.Optimizer): Optimizer to use for training
        """
        super().__init__()
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set hyperparameters
        self.lr = params.get("lr", 1e-3)
        self.batch_size = params.get("batch_size", 32)
        self.num_epochs = params.get("num_epochs", 10)
        # TODO add more hyperparameters if needed

    def forward(self, x):
        raise NotImplementedError

    def get_sample_weights(self, batch, class_weights):
        _, targets = batch 
        valid_targets = targets[targets != -1.]  

        # Convert target to integer before indexing class_weights
        sample_weights = [class_weights[int(target.item())] if int(target.item()) < len(class_weights) else 0 for target in valid_targets]

        return torch.tensor(sample_weights)


    def save_model(self, path: str):
        """
        Save the model and its weights to the given path.

        Args:
            path (str): Path where the model should be saved
        """
        model = self.model.to("cpu")
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(model.state_dict(), path)

    def _general_step(self, batch, loss_fn=F.cross_entropy, class_weights=None):
        images, targets = batch
        valid_mask = targets != -1.
        """
         !! Excluding -1 values (for testing purposes), implement a way to handle that else we lose much data !!
        """
        valid_images, valid_targets = images[valid_mask].to(self.device), targets[valid_mask].to(self.device)

        out = self.forward(valid_images)

        if class_weights is not None:
            loss = loss_fn(out, valid_targets, weight=class_weights)
        else:
            loss = loss_fn(out, valid_targets)

        preds = out.argmax(dim=1)
        n_correct = (preds == valid_targets).sum().item()
        return loss, n_correct

    def _general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + "_loss"] for x in outputs]).mean()
        total_correct = (
            torch.stack([x[mode + "_n_correct"] for x in outputs]).sum().cpu().numpy()
        )
        acc = total_correct / len(self.dataset[mode])
        return avg_loss, acc

    def _configure_optimizer(self, learning_rate=1e-3):
        optim = torch.optim.Adam(self.parameters(), learning_rate)
        return optim

    def train(self, train_dataset, val_dataset, loss_fn, tb_logger, epochs=None, weighted_sampling=False):
        epochs = epochs or self.num_epochs

        # Compute class weights
        class_counts = {}
        for _, targets in train_dataset:
            valid_targets = targets[targets != -1.]
            for target in valid_targets:
                class_counts[target.item()] = class_counts.get(target.item(), 0) + 1

        total_samples = sum(class_counts.values())
        class_weights = {label: total_samples / count for label, count in class_counts.items()}
        class_weights = torch.tensor(list(class_weights.values())).float().to(self.device)

        # DataLoaders with or without Weighted Sampling
        if weighted_sampling:
            sample_weights = []
            for batch in train_dataset:
                batch_weights = self.get_sample_weights(batch, class_weights)
                sample_weights.extend(batch_weights)

            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(train_dataset),
                replacement=True
            )
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler)
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        optimizer = self._configure_optimizer()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(train_loader) // 5, gamma=0.7)

        # Training and validation loop
        self.model = self.model.to(self.device)

        for epoch in range(epochs):
            # Training loop
            training_loss = 0
            training_loop = tqdm(
                enumerate(train_loader),
                desc=f"Training Epoch {epoch + 1}/{epochs}",
                total=len(train_loader),
                ncols=200
            )
            for train_iteration, batch in training_loop:
                optimizer.zero_grad()
                loss, _ = self._general_step(batch, loss_fn, class_weights)
                loss.backward()
                optimizer.step()
                scheduler.step()

                training_loss += loss.item()
                training_loop.set_postfix(train_loss=f"{training_loss / (train_iteration + 1):.6f}")

                tb_logger.add_scalar("train_loss", loss.item(), epoch * len(train_loader) + train_iteration)

            # Validation loop
            validation_loss = 0
            with torch.no_grad():
                val_loop = tqdm(
                    enumerate(val_loader),
                    desc=f"Validation Epoch {epoch + 1}/{epochs}",
                    total=len(val_loader),
                    ncols=200
                )
                for val_iteration, batch in val_loop:
                    loss, _ = self._general_step(batch, loss_fn, class_weights)
                    validation_loss += loss.item()

                    val_loop.set_postfix(val_loss=f"{validation_loss / (val_iteration + 1):.6f}")
                    tb_logger.add_scalar("val_loss", validation_loss / (val_iteration + 1), epoch * len(val_loader) + val_iteration)

class ResNet50OneStage(AbstractOneStageModel):

    def __init__(
        self,
        params: dict,
        input_channels: int = 1,
        num_classes: int = None,
        **kwargs,
    ):
        """
        Initialize the model with the given hyperparameters.

        Args:
            params (dict): Dictionary containing the hyperparameters.
            num_classes (int): Number of classes in the dataset
        """
        super().__init__(
            params=params,
            **kwargs,
        )

        # Load pretrained model
        # Best available weights (currently alias for IMAGENET1K_V2)
        self.model = torchvision.models.resnet50(weights="IMAGENET1K_V2")

        if input_channels != 3:
            self.model.conv1 = torch.nn.Conv2d(
                input_channels,
                self.model.conv1.out_channels,
                kernel_size=(3, 3),
                stride=self.model.conv1.stride,
                padding=self.model.conv1.padding,
                bias=False,
            )
            torch.nn.init.kaiming_normal_(self.model.conv1.weight, mode='fan_out', nonlinearity='relu')
            
        # Replace the output layer
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        print(self.model)

        # Set device
        self.model.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        return self.model(x)


class ResNet18OneStage(AbstractOneStageModel):

    def __init__(
        self,
        params: dict,
        input_channels: int = 1,
        num_classes: int = None,
        **kwargs,
    ):
        """
        Initialize the model with the given hyperparameters.

        Args:
            params (dict): Dictionary containing the hyperparameters.
            num_classes (int): Number of classes in the dataset
        """
        super().__init__(
            params=params,
            **kwargs,
        )

        # Load pretrained model
        self.model = torchvision.models.resnet18(weights="IMAGENET1K_V1")

        # Adapt input size of model to the image channels
        if input_channels != 3:
            self.model.conv1 = torch.nn.Conv2d(
                input_channels,
                self.model.conv1.out_channels,
                kernel_size=self.model.conv1.kernel_size,
                stride=self.model.conv1.stride,
                padding=self.model.conv1.padding,
                bias=False,
            )

        # Replace the output layer
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        print(self.model)

        # Set device
        self.model.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        return self.model(x)
