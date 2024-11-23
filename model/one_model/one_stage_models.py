import torch
import json
import torchvision
import os
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
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
        self.class_weights = None  # Initialize class_weights for WeightedRandomSampler

        # Set hyperparameters
        if "lr" in params:
            self.lr = params["lr"]
        if "batch_size" in params:
            self.batch_size = params["batch_size"]
        if "num_epochs" in params:
            self.num_epochs = params["num_epochs"]
        if "optimizer" in params:
            self.optimizer = params["optimizer"]
        if "loss_fn" in params:
            self.loss_fn = params["loss_fn"]
        if "save_epoch" in params:
            self.save_epoch = params["save_epoch"]
        if "use_weighted_sample" in params:
            self.use_weighted_sampler = params["use_weighted_sampler"]
        # TODO add more hyperparameters if needed

        ### Wouldn't it be better to just self.x = params.get('x', default_value)?

        # Save hyperparameters
        self.params = params

        # Save results
        self.results = {}

    def forward(self, x):
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError

    def save_model(self, path: str, epoch: int = None):
        """
        Save the model and its weights to the given path.

        Args:
            path (str): Path where the model should be saved
        """
        model = self.model.to("cpu")
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        if epoch is not None:
            path = os.path.join(path, f"model_epoch_{epoch}.pth")
        else:
            path = os.path.join(path, "model.pth")
        torch.save(model.state_dict(), path)

    def save_hparams(self, path: str):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        path = os.path.join(path, "hparams.json")
        with open(path, "w") as f:
            json.dump(self.params, f)

    def load_hparams(self, path: str):
        self.params = json.load(open(path, "r"))

    def create_weighted_sampler(self, dataset):
        """
        Create a WeightedRandomSampler for class imbalance.
        Args: dataset (Dataset): PyTorch dataset with `targets` attribute.
        Returns: WeightedRandomSampler: Sampler for the DataLoader.
        """
        if hasattr(dataset, "targets"):
            targets = dataset.targets  
            class_sample_count = np.array([np.sum(targets == t) for t in np.unique(targets)])
            class_weights = 1.0 / class_sample_count
            sample_weights = np.array([class_weights[t] for t in targets])

            sampler = WeightedRandomSampler(
                weights=torch.tensor(sample_weights, dtype=torch.float32),
                num_samples=len(sample_weights),
                replacement=True,
            )
            return sampler

    def _general_step(self, batch, loss_fn=F.cross_entropy):
        images, targets = batch
        # load X, y to device!
        images, targets = images.to(self.device), targets.to(self.device)
        # forward pass
        out = self.forward(images)
        # loss
        loss = loss_fn(out, targets)
        # calculate n_correct
        preds = out.argmax(dim=1)
        n_correct = (preds == targets).sum().item()
        return loss, n_correct

    def _general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + "_loss"] for x in outputs]).mean()
        total_correct = (
            torch.stack([x[mode + "_n_correct"] for x in outputs]).sum().cpu().numpy()
        )
        acc = total_correct / len(self.dataset[mode])
        return avg_loss, acc

    def _training_step(self, batch, loss_fn):
        loss, n_correct = self._general_step(batch, loss_fn=loss_fn)
        return loss

    def _validation_step(self, batch, loss_fn=F.cross_entropy):
        loss, n_correct = self._general_step(batch, loss_fn=loss_fn)
        return loss, n_correct

    def _test_step(self, batch, loss_fn=F.cross_entropy):
        loss, n_correct = self._general_step(batch, loss_fn=loss_fn)
        return loss

    def _configure_optimizer(self, learning_rate=1e-3):
        if self.optimizer == "adam":
            optim = torch.optim.Adam(self.parameters(), learning_rate)
        else:
            optim = torch.optim.Adam(self.parameters(), learning_rate)
        return optim

    def train(self, train_dataset, val_dataset, tb_logger, path):
        """
        Train the model with optional WeightedRandomSampler.
        """
        if self.use_weighted_sampler:
            sampler = self.create_weighted_sampler(train_dataset)
            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, sampler=sampler
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            )

        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )
        optimizer = self._configure_optimizer()
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.num_epochs * len(train_loader) / 5, gamma=0.7
        )
        loss_fn = F.cross_entropy
        if self.class_weights is not None:
            class_weights_tensor = torch.tensor(self.class_weights).to(self.device)
            loss_fn = lambda output, target: F.cross_entropy(output, target, weight=class_weights_tensor)

        self.model = self.model.to(self.device)
        for epoch in range(self.num_epochs):
            # Training loop
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}"):
                optimizer.zero_grad()
                loss, _ = self._general_step(batch, loss_fn)
                loss.backward()
                optimizer.step()
                scheduler.step()

            # Validation
            val_loop = tqdm(
                enumerate(val_loader),
                desc=f"Validation Epoch {epoch + 1}/{self.num_epochs}",
                total=len(train_loader),
                ncols=200,
            )
            validation_loss = 0
            total_correct = 0
            with torch.no_grad():
                for val_iteration, batch in val_loop:
                    loss, n_correct = self._validation_step(
                        batch, loss_fn
                    )  # You need to implement this function.
                    validation_loss += loss.item()
                    total_correct += n_correct

                    val_loop.set_postfix(
                        val_loss="{:.8f}".format(validation_loss / (val_iteration + 1)),
                    )

                    # Update the tensorboard logger.
                    tb_logger.add_scalar(
                        "Loss/val",
                        validation_loss / (val_iteration + 1),
                        epoch * len(val_loader) + val_iteration,
                    )

                if self.save_epoch and epoch % self.save_epoch == 0 and epoch != 0:
                    save_path = os.path.join(path)
                    self.save_model(save_path, epoch)

            # This value is for the progress bar of the training loop.
            validation_loss /= len(val_loader)
            validation_acc = 100.0 * total_correct / len(val_loader.dataset)
            tb_logger.add_scalar(
                "Acc/acc",
                validation_acc,
                epoch * len(val_loader) + val_iteration,
            )


class ResNet50OneStage(AbstractOneStageModel):
    def __init__(
        self,
        params: dict,
        input_channels: int = 1,
        num_labels: int = None,
        use_weighted_sampler: bool = False,
        **kwargs,
    ):
        """
        Initialize the model with the given hyperparameters.

        Args:
            params (dict): Dictionary containing the hyperparameters.
            # input_size (np.array): Size of the input image. Shape: [channels, height, width]
            num_labels (int): Number of classes in the dataset
            use_weighted_sampler (bool): If True, balances target classes.
        """
        super().__init__(
            params=params,
            **kwargs,
        )

        # Assign use_weighted_sampler as an instance attribute
        self.use_weighted_sampler = use_weighted_sampler

        # Load pretrained model
        # Best available weights (currently alias for IMAGENET1K_V2)
        self.model = torchvision.models.resnet50(weights="IMAGENET1K_V2")

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
        self.model.fc = torch.nn.Linear(
            self.model.fc.in_features,
            num_labels,
        )

        # Set device
        self.model.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        return self.model(x)

    @property
    def name(self):
        return "ResNet50OneStage"


class ResNet18OneStage(AbstractOneStageModel):
    def __init__(
        self,
        params: dict,
        input_channels: int = 1,
        num_labels: int = None,
        use_weighted_sampler: bool = False,
        **kwargs,
    ):
        """
        Initialize the model with the given hyperparameters.

        Args:
            params (dict): Dictionary containing the hyperparameters.
            # input_size (np.array): Size of the input image. Shape: [channels, height, width]
            num_labels (int): Number of classes in the dataset
            use_weighted_sampler (bool): If True, balances target classes.
        """
        super().__init__(
            params=params,
            **kwargs,
        )

        # Assign use_weighted_sampler as an instance attribute
        self.use_weighted_sampler = use_weighted_sampler

        # Load pretrained model
        self.model = torchvision.models.resnet18(weights="IMAGENET1K_V1:")

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
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_labels)

        # Set device
        self.model.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        return self.model(x)

    @property
    def name(self):
        return "ResNet18OneStage"