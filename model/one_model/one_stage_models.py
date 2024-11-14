import torch
import torchvision
import os
import torch.nn.functional as F

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
        if "lr" in params:
            self.lr = params["lr"]
        if "batch_size" in params:
            self.batch_size = params["batch_size"]
        if "num_epochs" in params:
            self.num_epochs = params["num_epochs"]
        # TODO add more hyperparameters if needed

    def forward(self, x):
        raise NotImplementedError

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

    def _general_step(self, batch, loss_fn=F.cross_entropy):

        images, targets = batch

        # load X, y to device!
        images, targets = images.to(self.device), targets.to(self.device)

        # forward pass
        out = self.forward(images)

        # loss
        loss = loss_fn(out, targets)

        preds = out.argmax(axis=1)
        n_correct = (targets == preds).sum()
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
        return loss

    def _test_step(self, batch, loss_fn=F.cross_entropy):
        loss, n_correct = self._general_step(batch, loss_fn=loss_fn)
        return loss

    def _configure_optimizer(self, learning_rate=1e-3):
        optim = torch.optim.Adam(self.parameters(), learning_rate)
        return optim

    def train(self, train_dataset, val_dataset, loss_fn, tb_logger, epochs=10):
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )
        optimizer = self._configure_optimizer()
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=epochs * len(train_loader) / 5, gamma=0.7
        )
        validation_loss = 0
        self.model = self.model.to(self.device)
        for epoch in range(epochs):

            # Train
            training_loop = tqdm(
                enumerate(train_loader),
                desc=f"Training Epoch {epoch + 1}/{self.num_epochs}",
                total=len(train_loader),
                ncols=200,
            )
            training_loss = 0
            for train_iteration, batch in training_loop:
                optimizer.zero_grad()
                loss = self._training_step(batch, loss_fn)
                loss.backward()
                optimizer.step()
                scheduler.step()

                training_loss += loss.item()

                # Update the progress bar.
                training_loop.set_postfix(
                    train_loss="{:.8f}".format(training_loss / (train_iteration + 1)),
                    val_loss="{:.8f}".format(validation_loss),
                )

                # Update the tensorboard logger.
                tb_logger.add_scalar(
                    f"train_loss",
                    loss.item(),
                    epoch * len(train_loader) + train_iteration,
                )

            # Validation
            val_loop = tqdm(
                enumerate(val_loader),
                desc=f"Validation Epoch {epoch + 1}/{self.num_epochs}",
                total=len(train_loader),
                ncols=200,
            )
            validation_loss = 0
            with torch.no_grad():
                for val_iteration, batch in val_loop:
                    loss = self._validation_step(
                        batch, loss_fn
                    )  # You need to implement this function.
                    validation_loss += loss.item()

                    # Update the progress bar.
                    val_loop.set_postfix(
                        val_loss="{:.8f}".format(validation_loss / (val_iteration + 1))
                    )

                    # Update the tensorboard logger.
                    tb_logger.add_scalar(
                        f"val_loss",
                        validation_loss / (val_iteration + 1),
                        epoch * len(val_loader) + val_iteration,
                    )
            # This value is for the progress bar of the training loop.
            validation_loss /= len(val_loader)


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
            # input_size (np.array): Size of the input image. Shape: [channels, height, width]
            num_classes (int): Number of classes in the dataset
        """
        super().__init__(
            params=params,
            **kwargs,
        )

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
            # input_size (np.array): Size of the input image. Shape: [channels, height, width]
            num_classes (int): Number of classes in the dataset
        """
        super().__init__(
            params=params,
            **kwargs,
        )

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
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        print(self.model)

        # Set device
        self.model.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        return self.model(x)
