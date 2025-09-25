import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import ResNet18_Weights, resnet18, swin_v2_t

np.set_printoptions(precision=3)


class ExposureNet(LightningModule):

    def __init__(self, lr=0.0001):
        super().__init__()

        self.lr = lr
        # torch.manual_seed(42)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.forward(images.half())

        loss = self.loss_factor(outputs, targets)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.forward(images.half())

        loss = self.loss_factor(outputs, targets)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.forward(images.half())

        loss = self.loss_factor(outputs, targets)
        self.log("test_loss", loss, prog_bar=False)
        return loss
    
    def loss_factor(self, outputs, targets):
        """
        Calculate the loss factor based on the outputs and targets.
        This is a placeholder method that can be overridden in subclasses.
        """

        # Handle different target shapes
        if len(targets.shape) > 1:
            # For multi-dimensional targets (e.g., covariance matrices)
            targets_flat = targets.view(-1).float()
            outputs_flat = outputs.view(-1)
            
            # Ensure outputs and targets have compatible shapes
            if outputs_flat.shape[0] != targets_flat.shape[0]:
                min_size = min(outputs_flat.shape[0], targets_flat.shape[0])
                outputs_flat = outputs_flat[:min_size]
                targets_flat = targets_flat[:min_size]
                print(f"Warning: Shape mismatch. Truncating to size {min_size}")
        else:
            # For 1D targets (e.g., single exposure values)
            targets_flat = targets.float()
            outputs_flat = outputs.view(-1)
        
        # Debug prints (uncomment if needed)
        # print(f"Outputs shape: {outputs.shape} -> flattened: {outputs_flat.shape}")
        # print(f"Targets shape: {targets.shape} -> flattened: {targets_flat.shape}")
        
        # print(f"Outputs: {outputs_flat[:5]}")
        # print(f"Targets: {targets_flat[:5]}")

        loss = F.mse_loss(outputs_flat, targets_flat, reduction="mean")

        # print(f"Loss: {loss.item()}")

        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)
        return [optimizer], [{"scheduler": scheduler, "monitor": "val_loss"}]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(metric)


class ExposureResNet(ExposureNet):

    def __init__(self, output_size=1, lr=0.0001):
        super().__init__(lr=lr)

        self.model = resnet18(weights=None)
        self.in_features = self.model.fc.in_features
        self.input_depth(1)
        self.replace_head(output_size)

    def forward(self, x):
        x = self.model(x)
        return x
    
    def input_depth(self, depth):
        self.model.conv1 = torch.nn.Conv2d(depth, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    def freeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True
        self.model.fc.train()

    def replace_head(self, output_size):
        self.model.fc  = torch.nn.Sequential(
            torch.nn.Linear(self.in_features, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, output_size),
        )