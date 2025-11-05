import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import ResNet18_Weights, resnet18, swin_v2_t

from BorealHDR.scripts.classes.class_image_emulator import ImageEmulatorOneSequence 

np.set_printoptions(precision=3)


class BaseNet(LightningModule):

    def __init__(self, lr=0.0001):
        super().__init__()

        self.lr = lr
        # torch.manual_seed(42)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        images, first_exposure, next_brackets, targets = batch["image"], batch["first_exposure"], batch["next_brackets"], batch["target"]
        outputs = self.forward(images.half(), first_exposure, next_brackets)

        loss = self.loss_factor(outputs, targets)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, first_exposure, next_brackets, targets = batch["image"], batch["first_exposure"], batch["next_brackets"], batch["target"]
        outputs = self.forward(images.half(), first_exposure, next_brackets)

        loss = self.loss_factor(outputs, targets)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, first_exposure, next_brackets, targets = batch["image"], batch["first_exposure"], batch["next_brackets"], batch["target"]
        outputs = self.forward(images.half(), first_exposure, next_brackets)

        loss = self.loss_factor(outputs, targets)
        self.log("test_loss", loss, prog_bar=False)
        return loss
    
    def loss_factor(self, outputs, targets):
        """
        Calculate the loss factor based on the outputs and targets.
        This is a placeholder method that can be overridden in subclasses.
        """
        loss = F.mse_loss(outputs, targets)

        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)
        return [optimizer], [{"scheduler": scheduler, "monitor": "val_loss"}]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(metric)


class ExposureResNet(BaseNet):

    def __init__(self, output_size=1, lr=0.0001):
        super().__init__(lr=lr)

        self.model = resnet18(weights=None)
        self.in_features = self.model.fc.in_features
        self.input_depth(1)
        self.replace_head(output_size)

        self.bracketing_values = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
        self.emulator = ImageEmulatorOneSequence(
            calibration_file="pcalib_inside1.txt",
            exposure_times_bracketing=self.bracketing_values,
            color_bayer=False,
            device='cuda'
        )

    def forward(self, x, first_exposure, next_brackets):
        delta_next_exposure = self.model(x)
        next_exposure = torch.abs(first_exposure * delta_next_exposure)
        
        # Process each sample in the batch individually
        batch_size = x.shape[0]
        emulated_images = []
        for i in range(batch_size):
            exposure_value = next_exposure[i].item()
            bracket_paths = next_brackets[i]
            
            emulated_img = self.emulator.emulate_image(
                exposure_value, 
                bracket_paths
            )["emulated_img"]
            emulated_images.append(emulated_img)

        # Stack the emulated images back into a batch
        emulated_next_image = torch.stack(emulated_images)
        
        # TODO: Use next image and the actual (x) to pass into DROID-SLAM
        # TODO: Compute loss by comparing DROID-SLAM result with lidar trajectory
        return emulated_next_image
    
    def loss_factor(self, outputs, targets):
        # TODO: Implement a custom loss function
        return super().loss_factor(outputs, targets)
    
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