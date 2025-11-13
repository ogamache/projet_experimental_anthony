from cgi import print_environ_usage

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import ResNet18_Weights, resnet18, swin_v2_t
from typing import Optional
from lietorch import SE3

from BorealHDR.scripts.classes.class_image_emulator import ImageEmulatorOneSequence 
from neural_network.slam import DroidSlam
from neural_network.geodesic_loss import geodesic_loss
from neural_network.datasets import pose_from_lidar

np.set_printoptions(precision=3)

class CustomModel(torch.nn.Module):
    def __init__(self, weights: Optional[ResNet18_Weights] = None, in_depth: int = 3, output_size: int = 1):
        super().__init__()
        if in_depth != 3:
            raise NotImplementedError("Current DROID-SLAM only support 3 channel inputs.")
        self.backbone = resnet18(weights=weights)
        self.in_features = self.backbone.fc.in_features
        self.head = torch.nn.Sequential(
            torch.nn.Linear(self.in_features, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, output_size),
        )
        self.backbone.fc = torch.nn.Identity()
        # self.backbone.conv1 = torch.nn.Conv2d(in_depth, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, img1, img2):
        features1 = self.backbone(img1)
        features2 = self.backbone(img2)
        # Add features for now
        combined_features = features1 + features2
        output = self.head(combined_features)
        return output

class BaseNet(LightningModule):

    def __init__(self, lr=0.0001):
        super().__init__()

        self.lr = lr
        # torch.manual_seed(42)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # images, first_exposure, next_brackets, targets = batch["image"], batch["first_exposure"], batch["next_brackets"], batch["target"]
        # outputs = self.forward(images.half(), first_exposure, next_brackets)
        outputs, gt_poses = self.forward(**batch)

        loss = self.loss_factor(outputs, gt_poses)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # images, first_exposure, next_brackets, targets = batch["image"], batch["first_exposure"], batch["next_brackets"], batch["target"]
        outputs, gt_poses = self.forward(**batch)

        loss = self.loss_factor(outputs, gt_poses)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, first_exposure, next_brackets, targets = batch["image"], batch["first_exposure"], batch["next_brackets"], batch["target"]
        outputs = self.forward(images.half(), first_exposure, next_brackets)

        loss = self.loss_factor(outputs, targets)
        self.log("test_loss", loss, prog_bar=False)
        return loss
    
    def loss_factor(self, predicted_poses, target_poses):
        """
        Calculate the loss factor based on the outputs and targets.
        This is a placeholder method that can be overridden in subclasses.
        """
        loss = F.mse_loss(predicted_poses, target_poses)

        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)
        return [optimizer], [{"scheduler": scheduler, "monitor": "val_loss"}]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(metric)


class ExposureResNet(BaseNet):

    def __init__(self, n_images: int, calib: np.ndarray, output_size=1, lr=0.0001):
        super().__init__(lr=lr)

        self.model = CustomModel(output_size=output_size)
        self.in_features = self.model.in_features

        self.bracketing_values = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
        self.emulator = ImageEmulatorOneSequence(
            calibration_file="pcalib_inside1.txt",
            exposure_times_bracketing=self.bracketing_values,
            color_bayer=False,
            device='cuda'
        )
        self.slam = DroidSlam(n_images, calib)
        self.slam.eval()

    def forward(self, image0, image1, exposure0, exposure1, pose0, pose1, next_brackets, target):
        delta_next_exposure = self.model(image0, image1)
        next_exposure = torch.abs(exposure1 * delta_next_exposure)
        
        # Process each sample in the batch individually
        batch_size = image1.shape[0]
        emulated_images = []
        lidar_targets = []

        for i in range(batch_size):
            exposure_value = next_exposure[i]
            bracket_paths = next_brackets[i]
            
            res = self.emulator.emulate_image(
                exposure_value, 
                bracket_paths
            )
            emulated_img = res["emulated_img"]
            emulated_img = emulated_img.tile(3, 1, 1)
            timestamp = res["timestamp"]
            final_target = pose_from_lidar(target[i], timestamp)
            lidar_target = torch.stack([pose0[i], pose1[i], final_target.to(pose1.device)])

            lidar_targets.append(lidar_target)
            emulated_images.append(emulated_img)

        # Stack the emulated images back into a batch
        emulated_next_image = torch.stack(emulated_images)
        poses = torch.stack(lidar_targets).float()[1:]

        predicted_poses = self.slam(
            torch.concat((image0.unsqueeze(1), image1.unsqueeze(1), emulated_next_image.unsqueeze(1)), dim=1),
            poses
        )

        return predicted_poses, SE3(poses).inv()
    
    def loss_factor(self, predicted_poses, target_poses):
        loss = geodesic_loss(target_poses, predicted_poses, self.slam.graph)
        return loss
    
    def freeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True
        self.model.fc.train()
