import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import ResNet18_Weights, resnet18, swin_v2_t
from typing import Optional
from lietorch import SE3
from kornia.color import rgb_to_grayscale, raw_to_rgb, CFA
import kornia.geometry.transform as KGT
from kornia.enhance import normalize
import matplotlib.pyplot as plt

from BorealHDR.scripts.classes.class_image_emulator import ImageEmulatorOneSequence 
from neural_network.slam import DroidSlam
from neural_network.geodesic_loss import geodesic_loss
from neural_network.datasets import pose_from_lidar

np.set_printoptions(precision=3)

class CustomModel(torch.nn.Module):
    def __init__(self, weights: Optional[ResNet18_Weights] = ResNet18_Weights.IMAGENET1K_V1, in_depth: int = 3, output_size: int = 1):
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

    def forward(self, img0, img1, img2, img3, img4, img5, img6):
        features0 = self.backbone(img0)
        features1 = self.backbone(img1)
        features2 = self.backbone(img2)
        features3 = self.backbone(img3)
        features4 = self.backbone(img4)
        features5 = self.backbone(img5)
        features6 = self.backbone(img6)
        # Add features for now
        combined_features = features0 + features1 + features2 + features3 + features4 + features5 + features6
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
    
    def on_after_backward(self):
    # pick a parameter to inspect
        for name, param in self.named_parameters():
            # print(f"Name: {name}")
            if param.grad is not None and name == "model.backbone.conv1.weight":
                print("--- Gradient Inspection ---")
                print(f"{name} grad mean: {param.grad.mean().item()}")

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
            color_bayer=True,
            device='cuda'
        )
        self.slam = DroidSlam(n_images, calib)
        self.slam.eval()

    def forward(self, image0, image1, image2, image3, image4, image5, image6, exposure0, exposure1, exposure2, exposure3, exposure4, exposure5, exposure6, pose0, pose1, pose2, pose3, pose4, pose5, pose6, next_brackets, target):
        image0_norm = normalize(image0, mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225]))
        image1_norm = normalize(image1, mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225]))
        image2_norm = normalize(image2, mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225]))
        image3_norm = normalize(image3, mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225]))
        image4_norm = normalize(image4, mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225]))
        image5_norm = normalize(image5, mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225]))
        image6_norm = normalize(image6, mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225]))

        output_model = self.model(image0_norm, image1_norm, image2_norm, image3_norm, image4_norm, image5_norm, image6_norm)
        delta_next_exposure = torch.abs(output_model) # TODO: Verify
        delta_next_exposure = torch.clamp(delta_next_exposure, min=1e-3, max=1e3)
        next_exposure = torch.clamp(exposure6 * delta_next_exposure, min=0.5, max=50.0)

        # next_exposure = torch.tensor([10.0, 10.0], requires_grad=True).to(exposure1.device)
        # print(f"Predicted next exposure: {next_exposure.squeeze().detach().cpu().numpy()}")
        
        # Process each sample in the batch individually
        batch_size = image0.shape[0]
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
            emulated_img = emulated_img / 16.0
            emulated_img = raw_to_rgb(emulated_img.unsqueeze(0).unsqueeze(0), cfa=CFA.RG).squeeze(0)

            # Apply transforms
            emulated_img = KGT.resize(emulated_img, size=image0.shape[2:])
            # emulated_img_normalized = normalize(emulated_img_resized.unsqueeze(0), mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])).squeeze(0)
            # emulated_img = augmented_emulated["image"].to(next_exposure.device)
            ###

            timestamp = res["timestamp"]
            final_target = pose_from_lidar(target[i], timestamp)
            lidar_target = torch.stack([pose0[i], pose1[i], pose2[i], pose3[i], pose4[i], pose5[i], pose6[i], final_target.to(pose0.device)])

            lidar_targets.append(lidar_target)
            emulated_images.append(emulated_img)

        # Stack the emulated images back into a batch
        emulated_next_image = torch.stack(emulated_images)
        poses = torch.stack(lidar_targets).float()#[1:]

        # Display all 7 images and emulated_next_image side-by-side for the first sample in the batch
        img0_np = image0[0].detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        img1_np = image1[0].detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        img2_np = image2[0].detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        img3_np = image3[0].detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        img4_np = image4[0].detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        img5_np = image5[0].detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        img6_np = image6[0].detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        emu_np = emulated_next_image[0].detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)

        fig, axes = plt.subplots(1, 8, figsize=(32, 4))
        axes[0].imshow(img0_np)
        axes[0].set_title(f"{exposure0[0].item():.2f} ms")
        axes[0].axis('off')

        axes[1].imshow(img1_np)
        axes[1].set_title(f"{exposure1[0].item():.2f} ms")
        axes[1].axis('off')

        axes[2].imshow(img2_np)
        axes[2].set_title(f"{exposure2[0].item():.2f} ms")
        axes[2].axis('off')

        axes[3].imshow(img3_np)
        axes[3].set_title(f"{exposure3[0].item():.2f} ms")
        axes[3].axis('off')

        axes[4].imshow(img4_np)
        axes[4].set_title(f"{exposure4[0].item():.2f} ms")
        axes[4].axis('off')

        axes[5].imshow(img5_np)
        axes[5].set_title(f"{exposure5[0].item():.2f} ms")
        axes[5].axis('off')

        axes[6].imshow(img6_np)
        axes[6].set_title(f"{exposure6[0].item():.2f} ms")
        axes[6].axis('off')

        axes[7].imshow(emu_np)
        axes[7].set_title(f"Pred: {next_exposure[0].item():.2f} ms")
        axes[7].axis('off')
        plt.tight_layout()
        plt.show()

        print(f"emulated_next_image ({next_exposure[0].item():.2f} ms)")

        predicted_poses = self.slam(
            torch.concat((image0.unsqueeze(1), image1.unsqueeze(1), image2.unsqueeze(1), image3.unsqueeze(1), 
                         image4.unsqueeze(1), image5.unsqueeze(1), image6.unsqueeze(1), emulated_next_image.unsqueeze(1)), dim=1),
            poses
        )

        return predicted_poses, SE3(poses).inv()
    
    def loss_factor(self, predicted_poses, target_poses):


        # Extract translation parts (last column of the 4x4 pose matrices)
        print(predicted_poses[-1].shape)
        pred_matrix = predicted_poses[-1].matrix().detach().cpu().numpy()
        target_matrix = target_poses.matrix().detach().cpu().numpy()

        # If batch size > 1, shape is (B, 4, 4)
        pred_trans = pred_matrix[0, :, :3, 3]
        # pred_trans /= np.nanmax(pred_trans, axis=0, keepdims=True)  # Normalize for better visualization
        target_trans = target_matrix[0, :, :3, 3]
        # target_trans /= np.nanmax(target_trans, axis=0, keepdims=True)  # Normalize for better visualization

        print(pred_trans)
        print(target_trans)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        ax_target = plt.subplot(1, 2, 1)
        ax_predicted = plt.subplot(1, 2, 2)
        ax_predicted.scatter(pred_trans[:,0], pred_trans[:,1], c=np.arange(pred_trans.shape[0]), cmap='plasma', label='Predicted')
        ax_target.scatter(target_trans[:,0], target_trans[:,1], c=np.arange(target_trans.shape[0]), cmap='plasma', label='Target')
        ax_predicted.set_xlabel('x')
        ax_target.set_xlabel('x')
        ax_predicted.set_ylabel('y')
        ax_target.set_ylabel('y')
        ax_predicted.legend()
        ax_target.legend()

        plt.tight_layout()
        plt.show()

        loss = geodesic_loss(target_poses, predicted_poses, self.slam.graph)
        return loss
    
    def freeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True
        self.model.fc.train()
