import sys
if "DROID-SLAM/droid_slam" not in sys.path: sys.path.append("DROID-SLAM/droid_slam")
if "DROID-SLAM/droid_slam/modules" not in sys.path: sys.path.append("DROID-SLAM/droid_slam/modules")

# Generic imports
from lietorch import SE3
from collections import OrderedDict
import numpy as np
import torch

# Droid slam imports
from droid_slam.droid_net import DroidNet

class DroidSlam(torch.nn.Module):
    def __init__(self, num_images: int, calib, num_steps: int = 15):
        super().__init__()
        self.model = DroidNet()
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.num_steps = num_steps
        self.num_images = num_images

        graph = OrderedDict()
        for i in range(num_images):
            graph[i] = [j for j in range(num_images) if i != j and abs(i - j) <= 2]
        self.graph = graph

        fx, fy, cx, cy = calib[:4]
        self.intrinsics = torch.tensor([fx, fy, cx, cy]).tile(1, self.num_images, 1).float().cuda()

        # Initialized at runtime
        self.Gs = None
        self.disp0 = None

    def forward(self, images, poses):
        if self.Gs is None:
            Ps = SE3(poses).inv()
            self.Gs = SE3.IdentityLike(Ps)

        if self.disp0 is None:
            self.disp0 = torch.ones((images.size(0), images.shape[2], images.shape[3]//8, images.shape[4]//8), device=images.device)
        if self.intrinsics.size(0) != images.size(0):
            self.intrinsics = self.intrinsics.tile(images.size(0), 1, 1)

        poses_est, _, _ = self.model(self.Gs, images, self.disp0, self.intrinsics,
                                                    self.graph, num_steps=self.num_steps)
        return poses_est