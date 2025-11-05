import sys
sys.path.append("DROID-SLAM/droid_slam")

# Generic imports
from lietorch import SE3
from collections import OrderedDict
import numpy as np
import torch
import cv2
import os
from typing import NamedTuple
import pandas as pd

# Droid slam imports
from droid_slam.geom.graph_utils import graph_to_edge_list
from droid_slam.droid_net import DroidNet


def geodesic_loss(Ps, Gs, graph, gamma=0.9) -> torch.Tensor:
    """ Loss function for training network """

    # relative pose
    ii, jj, kk = graph_to_edge_list(graph)
    dP = Ps[:, jj] * Ps[:, ii].inv() # Convert to relative pose (From i -> j) Because P is the absolute pose (0 -> t)

    n = len(Gs)
    geodesic_loss = 0.0

    # We loop over all iterations with a decaying loss weight. This forces the network to improve its estimates at each
    # iteration. Effectively learning to iteratively optimize its pose estimates.
    for i in range(n):
        w = gamma ** (n - i - 1) # Decay weight
        dG = Gs[i][:, jj] * Gs[i][:, ii].inv() # Convert to relative pose displacement


        # pose error
        # log maps to tangent space at identity so that its norm is the geodesic distance
        d = (dG * dP.inv()).log() # (Compute the displacement between the estimated relative pose and the ground truth relative pose)

        tau, phi = d.split([3, 3], dim=-1)
        geodesic_loss += w * (
                tau.norm(dim=-1).mean() +
                phi.norm(dim=-1).mean())

    return geodesic_loss

class ImgDims(NamedTuple):
    h0: int
    w0: int
    h1: int
    w1: int

def load_image(imagedir, imfile, get_img_dims=False):
    image = cv2.imread(os.path.join(imagedir, imfile), cv2.IMREAD_ANYDEPTH)
    image = (image / 16.0).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2RGB)
    if len(calib) > 4:
        image = cv2.undistort(image, K, calib[4:])

    h0, w0, _ = image.shape
    h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
    w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

    image = cv2.resize(image, (w1, h1))
    image = image[:h1 - h1 % 8, :w1 - w1 % 8]
    image = torch.as_tensor(image).permute(2, 0, 1)
    if get_img_dims:
        return image, ImgDims(h0, w0, h1, w1)
    else:
        return image


# Number of frames
N = 7
path = "/home/alienware/Desktop/tmp/training_trajs/backpack_2023-04-20-09-29-14/16.0"

# Load intrisics
calib = "DROID-SLAM/calib/belair.txt"
calib = np.loadtxt(calib, delimiter=" ")
fx, fy, cx, cy = calib[:4]
K = np.eye(3)
K[0,0] = fx
K[0,2] = cx
K[1,1] = fy
K[1,2] = cy
intrinsics = torch.as_tensor([fx, fy, cx, cy])

_, img_dims = load_image(path, "1681997354956098048.png", get_img_dims=True)
# Adjust intrinsics for image resizing
intrinsics[0::2] *= (img_dims.w1 / img_dims.w0)
intrinsics[1::2] *= (img_dims.h1 / img_dims.h0)
intrinsics0 = (intrinsics / 8.0).tile(1, N, 1).float()


# Load model
model = DroidNet()
model.cuda()
model.train()

# Load data
filenames = sorted(os.listdir(path), key=lambda x: int(x.split(".")[0]))
images = torch.concat([
    load_image(path, filenames[i]).unsqueeze(0)
    for i in range(N)
]).unsqueeze(0).float() / 255.0  # [B=1, N, 3, H, W]

# Load poses
lidar_file = "/home/alienware/Desktop/tmp/training_trajs/backpack_2023-04-20-09-29-14/lidar_trajectory.csv"
lidar_df = pd.read_csv(lidar_file, names=["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"], sep=" ", header=None)
poses = torch.from_numpy(lidar_df[["ty", "tz", "tx", "qy", "qz", "qx", "qw"]].iloc[:N].values).unsqueeze(0).float() # Unsqueeze for batch dim
Ps = SE3(poses).inv()
Gs = SE3.IdentityLike(Ps)

# Init
disp0 = torch.ones((1, N, img_dims.h1//8, img_dims.w1//8)) # Initiate to ones



# Load graph. We can also build a smart graph from depth. TODO: Implement load depth from stereo and use smart graph
graph = OrderedDict()
for i in range(N):
    graph[i] = [j for j in range(N) if i!=j and abs(i-j) <= 2]

Gs, images, disp0, intrinsics0 = Gs.cuda(), images.cuda(), disp0.cuda(), intrinsics0.cuda()

images.requires_grad = True

poses_est, disps_est, residuals = model(Gs, images, disp0, intrinsics0,
                    graph, num_steps=15, fixedp=2)
# Poses est has a length of 15, which is the number of refinement steps

# ii, jj, kk = graph_to_edge_list(graph)
# ii -> jj = all pairs of edges of connected frames in the graph (src -> dest)

print(poses_est[-2].shape)
loss = geodesic_loss(Ps.cuda(), poses_est, graph)
loss.backward()
grad = images.grad.squeeze()
for image in grad:
    print(image.norm())
