import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from loftr import LoFTR
import numpy as np
from datetime import datetime
from kornia.geometry import epipolar
from pathlib import Path
import glob
import cv2
import yaml
import kornia.feature as KF
from kornia_moons.viz import *
from pyutils import progress, Colors

class DifferentiableVO(torch.nn.Module):
    def __init__(self, K: torch.Tensor, homogeneous: bool = False):
        super().__init__()
        self.K = K
        self.homogeneous = homogeneous
        self.matcher = LoFTR(pretrained='outdoor')

    def get_keypoints(self, img1: torch.Tensor, img2: torch.Tensor):
        input_dict = {
            'image0': img1,
            'image1': img2,
        }
        correspondences = self.matcher(input_dict)

        kp1 = correspondences['keypoints0']  # [N, 2]
        kp2 = correspondences['keypoints1']  # [N, 2]
        confidence = correspondences['confidence']  # [N]
        return kp1, kp2, confidence

    def motion_from_keypoints(self, kp1: torch.Tensor, kp2: torch.Tensor, confidence: torch.Tensor):
        F = epipolar.find_fundamental(kp1.unsqueeze(0), kp2.unsqueeze(0), confidence.unsqueeze(0))
        E = epipolar.essential_from_fundamental(F, self.K.squeeze(0), self.K.squeeze(0))
        R, t, _ = epipolar.motion_from_essential_choose_solution(E, self.K, self.K, kp1.unsqueeze(0), kp2.unsqueeze(0))
        return R, t

    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        print("Getting keypoints...")
        kp1, kp2, confidence = self.get_keypoints(img1, img2)
        print("Estimating motion...")
        R, t = self.motion_from_keypoints(kp1, kp2, confidence)
        if self.homogeneous:
            # Differentiable homogeneous transformation matrix [B, 4, 4]
            R_tmp = torch.cat((R, torch.zeros(1, 1, 3)), dim=1)
            t_tmp = torch.cat((t, torch.ones(1, 1, 1)), dim=1)
            M = torch.cat((R_tmp, t_tmp), dim=2)
            return M
        else:
            return R, t

    def plot_keypoints(self, img1: torch.Tensor, img2: torch.Tensor, n: int = 64):
        kp1, kp2, confidence = self.get_keypoints(img1, img2)
        idx = np.arange(len(kp1))
        np.random.shuffle(idx)
        idx = idx[:n]  # Plot only n matches
        kp1s = kp1[idx]
        kp2s = kp2[idx]
        draw_LAF_matches(
            KF.laf_from_center_scale_ori(
                kp1s.view(1, -1, 2),
                torch.ones(kp1s.shape[0]).view(1, -1, 1, 1),
                torch.ones(kp1s.shape[0]).view(1, -1, 1),
            ),
            KF.laf_from_center_scale_ori(
                kp2s.view(1, -1, 2),
                torch.ones(kp2s.shape[0]).view(1, -1, 1, 1),
                torch.ones(kp2s.shape[0]).view(1, -1, 1),
            ),
            torch.arange(kp1s.shape[0]).view(-1, 1).repeat(1, 2),
            img1.detach().cpu(),
            img2.detach().cpu(),
            draw_dict={"inlier_color": (0.2, 1, 0.2), "tentative_color": (1, 1, 0.2, 0.3), "feature_color": None,
                       "vertical": False},
        )
        return


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B_size = 1
    # Camera intrinsics
    with open('BorealHDR/calibration_files/calibration_vo/september_2023/left.yaml', 'r') as f:
        calib = yaml.safe_load(f)
    K_data = calib['camera_matrix']['data']
    K = torch.tensor([
        [K_data[0], K_data[1], K_data[2]],
        [K_data[3], K_data[4], K_data[5]],
        [K_data[6], K_data[7], K_data[8]]
    ], dtype=torch.float32).unsqueeze(0)

    # Loading VO model
    vo = DifferentiableVO(K, homogeneous=True)

    # Load images
    path = Path("../BorealHRD_dataset/backpack_2023-09-27-12-46-32/camera_left")
    bracket = str(16.0)

    files = [Path(filepath).name for filepath in glob.glob(str(path / bracket / "*.png"))]
    files = sorted(files, key=lambda x: int(x.split(".")[0]))

    ref_img = files[0]
    imgs1 = []
    imgs2 = []
    for f in progress(files[1:]):
        img1 = torch.from_numpy(cv2.imread(str(path / bracket / ref_img), cv2.IMREAD_ANYDEPTH) / 16).to(torch.float)
        img1 = img1.reshape(1, 1, img1.shape[0], img1.shape[1]) / 256
        img2 = torch.from_numpy(cv2.imread(str(path / bracket / f), cv2.IMREAD_ANYDEPTH) / 16).to(torch.float)
        img2 = img2.reshape(1, 1, img2.shape[0], img2.shape[1]) / 256
        ref_img = f
        imgs1.append(img1)
        imgs2.append(img2)

    img1s = torch.cat(imgs1, dim=0)
    img2s = torch.cat(imgs2, dim=0)

    Ms = []
    with torch.inference_mode():
        # Run batched VO
        for i in progress(range(0, img1s.shape[0], B_size), desc="Running VO"):
            img1 = img1s[i:i+B_size].to(device)
            img2 = img2s[i:i+B_size].to(device)
            M = vo(img1, img2)
            Ms.append(M.cpu())

    Ms = torch.cat(Ms, dim=0)
    torch.save(Ms, 'vo_poses.pth')

if __name__ == '__main__':
    main()