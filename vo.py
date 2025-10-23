import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
# from loftr import LoFTR
from kornia.feature import LoFTR
import numpy as np
from datetime import datetime
from kornia.geometry import epipolar
from kornia.geometry.conversions import Rt_to_matrix4x4, matrix4x4_to_Rt
from kornia.geometry.ransac import RANSAC
from kornia.geometry.calibration import undistort_image
from kornia.geometry.transform import resize
from pathlib import Path
import glob
import cv2
import yaml
import kornia.feature as KF
from kornia_moons.viz import *
from pyutils import progress, Colors
from scipy.spatial.transform import Rotation
import time
import kornia

class DifferentiableVO(torch.nn.Module):
    def __init__(self, K: torch.Tensor, homogeneous: bool = False):
        super().__init__()
        self.K = K
        self.homogeneous = homogeneous
        self.matcher = LoFTR(pretrained='outdoor')
        self.ransac = RANSAC(model_type='fundamental', inl_th=1.0, max_iter=100, confidence=0.99)

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
        R, t, points_3d = epipolar.motion_from_essential_choose_solution(E, self.K, self.K, kp1.unsqueeze(0), kp2.unsqueeze(0))

        # # --- Step 1: RANSAC expects (N, 2), not (1, N, 2)
        # if kp1.dim() == 3:
        #     kp1 = kp1.squeeze(0)
        # if kp2.dim() == 3:
        #     kp2 = kp2.squeeze(0)
        # if confidence.dim() == 2:
        #     confidence = confidence.squeeze(0)

        # # --- Step 2: Estimate Fundamental matrix
        # F, _ = self.ransac(kp1, kp2, confidence)  # shapes: F -> (1, 3, 3)

        # # --- Step 3: Compute Essential matrix
        # K = self.K if self.K.dim() == 3 else self.K.unsqueeze(0)
        # E = epipolar.essential_from_fundamental(F, K, K)  # (1, 3, 3)

        # # --- Step 4: motion_from_essential_choose_solution expects batched inputs (1, N, 2)
        # kp1_b = kp1.unsqueeze(0)
        # kp2_b = kp2.unsqueeze(0)

        # R, t, points_3d = epipolar.motion_from_essential_choose_solution(E, K, K, kp1_b, kp2_b)
        return R, t, points_3d

    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        # self.plot_keypoints(img1, img2, n=64)
        kp1, kp2, confidence = self.get_keypoints(img1, img2)
        R, t, points = self.motion_from_keypoints(kp1, kp2, confidence)
        # self.plot_3d_points(points)
        if self.homogeneous:
            # # Differentiable homogeneous transformation matrix [B, 4, 4]
            # R_tmp = torch.cat((R, torch.zeros(R.size(0), 1, 3, device=R.device)), dim=1)
            # t_tmp = torch.cat((t, torch.ones(t.size(0), 1, 1, device=t.device)), dim=1)
            # M = torch.cat((R_tmp, t_tmp), dim=2)
            M = Rt_to_matrix4x4(R, t)
            return M
        else:
            return R, t
        
    def plot_3d_points(self, points_3d: torch.Tensor):
        """
        Plots the 3D points using matplotlib.
        Args:
            points_3d (torch.Tensor): 3D points of shape [B, N, 3] or [N, 3]
        """
        pts = points_3d.detach().cpu().numpy()
        if pts.ndim == 3:
            pts = pts.reshape(-1, 3)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

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
                torch.ones(kp1s.shape[0], device=kp1s.device).view(1, -1, 1, 1),
                torch.ones(kp1s.shape[0], device=kp1s.device).view(1, -1, 1),
            ),
            KF.laf_from_center_scale_ori(
                kp2s.view(1, -1, 2),
                torch.ones(kp2s.shape[0], device=kp2s.device).view(1, -1, 1, 1),
                torch.ones(kp2s.shape[0], device=kp2s.device).view(1, -1, 1),
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
    K = 1/3 * torch.tensor([
        [K_data[0], K_data[1], K_data[2]],
        [K_data[3], K_data[4], K_data[5]],
        [K_data[6], K_data[7], K_data[8]]
    ], dtype=torch.float32, device=device).unsqueeze(0)

    distortion_coeffs = calib['distortion_coefficients']['data']
    dist_coeffs = torch.tensor(distortion_coeffs, dtype=torch.float32, device=device).unsqueeze(0)

    # Loading VO model
    vo = DifferentiableVO(K, homogeneous=True).to(device)

    # Load images
    path = Path("/media/alienware/T7_Shield/ICRA2024_OG/dataset/belair-09-27-2023/data_high_resolution/backpack_2023-09-27-12-51-03/camera_left")
    bracket = str(16.0)

    files = [Path(filepath).name for filepath in glob.glob(str(path / bracket / "*.png"))]
    files = sorted(files, key=lambda x: int(x.split(".")[0]))

    ref_img = files[0]
    imgs1 = []
    imgs2 = []
    new_size = (400, 640)  # (W, H)
    for f in progress(files[1:], desc="Loading images"):
        img1_distorted = torch.from_numpy(cv2.imread(str(path / bracket / ref_img), cv2.IMREAD_ANYDEPTH) / 16.0).to(torch.float).to(device)
        img1_distorted = img1_distorted.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        img1_undistorted = undistort_image(img1_distorted, K*3, dist_coeffs)
        img1_resized = resize(img1_undistorted, (img1_distorted.shape[2] // 3, img1_distorted.shape[3] // 3), interpolation='bilinear', align_corners=False, antialias=True)
        img1_resized = img1_resized / 256.0

        img2_distorted = torch.from_numpy(cv2.imread(str(path / bracket / f), cv2.IMREAD_ANYDEPTH) / 16).to(torch.float).to(device)
        img2_distorted = img2_distorted.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        img2_undistorted = undistort_image(img2_distorted, K*3, dist_coeffs)
        img2_resized = resize(img2_undistorted, (img2_distorted.shape[2] // 3, img2_distorted.shape[3] // 3), interpolation='bilinear', align_corners=False, antialias=True)
        # Normalize img2_resized after resizing using kornia
        img2_resized = img2_resized / 256.0

        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.title("Image 1")
        # plt.imshow(img1_resized.squeeze().detach().cpu().numpy(), cmap='gray')
        # plt.axis('off')
        # plt.subplot(1, 2, 2)
        # plt.title("Image 1 undistorted and resized")
        # plt.imshow(img2_resized.squeeze().detach().cpu().numpy(), cmap='gray')
        # plt.axis('off')
        # plt.show()

        ref_img = f
        imgs1.append(img1_resized)
        imgs2.append(img2_resized)

    img1s = torch.cat(imgs1, dim=0)
    img2s = torch.cat(imgs2, dim=0)

    Ms = [torch.eye(4, device=device).unsqueeze(0)]  # Initial pose
    with torch.inference_mode():
        # Run batched VO
        for i in progress(range(0, img1s.shape[0], B_size), desc="Running VO"):
            img1 = img1s[i:i+B_size].to(device)
            img2 = img2s[i:i+B_size].to(device)
            M = vo(img1, img2)
            Ms.append(Ms[-1] @ M)

    # Ms = torch.cat(Ms, dim=0)
    # torch.save(Ms, 'vo_poses.pth')
    # Ms = torch.load('vo_poses.pth')
    # Save as tum format
    timestamps = files
    timestamps = [int(f.split(".")[0]) / 1e9 for f in timestamps]
    # M = [torch.eye(4)]
    # print(Ms.shape)
    with open('trajectory3.txt', 'w') as f:
        for i in range(len(Ms)):
            R, t = matrix4x4_to_Rt(Ms[i])
            R = R.squeeze(0).detach().cpu().numpy()
            t = t.squeeze(0).detach().cpu().numpy().squeeze()
            q = Rotation.from_matrix(R).as_quat()
            f.write(f"{timestamps[i]} {t[0]} {t[1]} {t[2]} {q[0]} {q[1]} {q[2]} {q[3]}\n")


if __name__ == '__main__':
    main()