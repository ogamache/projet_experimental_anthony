import os
from pathlib import Path
import re
from os.path import exists, join
import sys
sys.path.append(str(Path(__file__).parents[1]))

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from BorealHDR.scripts.classes.class_image_emulator import ImageEmulatorOneSequence
from kornia.color import rgb_to_grayscale, raw_to_rgb, CFA

def pose_from_lidar(lidar, timestamp):
    return torch.from_numpy(lidar.loc[np.argmin(np.abs(lidar['timestamp'].values - (float(timestamp))*1e-9))][
                         ["ty", "tz", "tx", "qy", "qz", "qx", "qw"]].values)

class CustomDataset(Dataset):
    def __init__(self, data_folders, transform=None, fract: float = 1.):
        """
        Args:
            data_folder: The folder containing the dataset
            transform: The transformations to apply to the images
        """
        self.data_folder = data_folders
        self.transform = transform
        self.bracketing_values = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
        self.emulator = ImageEmulatorOneSequence(
            calibration_file="pcalib_inside1.txt",
            exposure_times_bracketing=self.bracketing_values,
            color_bayer=True,
            device='cpu'
        )
        self.inputs = []
        self.labels = []

        for folder in data_folders:
            print(f"Loading data from folder {folder}")
            inputs, labels = self.load_data_from_folder(folder)
            self.inputs.extend(inputs)
            self.labels.extend(labels)

        if fract < 1.0:
            n_samples = int(len(self.inputs) * fract)
            self.inputs = self.inputs[:n_samples]
            self.labels = self.labels[:n_samples]
            print(f"Using only {n_samples} samples from the dataset.")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        label = self.labels[idx].copy()
        label = label[["timestamp", "ty", "tz", "tx", "qy", "qz", "qx", "qw"]]

        bracket_images_t0, bracket_images_t1, bracket_images_t2, bracket_images_t3, bracket_images_t4, bracket_images_t5, bracket_images_t6, bracket_images_t7, target_exposure_t0, target_exposure_t1, target_exposure_t2, target_exposure_t3, target_exposure_t4, target_exposure_t5, target_exposure_t6 = self.inputs[idx]
        
        res = self.emulator.emulate_image(target_exposure_t0, bracket_images_t0)
        img_t0 = res["emulated_img"]
        img_p0 = pose_from_lidar(label, res["timestamp"])
        
        res = self.emulator.emulate_image(target_exposure_t1, bracket_images_t1)
        img_t1 = res["emulated_img"]
        img_p1 = pose_from_lidar(label, res["timestamp"])
        
        res = self.emulator.emulate_image(target_exposure_t2, bracket_images_t2)
        img_t2 = res["emulated_img"]
        img_p2 = pose_from_lidar(label, res["timestamp"])
        
        res = self.emulator.emulate_image(target_exposure_t3, bracket_images_t3)
        img_t3 = res["emulated_img"]
        img_p3 = pose_from_lidar(label, res["timestamp"])
        
        res = self.emulator.emulate_image(target_exposure_t4, bracket_images_t4)
        img_t4 = res["emulated_img"]
        img_p4 = pose_from_lidar(label, res["timestamp"])
        
        res = self.emulator.emulate_image(target_exposure_t5, bracket_images_t5)
        img_t5 = res["emulated_img"]
        img_p5 = pose_from_lidar(label, res["timestamp"])
        
        res = self.emulator.emulate_image(target_exposure_t6, bracket_images_t6)
        img_t6 = res["emulated_img"]
        img_p6 = pose_from_lidar(label, res["timestamp"])

        # Convert 12-bit bayer images to RGB
        img_t0 = img_t0 / 16.0
        image0 = raw_to_rgb(img_t0.unsqueeze(0).unsqueeze(0), cfa=CFA.RG).squeeze(0)
        img_t1 = img_t1 / 16.0
        image1 = raw_to_rgb(img_t1.unsqueeze(0).unsqueeze(0), cfa=CFA.RG).squeeze(0)
        img_t2 = img_t2 / 16.0
        image2 = raw_to_rgb(img_t2.unsqueeze(0).unsqueeze(0), cfa=CFA.RG).squeeze(0)
        img_t3 = img_t3 / 16.0
        image3 = raw_to_rgb(img_t3.unsqueeze(0).unsqueeze(0), cfa=CFA.RG).squeeze(0)
        img_t4 = img_t4 / 16.0
        image4 = raw_to_rgb(img_t4.unsqueeze(0).unsqueeze(0), cfa=CFA.RG).squeeze(0)
        img_t5 = img_t5 / 16.0
        image5 = raw_to_rgb(img_t5.unsqueeze(0).unsqueeze(0), cfa=CFA.RG).squeeze(0)
        img_t6 = img_t6 / 16.0
        image6 = raw_to_rgb(img_t6.unsqueeze(0).unsqueeze(0), cfa=CFA.RG).squeeze(0)

        if self.transform:
            image0 = image0.permute(1, 2, 0).numpy()
            augmented0 = self.transform(image=image0)
            image0 = augmented0["image"]

            image1 = image1.permute(1, 2, 0).numpy()
            augmented1 = self.transform(image=image1)
            image1 = augmented1["image"]

            image2 = image2.permute(1, 2, 0).numpy()
            augmented2 = self.transform(image=image2)
            image2 = augmented2["image"]

            image3 = image3.permute(1, 2, 0).numpy()
            augmented3 = self.transform(image=image3)
            image3 = augmented3["image"]

            image4 = image4.permute(1, 2, 0).numpy()
            augmented4 = self.transform(image=image4)
            image4 = augmented4["image"]

            image5 = image5.permute(1, 2, 0).numpy()
            augmented5 = self.transform(image=image5)
            image5 = augmented5["image"]

            image6 = image6.permute(1, 2, 0).numpy()
            augmented6 = self.transform(image=image6)
            image6 = augmented6["image"]

        training_data = {
            "image0": image0,
            "image1": image1,
            "image2": image2,
            "image3": image3,
            "image4": image4,
            "image5": image5,
            "image6": image6,
            "exposure0": torch.Tensor([target_exposure_t0]),
            "exposure1": torch.Tensor([target_exposure_t1]),
            "exposure2": torch.Tensor([target_exposure_t2]),
            "exposure3": torch.Tensor([target_exposure_t3]),
            "exposure4": torch.Tensor([target_exposure_t4]),
            "exposure5": torch.Tensor([target_exposure_t5]),
            "exposure6": torch.Tensor([target_exposure_t6]),
            "pose0": img_p0,
            "pose1": img_p1,
            "pose2": img_p2,
            "pose3": img_p3,
            "pose4": img_p4,
            "pose5": img_p5,
            "pose6": img_p6,
            "next_brackets": bracket_images_t7,
            "target": label
        }

        return training_data
    
    def load_data_from_folder(self, folder):

        training_data = []
        labels = []
        for traj in tqdm(os.listdir(folder)):
            data_path = join(folder, traj)
            exposure_file = join(data_path, "optimal_exposure_times.txt")
            lidar_file = join(data_path, "lidar_trajectory.csv")
            if not exists(exposure_file) or not exists(lidar_file):
                print(f"Exposure or Lidar file not found for {data_path}, skipping...")
                continue

            exposure_df = pd.read_csv(exposure_file, header=None, names=["exposure_time"])
            lidar_df = pd.read_csv(lidar_file, names=["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"], sep=" ", header=None)
            images_df = self.create_dataframe(data_path, self.bracketing_values)

            for index in exposure_df.index[:-35]:
                optimal_exposure_t0 = exposure_df["exposure_time"].iloc[index]
                optimal_exposure_t1 = exposure_df["exposure_time"].iloc[index + 5]
                optimal_exposure_t2 = exposure_df["exposure_time"].iloc[index + 10]
                optimal_exposure_t3 = exposure_df["exposure_time"].iloc[index + 15]
                optimal_exposure_t4 = exposure_df["exposure_time"].iloc[index + 20]
                optimal_exposure_t5 = exposure_df["exposure_time"].iloc[index + 25]
                optimal_exposure_t6 = exposure_df["exposure_time"].iloc[index + 30]
                
                brackets_t0 = images_df.iloc[:, index].dropna().values
                brackets_t1 = images_df.iloc[:, index + 5].dropna().values
                brackets_t2 = images_df.iloc[:, index + 10].dropna().values
                brackets_t3 = images_df.iloc[:, index + 15].dropna().values
                brackets_t4 = images_df.iloc[:, index + 20].dropna().values
                brackets_t5 = images_df.iloc[:, index + 25].dropna().values
                brackets_t6 = images_df.iloc[:, index + 30].dropna().values
                brackets_t7 = images_df.iloc[:, index + 35].dropna().values
                labels.append(lidar_df)
                training_data.append((brackets_t0, brackets_t1, brackets_t2, brackets_t3, brackets_t4, brackets_t5, brackets_t6, brackets_t7, 
                                     optimal_exposure_t0, optimal_exposure_t1, optimal_exposure_t2, optimal_exposure_t3, optimal_exposure_t4, optimal_exposure_t5, optimal_exposure_t6))

        return training_data, labels
    
    def create_dataframe(self, path_imgs, bracket_values):
        bracket_lists = []
        for bracket_value in bracket_values:
            path_bracket = Path(path_imgs, str(bracket_value))
            bracket_list = []
            for img_filename in sorted(os.listdir(path_bracket)):
                bracket_list.append(str(path_bracket / img_filename))
            bracket_lists.append(bracket_list)
        df = pd.DataFrame(bracket_lists)
        df.index = bracket_values
        return df
    

def lightning_collate_fn(batch):
    elem = batch[0]
    if isinstance(elem, dict):
        return {k: lightning_collate_fn([d[k] for d in batch]) for k in elem}
    elif isinstance(elem, np.ndarray):
        # Check if it's a numpy array of objects (like strings)
        if elem.dtype == np.object_ or elem.dtype.kind in ('U', 'S', 'O'):
            return batch  # list of numpy arrays of strings, keep as-is
        else:
            return default_collate(batch)
    elif isinstance(elem, (torch.Tensor, int, float)):
        return default_collate(batch)
    elif isinstance(elem, str):
        return list(batch)
    elif isinstance(elem, list) and all(isinstance(x, str) for x in elem):
        return batch  # list of strings, one per sample
    elif isinstance(elem, pd.DataFrame):
        return batch  # list of DataFrames, one per sample
    else:
        return batch  # fallback (keep as-is)