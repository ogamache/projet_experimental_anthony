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

class CustomDataset(Dataset):
    def __init__(self, data_folders, transform=None):
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
            color_bayer=False,
            device='cpu'
        )
        self.inputs = []
        self.labels = []

        for folder in data_folders:
            print(f"Loading data from folder {folder}")
            inputs, labels = self.load_data_from_folder(folder)
            self.inputs.extend(inputs)
            self.labels.extend(labels)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):

        bracket_images_t0, bracket_images_t1, target_exposure_t0 = self.inputs[idx]
        img_t0 = self.emulator.emulate_image(target_exposure_t0, bracket_images_t0)["emulated_img"]

        # img_t0 #12bits bayer image
        img_t0 = img_t0 / 16.0
        image = raw_to_rgb(img_t0.unsqueeze(0).unsqueeze(0), cfa=CFA.RG)
        image = rgb_to_grayscale(image).squeeze(0)

        if self.transform:
            image = image.permute(1, 2, 0).numpy()
            augmented = self.transform(image=image)
            image = augmented["image"]
        
        label = self.labels[idx].copy()
        label = label[["timestamp", "ty", "tz", "tx", "qy", "qz", "qx", "qw"]]

        training_data = {
            "image": image,
            "first_exposure": torch.Tensor([target_exposure_t0]),
            "next_brackets": bracket_images_t1,
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

            for index in exposure_df.index[:-1]:
                optimal_exposure_t0 = exposure_df["exposure_time"].iloc[index]
                brackets_t0 = images_df.iloc[:, index].dropna().values
                brackets_t1 = images_df.iloc[:, index + 1].dropna().values

                labels.append(lidar_df)
                training_data.append((brackets_t0, brackets_t1, optimal_exposure_t0))

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