import os
import re
from os.path import exists, join

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_folders, transform=None):
        """
        Args:
            data_folder: The folder containing the dataset
            transform: The transformations to apply to the images
        """
        self.data_folder = data_folders
        self.transform = transform
        self.bracketing_values = np.array([2.0, 4.0, 8.0, 16.0, 32.0])
        self.image_paths = []
        self.labels = []

        for folder in data_folders:
            print(f"Loading data from folder {folder}")
            inputs, labels = self.load_data_from_folder(folder)
            self.image_paths.extend(inputs)
            self.labels.extend(labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image_og = cv2.imread(self.image_paths[idx], cv2.IMREAD_UNCHANGED)
        image_gray = cv2.cvtColor(image_og, cv2.COLOR_BAYER_RG2GRAY)
        image = (image_gray / 16.0).astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        
        label = self.labels[idx]

        return image, label
    
    def load_data_from_folder(self, folder):
        raise NotImplementedError("Subclasses should implement this method.")

###############################################################################################################################
class AutoExposureDataset(CustomDataset):
    def __init__(self, data_folders, transform=None):
        """
        Args:
            data_folder: The folder containing the dataset
            transform: The transformations to apply to the images
        """
        super().__init__(data_folders, transform)
    
    def load_data_from_folder(self, folder):

        image_paths = []
        labels = []
        for traj in tqdm(os.listdir(folder)):
            data_path = join(folder, traj)
            exposure_file = join(data_path, "optimal_exposure_times.txt")
            if not exists(exposure_file):
                print(f"Exposure file not found for {data_path}, skipping...")
                continue
            df = pd.read_csv(exposure_file, header=None, names=["exposure_time"])

            for bracket in self.bracketing_values:
                bracket_path = join(data_path, str(bracket))
                if not exists(bracket_path):
                    print(f"Bracket path {bracket_path} does not exist, skipping...")
                    continue
                sorted_files = sorted([f for f in os.listdir(bracket_path) if f.endswith(".png")])

                for idx in range(len(df)-1):
                    labels.append(np.log2(df["exposure_time"][idx+1]/ bracket))
                    image_paths.append(join(bracket_path, sorted_files[idx]))

        return image_paths, labels
    

###############################################################################################################################
class AutoExposureCovarianceDataset(CustomDataset):
    def __init__(self, data_folders, transform=None):
        """
        Args:
            data_folder: The folder containing the dataset
            transform: The transformations to apply to the images
        """
        super().__init__(data_folders, transform)
    
    def load_data_from_folder(self, folder):

        # TODO: Skip Nan values in label (and related images)

        image_paths = []
        labels = []
        for traj in tqdm(os.listdir(folder)):
            # if traj == "backpack_2023-04-21-09-41-22" or traj == "backpack_2023-04-21-10-46-54":
            if "2023-04-21" in traj or "2023-04-20" in traj:
                print(f"Processing trajectory: {traj}")
                data_path = join(folder, traj)
                exposure_file = join(data_path, "minimal_covariance_exposure_times_trans_video.txt")
                if not exists(exposure_file):
                    print(f"Exposure file not found for {data_path}, skipping...")
                    continue
                df = pd.read_csv(exposure_file, header=None, names=["exposure_time"])
                if df["exposure_time"].isnull().any():
                    print(f"NaN values found in {exposure_file}, skipping trajectory {traj}...")
                    continue

                ################################################################################
                # Apply exponential moving average to exposure_time values
                alpha = 0.3  # Smoothing factor, adjust as needed
                ema_exposure = []
                for idx, val in enumerate(df["exposure_time"]):
                    if idx == 0:
                        ema_exposure.append(val)
                    else:
                        ema_exposure.append(alpha * val + (1 - alpha) * ema_exposure[-1])
                df["exposure_time"] = ema_exposure
                ################################################################################

                for bracket in self.bracketing_values:
                    bracket_path = join(data_path, str(bracket))
                    if not exists(bracket_path):
                        print(f"Bracket path {bracket_path} does not exist, skipping...")
                        continue
                    sorted_files = sorted([f for f in os.listdir(bracket_path) if f.endswith(".png")])

                    for idx in range(len(df)-1):
                        if abs(df["exposure_time"][idx+1] - df["exposure_time"][idx]) < 5.0:
                            labels.append(np.log2(df["exposure_time"][idx+1]/ bracket))
                            image_paths.append(join(bracket_path, sorted_files[idx]))
                        # else:
                        #     print(f"Skipping difference is: {abs(df['exposure_time'][idx+1] - df['exposure_time'][idx])}")
            else:
                continue

        if labels:
            import matplotlib.pyplot as plt
            all_labels = np.array(labels).flatten()
            print(f"Labels distribution:")
            print(f"Min: {np.min(all_labels)}, Max: {np.max(all_labels)}, Mean: {np.mean(all_labels)}, Std: {np.std(all_labels)}")

            plt.figure(figsize=(8, 4))
            plt.hist(all_labels, bins=50, color='skyblue', edgecolor='black')
            plt.title("Distribution of Labels")
            plt.xlabel("Label Value")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.show()

        return image_paths, labels

###############################################################################################################################
class CovarianceDataset(CustomDataset):
    def __init__(self, data_folders, transform=None):
        """
        Args:
            data_folder: The folder containing the dataset
            transform: The transformations to apply to the images
        """
        super().__init__(data_folders, transform)

    def __len__(self):
        return len(self.image_paths)
    
    def load_data_from_folder(self, folder):

        image_paths = []
        labels = []
        for traj in tqdm(os.listdir(folder)):
            data_path = join(folder, traj)
            covariance_folder = join(data_path, "covariance")
            if not exists(covariance_folder):
                print(f"Covariance folder not found for {data_path}, skipping...")
                break  # Skip the whole loop for this trajectory

            for bracket in self.bracketing_values:
                img_bracket_path = join(data_path, str(bracket))
                covariance_bracket_path = join(covariance_folder, str(bracket))
                if not exists(covariance_bracket_path) or not exists(img_bracket_path):
                    print(f"Covariance or image bracket path {covariance_bracket_path} does not exist, skipping...")
                    break # Skip the whole loop for this bracket

                sorted_files = sorted([f for f in os.listdir(img_bracket_path) if f.endswith(".png")])

                for file in sorted_files:
                    try:
                        covariance = np.load(join(covariance_bracket_path, f"{file.split('.')[0]}.npy"))
                    except Exception as e:
                        print(f"Covariance file does not exist for {file} in {covariance_bracket_path}, skipping...")
                        continue
                    if np.mean(np.diag(covariance)) > 2:
                        print("################################################################################")
                        print(f"Covariance matrix for {file} has a mean diagonal value greater than 2, skipping...")
                        print(f"Mean diagonal value: {np.mean(np.diag(covariance))}")
                        print("################################################################################")
                        continue
                    # Create a writable copy of the diagonal to avoid PyTorch warnings
                    diagonal_cov = np.diag(covariance).copy().reshape(6, 1)  # Assuming covariance is 6x6
                    labels.append(diagonal_cov)
                    image_paths.append(join(img_bracket_path, file))

        # Print and display the distribution of the diagonal values in the labels
        if labels:
            import matplotlib.pyplot as plt
            print(np.array(labels).shape)
            all_diags = np.array(labels).flatten()
            print(f"Shape of all_diags: {all_diags.shape}")
            print("Labels distribution:")
            print(f"Min: {np.min(all_diags)}, Max: {np.max(all_diags)}, Mean: {np.mean(all_diags)}, Std: {np.std(all_diags)}")

            plt.figure(figsize=(8, 4))
            plt.hist(all_diags, density=False, color='skyblue', edgecolor='black', range=(np.min(all_diags), 1e-6))
            plt.title("Distribution of Diagonal Covariance Values")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.show()

        return image_paths, labels