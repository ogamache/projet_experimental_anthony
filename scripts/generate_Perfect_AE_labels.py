import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from tqdm import tqdm
from pathlib import Path

from BorealHDR.scripts.emulator_threads import create_dataframe
from BorealHDR.scripts.classes.class_image_emulator import Image_Emulator
from scipy.interpolate import interp1d

PATH_BRACKETING_IMGS_LEFT = Path("/media/alienware/T7_Shield/ICRA2024_OG/dataset/forest-04-21-2023/data_high_resolution")
BRACKETING_VALUES = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
# BRACKETING_VALUES = np.array([0.025, 0.1, 0.4, 1.6, 6.4, 25.6])

def approximate_exposure_time(images_list):
    averages = []
    for img in images_list:
        averages.append(np.mean(img))
    averages = np.array(averages)
    f = interp1d(averages, BRACKETING_VALUES, kind='linear', fill_value="extrapolate")
    exposure_time = f(2048.0)
    return exposure_time

def iterative_emulation(emulator_class, initial_exposure_time, kp=0.001, tolerance=0.1, max_iterations=10):
    exposure_time = initial_exposure_time
    error = float("inf")
    iterations = 0
    while abs(error) > tolerance and iterations < max_iterations:
        # print(f"Current error: {error}, Current exposure time: {exposure_time}")
        emulated_image = emulator_class.emulate_image(exposure_time)
        average_intensity = np.mean(emulated_image["emulated_img"])
        error = 2048 - average_intensity
        exposure_time *= (error * kp + 1)  # Increase exposure time
        iterations += 1
    return exposure_time

def main():

    for traj in sorted(os.listdir(PATH_BRACKETING_IMGS_LEFT)):
        print(f"Processing trajectory: {traj}")
        images_path = PATH_BRACKETING_IMGS_LEFT / traj / "camera_left"
        if not images_path.is_dir():
            continue
        SAVE_PATH = images_path / "optimal_exposure_times.txt"
        if SAVE_PATH.exists():
            os.remove(SAVE_PATH)
        dataframe_left = create_dataframe(images_path, BRACKETING_VALUES)

        emulator_left_class = Image_Emulator(images_path, "radiance", "closer_least_sat", False)
        for index in tqdm(range(0, dataframe_left.shape[1]-1)): # Should remove the -1 ...
            emulator_left_class.update_image_list(dataframe_left.loc[:][index].to_list())
            initial_exposure_time = approximate_exposure_time(emulator_left_class.bracket_images)
            optimal_exposure_time = iterative_emulation(emulator_left_class, initial_exposure_time)

            with open(images_path /"optimal_exposure_times.txt", "a") as f:
                f.write(f"{optimal_exposure_time}\n")

if __name__ == "__main__":
    main()
