import os
import cv2
from tqdm import tqdm

# Set input and output directories
input_dir = f'/media/alienware/T7_Shield/ICRA2024_OG/dataset/campus-11-13-2024/data_high_resolution'
output_dir = '/home/alienware/Desktop/tmp2'
bracketing_values = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
# bracketing_values = [0.025, 0.1, 0.4, 1.6, 6.4, 25.6]  # Adjusted values for lower resolution

# Set the desired lower resolution (width, height)
new_size = (640, 480)

# Supported image extensions
image_extensions = ('.png',)

for traj in sorted(os.listdir(input_dir)):

    print(f"Processing trajectory: {traj}")
    traj_path = os.path.join(input_dir, traj, 'camera_left')
    if not os.path.isdir(traj_path):
        print(f"Skipping {traj_path}, not a directory.")
        continue

    src_txt = os.path.join(traj_path, 'optimal_exposure_times.txt')
    dst_txt = os.path.join(output_dir, traj, 'optimal_exposure_times.txt')
    os.makedirs(os.path.dirname(dst_txt), exist_ok=True)
    try:
        with open(src_txt, 'r') as f_src, open(dst_txt, 'w') as f_dst:
            f_dst.write(f_src.read())
    except Exception as e:
        print(f"Failed to copy {src_txt} to {dst_txt}: {e}")
    for bracket in bracketing_values:
        path = os.path.join(traj_path, str(bracket))
        output = os.path.join(output_dir, traj, str(bracket))
        os.makedirs(output, exist_ok=True)
        if not os.path.exists(path):
            print(f"Directory for bracket {bracket} does not exist: {path}")
            continue
        for filename in tqdm(os.listdir(path)):
            if filename.lower().endswith(image_extensions):
                input_path = os.path.join(path, filename)
                output_path = os.path.join(output, filename)
                try:
                    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        print(f"Failed to read {input_path}")
                        continue
                    resized_img = cv2.resize(img, new_size)
                    cv2.imwrite(output_path, resized_img)
                except Exception as e:
                    print(f"Failed to process {input_path}: {e}")