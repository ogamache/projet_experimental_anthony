import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

# Set equal aspect ratio for 3D plot
def set_axes_equal(ax):
    '''Set 3D plot axes to equal scale.'''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def to_3D(fx, fy, depth, cx, cy, u, v):
    x = (u-cx)*depth/fx
    y = (v-cy)*depth/fy
    z = depth
    x = np.expand_dims(x, axis = -1)
    y = np.expand_dims(y, axis = -1)
    z = np.expand_dims(z, axis = -1)
    return np.concatenate((x,y,z), axis=-1)

def select_calib_file(traj_name):
    if 'backpack_2023-04-20' in traj_name or 'backpack_2023-04-21' in traj_name:
        print("Using April 2023 calibration file.")
        return '../BorealHDR/calibration_files/calibration_vo/april_2023/left.yaml'
    elif 'backpack_2023-09-25' in traj_name or 'backpack_2023-09-27' in traj_name:
        print("Using September 2023 calibration file.")
        return '../BorealHDR/calibration_files/calibration_vo/september_2023/left.yaml'
    else:
        print(f"Calibration file not found for trajectory {traj_name}. Skipping.")
        return None