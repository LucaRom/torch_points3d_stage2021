import os
import numpy as np
import logging
import torch
import laspy
import random
import os.path as osp
import omegaconf
from scipy import stats
import numpy as np

from tqdm import tqdm
import multiprocessing

from torch_geometric.data import InMemoryDataset, Data
from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.core.data_transform.transforms import RandomSphere, GridCylinderSampling, GridSphereSampling, RandomCylinder
from torch_points3d.core.data_transform.grid_transform import GridSampling3D
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker


# Settings paths to raw dataset and splitting into train/val/test
DIR = os.path.dirname(os.path.realpath(__file__))
dataroot = os.path.join(DIR + "/../../data/stjohns/raw")

# Full raw dataset
files_list = [f for f in os.listdir(os.path.join(dataroot, "train")) if f.endswith('.las')]
files_list_small = random.choices(files_list, k=2)

# Samplers
# gss_sampler = GridSphereSampling(radius=gss_radius, grid_size=gss_grid, delattr_kd_tree=True,
#                                  center=False)

rs_sampler = RandomSphere(radius=10, strategy="freq_class_based")
rc_sampler = RandomCylinder(radius=10, strategy="freq_class_based")
#
# _grid_sampler = GridSampling3D(size=gs3d_grid)

# List


def points_list_create(data):
    new_data = rs_sampler(data)
    nb_points = len(new_data.y)
    pts_len_list.append(nb_points)
    print(pts_len_list)

pts_len_list = []
for i in files_list:
    # Load the .las file and extract needed data
    las_file = laspy.read(os.path.join(dataroot, "train", i))
    las_xyz = np.stack([las_file.x, las_file.y, las_file.z], axis=1)
    las_label = np.array(las_file.classification).astype(np.int)
    y = torch.from_numpy(las_label)
    # y = self._remap_labels(y)  # Remapping label necessary is not [0, n] already

    # Feed extracted data to the Data() class. Data is also resampled for train and val.
    data = Data(pos=torch.from_numpy(las_xyz).type(torch.float), y=y)

    needed_range = 15000 // len(files_list)

    for j in tqdm(range(needed_range)):
        new_data = rs_sampler(data)
        nb_points = len(new_data.y)
        pts_len_list.append(nb_points)

print(pts_len_list)

z = np.abs(stats.zscore(pts_len_list))
print(z)

out_index = list(np.where(z > 3)[0])
print(out_index)

outliers = []
for out in out_index:
    outliers.append(pts_len_list[out])
print(outliers)

with open('outliers_output.txt','w') as f:
    for num in pts_len_list:
        f.write(str(num) + ', ')

with open('outliers_num_points.txt','w') as f:
    for num in outliers:
        f.write(str(num) + ', ')

# from scipy import stats
# import numpy as np
#
# outliers_saved_path = '/export/sata01/wspace/lidar/classification_pts/torch-points3d/torch_points3d/utils/outliers_output.txt'
# loaded_outliers = open(outliers_saved_path).read()
# int_list = [x.strip() for x in loaded_outliers.split(',') if len(x) > 1]
# print(int_list)
# # for i in loaded_outliers:
# #     int_list.append(i)
# outliers_list = [int(i) for i in int_list]
# print(outliers_list)
#
# z = np.abs(stats.zscore(outliers_list))
# print(z)
#
# out_index = list(np.where(z > 2)[0])
# print(out_index)
#
# outliers = []
# for out in out_index:
#     outliers.append(outliers_list[out])
#
# print(f"list of outliers : {outliers}")
# print(f"total of samples : {len(outliers_list)}")
# print(f"number of outliers : {len(outliers)}")
# print(f"outliers ration : {len(outliers) / len(outliers_list)}")
# print(f"min outlier : {min(outliers)}")