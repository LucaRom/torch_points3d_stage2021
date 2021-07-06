# import os
# import os.path as osp
#
# path = "/wspace/disk01/lidar/classification_pts/torch-points3d/data/dales/raw"
# files = os.listdir(path)
#
# toe = [os.path.splitext(x)[0] for x in files]
# print(toe)
#
# # for i in files:
# #     toe = os.path.splitext(i)[0] # 0 pour enlever l'extension, 1 pour imprimer l'extension
# #     print (toe)


import laspy
from laspy.file import File
import numpy as np

# las = laspy.read('/wspace/disk01/lidar/classification_pts/torch-points3d/data/dales/raw/5080_54400.las')

inFile = laspy.read('/wspace/disk01/lidar/classification_pts/torch-points3d/data/dales/raw/5080_54400.las')

pointformat = inFile.point_format
for spec in inFile.point_format:
    print(spec.name)

points = np.vstack((inFile['X'], inFile['Y'], inFile['Z'])).astype(np.float32).T
points_x = np.vstack((inFile.x, inFile.y, inFile.y)).astype(np.float32).T
las_label = np.reshape(inFile.classification, (len(inFile), 1))
#label = inFile.classification
#las_label = label.reshape(-1,1)
print(points)
print(points_x)
print(las_label.T)

with laspy.open('/wspace/disk01/lidar/classification_pts/torch-points3d/data/dales/raw/5080_54400.las') as f:
    print(f"Point format:       {f.header.point_format}")
    print(f"Number of points:   {f.header.point_count}")
    print(f"Number of vlrs:     {len(f.header.vlrs)}")



# import os
# from glob import glob
# import numpy as np
# import multiprocessing
# import logging
# import torch
#
# import shutil
# import os.path as osp
#
# from torch_geometric.data import InMemoryDataset, Dataset, Data
# from torch_geometric.io import read_ply
#
# from torch_points3d.datasets.base_dataset import BaseDataset
# from torch_points3d.metrics.segmentation_tracker import SegmentationTracker
#
# # Read ply file
# data = read_ply('/wspace/disk01/lidar/classification_pts/torch-points3d/data/dales/raw/5080_54435.ply')
#
# print(data)

# points = np.vstack((data['x'], data['y'], data['z'])).astype(np.float32).T
# reflectance = np.expand_dims(data['reflectance'], 1).astype(np.float32)

# log = logging.getLogger(__name__)
#
# path = "/wspace/disk01/lidar/classification_pts/torch-points3d/data/dales/raw"
# files = os.listdir(path)
#
# toe = [os.path.splitext(x)[0] for x in files]
#
#
# class ply_dales(InMemoryDataset):
#     def __init__(self, transform=None, pre_transform=None, pre_filter=None):
#         super().__init__(transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
#         path = "/wspace/disk01/lidar/classification_pts/torch-points3d/data/dales/raw/"
#         #path = self.processed_paths[0] if train else self.processed_paths[1]
#
#         print(self.processed_paths[0])
#         print(self.processed_paths[1])
#
#         self.data, self.slices = torch.load(path)
#
#         print(self.data)
#         print(torch.load(path))
#     # @property
    # def raw_file_names(self) -> str:
    #     return 'MPI-FAUST.zip'
    #
    #
    # @property
    # def processed_file_names(self) -> List[str]:
    #     return ['training.pt', 'test.pt']
    #
    #
    # def download(self):
    #     raise RuntimeError(
    #         f"Dataset not found. Please download '{self.raw_file_names}' from "
    #         f"'{self.url}' and move it to '{self.raw_dir}'")

    #
    # def process(self):
    #     # extract_zip(self.raw_paths[0], self.raw_dir, log=False)
    #     #
    #     # path = osp.join(self.raw_dir, 'MPI-FAUST', 'training', 'registrations')
    #     # path = osp.join(path, 'tr_reg_{0:03d}.ply')
    #     data_list = []
    #     for i in range(100):
    #         data = read_ply(path.format(i))
    #         data.y = torch.tensor([i % 10], dtype=torch.long)
    #         if self.pre_filter is not None and not self.pre_filter(data):
    #             continue
    #         if self.pre_transform is not None:
    #             data = self.pre_transform(data)
    #         data_list.append(data)
    #
    #     torch.save(self.collate(data_list[:80]), self.processed_paths[0])
    #     torch.save(self.collate(data_list[80:]), self.processed_paths[1])
    #
    #     shutil.rmtree(osp.join(self.raw_dir, 'MPI-FAUST'))

# ply_dales()