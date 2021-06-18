# Links used to create the dataset input (To refer properly when 'final' version is made) :
# 1. https://github.com/HuguesTHOMAS/KPConv-PyTorch/issues/72 (Buildind dales dataset in KPConv PyTorch)
# 2. https://github.com/Arjun-NA/KPConv_for_DALES/blob/master/datasets/DALES.py (dataset files taken adapted form #1 in
#    a tensorflow model. Can also be found temporary in the 'dales_ref_temp.py' file from this repository.)
# 3. https://github.com/HuguesTHOMAS/KPConv-PyTorch/blob/master/datasets/S3DIS.py (S3DIS dataset file as proposed by the
#    author in #1 thread.  Can also be found temporary in the 'dales_ref_temp2.py' file from this repository.).
# -> Files in #3 was compared with s3dis.py form torch_points3d to create the DALES dataset file.


################################### IMPORTS #######################################
# From s3dis.py (torch_points3d) (to merge with upper part)
import os
import os.path as osp
from itertools import repeat, product
import numpy as np
import h5py
import torch
import random
import glob
from plyfile import PlyData, PlyElement
from torch_geometric.data import InMemoryDataset, Data, extract_zip, Dataset
from torch_geometric.data.dataset import files_exist
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
import logging
from sklearn.neighbors import NearestNeighbors, KDTree
from tqdm.auto import tqdm as tq
import csv
import pandas as pd
import pickle
import gdown
import shutil

from torch_points3d.datasets.samplers import BalancedRandomSampler
import torch_points3d.core.data_transform as cT
from torch_points3d.datasets.base_dataset import BaseDataset

DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)




import torch
from torch_geometric.data import InMemoryDataset, download_url


class DALESDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def process(self):
        # Read data into huge `Data` list.
        data_list = [...]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])