import os
from glob import glob
import numpy as np
import multiprocessing
import logging
import torch
import laspy
import time
import random

from functools import partial

from multiprocessing import Pool, set_start_method, get_context, Process
#set_start_method("spawn")


from torch_geometric.data import InMemoryDataset, Dataset, Data
from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.core.data_transform.transforms import RandomSphere, GridSphereSampling
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker

log = logging.getLogger(__name__)

# reference for dales dataset : https://arxiv.org/abs/2004.11985
# The dataset must be downloaded at go.udayton.edu/dales3d.

################################### Utils ###################################


################################### Initial variables and paths ###################################

DIR = os.path.dirname(os.path.realpath(__file__))
dataroot = os.path.join(DIR, "..", "..", "..", "data", "dales", "raw")

train_list = [f for f in os.listdir(os.path.join(dataroot, "train")) if
            f.endswith('.las')]  # liste des fichiers avec extension .las
train_num = [os.path.splitext(x)[0] for x in train_list]  # liste des fichiers sans extension

test_list = [f for f in os.listdir(os.path.join(dataroot, "test")) if
            f.endswith('.las')]  # liste des fichiers avec extension .las
test_num = [os.path.splitext(x)[0] for x in test_list]  # liste des fichiers sans extension

las_list = train_list + test_list
#print(f"this is las list : {las_list}")

# Dict from labels to names
dales_class_names = {0: 'unknown',
                     1: 'Ground',
                     2: 'Vegetation',
                     3: 'Cars',
                     4: 'Trucks',
                     5: 'Power lines',
                     6: 'Fences',
                     7: 'Poles',
                     8: 'Buildings'}

dales_num_classes = len(dales_class_names)
#
# print(dales_num_names)

################################### Memory dataset Main Class ###################################

class Dales(InMemoryDataset):
    """
    Class to handle DALES dataset for segmentation task.
    """

    def __init__(self, root,  sample_per_epoch=10000, radius=15, grid_size_d=15, split=None, transform=None,
                 pre_transform=None, pre_filter=None):

        self._split = split
        self._sample_per_epoch = sample_per_epoch
        self._radius = radius
        self._grid = grid_size_d

        super().__init__(root, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)

        # Processed path : a mieux expliquer pour final -> doit pointer vert les dossier "processed"
        # les processed path sont probablement créé par la section "processed file names" qui retourne actuellement train.pt et test.pt

        # Load the appropriate .pt file according to the wrapper class argument (DalesDataset())
        if split == "train":
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif split == "test":
            self.data, self.slices = torch.load(self.processed_paths[1])
        else:
            raise ValueError("Split %s not recognised" % split)

    @property
    def raw_file_names(self):
        """
        La fonction vérifie la liste des données à télécharger, puisque nos données sont déjà téléchargées,
        on réfère simplement à la liste des fichiers .las
        """
        return las_list

    @property
    def processed_file_names(self):
        """
        Ici on définit le nom des fichier processed qui deviendront les "processed_paths" appelé plus haut

        """
        return ['train.pt', 'test.pt']

    def download(self):
        # pas besoin de downloader, mais il faut appeler la fonction quand même
        pass

    def process(self):
        """
        On parcours la listw des fichiers "raw" qui ne sont pas "processed" et on les traite pour les
        sauvegarder en format .pt dans le dossier processed

        Chaque fichier .las devient un fichier .pt qui est alimenté par la fonction Data(pos= , x= et y)
        ou pos = x, y, z , x = liste des features et y les labels

        """

        ### Create train dataset
        """
        First part load the point cloud in .las format and store as tensor data

        Second part create subsamples (according to the range) using the RandomSphere fonction from torch-points3d
        The subsampling range will give the number of total sample available for batching per .las file
        
        While spawning can be slower than forking from pools, it avoids alot of 'hanging' problem encounter while simply 
        using Pools in multiprocessing. It is still alot faster than no multiprocessing at all. It also seems to make 
        the logging more complex. As a workaround, a print fonction was placed in the fonction, but the log should be 
        fix if possible.

        """

        if self._split == "train":
            data_list = []
            for i in train_num:
                las_file = laspy.read(os.path.join(dataroot, "train", "{}.las".format(i)))
                # print(las_file)

                las_xyz = np.stack([las_file.x, las_file.y, las_file.z], axis=1)

                las_label = np.array(las_file.classification).astype(np.int)
                # print(las_xyz)
                y = torch.from_numpy(las_label)
                # y = self._remap_labels(y)
                data = Data(pos=torch.from_numpy(las_xyz).type(torch.float), y=y)

                # Calling sampler
                my_sampler = GridSphereSampling(radius=15, grid_size=20, delattr_kd_tree=True, center=True)
                data_samples = my_sampler(data.clone())  # create a whole list of samples
                # data_samples = sampler(data) # create a whole list of samples

                # print(f"this is data {data}")
                # print(f"this is sampler {data_sample}")

                # Removing samples with length zero
                for my_sample in data_samples:
                    if len(my_sample.y) > 0:
                        data_list.append(my_sample)

                log.info("Processed file %s, nb points = %i, nb samples = %i", i, data.pos.shape[0],
                         len(data_samples))

            # print(data_list)

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])

        elif self._split == "test":
            data_list = []
            for i in test_num:
                las_file = laspy.read(os.path.join(dataroot, "test", "{}.las".format(i)))
                # print(las_file)

                las_xyz = np.stack([las_file.x, las_file.y, las_file.z], axis=1)

                las_label = np.array(las_file.classification).astype(np.int)
                # print(las_xyz)
                y = torch.from_numpy(las_label)
                # y = self._remap_labels(y)
                data = Data(pos=torch.from_numpy(las_xyz).type(torch.float), y=y)

                # Calling sampler
                my_sampler = GridSphereSampling(radius=15, grid_size=20, delattr_kd_tree=True, center=True)
                data_samples = my_sampler(data.clone())  # Creates a whole list of samples
                # data_samples = sampler(data)  # Creates a whole list of samples

                # print(f"this is data {data}")
                # print(f"this is sampler {data_sample}")

                # Removing samples with length zero
                for my_sample in data_samples:
                    if len(my_sample.y) > 0:
                        data_list.append(my_sample)

                log.info("Processed file %s, nb points = %i, nb samples = %i", i, data.pos.shape[0],
                         len(data_samples))

            # print(data_list)

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[1])

        else:
            raise ValueError("Split %s not recognised" % split)

    @property
    def num_classes(self):
        return 9


### Dataset pour sampling
class DalesSphere(Dales):
    """
    Class to handle DALES dataset for segmentation task.
    """

    """ Wrapper around Dales that creates train and test datasets.
    Parameters
    ----------
    dataset_opt: omegaconf.DictConfig
        Config dictionary that should contain
            - root,
            - transform,
            - pre_transform
            - process_workers
    """

    def __init__(self, root, sample_per_epoch=10000, radius=15, grid_size_d=15, *args, **kwargs):
        self._samples_per_epoch = sample_per_epoch
        self._radius = radius
        self._grid = grid_size_d

        super().__init__(root, *args, **kwargs)

        self.train_dataset = Dales(
            self._data_path,
            split="train",
            transform=self.train_transform,
            pre_transform=self.pre_transform,
        )

        self.test_dataset = Dales(
            self._data_path,
            split="test",
            transform=self.test_transform,
            pre_transform=self.pre_transform,
        )

    def __len__(self):
        if self._samples_per_epoch > 0:
            return self._samples_per_epoch
        else:
            return len(self._test_spheres)

    def get(self, idx):
        if super()._split == "train":
            if self._sample_per_epoch > 0:
                return self._get_random(self.train_dataset)
            else:
                return self.train_dataset[idx].clone()
        elif super()._split == "test":
            if self._sample_per_epoch > 0:
                return self._get_random(self.test_dataset)
            else:
                return self.test_dataset[idx].clone()
        else:
            print("Nothing worked, we are doomed")

    def process(self):  # We have to include this method, otherwise the parent class skips processing
        super().process()

    def download(self):  # We have to include this method, otherwise the parent class skips processing
        super().download()

    def _get_random(self, random_split):
        # Random spheres biased towards getting more low frequency classes
        # gridsphere_sampler = cT.SphereSampling(self._radius, centre[:3], align_origin=False)
        my_gridsampler = GridSphereSampling(radius=self._radius, grid_size=self._grid, delattr_kd_tree=True,
                                            center=True)
        return my_gridsampler(random_split)

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker
        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        return SegmentationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)


class DalesDataset(BaseDataset):
    """ Wrapper around Dales that creates train and test datasets.
    Parameters
    ----------
    dataset_opt: omegaconf.DictConfig
        Config dictionary that should contain
            - root,
            - transform,
            - pre_transform
            - process_workers
    """

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        self.train_dataset = Dales(
            self._data_path,
            split="train",
            transform=self.train_transform,
            pre_transform=self.pre_transform,
        )

        # self.val_dataset = Dales(
        #     self._data_path,
        #     #split="val",
        #     transform=self.val_transform,
        #     pre_transform=self.pre_transform,
        # )

        self.test_dataset = Dales(
            self._data_path,
            split="test",
            transform=self.test_transform,
            pre_transform=self.pre_transform,
        )

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker
        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        return SegmentationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)
