import os
from glob import glob
import numpy as np
import multiprocessing
import logging
import torch
import laspy
import time
import random
import os.path as osp
import omegaconf

from multiprocessing import Pool, set_start_method, get_context, Process
from functools import partial


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

# Calling dataset conf file to use extra parameters
# This is a quick workaroud to pass argument from the config file and could/should be recoded so it is passed from
# the class
dales_confpath = os.path.join(DIR, "..", "..", "..", "conf", "data", "segmentation", "dales.yaml")
dales_cfg = omegaconf.OmegaConf.load(dales_confpath)


if dales_cfg.raw_folder_param == "small":
    dir_raw = "raw"
else:
    dir_raw = "raw (full)"

dataroot = os.path.join(DIR, "..", "..", "..", "data", "dales", dir_raw)

print(f"this is my dataroot {dataroot}")
# path = "/wspace/disk01/lidar/classification_pts/torch-points3d/data/dales/raw"  # à remplacer avec chemmin hydra ou points3D

train_list = [f for f in os.listdir(os.path.join(dataroot, "train")) if
            f.endswith('.las')]  # liste des fichiers avec extension .las
train_num = [os.path.splitext(x)[0] for x in train_list]  # liste des fichiers sans extension

#val_list = [f for f in os.listdir(os.path.join(dataroot, "val")) if
#             f.endswith('.las')]  # liste des fichiers avec extension .las
#val_num = [os.path.splitext(x)[0] for x in val_list]  # liste des fichiers sans extension

test_list = [f for f in os.listdir(os.path.join(dataroot, "test")) if
            f.endswith('.las')]  # liste des fichiers avec extension .las
test_num = [os.path.splitext(x)[0] for x in test_list]  # liste des fichiers sans extension

#las_list = train_list + val_list + test_list
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

# print(newcfg)
# print("is this myu life?")

# print(dales_num_names)

#print(f"from data yaml {grid_test}, {radius_test}, {processed_name_custom}")


def random_samples(data, i):
    print(f"picking sample number {i}")
    my_second_sampler = RandomSphere(radius=10, strategy="freq_class_based")
    my_samples_sample = my_second_sampler(data)
    if len(my_samples_sample.y) > 1000:
        return my_samples_sample
        # data_list.append(my_samples_sample)


################################### Memory dataset Main Class ###################################

class Dales(InMemoryDataset):
    """
    Class to handle DALES dataset for segmentation task.
    """

    def __init__(self, root, split=None, radius=None, grid_size_d=None, transform=None, pre_transform=None, pre_filter=None):

        self._split = split
        self._radius = radius
        self._grid_size_d = grid_size_d

        log.info(f"Actual split is {self._split}")
        log.info(f"GridSphere sampling parameters : Radius = {self._radius}, Grid = {self._grid_size_d}")
        log.info(f"Parameter first_subsampling is set to : {dales_cfg.first_subsampling}")
        log.info(f"Number of processed files (train + test): {len(las_list)} using file from {dales_cfg.processed_folder_name}")

        super().__init__(root, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)

        # Processed path : a mieux expliquer pour final -> doit pointer vert les dossier "processed_ba"
        # les processed_ba path sont probablement créé par la section "processed_ba file names" qui retourne actuellement train.pt et test.pt

        # # Load the appropriate .pt file according to the wrapper class argument (DalesDataset())
        # if split == "train":
        #     self.data, self.slices = torch.load(self.processed_paths[0])
        # elif split == "test":
        #     self.data, self.slices = torch.load(self.processed_paths[1])
        # else:
        #     raise ValueError("Split %s not recognised" % split)

        # Load the appropriate .pt file according to the wrapper class argument (DalesDataset())

        if split == "train":
            path = self.processed_paths[0]
        elif split == "val":
            path = self.processed_paths[1]
        elif split == "test":
            path = self.processed_paths[2]
        else:
            raise ValueError("Split %s not recognised" % split)

        self._load_data(path)

    @property
    def processed_dir(self):
        """
        Inherited from Dataset class in dataset.py
        Set the name of the processed_ba files folder
        """
        return osp.join(self.root, dales_cfg.processed_folder_name)

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
        Ici on définit le nom des fichier processed_ba qui deviendront les "processed_paths" appelé plus haut

        """
        #return ['train.pt', 'val.pt', 'test.pt']
        return ['train.pt', 'val.pt', 'test.pt'] #val.pt is not neeeded since we use test dataset for it

    def download(self):
        # pas besoin de downloader, mais il faut appeler la fonction quand même
        pass

    def process(self):
        """
        On parcours la listw des fichiers "raw" qui ne sont pas "processed_ba" et on les traite pour les
        sauvegarder en format .pt dans le dossier processed_ba

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

        #Creating the sampler for each set
        my_sampler = GridSphereSampling(radius=self._radius, grid_size=self._grid_size_d, delattr_kd_tree=True,
                                        center=False)
        my_second_sampler = RandomSphere(radius=10, strategy="freq_class_based")

        # Check if the processed file already exist for this split, if not, proceed with the processing
        if self._split == "train":
            if os.path.exists(self.processed_paths[0]):
                print(f"Processed file for {self._split} already exists, skipping processing")
            else:
                data_list = []
                for idx, filenum in enumerate(train_num, 1):
                    las_file = laspy.read(os.path.join(dataroot, "train", "{}.las".format(filenum)))
                    # print(las_file)

                    las_xyz = np.stack([las_file.x, las_file.y, las_file.z], axis=1)

                    las_label = np.array(las_file.classification).astype(np.int)
                    # print(las_xyz)
                    y = torch.from_numpy(las_label)
                    # y = self._remap_labels(y)
                    data = Data(pos=torch.from_numpy(las_xyz).type(torch.float), y=y)

                    total_sample_needed = 10000
                    need_per_i = int(total_sample_needed/len(train_num))

                    # def random_samples():
                    #     my_samples_sample = my_second_sampler(data)
                    #     if len(my_samples_sample.y) > 1000:
                    #         return my_samples_sample
                    #         #data_list.append(my_samples_sample)

                    while len(data_list) < int(need_per_i*idx):
                        my_samples_sample = my_second_sampler(data)
                        if len(my_samples_sample.y) > 1000:
                            data_list.append(my_samples_sample)
                        #print(len(data_list))

                    # print(f"this is data {data}")
                    # print(f"this is sampler {data_sample}")

                    #Removing samples with length zero
                    # for my_sample in data_samples:
                    #     if len(my_sample.y) > 1000:
                    #         data_list.append(my_sample)

                    #log.info("Processed file %s, nb points = %i, nb samples = %i", i, data.pos.shape[0], len(data_samples))
                    log.info("Processed file %s, nb points = %i", filenum, data.pos.shape[0])

                #print(data_list)

                #data, slices = self.collate(data_list)
                #torch.save((data, slices), self.processed_paths[0])
                self._save_data(data_list, self.processed_paths[0])

        # Check if the processed file already exist for this split, if not, proceed with the processing
        elif self._split == "val":
            if os.path.exists(self.processed_paths[1]):
                print(f"Processed file for {self._split} already exists, skipping processing")
            else:
                data_list = []
                for idx, filenum in enumerate(test_num, 1):
                    las_file = laspy.read(os.path.join(dataroot, "test", "{}.las".format(filenum)))
                    # print(las_file)

                    las_xyz = np.stack([las_file.x, las_file.y, las_file.z], axis=1)

                    las_label = np.array(las_file.classification).astype(np.int)
                    y = torch.from_numpy(las_label)
                    data = Data(pos=torch.from_numpy(las_xyz).type(torch.float), y=y)

                    total_sample_needed = 2000
                    need_per_i = int(total_sample_needed/len(train_num))

                    # def random_samples():
                    #     my_samples_sample = my_second_sampler(data)
                    #     if len(my_samples_sample.y) > 1000:
                    #         return my_samples_sample
                    #         #data_list.append(my_samples_sample)

                    while len(data_list) < int(need_per_i*idx):
                        my_samples_sample = my_second_sampler(data)
                        if len(my_samples_sample.y) > 1000:
                            data_list.append(my_samples_sample)
                        #print(len(data_list))

                    # log.info("Processed file %s, nb points = %i, nb samples = %i", i, data.pos.shape[0], len(data_samples))
                    log.info("Processed file %s, nb points = %i", filenum, data.pos.shape[0])

                self._save_data(data_list, self.processed_paths[1])

        # Not necessary to skip the last split, since whole method is skipped is all files (processed_paths) exist
        elif self._split == "test":
            data_list = []
            for i in test_num:
                las_file = laspy.read(os.path.join(dataroot, "test", "{}.las".format(i)))
                # print(las_file)

                las_xyz = np.stack([las_file.x, las_file.y, las_file.z], axis=1)

                las_label = np.array(las_file.classification).astype(np.int)
                y = torch.from_numpy(las_label)
                data = Data(pos=torch.from_numpy(las_xyz).type(torch.float), y=y)

                data_list.append(data)

                #log.info("Processed file %s, nb points = %i, nb samples = %i", i, data.pos.shape[0], len(data_samples))
                log.info("Processed file %s, nb points = %i", i, data.pos.shape[0])


            self._save_data(data_list, self.processed_paths[2])

        else:
            raise ValueError("Split %s not recognised" % split)

    @property
    def num_classes(self):
        return 9

    def _save_data(self, data_list, pp_path):
        data, slices = self.collate(data_list)
        torch.save((data, slices), pp_path)

    def _load_data(self, path):
        self.data, self.slices = torch.load(path)

class DalesSampled(Dales):
    """
    Class to handle DALES dataset for segmentation task.
    """

    # Actually, radius and grid_size_d are not used for the moment
    def __init__(self, root, sample_per_epoch=None, radius=None, grid_size_d=None, split=None, transform=None,
                 pre_transform=None, pre_filter=None):

        self._split = split
        self._sample_per_epoch = sample_per_epoch
        self._radius = radius
        self._grid = grid_size_d

        super().__init__(root, split=split, radius=radius, grid_size_d=grid_size_d, transform=transform,
                         pre_transform=pre_transform, pre_filter=pre_filter)
        #super().__init__(root, *args, **kwargs)

    def __len__(self):
        # if self._sample_per_epoch > 0:
        #     return self._sample_per_epoch
        # else:
        #     return len(self._datas)
        return len(self._datas)

    def get(self, idx):

        #my_new_sampler = RandomSphere(radius=15, strategy="freq_class_based")

        #my_random_data = my_new_sampler(self._datas)
        #print(my_random_data)

        #random_idx = random.randint(0, len(self._datas)-1)

        #print(random_idx)
        #print(self._datas[random_idx])
        return self._datas[idx].clone()
        #return my_random_data

    # def __get__(self, idx):
    #     return self.data23[idx], self.slices[idx]

    def process(self):  # We have to include this method, otherwise the parent class skips processing
        super().process()

    def download(self):  # We have to include this method, otherwise the parent class skips download
        super().download()

    def _save_data(self, data_list, pp_path):
        #data, slices = self.collate(data_list)
        torch.save((data_list), pp_path)

    def _load_data(self, path):
        self._datas = torch.load(path)

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
        #     split="val",
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

class DalesSphere(BaseDataset):
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

        my_new_sampler = RandomSphere(radius=15, strategy="freq_class_based")  ##NEW

        self.train_dataset = DalesSampled(
            self._data_path,
            split="train",
            sample_per_epoch=200, #-1 for all
            radius=dales_cfg.radius_param,
            grid_size_d=dales_cfg.grid_param,
            transform=self.train_transform,
            pre_transform=self.pre_transform,
        )

        self.val_dataset = DalesSampled(
            self._data_path,
            split="val",
            sample_per_epoch=200, #-1 for all
            radius=dales_cfg.radius_param,
            grid_size_d=dales_cfg.grid_param,
            transform=self.val_transform,
            pre_transform=self.pre_transform,
        )

        self.test_dataset = DalesSampled(
            self._data_path,
            split="test",
            sample_per_epoch=-1,  #for all
            radius=dales_cfg.radius_param,
            grid_size_d=dales_cfg.grid_param,
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
