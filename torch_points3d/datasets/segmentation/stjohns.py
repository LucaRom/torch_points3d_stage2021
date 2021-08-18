import os
import numpy as np
import logging
import torch
import laspy
import random
import os.path as osp
import omegaconf

from torch_geometric.data import InMemoryDataset, Dataset, Data
from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.core.data_transform.transforms import RandomSphere, GridSphereSampling
from torch_points3d.core.data_transform.grid_transform import GridSampling3D
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker

log = logging.getLogger(__name__)

################################### Datasets general notes and structure ###################################
"""
    The dataset is defided in 3 main class : 
        - stjohns
        - stjohnsWrapper
        - 
        
    - stjonhs2021 is the main one ... it stores the basic dataset ... etc etc...
    - the second one sampleds the first one as needed
    - the third one is a final wrapper that created the different dataset as needed in the framework..

    NOTES : 
"""


################################### Initial variables and paths ###################################

# Settings paths to raw dataset and splitting into train/val/test
DIR = os.path.dirname(os.path.realpath(__file__))
dataroot = os.path.join(DIR + "/../../../data/st_john_data")

# Full raw dataset
train_list = [f for f in os.listdir(os.path.join(dataroot, "train")) if f.endswith('.las')]
val_list = [f for f in os.listdir(os.path.join(dataroot, "val")) if f.endswith('.las')]
test_list = [f for f in os.listdir(os.path.join(dataroot, "test")) if f.endswith('.las')]

# Small dataset for debug/tests
train_list_small = random.choices(train_list, k=2)
val_list_small = random.choices(val_list, k=1)
test_list_small = random.choices(test_list, k=1)

################################### Config files integration ###################################
# Calling overrided conf .yaml file to use extra parameters
"""
    This is a quick workaround to pass arguments/parameters from the config file
    
    NOTES : The config file called is the overrided one from the OUTPUT folder of the current training/eval/etc run
            This way you can call parameters from all conf files since they are grouped in the output conf, but more
            importantly, it allows to use overrided parameters passed to a specific job (ie, from the command line).
            THEREFORE following lines and called parameters/arguments will raise errors if called outside of a run.          
"""

# output_dir = os.getcwd()
# config_path = os.path.join(output_dir, ".hydra/config.yaml")
# config_cfg = omegaconf.OmegaConf.load(config_path)
# #config_data = config_cfg.data

################################### Fonctions and Utils ###################################


################################### Memory dataset Main Class ###################################

class stjohns2021(InMemoryDataset):
    """
    Class to handle stjohns dataset for segmentation task.
    Most of the methods are inherited from the InMemoryDataset (in_memory_dataset.py) which also inherit methods
    from Dataset (dataset.py) wich is based on pytorch dataset class.

    NOTES : - If a method is called, it's either an existing one that is being override (from the parent class) or a
            new one created for the dataset.
            - Remember that Errors may arise from methods called by default in parent classes.
    """

    def __init__(self, root, split=None, radius=None, grid_size_d=None, transform=None, pre_transform=None, pre_filter=None):

        # Variable are set in the class from the arguments so the can be called in the class' methods
        self._split = split
        self._radius = radius
        self._grid_size_d = grid_size_d

        # log infos are printed in the output log file of each run
        log.info(f"Actual split is {self._split}")
        log.info(f"GridSphere sampling parameters : Radius = {self._radius}, Grid = {self._grid_size_d}")
        log.info(f"Parameter first_subsampling is set to : {config_cfg.data.first_subsampling}")
        log.info(f"Number of files : {len(las_list)} using file from {config_cfg.data.processed_folder_name}")

        # super() specifies methods inherited from the superclass (here "InMemoryDataset")
        super().__init__(root, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)

        # This loads existing processed files (it runs after the process method when they do not exist yet)
        # processed_paths are defined in the "processed_file_names(self)" method below
        if split == "train":
            path = self.processed_paths[0]
        elif split == "val":
            path = self.processed_paths[1]
        elif split == "test":
            path = self.processed_paths[2]
        else:
            raise ValueError("Split %s not recognised" % split)

        # Calls the _load_data method to load selected path
        self._load_data(path)

    @property
    def processed_dir(self):
        """
        Set the name of the processed files folder
        Here it is fetched from the config parameter "processed_folder_name" in stjohns.yaml
        """
        return osp.join(self.root, config_cfg.data.processed_folder_name)

    @property
    def raw_file_names(self):
        """
        This method returns a list of file names for the raw files needed for the dataset. If file names are missing, it
        will trigger the download method. Since the dataset is managed manually outside the class, the method is passed.

        NOTES : This method is needed for the dataset to work, that's why 'pass' is used.
        """
        pass

    @property
    def processed_file_names(self):
        """
        This method defines the name of the processed datasets that are processed and goup in a .pt file. Those file
        names also define the processed_paths list (starting from 0).
        """
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        """
        This method would be trigger by an incomplete list of raw files (raw_file_names). Since the raw files are
        managed manually, this method is passed.

        NOTES : This method is needed for the dataset to work, that's why 'pass' is used.
        """
        pass

    def process(self):
        """
        This method gives the instructions to handle the raw data and process it if they are not already according to
        the specific dataset (_split). Each .las file is processed and group to respective dataset in the .pt format
        (train.pt, val.pt, test.pt).

        Information is extracted from .las file to feed the Data(pos, x, y) class where :
            pos = x, y, z (coordinates)
            x = list of features (if any)
            y = labels (if any)

        Note that the x and y arguments from the Data() class are not related to the x and y coordinates needed for
        pos.
        """

        if self._split == "train":
            pp_paths = self.processed_paths[0]
            current_split = "train"
            raw_list = train_list_small
        elif self._split == "val":
            pp_paths = self.processed_paths[1]
            current_split = "train"
            raw_list = val_list_small
        elif self._split == "test":
            pp_paths = self.processed_paths[2]
            current_split = "train"
            raw_list = test_list_small
        else:
            raise ValueError("Split %s not recognised" % self._split)

        self.split_process(self, current_split=current_split, raw_list=raw_list, pp_paths=pp_paths)

    @property
    def num_classes(self):
        return 9

    def _save_data(self, data_list, pp_path):
        data, slices = self.collate(data_list)
        torch.save((data, slices), pp_path)

    def _load_data(self, path):
        self.data, self.slices = torch.load(path)

    def split_process(self, current_split, raw_list, pp_paths):
        # Samplers used in dataset creation
        # GridSphereSampling fit the point clouds to a grid and samples a sphere around the center point. The radius
        # and grid_size are fed from the dataset conf file.
        _sampler = GridSphereSampling(radius=self._radius, grid_size=self._grid_size_d, delattr_kd_tree=True,
                                      center=False)

        # The GridSampling3d resamples the dataset with the center point of a voxel of the set size. This is used to
        # reduce the dataset size
        _grid_sampler = GridSampling3D(size=0.1)

        # Check if the processed file already exist for this split, if not, proceed with the processing
        if os.path.exists(pp_paths):
            print(f"Processed file for {current_split} already exists, skipping processing")
        else:
            data_list = []
            for i in raw_list:
                # Load the .las file and extract needed data
                las_file = laspy.read(os.path.join(dataroot, current_split, i))
                las_xyz = np.stack([las_file.x, las_file.y, las_file.z], axis=1)
                las_label = np.array(las_file.classification).astype(np.int)
                y = torch.from_numpy(las_label)

                # Feed extracted data to the Data() class. Data is also resampled for train and val.
                data = Data(pos=torch.from_numpy(las_xyz).type(torch.float), y=y)

                if current_split == "train" or self._split == "val":
                    reduced_data = _grid_sampler(data.clone())  # Resampling
                    data = Data(pos=reduced_data.pos, y=reduced_data.y)
                    data_list.append(data)

                    log.info("Processed file %s, nb points = %i", i, data.pos.shape[0])

                elif current_split == "test":
                    data_samples = _sampler(data.clone())

                    # If sampler results in a list, we first remove samples with no points, else we save the whole data
                    if isinstance(data_samples, list):
                        for my_sample in data_samples:
                            if len(my_sample.y) > 0:
                                data_list.append(my_sample)
                    else:
                        data_list.append(data_samples)

                    log.info("Processed file %s, nb points = %i, nb samples = %i", i, data.pos.shape[0],
                             len(data_samples))
                else:
                    raise ValueError("Something is wrong in the split_process method")

            print(f"Saving full data list in {pp_paths}")
            self._save_data(data_list, pp_paths)
            print("Saving done")
