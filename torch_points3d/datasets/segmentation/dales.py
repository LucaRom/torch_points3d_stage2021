import os
from glob import glob
import numpy as np
import multiprocessing
import logging
import torch
import laspy
import time

from functools import partial

from multiprocessing import Pool, set_start_method, get_context
#set_start_method("spawn")


from torch_geometric.data import InMemoryDataset, Dataset, Data
from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.core.data_transform.transforms import RandomSphere
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker

log = logging.getLogger(__name__)

# reference for dales dataset : https://arxiv.org/abs/2004.11985
# The dataset must be downloaded at go.udayton.edu/dales3d.

################################### Utils ###################################

def create_samples(las_image, set_val):
    '''
    This fonction was created to facilitate multiprocessing integration. It might be faster/better to multiprocess at the
    subsampliong level only. This would mean that the first part of this fonction would be called in an iteration from
    the appropritate process section wich would call the subsampling fonction as a multiprocess process...

    :param las_image:
    :return:
    '''

    las_data_list = []
    las_file = laspy.read(os.path.join(dataroot, set_val, "{}.las".format(las_image)))

    las_file_path = os.path.join(dataroot, set_val, "{}.las".format(las_image))

    #print(f"this is las file path {las_file_path}") #debug

    las_xyz = np.stack([las_file.x, las_file.y, las_file.z], axis=1)

    las_label = np.array(las_file.classification).astype(np.int)
    # print(las_xyz)
    y = torch.from_numpy(las_label)
    # y = self._remap_labels(y)
    data = Data(pos=torch.from_numpy(las_xyz).type(torch.float), y=y)

    #print(data) #debug

    # Subsampling
    for sample_no in range(500):
        random_sphere = RandomSphere(1, strategy="RANDOM")
        data_sample = random_sphere(data.clone())

        log.info("Processed file %s, sample_no = %s nb points = %i", las_image, sample_no, data.pos.shape[0])
        print(f"Processed file {las_image}, sample_no = {sample_no} nb points = {data.pos.shape[0]}")

        las_data_list.append(data_sample)

    return las_data_list


################################### Initial variables and paths ###################################

DIR = os.path.dirname(os.path.realpath(__file__))
dataroot = os.path.join(DIR, "..", "..", "..", "data", "dales", "raw")

# path = "/wspace/disk01/lidar/classification_pts/torch-points3d/data/dales/raw"  # à remplacer avec chemmin hydra ou points3D

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

    def __init__(self, root, split=None, transform=None, pre_transform=None, pre_filter=None):

        self._split = split

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
        # Load the appropriate .pt file according to the wrapper class argument (DalesDataset())
        if self._split == "train":

            t1 = time.perf_counter()  # Start time of processing (debugging)

            ### Create train dataset
            pool = get_context("spawn").Pool()                              # Creating pools for multiprocess
            create_samples_x = partial(create_samples, set_val="train")     # Set 'static' variable in iteration
            data_list_temp = pool.map(create_samples_x, train_num)          # Mapping process with train_num range
            flat_data_list = [x for z in data_list_temp for x in z]         # Flattening list of lists from mapping

            # Create tensor and save it as "train.pt" wich is 'processed_paths[0]
            data, slices = self.collate(flat_data_list)
            torch.save((data, slices), self.processed_paths[0])

            # pool.terminate()
            # pool.join()
            # pool.close()

            t2 = time.perf_counter()  # end time of processing training data

            print(f'Processing training data finished in {t2 - t1} seconds')

        elif self._split == "test":
            pool_test = get_context("spawn").Pool()  # Creating pools for multiprocess
            create_samples_x = partial(create_samples, set_val="test")
            data_list_temp = pool_test.map(create_samples_x, test_num)  # Mapping process with test_num range
            flat_data_list = [x for z in data_list_temp for x in z]  # Flattening list since pool.map return list of lists

            # Create tensor and save it as "train.pt" wich is 'processed_paths[0]
            data, slices = self.collate(flat_data_list)
            torch.save((data, slices), self.processed_paths[1])

            # pool_test.terminate()
            # pool_test.join()
            # pool_test.close()

        else:
            raise ValueError("Split %s not recognised" % split)

        #
        # t1 = time.perf_counter() # Start time of processing (debugging)
        #
        # ### Create train dataset
        # pool = Pool()                                               # Creating pools for multiprocess
        # create_samples_x = partial(create_samples, set_val="train")
        # data_list_temp = pool.map(create_samples_x, train_num)      # Mapping process with train_num range
        # flat_data_list = [x for z in data_list_temp for x in z]     # Flattening list since pool.map return list of lists
        #
        # # Create tensor and save it as "train.pt" wich is 'processed_paths[0]
        # data, slices = self.collate(flat_data_list)
        # torch.save((data, slices), self.processed_paths[0])
        #
        # pool.terminate()
        # pool.join()
        # pool.close()
        #
        # t2 = time.perf_counter() #end time of processing training data
        #
        # print(f'Processing training data finished in {t2-t1} seconds')

        ### Create test dataset
        # Same as train dataset, but using 'test_num' for mapping and processed_paths[1] for 'test.pt'



        # pool = Pool()                                               # Creating pools for multiprocess
        # create_samples_x = partial(create_samples, set_val="test")
        # data_list_temp = pool.map(create_samples_x, test_num)       # Mapping process with test_num range
        # flat_data_list = [x for z in data_list_temp for x in z]     # Flattening list since pool.map return list of lists
        #
        # # Create tensor and save it as "train.pt" wich is 'processed_paths[0]
        # data, slices = self.collate(flat_data_list)
        # torch.save((data, slices), self.processed_paths[1])
        #
        # pool.terminate()
        # pool.join()
        # pool.close()

    @property
    def num_classes(self):
        return 9

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
        transform = self.train_transform,
        pre_transform = self.pre_transform,
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
