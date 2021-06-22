import os
from glob import glob
import numpy as np
import multiprocessing
import logging
import torch
import laspy

from torch_geometric.data import InMemoryDataset, Dataset, Data
from torch_geometric.io import read_ply

from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker

log = logging.getLogger(__name__)


# "logic of your raw dataset Dales that should probably inherit from the torch_geometric InMemory dataset but does
# not have to, it could just be a raw pytorch dataset"
# https://github.com/nicolas-chaulet/torch-points3d/issues/471

###################################Memory dataset ###################################

DIR = os.path.dirname(os.path.realpath(__file__))
dataroot = os.path.join(DIR, "..", "..", "..", "data", "dales")

# path = "/wspace/disk01/lidar/classification_pts/torch-points3d/data/dales/raw"  # à remplacer avec chemmin hydra ou points3D

las_list = [f for f in os.listdir(os.path.join(dataroot, "raw")) if
            f.endswith('.las')]  # liste des fichiers avec extension .las
las_num = [os.path.splitext(x)[0] for x in las_list]  # liste des fichiers sans extension

class Dales(InMemoryDataset):
    """
    Class to handle DALES dataset for segmentation task.
    """

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
                                                                                             # super() inherits from
                                                                                             # parent class, but extent
                                                                                             #  the method (here __init_(read : https://rhettinger.wordpress.com/2011/05/26/super-considered-super/_
                                                                                             # or https://stackoverflow.com/questions/576169/understanding-python-super-with-init-methods )
                                                                                             # or https://www.geeksforgeeks.org/python-super/

        print(self.processed_dir)

        # Debug LR (temp)
        # print(f"Written path is {path}")
        # print(f"processed_paths 0 is :{self.processed_paths[0]}")
        # print(f"processed_paths 0 is :{self.processed_paths[1]}")
        # print(f"processed_paths 0 is :{self.processed_paths[2]}")
        # print(f"processed_paths 0 is :{self.processed_paths[3]}")

        self.data, self.slices = torch.load(self.processed_paths[0])
        print(self.data)
        print(self.slices)

    @property
    def raw_file_names(self):
        """
        La fonction vérifie la liste des données à télécharger, puisque nos données sont déjà télécharger,
        on réfère simplement à la liste des fichier .las
        """
        return las_list

    @property
    def processed_file_names(self):
        """Ici on définit le nom des fichier processes , on tente de faire un gros fichier data avec toute mes raw, donc dans un seul
        fichier "data.pt"

        Donc, ici on réfère à la liste des .las en entrées (sans extension) et on recherche s'il sont déjà processed
        (présents dans le dossier processed et sous le format nom_du_fichier_las.pt"

        """
        #return ["{}.pt".format(s) for s in las_num]
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        """
        On parcours la list des fichiers "raw" qui ne sont pas "processed" et on les traite pour les
        sauvegarder en format .pt dans le dossier processed

        Chaque fichier .las devient un fichier .pt qui est alimenté par la fonction Data(pos= , x= et y)
        ou pos = x, y, z , x= liste des features et y les labels

        """
        data_list = []

        # for i in las_num:
        #     #las_file = laspy.read(os.path.join(dataroot,'raw', "{}.las".format(i)))
        #     las_file = laspy.read(os.path.join(dataroot, "raw", "{}.las".format(i)), mode='r')
        #     #print(las_file)
        #     las_xyz = np.vstack((las_file['X'], las_file['Y'], las_file['Z'])).astype(np.float32).T # .T -> transpose, si on le fait pas on a pas les bonne dimension des données
        #     las_label = np.reshape(las_file.classification, (len(las_file), 1))
        #     #print(las_xyz)
        #     data = Data(pos=las_xyz)
        #     data.y = las_label.T
        #     #print(data.y)

        for i in las_num:
            las_file = laspy.read(os.path.join(dataroot, "raw", "{}.las".format(i)))
            #print(las_file)

            las_xyz = np.stack([las_file.x, las_file.y, las_file.z], axis=1)

            las_label = np.array(las_file.classification).astype(np.int)
            #print(las_xyz)
            y = torch.from_numpy(las_label)
            #y = self._remap_labels(y)
            data = Data(pos=torch.from_numpy(las_xyz).type(torch.float), y=y)

            print(f"Processed file {i}, nb points = {data.pos.shape[0]}") #To remove once wrapper is set and calling log
            log.info("Processed file %s, nb points = %i", i, data.pos.shape[0])
            #torch.save(data, os.path.join(self.processed_dir, "{}.pt".format(i)))

            data_list.append(data)


        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

################################### Wrapper ###################################

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
            transform=self.train_transform,
            pre_transform=self.pre_transform
        )
        #
        # self.val_dataset = SemanticKitti(
        #     self._data_path,
        #     split="val",
        #     transform=self.val_transform,
        #     pre_transform=self.pre_transform,
        #     process_workers=process_workers,
        # )
        #
        # self.test_dataset = SemanticKitti(
        #     self._data_path,
        #     split="test",
        #     transform=self.test_transform,
        #     pre_transform=self.pre_transform,
        #     process_workers=process_workers,
        # )

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker
        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        return SegmentationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)

#
# if __name__ == "__main__":
#     DIR = os.path.dirname(os.path.realpath(__file__))
#     dataroot = os.path.join(DIR, "..", "..", "..", "data", "dales")
#
#     #path = "/wspace/disk01/lidar/classification_pts/torch-points3d/data/dales/raw"  # à remplacer avec chemmin hydra ou points3D
#
#     las_list = [f for f in os.listdir(os.path.join(dataroot, "raw")) if f.endswith('.las')]  # liste des fichiers avec extension .las
#     las_num = [os.path.splitext(x)[0] for x in las_list] # liste des fichiers sans extension
#
#     # SemanticKitti(
#     #     dataroot, split="train", process_workers=10,
#     # )
#
#     print(f"last_list : {las_list}")
#     print(f"last_num : {las_num}")
#
#     print(f"this is the DIR: {DIR}")
#     print(f"this is the dataroot: {dataroot}")



    #Dales(dataroot)
    #DalesDataset(Dales(dataroot))