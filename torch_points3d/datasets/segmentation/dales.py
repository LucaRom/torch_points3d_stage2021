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


class Dales(InMemoryDataset):
    """
    Class to handle DALES dataset for segmentation task.
    """

    # Initiation methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
                                                                                             # super() inherits from
                                                                                             # parent class, but extent
                                                                                             #  the method (here __init_(read : https://rhettinger.wordpress.com/2011/05/26/super-considered-super/_
                                                                                             # or https://stackoverflow.com/questions/576169/understanding-python-super-with-init-methods )
                                                                                             # or https://www.geeksforgeeks.org/python-super/

        print(self.processed_dir)

        # Debug LR (temp)
        print(f"Written path is {path}")
        print(f"processed_paths 0 is :{self.processed_paths[0]}")
        print(f"processed_paths 0 is :{self.processed_paths[1]}")
        print(f"processed_paths 0 is :{self.processed_paths[2]}")
        print(f"processed_paths 0 is :{self.processed_paths[3]}")

        # self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        """
        La fonction vérifie la liste des données à télécharger, puisque nos données sont déjà télécharger,
        on réfère simplement à la liste des fichier .las
        """
        return las_list

    @property
    def processed_file_names(self):
        """Il faut lister les fichiers qui sont processed en format pytorch. Les fichiers "processed" sont
        produit dans la method "process". Cependant le dataset  vérifie la liste des fichiers déjà processed
        avant d'activer process() pour ne pas traiter les fichiers déjà traités

        Donc, ici on réfère à la liste des .las en entrées (sans extension) et on recherche s'il sont déjà processed
        (présents dans le dossier processed et sous le format nom_du_fichier_las.pt"

        """
        return ["{}.pt".format(s) for s in las_num]

    def download(self):
        pass

    def process(self):
        """
        On parcours la list des fichiers "raw" qui ne sont pas "processed" et on les traite pour les
        sauvegarder en format .pt dans le dossier processed

        Chaque fichier .las devient un fichier .pt qui est alimenté par la fonction Data(pos= , x= et y)
        ou pos = x, y, z , x= liste des features et y les labels

        """
        for i in las_num:
            las_file = laspy.read(os.path.join(path, "{}.las".format(i)))
            las_xyz = np.vstack((las_file['X'], las_file['Y'], las_file['Z'])).astype(np.float32).T
            las_label = np.reshape(las_file.classification, (len(las_file), 1))
            data = Data(pos=las_xyz)
            data.y = las_label.T

            print(f"Processed file {i}, nb points = {data.pos.shape[0]}") #To remove once wrapper is set and calling log
            log.info("Processed file %s, nb points = %i", i, data.pos.shape[0])
            torch.save(data, os.path.join(self.processed_dir, "{}.pt".format(i)))





        # # Debug LR
        # print(f"raw_paths in process : {raw_paths}")
        # print(f"processed_paths in process : {processed_paths}")

        # data_list = []
        # for i in range(100):
        #     print(f"i dans process: {i}")
        #     data = read_ply(path.format(i))
        #     data.y = torch.tensor([i % 10], dtype=torch.long)
        #     if self.pre_filter is not None and not self.pre_filter(data):
        #         continue
        #     if self.pre_transform is not None:
        #         data = self.pre_transform(data)
        #     data_list.append(data)

        # torch.save(self.collate(data_list[:80]), self.processed_paths[0])
        # torch.save(self.collate(data_list[80:]), self.processed_paths[1])

        #shutil.rmtree(osp.join(self.raw_dir, 'MPI-FAUST'))




        # for i in self.raw_paths:
        #     if os.path.exists(self.processed_paths[i]):
        #         continue
        #     os.makedirs(self.processed_paths[i])
        #
        #     # seqs = self.SPLIT[split]
        #     # scan_paths, label_paths = self._load_paths(seqs)
        #     # scan_names = []
        #     # for scan in scan_paths:
        #     #     scan = os.path.splitext(scan)[0]
        #     #     seq, _, scan_id = scan.split(os.path.sep)[-3:]
        #     #     scan_names.append("{}_{}".format(seq, scan_id))
        #
        #     ply_names = [os.path.splitext(x)[0] for x in os.listdir(path)]
        #
        #     #Debug LR
        #     print(f"liste ply_names : {ply_names}")
        #
        #     out_files = [os.path.join(self.processed_paths[i], "{}.pt".format(ply_name)) for ply_name in
        #                  ply_names]
        #
        #     #Debug LR
        #     print(f"liste out_files : {out_files}")
        #
        #     # Read ply file
        #     data = read_ply(file_path)
        #     points = np.vstack((data['x'], data['y'], data['z'])).astype(np.float32).T
        #     reflectance = np.expand_dims(data['reflectance'], 1).astype(np.float32)
        #     if cloud_split == 'test':
        #         int_features = None
        #     else:
        #         int_features = data['class'].astype(np.int32)
        #
        #     # args = zip(scan_paths, label_paths, [self.pre_transform for i in range(len(scan_paths))], out_files)
        #     # if self.use_multiprocessing:
        #     #     with multiprocessing.Pool(processes=self.process_workers) as pool:
        #     #         pool.starmap(self.process_one, args)
        #     # else:
        #     #     for arg in args:
        #     #         self.process_one(*arg)

    # def len(self):
    #     return len(self.processed_file_names)
    #
    # def get(self, idx):
    #     data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
    #     return data

        # # Path of the folder containing ply files
        # self.path = '../Data'
        #
        # # Path of the training files
        # self.train_path = join(self.path, 'train_bin')
        # self.test_path = join(self.path, 'test_bin')
        #
        # # List of training and test files
        # self.train_files = np.sort([join(self.train_path, f) for f in listdir(self.train_path) if f[-4:] == '.ply'])
        # self.test_files = np.sort([join(self.test_path, f) for f in listdir(self.test_path) if f[-4:] == '.ply'])
        #
        # # Proportion of validation scenes
        # self.all_splits = [i for i in range(29)]
        # self.validation_split = 1


################################### DALES wrapper ###################################

# class DalesDataset(BaseDataset):
#     """ Wrapper around Dales that creates train and test datasets.
#     Parameters
#     ----------
#     dataset_opt: omegaconf.DictConfig
#         Config dictionary that should contain
#             - root,
#             - transform,
#             - pre_transform
#             - process_workers
#     """
#
#     def __init__(self, dataset_opt):
#         super().__init__(dataset_opt)
#         self.train_dataset = Dales(
#             self._data_path,
#             transform=self.train_transform,
#             pre_transform=self.pre_transform
#         )
#
#         self.val_dataset = SemanticKitti(
#             self._data_path,
#             split="val",
#             transform=self.val_transform,
#             pre_transform=self.pre_transform,
#             process_workers=process_workers,
#         )
#
#         self.test_dataset = SemanticKitti(
#             self._data_path,
#             split="test",
#             transform=self.test_transform,
#             pre_transform=self.pre_transform,
#             process_workers=process_workers,
#         )
#
#     def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
#         """Factory method for the tracker
#         Arguments:
#             wandb_log - Log using weight and biases
#             tensorboard_log - Log using tensorboard
#         Returns:
#             [BaseTracker] -- tracker
#         """
#         return SegmentationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)


if __name__ == "__main__":
    DIR = os.path.dirname(os.path.realpath(__file__))
    dataroot = os.path.join(DIR, "..", "..", "..", "data", "dales")

    #path = "/wspace/disk01/lidar/classification_pts/torch-points3d/data/dales/raw"  # à remplacer avec chemmin hydra ou points3D

    las_list = [f for f in os.listdir(os.path.join(dataroot, "raw")) if f.endswith('.las')]  # liste des fichiers avec extension .las
    las_num = [os.path.splitext(x)[0] for x in las_list] # liste des fichiers sans extension

    # SemanticKitti(
    #     dataroot, split="train", process_workers=10,
    # )

    print(f"last_list : {las_list}")
    print(f"last_num : {las_num}")

    print(f"this is the DIR: {DIR}")
    print(f"this is the dataroot: {dataroot}")



    #Dales(dataroot)