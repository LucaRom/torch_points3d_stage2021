import os
from glob import glob
import numpy as np
import multiprocessing
import logging
import torch
import laspy

from torch_geometric.data import InMemoryDataset, Dataset, Data
#from torch_geometric.io import read_ply

from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.core.data_transform.transforms import RandomSphere
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker

log = logging.getLogger(__name__)


# "logic of your raw dataset Dales that should probably inherit from the torch_geometric InMemory dataset but does
# not have to, it could just be a raw pytorch dataset"
# https://github.com/nicolas-chaulet/torch-points3d/issues/471

# reference for dales dataset : https://arxiv.org/abs/2004.11985
# The dataset must be downloaded at go.udayton.edu/dales3d.

###################################Memory dataset ###################################

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
print(f"this is las list : {las_list}")

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

class Dales(InMemoryDataset):
    """
    Class to handle DALES dataset for segmentation task.
    """

    def __init__(self, root, split=None, transform=None, pre_transform=None, pre_filter=None):

        super().__init__(root, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
                                                                                             # super() inherits from

                                                                             # parent class, but extent
                                                                                             #  the method (here __init_(read : https://rhettinger.wordpress.com/2011/05/26/super-considered-super/_
                                                                                             # or https://stackoverflow.com/questions/576169/understanding-python-super-with-init-methods )
                                                                                             # or https://www.geeksforgeeks.org/python-super/


        # Debug LR (temp)
        # print(self.processed_dir)
        print(f"processed_paths 0 is :{self.processed_paths[0]}") #should point to train.pt
        print(f"processed_paths 1 is :{self.processed_paths[1]}") #should point to test.pt
        print(f"processed_paths 1 is : test 1,2, 3")

        # les processed path sont probablement créé par la section "processed file names" qui retourne actuellement train.pt et test.pt


        #Processed path : a mieux expliquer -> doit pointer vert les dossier "processed"

        if split == "train":
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif split == "test":
            self.data, self.slices = torch.load(self.processed_paths[1])
        else:
            raise ValueError("Split %s not recognised" % split)

        #self.data, self.slices = torch.load(self.processed_paths[0])


        #print(self.data)
        #print(self.slices)

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

        # if self.split == "train":
        #     return ['train.pt']
        # elif self.split == "test":
        #     return ['test.pt']
        # else:
        #     raise ValueError("Error in processed_file_names section")

        return ['train.pt', 'test.pt']

    def download(self):
        pass

    def process(self):
        """
        On parcours la list des fichiers "raw" qui ne sont pas "processed" et on les traite pour les
        sauvegarder en format .pt dans le dossier processed

        Chaque fichier .las devient un fichier .pt qui est alimenté par la fonction Data(pos= , x= et y)
        ou pos = x, y, z , x= liste des features et y les labels

        """
        #data_list = []

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

        ### Create train dataset
        """
        First part load the point cloud in .las format and store as tensor data

        Second part create subsamples (according to the range) using the RandomSphere fonction from torch-points3d
        The subsampling range will give the number of total sample available for batching per .las file

        """
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

            # Subsampling
            for sample_no in range (5):
                random_sphere = RandomSphere(0.1, strategy="RANDOM")
                data_sample = random_sphere(data.clone())

                #print(f"Processed file {i}, nb points = {data.pos.shape[0]}") #To remove once wrapper is set and calling log
                log.info("Processed file %s, sample_no = %s nb points = %i", i, sample_no, data.pos.shape[0])
                #torch.save(data, os.path.join(self.processed_dir, "{}.pt".format(i)))

                data_list.append(data_sample)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        ### Create test dataset
        data_list = []
        for i in test_num:
            las_xyz = np.stack([las_file.x, las_file.y, las_file.z], axis=1)

            las_label = np.array(las_file.classification).astype(np.int)
            # print(las_xyz)
            y = torch.from_numpy(las_label)
            # y = self._remap_labels(y)
            data = Data(pos=torch.from_numpy(las_xyz).type(torch.float), y=y)

            for sample_no in range (5):
                random_sphere = RandomSphere(0.1, strategy="RANDOM")
                data_sample = random_sphere(data.clone())

                #print(f"Processed file {i}, nb points = {data.pos.shape[0]}") #To remove once wrapper is set and calling log
                log.info("Processed file %s, sample_no = %s nb points = %i", i, sample_no, data.pos.shape[0])
                #torch.save(data, os.path.join(self.processed_dir, "{}.pt".format(i)))

                data_list.append(data_sample)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[1])

    @property
    def num_classes(self):
        return 9

### Sampling test

# class DALESsphere(Dales):
#
#     def __init__(self, root, sample_per_epoch=100, radius=2, *args, **kwargs):
#         self._sample_per_epoch = sample_per_epoch
#         self._radius = radius
#         self._grid_sphere_sampling = cT.GridSampling3D(size=radius / 10.0)
#         super().__init__(root, *args, **kwargs)
#
#     def __len__(self):
#         if self._sample_per_epoch > 0:
#             return self._sample_per_epoch
#         else:
#             return len(self._test_spheres)
#
#     def get(self, idx):
#         if self._sample_per_epoch > 0:
#             return self._get_random()
#         else:
#             return self._test_spheres[idx].clone()
#
#     def process(self):  # We have to include this method, otherwise the parent class skips processing
#         super().process()
#
#     def download(self):  # We have to include this method, otherwise the parent class skips download
#         super().download()
#
#     def _get_random(self):
#         # Random spheres biased towards getting more low frequency classes
#         chosen_label = np.random.choice(self._labels, p=self._label_counts)
#         valid_centres = self._centres_for_sampling[self._centres_for_sampling[:, 4] == chosen_label]
#         centre_idx = int(random.random() * (valid_centres.shape[0] - 1))
#         centre = valid_centres[centre_idx]
#         area_data = self._datas[centre[3].int()]
#         sphere_sampler = cT.SphereSampling(self._radius, centre[:3], align_origin=False)
#         return sphere_sampler(area_data)
#
#     def _save_data(self, train_data_list, val_data_list, test_data_list, trainval_data_list):
#         torch.save(train_data_list, self.processed_paths[0])
#         torch.save(val_data_list, self.processed_paths[1])
#         torch.save(test_data_list, self.processed_paths[2])
#         torch.save(trainval_data_list, self.processed_paths[3])
#
#     def _load_data(self, path):
#         self._datas = torch.load(path)
#         if not isinstance(self._datas, list):
#             self._datas = [self._datas]
#         if self._sample_per_epoch > 0:
#             self._centres_for_sampling = []
#             for i, data in enumerate(self._datas):
#                 assert not hasattr(
#                     data, cT.SphereSampling.KDTREE_KEY
#                 )  # Just to make we don't have some out of date data in there
#                 low_res = self._grid_sphere_sampling(data.clone())
#                 centres = torch.empty((low_res.pos.shape[0], 5), dtype=torch.float)
#                 centres[:, :3] = low_res.pos
#                 centres[:, 3] = i
#                 centres[:, 4] = low_res.y
#                 self._centres_for_sampling.append(centres)
#                 tree = KDTree(np.asarray(data.pos), leaf_size=10)
#                 setattr(data, cT.SphereSampling.KDTREE_KEY, tree)
#
#             self._centres_for_sampling = torch.cat(self._centres_for_sampling, 0)
#             uni, uni_counts = np.unique(np.asarray(self._centres_for_sampling[:, -1]), return_counts=True)
#             uni_counts = np.sqrt(uni_counts.mean() / uni_counts)
#             self._label_counts = uni_counts / np.sum(uni_counts)
#             self._labels = uni
#         else:
#             grid_sampler = cT.GridSphereSampling(self._radius, self._radius, center=False)
#             self._test_spheres = grid_sampler(self._datas)

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

#To confirm, no validation data

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