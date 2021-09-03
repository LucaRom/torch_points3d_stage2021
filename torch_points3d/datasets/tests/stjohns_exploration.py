import os
import laspy
import random
import os.path as osp

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

# Print for quick/lazy debug
# print(f"DIR {DIR}")
# print(f"dataroot {dataroot}")
# print(f"train_list {train_list}")
# print(f"train_list lenght = {len(train_list)}")
# print(f"val_list {val_list}")
# print(f"val_list lenght = {len(val_list)}")
# print(f"test_list {test_list}")
# print(f"test_list lenght = {len(test_list)}")
# print(f"train_list_small {train_list_small}")
# print(f"train_list_small lenght = {len(train_list_small)}")
# print(f"val_list_small {val_list_small}")
# print(f"val_list_small lenght = {len(val_list_small)}")
# print(f"test_list_small {test_list_small}")
# print(f"test_list_small lenght = {len(test_list_small)}")


laspath = '/export/sata01/wspace/lidar/classification_pts/torch-points3d/data/st_john_data/train/NL_StJohns_20210205_NAD83CSRS_UTMZ22_1km_E2730_N52400_CQL1_CLASS.las'
lazpath = "/export/sata01/wspace/lidar/classification_pts/data/st_jonhs2021/laz/NL_StJohns_20210205_NAD83CSRS_UTMZ22_1km_E2730_N52400_CQL1_CLASS.laz"

laspath1 = '/export/sata01/wspace/lidar/classification_pts/torch-points3d/data/st_john_data/train/NL_StJohns_20210205_repaired.las'
laspath2 = "/export/sata01/wspace/lidar/classification_pts/torch-points3d/data/st_john_data/train/NL_StJohns_20210205_NAD83CSRS_UTMZ22_1km_E2750_N52340_CQL1_CLASS.las"
laspath3 = "/export/sata01/wspace/lidar/classification_pts/torch-points3d/data/st_john_data/train/NL_StJohns_20210205_NAD83CSRS_UTMZ22_1km_E2750_N52370_CQL1_CLASS.las"
laspath4 = "/export/sata01/wspace/lidar/classification_pts/torch-points3d/data/st_john_data/train/NL_StJohns_20210205_NAD83CSRS_UTMZ22_1km_E3230_N53080_CQL1_CLASS.las"
laspath5 = "/export/sata01/wspace/lidar/classification_pts/torch-points3d/data/st_john_data/train/NL_StJohns_20210205_NAD83CSRS_UTMZ22_1km_E3300_N52710_CQL1_CLASS.las"

laspath6 = "/export/sata01/wspace/lidar/classification_pts/torch-points3d/data/st_john_data/train/NL_StJohns_20210205_NAD83CSRS_UTMZ22_1km_E3280_N52680_CQL1_CLASS.las"
laspath7 = "/export/sata01/wspace/lidar/classification_pts/torch-points3d/data/st_john_data/train/NL_StJohns_20210205_NAD83CSRS_UTMZ22_1km_E3290_N52760_CQL1_CLASS.las"
laspath8 = "/export/sata01/wspace/lidar/classification_pts/torch-points3d/data/st_john_data/train/NL_StJohns_20210205_NAD83CSRS_UTMZ22_1km_E3660_N52590_CQL1_CLASS.las"

laspath8 = "/export/sata01/wspace/lidar/classification_pts/torch-points3d/data/st_john_data/test/NL_StJohns_20210205_NAD83CSRS_UTMZ22_1km_E3510_N52170_CQL1_CLASS.las"

las_file = laspy.read(laspath2)
print("allo")

with laspy.open(laspath2) as f:
    print(f"Point format {f.header.point_format}")
    print(f"Number of points:   {f.header.point_count}")
    print(f"Number of vlrs:     {len(f.header.vlrs)}")
    print(f"allo :              {list(f.header.point_format.dimension_names)}")
    print(f"Point_format size :              {f.header.point_format.size}")
    print(f"allo :              {list(f.header.point_format.extra_dimension_names)}")
    print(f"allo :              {list(f.header.point_format.standard_dimension_names)}")
    # point_format = f.point_format
    # list(point_format.dimension_names)

# data_list = []
# current_split = "train"
# for i in train_list_small:
#     # Load the .las file and extract needed data
#     #print(os.path.join(dataroot, current_split, i))
#     las_file = laspy.read(os.path.join(dataroot, current_split, i))
#     print("allo")

# readable = []
# current_split = "test"
# for i in test_list:
#     try:
#         las_file = laspy.read(os.path.join(dataroot, current_split, i))
#         readable.append(i)
#     except Exception:
#         pass
#
# print(readable)

