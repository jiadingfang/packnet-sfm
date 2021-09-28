import os
from collections import defaultdict

# test for euroc file infexing
# # data_dir = '/data/datasets/euroc/MH_03_medium/mav0/cam0/data'
# data_dir = '/data/datasets/euroc/V2_03_difficult/mav0/cam0/data'

data_dir_list= []
# data_dir_list.append('/data/datasets/euroc/MH_01_easy/mav0/cam0/data')
# data_dir_list.append('/data/datasets/euroc/MH_02_easy/mav0/cam0/data')
# data_dir_list.append('/data/datasets/euroc/MH_03_medium/mav0/cam0/data')
# data_dir_list.append('/data/datasets/euroc/MH_04_difficult/mav0/cam0/data')
# data_dir_list.append('/data/datasets/euroc/MH_05_difficult/mav0/cam0/data')
# data_dir_list.append('/data/datasets/euroc/V1_01_easy/mav0/cam0/data')
# data_dir_list.append('/data/datasets/euroc/V1_02_medium/mav0/cam0/data')
# data_dir_list.append('/data/datasets/euroc/V1_03_difficult/mav0/cam0/data')
# data_dir_list.append('/data/datasets/euroc/V2_01_easy/mav0/cam0/data')
# data_dir_list.append('/data/datasets/euroc/V2_02_medium/mav0/cam0/data')
# data_dir_list.append('/data/datasets/euroc/V2_03_difficult/mav0/cam0/data')

data_dir_list.append('/data/datasets/euroc/MH_01_easy/mav0/cam1/data')
data_dir_list.append('/data/datasets/euroc/MH_02_easy/mav0/cam1/data')
data_dir_list.append('/data/datasets/euroc/MH_03_medium/mav0/cam1/data')
data_dir_list.append('/data/datasets/euroc/MH_04_difficult/mav0/cam1/data')
data_dir_list.append('/data/datasets/euroc/MH_05_difficult/mav0/cam1/data')
data_dir_list.append('/data/datasets/euroc/V1_01_easy/mav0/cam1/data')
data_dir_list.append('/data/datasets/euroc/V1_02_medium/mav0/cam1/data')
data_dir_list.append('/data/datasets/euroc/V1_03_difficult/mav0/cam1/data')
data_dir_list.append('/data/datasets/euroc/V2_01_easy/mav0/cam1/data')
data_dir_list.append('/data/datasets/euroc/V2_02_medium/mav0/cam1/data')
data_dir_list.append('/data/datasets/euroc/V2_03_difficult/mav0/cam1/data')

for data_dir in data_dir_list:
    file_list = os.listdir(data_dir)
    idx_list = [int(fname.split('.')[0]) for fname in file_list]
    sorted_idx_list = sorted(idx_list)

    diff_list = []
    for i in range(len(sorted_idx_list) - 1):
        diff_list.append( sorted_idx_list[i+1] - sorted_idx_list[i] )

    unique_list = []
    for x in diff_list:
            # check if exists in unique_list or not
            if x not in unique_list:
                unique_list.append(x)
    print(data_dir)
    print(sorted(unique_list))

# # test for read file method
# def read_files(directory, file_tree, ext=('.png', '.jpg', '.jpeg'), skip_empty=True):
#     files = defaultdict(list)
#     for entry in os.scandir(directory):
#         # print()
#         # print('entry')
#         # print(entry)
#         relpath = os.path.relpath(entry.path, directory)
#         if entry.is_dir():
#             # print('is_dir')
#             # print(entry.path)
#             # print(ext)
#             # print(skip_empty)
#             d_files = read_files(entry.path, file_tree, ext=ext, skip_empty=skip_empty)
#             # print('d_files')
#             # print('{} [--] {}'.format(d_files, entry.path))
#             if skip_empty and not len(d_files):
#                 # print('skip entry {}'.format(entry.path))
#                 # print(entry.path)
#                 continue
#             file_tree[entry.path] = d_files[entry.path]
#             # files[relpath] = d_files[entry.path]
#             # print('{} <--> {}'.format(relpath, files))
#         elif entry.is_file():
#             # print('is_file')
#             # print(entry.path.lower())
#             if ext is None or entry.path.lower().endswith(tuple(ext)):
#                 # print('append')
#                 # print(directory)
#                 files[directory].append(relpath)
#                 # print(files[directory])
#     return files

# directory = '/data/datasets/euroc_cam0'
# file_tree = defaultdict(list)
# read_files(directory, file_tree)

# print()
# print('file_tree')
# print(file_tree)
# print(len(file_tree))
# print(type(file_tree))
# print(file_tree['/data/datasets/euroc_cam0/cam0/V1_03_difficult/mav0/cam0/data'])
# for k in file_tree:
#     print(k)