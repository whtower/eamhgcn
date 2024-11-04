# -*- encoding: utf-8 -*-
# @File    :   main.py
# @Time    :   2023/11/05 10:49:13
# @Author  :   
# @Contact :   
# @Desc    :   None
from BuildingDataLoader import BuildingDataLoader
import numpy as np
import os
import argparse
import tqdm
import json
import shutil
import random
import jenkspy
import math
from sklearn.preprocessing import RobustScaler

parser = argparse.ArgumentParser()
dataset = "NEW_1000"
labels = None
if dataset == "NEW_1000":
    parser.add_argument("--read_dir", type=str, default=r"/to/path/DATA/NEW_1000/split_data",
                        help="shp文件根目录")
    parser.add_argument("--save_path", type=str, default=r"/to/path/DATA/NEW_1000/dataset",
                        help="json文件保存路径")
    parser.add_argument("--png_path", type=str, default=r'/to/path/DATA/NEW_1000/pic',
                        help="png文件保存路径")
    labels = {"不规则": 0, "直线": 1, "曲线": 2, "网格": 3, "规则轮廓": 4}
args = parser.parse_args()

def get_file_list(read_dir):
    """
    获取shp文件的路径
    :return: shp文件的路径 List
    """
    file_path_list = os.listdir(read_dir)
    file_list = []
    for idx in file_path_list:
        tmp_path1 = os.listdir(os.path.join(read_dir, idx))
        tmp_path2 = list(
            set([os.path.join(read_dir, idx, i.split('.')[0]) for i in tmp_path1 if i.endswith(".shp")]))
        file_list.append(tmp_path2)
    for i in file_list:
        random.shuffle(i)
    return file_list

def process(read_dir, save_path, dt, png_path):
    files = get_file_list(read_dir)
    res = []
    files = [j for i in files for j in i]
    for i in tqdm.tqdm(files,total=len(files)):
        file_path = i + ".shp"
        bdl = BuildingDataLoader(labels=labels,png_path=png_path)
        bdl.load_data_from_shp(file_path) # 从shp文件中加载数据
        ll = bdl.write_common_edge_building() # 将共边建筑物写入数据库
        bdl.create_delaunay() # 创建三角网
        bdl.create_skeleton() # 创建骨架
        # bdl.tmp_save_lines_to_shp()
        bdl.easy_create_next2_by_triangle()
        bdl.tmp_save_lines_to_shp()
        bdl.create_pictures() # 创建图片 # TODO
        re = bdl.get_result() # 获取结果
        res.append(re)

    with open(save_path.format(dt), "w") as f:
        json.dump(res, f, ensure_ascii=False)

def run(read_dir, save_path, dt, png_path):
    print("read_dir:\t", read_dir)
    print("save_path:\t", save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path,"{}.json")
    for i in dt:
        process(os.path.join(read_dir, i), save_path, i, png_path)



if __name__ == "__main__":
    dt = ['test_data_California_1000','test_data_usa','test_data_uk_10000','valid_data','train_data']#,
    if os.path.exists(args.png_path):
        shutil.rmtree(args.png_path)
    run(args.read_dir, args.save_path, dt, args.png_path)