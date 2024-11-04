# cython: language_level=3
# -*- encoding: utf-8 -*-
# @File    :   mutil_train.py
# @Time    :   2023/08/04 19:00:46
# @Author  :   
# @Contact :   
# @Desc    :   None

import os
from config_share import Config
from torch import optim
# import wandb
import torch.nn as nn
import torch
import warnings
warnings.filterwarnings("ignore")
import os.path
from utils.EarlyStopping import EarlyStopping
from utils.util_share import *
import argparse

paraser = argparse.ArgumentParser()
paraser.add_argument('--model',type=str,default='GCN_A_H_S')
paraser.add_argument('--model_num',type=int,default=5)
args = paraser.parse_args()
config_ = Config()
os.system('EXPORT PYTHONPATH={}:$PYTHONPATH'.format(config_.ROOT_PATH))

if args.model == 'GCN_A_H_S':
    from models.GCN_A_H_S.model import loss_function,get_model
    from models.GCN_A_H_S.config import Config as Config_private
    from models.GCN_A_H_S.graph import Graph


config_private = Config_private(config_.DATA_SET)

config_.add(config_private)

is_seed = True

if is_seed:
    setup_seed(2023)

def run(train_loader,valid_loader,test_loader=None,multi=10,restore=None, dir=None):

    config_.SAVE_PATH = config_.MULTI_SAVE_PATH
    DEVICE = torch.device(config_.DEVICE)
    
    # easy_test(test_loader[0],get_model(config_),config_.SAVE_PATH, DEVICE,multi)
    # easy_test(test_loader[1],get_model(config_),config_.SAVE_PATH, DEVICE,multi)
    # easy_test(test_loader[2],get_model(config_),config_.SAVE_PATH, DEVICE,multi)

    test_for_each_model(test_loader[0],get_model(config_),config_.SAVE_PATH, DEVICE,multi,config_,data_name='test1000')
    test_for_each_model(test_loader[1],get_model(config_),config_.SAVE_PATH, DEVICE,multi,config_,data_name='California1000')
    test_for_each_model(test_loader[2],get_model(config_),config_.SAVE_PATH, DEVICE,multi,config_,data_name='Britain10000')


if __name__ == '__main__':
    # train_loader = build_dataloader_from_file(config_.TRAIN_DATA_PATH, batch_size=config_.BATCH_SIZE, shuffle=True,Graph_class=Graph, num_workers=config_.NUMBER_WORKERS)
    # valid_loader = build_dataloader_from_file(config_.VALID_DATA_PATH, batch_size=config_.BATCH_SIZE, shuffle=False,Graph_class=Graph, num_workers=config_.NUMBER_WORKERS)
    test_loader1 = build_dataloader_from_file(config_.TEST_DATA_PATH1, batch_size=config_.BATCH_SIZE, shuffle=False,Graph_class=Graph, num_workers=config_.NUMBER_WORKERS)
    test_loader2 = build_dataloader_from_file(config_.TEST_DATA_PATH2, batch_size=config_.BATCH_SIZE, shuffle=False,Graph_class=Graph, num_workers=config_.NUMBER_WORKERS)
    test_loader3 = build_dataloader_from_file(config_.TEST_DATA_PATH3, batch_size=config_.BATCH_SIZE, shuffle=False,Graph_class=Graph, num_workers=config_.NUMBER_WORKERS)
    run(None,None,[test_loader1,test_loader2,test_loader3],args.model_num)
