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
    # train model
    for dir in range(0,multi):
        setup_seed(2023)
        print(f"Multi: {dir}")
        config_.SAVE_TIME = str(dir)

        best_acc = 0.0
        isDebug = False #print_info(config_=config_)
 
        save_path =  make_dirs(config_,isDebug,dir)

        early_stopping = EarlyStopping(config_.PATIENCE, verbose=False, path=os.path.join(save_path, "checkpoint.pt"))
        model = get_model(config_)

        model,start =  model_restore(model,restore)
        model.to(DEVICE)
        loss_func = loss_function(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=config_.LEARNING_RATE, weight_decay=1e-5)
        # cosine warmup scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config_.EPOCHS, T_mult=1, eta_min=1e-7)
        for epoch in range(start, config_.EPOCHS):
            epoch_loss_t, loss_tr, total_data, preds_tr, labels_tr = train_epoch(train_loader,optimizer,model,loss_func,DEVICE,epoch,config_,scheduler=True)
            scheduler.step(loss_tr)
            epoch_loss_t /= total_data
            f1_tr = accs(preds_tr, labels_tr)

            # valid model
            epoch_loss_v, total_data, preds_va, labels_va = valid_epoch(valid_loader, model, loss_func, DEVICE)
            epoch_loss_v /= total_data
            f1_val = accs(preds_va, labels_va)
            # if config_.WANDB and not isDebug:
            #     wandb.log({"Train Loss": epoch_loss_t, "Valid Loss": epoch_loss_v, "Train F1": f1_tr, "Valid F1": f1_val})
            print(f"Epoch [{epoch}/{config_.EPOCHS}] | Train Loss: {epoch_loss_t:.4f} | Valid Loss: {epoch_loss_v:.4f} | Train F1: {f1_tr:.4f} | Valid F1: {f1_val:.4f} | lr: {optimizer.param_groups[0]['lr']:.10f}")
            best_acc = save_model(model, save_path, epoch, f1_val, best_acc,config_,isDebug,epoch)
            early_stopping(epoch_loss_v, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    # test_for_each_model(test_loader,get_model(config_),config_.SAVE_PATH, DEVICE,multi,config_)
    
    # easy_test(test_loader[0],get_model(config_),config_.SAVE_PATH, DEVICE,multi)
    # easy_test(test_loader[1],get_model(config_),config_.SAVE_PATH, DEVICE,multi)
    # easy_test(test_loader[2],get_model(config_),config_.SAVE_PATH, DEVICE,multi)

    print("Training Successful")
    # if config_.WANDB:
        # wandb.finish()


if __name__ == '__main__':
    train_loader = build_dataloader_from_file(config_.TRAIN_DATA_PATH, batch_size=config_.BATCH_SIZE, shuffle=True,Graph_class=Graph, num_workers=config_.NUMBER_WORKERS)
    valid_loader = build_dataloader_from_file(config_.VALID_DATA_PATH, batch_size=config_.BATCH_SIZE, shuffle=False,Graph_class=Graph, num_workers=config_.NUMBER_WORKERS)
    test_loader1 = build_dataloader_from_file(config_.TEST_DATA_PATH1, batch_size=config_.BATCH_SIZE, shuffle=False,Graph_class=Graph, num_workers=config_.NUMBER_WORKERS)
    test_loader2 = build_dataloader_from_file(config_.TEST_DATA_PATH2, batch_size=config_.BATCH_SIZE, shuffle=False,Graph_class=Graph, num_workers=config_.NUMBER_WORKERS)
    test_loader3 = build_dataloader_from_file(config_.TEST_DATA_PATH3, batch_size=config_.BATCH_SIZE, shuffle=False,Graph_class=Graph, num_workers=config_.NUMBER_WORKERS)
    test_loaders = [test_loader1,test_loader2,test_loader3]
    # test_loaders = [test_loader1]
    run(train_loader,valid_loader,test_loaders,args.model_num)
