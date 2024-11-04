# cython: language_level=3
#!/user/bin/env python
# -*-coding: utf-8 -*-
# @Time : 2022/3/20 15:58
# @Author : 
import os
import time


class Config(object):
    def __init__(self, DATA_SET):
        self.DATA_SET = DATA_SET
        self.BASE_NAME = 'GCN_A_H_S'
        self.WANDB = True
        self.WANDB_PROJ = f"{self.BASE_NAME}_{self.DATA_SET}"
        self.WANDB_ENTITY = "proj_name"

        self.SAVE_PATH = rf'/to/path/SAVE/{self.DATA_SET}/ckpt/{self.BASE_NAME}'

        
        self.MULTI_SAVE_PATH = rf'/to/path/SAVE/{self.DATA_SET}/multi/ckpt/{self.BASE_NAME}'

        self.ERROR_SAVE_PATH = rf'/to/path/SAVE/{self.DATA_SET}/multi/errors/{self.BASE_NAME}'
        self.JSON_SAVE_PATH = rf'/to/path/CODE/main/json_result/{self.BASE_NAME}.json'

        self.NODE_FEATURE_NUM = 6