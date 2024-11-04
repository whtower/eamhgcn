# cython: language_level=3
#!/user/bin/env python
# -*-coding: utf-8 -*-
# @Time : 2022/3/20 15:58
# @Author : 
import os
import time


class Config(object):
    def __init__(self):
        self.DATA_SET = "NEW_1000"
        self.ROOT_PATH = '/to/path/CODE/main'
        self.SAVE_STEP = 1
        self.LEARNING_RATE = 5e-5
        self.EPOCHS = 200
        self.BATCH_SIZE = 48
        self.NUMBER_WORKERS = 0
        self.DROPOUT = 0.5
        self.NODE_FEATURE_NUM = 6
        self.EDGE_FEATURE_NUM = 5
        self.CLASSIFY_NUM = 5
        self.PATIENCE = 10
        self.SAVE_TIME = time.strftime("%Y_%m%d_%H_%M_%S", time.localtime())
        self.DEVICE = "cuda:0"
        self.data = {
            "NEW_1000": {
                "TRAIN_DATA_PATH": r'/to/path/DATA/NEW_1000/dataset/train_data.json',
                "VALID_DATA_PATH": r'/to/path/DATA/NEW_1000/dataset/valid_data.json',
                "TEST_DATA_PATH1": r'/to/path/DATA/NEW_1000/dataset/test_data_usa.json',
                "TEST_DATA_PATH2": r'/to/path/DATA/NEW_1000/dataset/test_data_California_1000.json',
                "TEST_DATA_PATH3": r'/to/path/DATA/NEW_1000/dataset/test_data_uk_10000.json',
                "PNG_PATH": r'/to/path/DATA/NEW_1000/pic'
            }
        }

        self.TRAIN_DATA_PATH = self.data[self.DATA_SET]["TRAIN_DATA_PATH"]
        self.VALID_DATA_PATH = self.data[self.DATA_SET]["VALID_DATA_PATH"]
        self.TEST_DATA_PATH1 = self.data[self.DATA_SET]["TEST_DATA_PATH1"]
        self.TEST_DATA_PATH2 = self.data[self.DATA_SET]["TEST_DATA_PATH2"]
        self.TEST_DATA_PATH3 = self.data[self.DATA_SET]["TEST_DATA_PATH3"]
        self.PNG_PATH = self.data[self.DATA_SET]["PNG_PATH"]

    def add(self,cfg):
        for attr, value in cfg.__dict__.items():
            setattr(self, attr, value)
