#! python
# -*- coding: utf-8 -*-


import os
import sys
import numpy as np
import torch
from torch.utils import data
import pandas as pd
from sklearn.utils import shuffle
from python_speech_features import delta

from extract_features import truncate_fea, frames_fea



########################################################################################################

def read_list(txt):
    f = open(txt, 'r')
    lines = f.readlines()
    list = []
    for line in lines:
        line = line.strip()
        line = line.split(' ')
        # print(line)
        list.append([line[0], int(line[1])])
    # print(len(list))
    return list


########################################################################################################


class TrainDatasetByFeatures(data.Dataset):
    def __init__(self, transform=None, *args, **kwargs):
        all_class = 0
        dict_list = []

        total_list = read_list("./train_audio_feas.txt")
        print("train list: ", len(total_list))

        # test_list = read_list("./test_audio_feas.txt")
        # print("test list: ", len(test_list))
        # list.extend(test_list)
        
        print("total_list: ",  len(total_list))
        
        dict = {}
        train_sample_num = 0
        for fea_path, label in total_list:
            fea_path = os.path.abspath(fea_path)

            if "_ts_" in fea_path:
                continue

            # if "_ps_" in fea_path:
                # continue


            if label not in dict:
                dict[label] = [fea_path]
            else:
                dict[label].append(fea_path)
            
            train_sample_num += 1

        print("train_sample_num: ", train_sample_num)

        self.num = train_sample_num * 2 # len(list) * 10
        
        print("samples num: ", self.num)

        all_class += len(dict)
        print("all_class: ", all_class)
        dict_list.append(dict)

        fea_list = []
        for dict in dict_list:
            for id in dict:
                fea_list.append(dict[id])

        self.transform = transform
        self.indices = fea_list
        self.class_num = len(self.indices)
        print("class_num: ", self.class_num)

    def load_all_features(self):
        return

    def __getitem__(self, index):
        def transform(feature):
            return self.transform(feature)

        c1 = np.random.randint(0, self.class_num)
        n1 = np.random.randint(0, len(self.indices[c1]))

        framefeas = np.load(self.indices[c1][n1])
        fea = truncate_fea(framefeas, mode="train")
        feature_a = transform(fea)
        return feature_a, c1

    def __len__(self):
        return self.num
