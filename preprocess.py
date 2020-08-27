#! python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import numpy as np
import shutil
from tqdm import tqdm
import csv
import codecs

from extract_features import frames_fea

train_audio_dir_path = "/data/Speech/XunFei/BabyCry/train"
train_audio_fea_dir_path = "/data/Speech/XunFei/BabyCry/train_feas"


############################################################################################################
def create_audio_lists(audio_dir, printable=True):
    if not os.path.exists(audio_dir):
        print("Image directory '" + audio_dir + "' not found.")
        return None
    file_list = []
    file_glob = os.path.join(audio_dir, '*.' + 'wav')
    file_list.extend(glob.glob(file_glob))
    file_glob = os.path.join(audio_dir, '*.' + 'flac')
    file_list.extend(glob.glob(file_glob))

    if printable:
        print(len(file_list))
    return file_list


def create_audio_lists_recursive(audio_dir):
    total_list = []
    for i in os.walk(audio_dir):
        cur_path = i[0]
        # print(cur_path)
        list = create_audio_lists(cur_path, printable=False)
        total_list.extend(list)
    print(len(total_list))
    return total_list


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
    print(len(list))
    return list


############################################################################################################
def create_train_audio_list():
    wav_list = create_audio_lists_recursive(train_audio_dir_path)
    dict = {}
    for wav in wav_list:
        tmp_dir = os.path.split(wav)[0]
        id_name = os.path.split(tmp_dir)[1]

        if id_name not in dict:
            dict[id_name] = [wav]
        else:
            dict[id_name].append(wav)

    print('ID num: ' + str(len(dict)))
    txt_path = r'train_audio.txt'
    f = open(txt_path, 'w')
    index = 0
    for id in dict:
        wav_list = dict[id]

        for wav in wav_list:
            f.write(wav + ' ' + str(index) + '\n')
            # print(wav+' '+str(index)+'/n')
        index += 1
    f.close()
    print('done')


def extract_train_audio_fea():
    txt_path = r'train_audio.txt'
    list = read_list(txt_path)

    txt_save_path = r'train_audio_feas.txt'
    f = open(txt_save_path, 'w')

    count = 0
    for wav_path, label in list:
        npy_path_tmp = wav_path.replace('.wav', '_FrameFeas_' + '.npy')
        npy_path_tmp = npy_path_tmp.replace(train_audio_dir_path, train_audio_fea_dir_path)

        if os.path.exists(npy_path_tmp):
            frames_features = np.load(npy_path_tmp)
            f.write(npy_path_tmp + ' ' + str(label) + '\n')
        else:
            try:
                frames_features = frames_fea(wav_path, mode="train", feat_opt="melspec")
                npy_path_tmp_dir = os.path.split(npy_path_tmp)[0]
                if not os.path.exists(npy_path_tmp_dir):
                    os.makedirs(npy_path_tmp_dir)
                np.save(npy_path_tmp, frames_features)
                f.write(npy_path_tmp + ' ' + str(label) + '\n')
            except Exception:
                print(wav_path)
                # os.remove(wav_path)
                raise Exception

        count += 1
        if count % 100 == 0:
            print(count)
    f.close()


if __name__ == '__main__':
    create_train_audio_list()
    extract_train_audio_fea()
