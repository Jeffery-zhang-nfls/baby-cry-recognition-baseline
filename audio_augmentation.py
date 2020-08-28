#-*-coding:utf-8-*-

import librosa
import numpy as np
import random
import os , time
import soundfile

from multiprocessing import Pool

sample_rate = 16000
pitch_shift1, pitch_shift2 = 0.01 , 5.0
time_stretch1, time_stretch2 = 0.05, 0.25
augmentation_num = 10


def audio_augmentation(wav_file):
    print("original wav file: ", wav_file)
    y, sr = librosa.load(wav_file, sr=sample_rate)
    for j in range(augmentation_num):
        rd1 = random.uniform(pitch_shift1, pitch_shift2)
        ii = random.choice((-1, 1))
        rd2 = random.uniform(time_stretch1, time_stretch2)
        rd2 = 1.0 + ii * rd2

        y_ps = librosa.effects.pitch_shift(y, sr, n_steps = rd1)
        y_ts = librosa.effects.time_stretch(y, rate = rd2)

        dir_path, wav_file_name = os.path.split(wav_file)
        wav_name = wav_file_name.split('.')[0]

        ps_wav = os.path.join(dir_path, wav_name + '_ps_' + str(j) + '.wav')
        ts_wav = os.path.join(dir_path, wav_name + '_ts_' + str(j) + '.wav')

        print("pitch_shift: ", ps_wav)
        print("time_stretch: ", ts_wav)
        
        # librosa.output.write_wav(ps_wav, y_ps, sample_rate)
        # librosa.output.write_wav(ts_wav, y_ts, sample_rate)
        soundfile.write(ps_wav, y_ps, sample_rate)
        soundfile.write(ts_wav, y_ts, sample_rate)
    return


if __name__ == "__main__":
    import sys
    from datasets import create_audio_lists_recursive
    train_audio_dir_path = "/data/Speech/XunFei/BabyCry/train"

    all_wav_list = create_audio_lists_recursive(train_audio_dir_path)

    random.shuffle(all_wav_list)
    all_file_len = len(all_wav_list)
    print('all wav file len:',all_file_len)

    print('start wav data augmentation ...')

    # multi processes
    with Pool(20) as p:
        p.map(audio_augmentation, all_wav_list)

    # single process
    # for i in range(all_file_len):
    #     audio_file = all_wav_list[i]
    #     audio_augmentation(audio_file)

    #     if (i % 100 == 0):
    #         now = time.localtime()
    #         now_time = time.strftime("%Y-%m-%d %H:%M:%S", now)
    #         print('time:', now_time)
    #         print('predict num:', i)

    print('wav data augmentation done ...')
