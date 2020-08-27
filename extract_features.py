#! python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import random
import librosa
from python_speech_features import logfbank, fbank, delta


NUM_PREVIOUS_FRAME = 0 
NUM_NEXT_FRAME = 256
NUM_FRAMES = NUM_PREVIOUS_FRAME + NUM_NEXT_FRAME

USE_LOGSCALE = True
USE_DELTA = False
USE_SCALE = False
SAMPLE_RATE = 16000
TRUNCATE_SOUND_FIRST_SECONDS = 0.5
FILTER_BANK = 64

USING_LOG1P = False  # True log 离散化

################################################################################################################

def feature_normalize(v):
    return (v - v.mean(axis=0)) / (v.std(axis=0) + 2e-12)

################################################################################################################

def truncate_fea(frames_features, mode="train"):
    frames_features_t = frames_features.T
    freq, time = frames_features_t.shape

    if "train" == mode:
        try:
            j = random.randint(NUM_PREVIOUS_FRAME, time - NUM_NEXT_FRAME)
            spec_mag = frames_features_t[:, j -
                                         NUM_PREVIOUS_FRAME: j + NUM_NEXT_FRAME]
        except:            
            spec_mag = cv2.resize(frames_features_t, (NUM_FRAMES, freq))
    else:
        spec_mag = frames_features_t

    final_data_t = feature_normalize(spec_mag)

    final_data = final_data_t.T

    final_feature = np.expand_dims(final_data, 0)
    return final_feature


def frames_fea(path, mode="train", feat_opt="melspec", double_length=False):
    try:
        (y, rate) = librosa.load(path, sr=SAMPLE_RATE)
    except Exception as e:
        print("Error audio file: ", path)
        return None

    # Trimming
    y, _ = librosa.effects.trim(y)

    # 预加重
    y = np.append(y[0], y[1:] - 0.97*y[:-1])  # TODO need check again!

    if double_length:
        if "train" == mode:
            sig = np.concatenate([y, y])
            # if np.random.random() < 0.3:
            # sig = sig[::-1]
        else:
            # sig = np.concatenate([y, y[::-1]])
            sig = np.concatenate([y, y])
    else:
        sig = y

    try:
        if "f26" == feat_opt:
            feat, energy = fbank(sig, rate, winfunc=np.hamming)
            filter_banks = np.log(feat)
            # print("f26: ", filter_banks.shape)
            return filter_banks
        elif "f40" == feat_opt:
            feat, energy = fbank(sig, rate, nfilt=40, winfunc=np.hamming)
            filter_banks = np.log(feat)
            # print("f40: ", filter_banks.shape)
            return filter_banks
        elif "melspec" == feat_opt:
            S = librosa.feature.melspectrogram(
                y=sig, sr=rate, window="hamm", n_fft=512, hop_length=160, win_length=400)
            feature = S.T
            # print("melspec: ", fea.shape)
            return feature
        elif "mel_pitch" == feat_opt:
            S = librosa.feature.melspectrogram(
                y=sig, sr=rate, window="hamm", n_fft=512, hop_length=160, win_length=400)
            # print("S: ", S.shape)
            pitches, magnitudes = librosa.piptrack(
                y=sig, sr=rate, window="hamm", n_fft=512, hop_length=160, win_length=400)
            # print("pitches: ", pitches.shape)
            # print("mags: ", magnitudes.shape)
            feature_t = np.concatenate([S, pitches])
            feature = feature_t.T
            return feature
        else:
            raise Exception("unsupported feat_opt: {}".format(feat_opt))

    except Exception:
        raise Exception("Error: %s" % path)


def get_truncated_features(wav_path, mode="train", feat_opt="stft"):
    frames_features = frames_fea(wav_path, mode=mode, feat_opt=feat_opt)
    truncated_features = truncate_fea(frames_features, mode=mode)
    return truncated_features
