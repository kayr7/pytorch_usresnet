from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
import struct
import matplotlib.pyplot as plt
import scipy
import scipy.signal as scipy_signal
import glob, os
import wave
import torch.utils.data as data

import os
import os.path
import transforms as transforms


AUDIO_EXTENSIONS = [
    '.wav',
]


def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)


def make_dataset(dir, sample_length, max=100):
    audios = []
    counter = 0
    for target in os.listdir(dir):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in fnames:
                if is_audio_file(fname):
                    item = os.path.join(root, fname)
                    try:
                        waveobject = wave.open(item)
                        sample_width = waveobject.getsampwidth()
                        samples = waveobject.getnframes()
                        if (samples > sample_length and
                            sample_width == 2 and
                            waveobject.getframerate() == 16000 and
                            waveobject.getnchannels() == 1):
                            frame_data = waveobject.readframes(samples)
                            if frame_data:
                                sample_width = waveobject.getsampwidth()
                                nb_samples = len(frame_data) // sample_width
                                if nb_samples < sample_length:
                                    continue
                                format = {1:"%db", 2:"<%dh", 4:"<%dl"}[sample_width] % nb_samples
                                audio = struct.unpack(format, frame_data)
                                audios.append(audio)
                                if len(audios) >= max:
                                    return audios
                            else:
                                continue


                    except Exception as e:
                        print('Error {} processing file {}'.format(e, item))

    return audios



class AudioFolder(data.Dataset):

    def __init__(self, root, sample_length=16384):
        self.sample_length = sample_length
        audios = make_dataset(root, max=8000, sample_length=sample_length)
        if len(audios) == 0:
            raise(RuntimeError("Found 0 wavs in subfolders of: " + root + "\n"
                               "Supported wav extensions are: " + ",".join(AUDIO_EXTENSIONS)))
        print('loaded {}'.format(len(audios)))
        self.audios = audios

    def __getitem__(self, index):
        index = random.randint(0, len(self.audios) - 1)
        data = self.audios[index]
        samples = len(data)
        sample_start = random.randint(0, samples - self.sample_length -1)
        target = np.asarray(data[sample_start : sample_start + self.sample_length])
        low_res = scipy_signal.resample(target, self.sample_length // 2)
#        target = np.ndarray(list(target))

        aud = torch.from_numpy(low_res[None, :]).float().div(32768) # +channel..
        tgt = torch.from_numpy(target[None, :]).float().div(32768)
        return aud, tgt

    def __len__(self):
        return len(self.audios)-1
