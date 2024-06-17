from numpy import random
from numpy.core.numeric import full
import torch
import numpy as np
import os
import scipy.ndimage
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import random
import zipfile
from torch._C import dtype, set_flush_denormal
import utils.geom
import glob
import json
import imageio
import cv2
import re
from torchvision.transforms import ColorJitter, GaussianBlur
from datasets.dataset import MaskDataset
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import albumentations as A

# np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')

import torchvision.transforms.functional as F

from einops import rearrange, repeat


def read_mp4(fn):
    try:
        vidcap = cv2.VideoCapture(fn)
        frames = []
        while(vidcap.isOpened()):
            ret, frame = vidcap.read()
            if ret == False:
                break
            frames.append(frame)
        vidcap.release()
        return frames
    except:
        print('some problem with file', fn)
        return []

# augment video sequences with the same augmenter, only support single bbox now
def augment_video(augmenter, rgbs):
    assert isinstance(augmenter, A.ReplayCompose)
    for i in range(len(rgbs)):
        data = augmenter(image=rgbs[i])
        rgbs[i] = data['image']
    return rgbs
    

class ExportDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_location='/orion/group/alltrack_export',
                 version='au_trainA',
                 dsets=None,
                 S=32,
                 horz_flip=False,
                 vert_flip=False,
                 time_flip=False,
                 use_augs=False,
                 is_training=True,
    ):
        print('loading export...')

        self.dataset_location = dataset_location
        self.S = S
        self.horz_flip = horz_flip
        self.vert_flip = vert_flip
        self.time_flip = time_flip
        self.use_augs = use_augs
        self.dataset_location = Path(self.dataset_location) / version

        self.mean = torch.as_tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1).float()
        self.std = torch.as_tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1).float()

        self.color_augmenter = A.ReplayCompose([
            A.GaussNoise(p=0.1),
            A.OneOf([
                A.MotionBlur(),
                A.MedianBlur(),
                A.Blur(),
            ], p=0.5),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
            ], p=0.5),
            A.RGBShift(),
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
            A.HueSaturationValue(),
            A.ImageCompression(),
        ], p=0.8)

        # self.photo_aug = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.25/3.14)
        
        dataset_names = self.dataset_location.glob('*/')
        self.dataset_names = [fn.stem for fn in dataset_names]

        print('dataset_names', self.dataset_names)

        folder_names = self.dataset_location.glob('*/*/*/')
        # folder_names = [fn for fn in folder_names if (fn.stem[-1] != '9' if is_training else fn.stem[-1] == '9')]
        
        folder_names = [str(fn) for fn in folder_names]

        folder_names = sorted(list(folder_names))
        # print('found {:d} {} folders in {}'.format(len(folder_names), ('train' if is_training else 'test'), self.dataset_location))
        print('found {:d} {} folders in {}'.format(len(folder_names), version, self.dataset_location))
        # print('found {:d} samples in {}'.format(len(self.all_folder_names), self.dataset_location))

        # print('folder_names', folder_names)
        if dsets is not None:
            print('dsets', dsets)
            new_folder_names = []
            for fn in folder_names:
                for dset in dsets:
                    if dset in fn:
                        new_folder_names.append(fn)
                        break
            folder_names = new_folder_names
            print('filtered to %d folders' % len(folder_names))

        self.all_folder_names = []
        for fi, folder in enumerate(folder_names):
            if os.path.isfile('%s/track.npz' % folder):
                self.all_folder_names.append(folder)
            else:
                print('missing track in %s' % folder)
        
        print('found {:d} {} samples in {}'.format(len(self.all_folder_names), version, self.dataset_location))
        
    # borrowed from https://github.com/shanice-l/gdrnpp_bop2022/blob/f3ca18632f4b68c15ab3306119c364a0497282a7/det/yolox/data/datasets/mosaicdetection.py#L198
    def __color_augment__(self, rgbs):
        augment_video(self.color_augmenter, rgbs)
        
    def __getitem__(self, index):
        folder = self.all_folder_names[index]
        # print('folder', folder)

        d = None
        while d is None:
            try: 
                d = dict(np.load(folder + '/track.npz', allow_pickle=True))
                # except: #zipfile.BadZipfile:
            except Exception as e:
                print(e)
                d = None
            if d is None:
                print('some problem with folder', folder)
            index = np.random.randint(len(self.all_folder_names))
            folder = self.all_folder_names[index]
        
        
        rgbs = d['rgbs']
        track_g = d['track_g']
        dname = str(d['dname'])

        S = track_g.shape[0]

        if self.use_augs:
            # the augmenter wants channels last
            rgbs = np.transpose(rgbs, [0,2,3,1])
            self.__color_augment__(rgbs)
            rgbs = np.transpose(rgbs, [0,3,1,2])

        # if np.random.rand() < 0.5:
        #     # random per-frame amount of aug
        #     rgbs = [np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs]
        #     rgbs = np.stack(rgbs, axis=0)
        
        rgbs = torch.from_numpy(rgbs).float()
        track_g = torch.from_numpy(track_g).float()

        if self.S < S:
            rgbs = rgbs[:self.S]
            track_g = track_g[:self.S]
        
        rgbs = rgbs / 255.0
        rgbs = (rgbs - self.mean)/self.std
        
        sample = {
            'rgbs': rgbs,
            'track_g': track_g,
            'dname': dname,
        }

        
        return sample
        

    def __len__(self):
        return len(self.all_folder_names)
