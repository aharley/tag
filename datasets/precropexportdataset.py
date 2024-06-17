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
# np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')
# import torchvision.transforms.functional as F
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
        print('loading precrop export...')

        self.dataset_location = dataset_location
        self.S = S
        self.horz_flip = horz_flip
        self.vert_flip = vert_flip
        self.time_flip = time_flip
        self.dataset_location = Path(self.dataset_location) / version

        self.mean = torch.as_tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1).float()
        self.std = torch.as_tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1).float()
        
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
        self.all_idxs = []

        # self.all_folder_names = folder_names

        # step through once and make sure all of the npzs are there
        self.all_folder_names = []
        for fi, folder in enumerate(folder_names):
            if os.path.isfile('%s/track.npz' % folder):
                self.all_folder_names.append(folder)
            else:
                print('missing track in %s' % folder)
        
        print('found {:d} {} samples in {}'.format(len(self.all_folder_names), version, self.dataset_location))
        
    def __getitem__(self, index):
        folder = self.all_folder_names[index]
        # print('folder', folder)

        scales = [0.25,1.0]
        N = len(scales)
        cH, cW = 128, 128
        stride = 32
        sH, sW = cH//stride, cW//stride
        final_stride = 4
        fH, fW = cH//final_stride, cW//final_stride

        fake_sample = {
            'crop_rgbs': torch.zeros((self.S, N, 3, cH, cW), dtype=torch.float32), 
            'crop_masks': torch.zeros((self.S, 1, fH, fW), dtype=torch.float32), 
            'crop_xys': torch.zeros((self.S, N, 2, sH, sW), dtype=torch.float32), 
            'crop_valids': torch.zeros((self.S, N, 1, sH, sW), dtype=torch.float32), 
            'track_g': torch.zeros((self.S, 8), dtype=torch.float32), 
            'xywhs_e': torch.zeros((self.S, 4), dtype=torch.float32), 
            'dname': 'none',
            'folder': 'none',
            'step': torch.zeros((), dtype=torch.int32), 
        }
        
        try: 
            d = dict(np.load(folder + '/track.npz', allow_pickle=True))
        except zipfile.BadZipfile:
            d = None
        if d is None:
            print('some problem with folder', folder)
            return fake_sample
            # index = np.random.randint(len(self.all_folder_names))

        # print('folder', folder, 'keys', d.keys())
        # print('crop_valids.shape', d['crop_valids'].shape)
        track_g = d['track_g']
        xywhs_e = d['xywhs_e']
        crop_rgbs = d['crop_rgbs']
        crop_masks = d['crop_masks']
        crop_xys = d['crop_xys']
        crop_valids = d['crop_valids']
        dname = str(d['dname'])
        # print('export dname', dname)

        S = track_g.shape[0]
        
        track_g = torch.from_numpy(track_g).float() # S,8
        xywhs_e = torch.from_numpy(xywhs_e).float() # S,4
        crop_rgbs = torch.from_numpy(crop_rgbs).float() # S,N,3,cH,cW
        crop_masks = torch.from_numpy(crop_masks).float() # S,N,1,cH,cW
        crop_xys = torch.from_numpy(crop_xys).float() # S,N,2,cH,cW
        crop_valids = torch.from_numpy(crop_valids).float() # S,N,1,cH,cW
        
        # undo the scaling we put for uint8
        crop_xys = (crop_xys - 128)*2.0
        
        # apply mean/std
        crop_rgbs_ = crop_rgbs.reshape(S*N,3,cH,cW)
        crop_rgbs_ = crop_rgbs_ / 255.0
        crop_rgbs_ = (crop_rgbs_ - self.mean)/self.std
        crop_rgbs = crop_rgbs_.reshape(S,N,3,cH,cW)
        step = torch.zeros((), dtype=torch.int32)

        crop_masks_pos = (crop_masks > 192.0) * 1.0
        crop_masks_ign = (crop_masks > 64.0) * 1.0
        crop_masks = (crop_masks_pos + crop_masks_ign)/2.0 

        xywhs_e = xywhs_e.reshape(S,4)
        track_g = track_g.reshape(S,8)

        crop_masks = crop_masks[:,-1] # last one, which is scale 1.0; S,1,cH,cW
        crop_masks = F.max_pool2d(crop_masks,
                                  kernel_size=final_stride,
                                  stride=final_stride).reshape(S,1,fH,fW)

        sample = {
            'crop_rgbs': crop_rgbs,
            'crop_masks': crop_masks,
            'crop_xys': crop_xys,
            'crop_valids': crop_valids,
            'track_g': track_g,
            'xywhs_e': xywhs_e,
            'dname': dname,
            'folder': folder,
            'step': step,
        }
        return sample
        

    def __len__(self):
        return len(self.all_folder_names)
