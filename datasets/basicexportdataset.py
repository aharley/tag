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

from torch._C import dtype, set_flush_denormal
# from detectron2.structures.masks import polygons_to_bitmask

# import utils.py
# import utils.basic
import utils.geom
# import utils.improc

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

def read_mp4(fn):
    vidcap = cv2.VideoCapture(fn)
    frames = []
    while(vidcap.isOpened()):
        ret, frame = vidcap.read()
        if ret == False:
            break
        frames.append(frame)
    vidcap.release()
    return frames

class ExportDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_location='../datasets/alltrack_export',
                 version='au_trainA',
                 dsets=None,
                 S=32,
                 rand_frames=False,
                 crop_size=(384,512), 
                 use_augs=False,
                 is_training=True,
    ):
        print('loading export...')

        self.dataset_location = dataset_location
        self.S = S
        self.H, self.W = crop_size
        
        self.dataset_location = Path(self.dataset_location) / version

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

        # load one and make sure it matches
        rgbs = read_mp4(folder_names[0] + '/rgb.mp4')
        S_local = len(rgbs)
        assert(self.S<=S_local)
        H, W, C = rgbs[0].shape
        assert(self.H==H)
        assert(self.W==W)

        self.all_idxs = [np.arange(self.S) for _ in range(len(folder_names))]
        self.all_folder_names = folder_names

        # # step through once and make sure all of the npzs are there
        # self.all_idxs = []
        # self.all_folder_names = []
        # for fi, folder in enumerate(folder_names):
        #     if os.path.isfile('%s/track.npz' % folder):
        #         self.all_idxs.append(np.arange(self.S))
        #         self.all_folder_names.append(folder)
        #     else:
        #         print('missing track in %s' % folder)
        
                
        # if S_local==S:
        #     self.all_idxs = [np.arange(self.S) for _ in range(len(folder_names))]
        #     self.all_folder_names = folder_names
        # else:
        #     for fi, folder in enumerate(folder_names):
        #         if os.path.isfile(str(folder / 'track.npz')): 
        #             for ii in range(0,S_local-self.S+1,8):
        #                 full_idx = ii + np.arange(self.S)
        #                 self.all_folder_names.append(folder)
        #                 self.all_idxs.append(full_idx)
        #                 sys.stdout.write('.')
        #                 sys.stdout.flush()
        # print('found {:d} {} samples in {}'.format(len(self.all_folder_names), ('train' if is_training else 'test'), self.dataset_location))
        print('found {:d} {} samples in {}'.format(len(self.all_folder_names), version, self.dataset_location))
        # print('found {:d} samples in {}'.format(len(self.all_folder_names), self.dataset_location))
        
    def __getitem__(self, index):
        folder = self.all_folder_names[index]
        full_idx = self.all_idxs[index]
        # print('full_idx', full_idx)
        # print('folder', folder)
        
        rgbs = read_mp4(folder  + '/rgb.mp4')
        masks = read_mp4(folder + '/mask.mp4')
        
        if len(masks)<self.S or len(rgbs)<self.S:
            print('mp4 rgb,mask len %d,%d in %s; returning fake' % (len(rgbs), len(masks), folder))
            fake_sample = {
                'rgbs': np.zeros((self.S,3,self.H,self.W), dtype=np.uint8), 
                'masks_g': np.zeros((self.S,3,self.H,self.W), dtype=np.float32), 
                'track_g': np.zeros((self.S,8), dtype=np.float32),
                'dname': 'none',
                'folder': 'none',
                'index': 0,
            }
            return fake_sample, False
        
        rgbs = np.stack(rgbs, axis=0) # S,H,W,3
        masks = np.stack(masks, axis=0) # S,H,W,3
        
        d = dict(np.load(folder + '/track.npz', allow_pickle=True))
        track = d['track_g']
        dname = str(d['dname'])
        # print('export dname', dname)
        
        rgbs = rgbs[full_idx]
        masks = masks[full_idx]
        track = track[full_idx]

        rgbs = rgbs.transpose(0,3,1,2)
        rgbs = rgbs[:,::-1].copy() # BGR->RGB

        masks = masks.transpose(0,3,1,2)#[:,0:1]
        masks_pos = (masks > 192.0) * 1.0
        masks_ign = (masks > 64.0) * 1.0
        masks = (masks_pos + masks_ign)/2.0 


        # print('folder, rgbs', folder, rgbs.shape)
        # print('rgbs', rgbs.shape)
        # print('masks', masks.shape)
        # print('track', track.shape)
        # print('dname', dname)
        # print('folder', folder)
        # print('index', index)
        
        assert(not np.any(np.isnan(rgbs)))
        assert(not np.any(np.isnan(masks)))
        assert(not np.any(np.isnan(track)))


        rgbs = torch.from_numpy(rgbs).float()
        masks_g = torch.from_numpy(masks).float()
        track_g = torch.from_numpy(track).float()

        S,C,H,W = rgbs.shape
        
        
        xywhs_g = track_g[:,:4]

        xys_valid = track_g[:,5]
        
        # touching any border means invalid
        xs_val0 = xywhs_g[:,0] >= 1
        xs_val1 = xywhs_g[:,0] <= W-2
        ys_val0 = xywhs_g[:,1] >= 1
        ys_val1 = xywhs_g[:,1] <= H-2
        xys_valid = xys_valid * (xs_val0.float() * xs_val1.float() * ys_val0.float() * ys_val1.float())
        track_g[:,5] = xys_valid

        xywhs_e0 = torch.rand((S,4), dtype=torch.float32) * min(H,W)
        xywhs_e0 = xywhs_e0
        # move closer to zero
        xywhs_e0 = xywhs_e0*0.1 + xywhs_g[0:1]*0.9
        # reset zeroth, so that it's clear what to track
        xywhs_e0[0] = xywhs_g[0]
        
        # xys_g = track_g[:,:,0:2]
        # whs_g = track_g[:,:,2:4]
        # vis_g = track_g[:,:,4]
        # xys_valid = track_g[:,:,5]
        # whs_valid = track_g[:,:,6]
        # vis_valid = track_g[:,:,7]
        step = 0#np.zeros((1), dtype=np.int32)
        
        sample = {
            'rgbs': rgbs,
            'masks_g': masks,
            'track_g': track,
            'xywhs_e0': xywhs_e0,
            'dname': dname,
            'folder': folder.split('/')[-1],
            'index': index,
            'step': step,
        }
        return sample, True

    def __len__(self):
        return len(self.all_folder_names)
