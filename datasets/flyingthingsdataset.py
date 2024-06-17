import time
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
import utils.basic
import utils.improc
import glob
import json
import imageio
import cv2
import re
import sys
from torchvision.transforms import ColorJitter, GaussianBlur
from datasets.dataset import PointDataset
from pathlib import Path
# np.random.seed(125)
# torch.multiprocessing.set_sharing_strategy('file_system')


def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data

def readImage(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        data = readPFM(name)
        if len(data.shape)==3:
            return data[:,:,0:3]
        else:
            return data
    return imageio.imread(name)

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

class FlyingThingsDataset(PointDataset):
    def __init__(self,
                 dataset_location='../datasets/flt_export',
                 version='aj',
                 S=32,
                 chunk=None,
                 rand_frames=False,
                 crop_size=(384,512), 
                 use_augs=False,
                 is_training=True):
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            rand_frames=rand_frames,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training
        )
        print('loading flyingthingsdataset...')
        
        self.dataset_location = Path(self.dataset_location) / version / 'export'

        folder_names = sorted(self.dataset_location.glob('*/*/'))
        folder_names = [fn for fn in folder_names if (fn.stem[-1] != '9' if is_training else fn.stem[-1] == '9')]
        # print('found {:d} {} samples in {}'.format(len(self.folder_names), ('train' if is_training else 'test'), self.dataset_location))

        if chunk is not None:
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            folder_names = chunkify(folder_names,100)[chunk]
        
        self.all_tids = []
        self.all_folder_names = []

        self.N_per = 8

        if not self.is_training:
            self.N_per = 1

        for folder_name in folder_names:
            for ni in range(self.N_per):
                self.all_tids.append(ni)
                self.all_folder_names.append(folder_name)
            
        print('found {:d} {} samples in {}'.format(len(self.all_folder_names), ('train' if is_training else 'test'), self.dataset_location))


    def getitem_helper(self, index):
        
        folder = self.all_folder_names[index]
        tid = self.all_tids[index]
        
        try: 
            d = dict(np.load(folder / 'trajs.npz', allow_pickle=True))
        except:
            print('problem with %s; returning None' % folder)
            return None

        trajs = d['trajs'].astype(int) # S,N,2
        visibs = d['visibs'] # S,N
        valids = np.ones_like(visibs)

        S = trajs.shape[0]

        rgbs = read_mp4(str(folder / 'rgb.mp4'))
        rgbs = np.stack(rgbs, axis=0) # S,H,W,3
        S,H,W,C = rgbs.shape
        
        if S > self.S:
            rgbs = rgbs[:self.S]
            trajs = trajs[:self.S]
            visibs = visibs[:self.S]
            valids = valids[:self.S]
            S = self.S

        # update visibility annotations
        for si in range(S):
            # avoid 1px edge
            oob_inds = np.logical_or(
                np.logical_or(trajs[si,:,0] < 1, trajs[si,:,0] > W-2),
                np.logical_or(trajs[si,:,1] < 1, trajs[si,:,1] > H-2))
            visibs[si,oob_inds] = 0

            # when a point moves far oob, don't supervise with it
            very_oob_inds = np.logical_or(
                np.logical_or(trajs[si,:,0] < -64, trajs[si,:,0] > W+64),
                np.logical_or(trajs[si,:,1] < -64, trajs[si,:,1] > H+64))
            valids[si,very_oob_inds] = 0

        # during preprocessing, we already checked for basic visibility in 0 and 1, and motion thresh.
        # extra: 
        # ensure that the point is good in at least S//2 frames (since this is already super hard)
        # vis_ok = np.sum(visibs, axis=0) >= max(S//2,2)
        # trajs = trajs[:,vis_ok]
        # visibs = visibs[:,vis_ok]
        # valids = valids[:,vis_ok]
        vis_valid = valids * visibs
        vis_safe = vis_valid*0
        for si in range(1,S-1):
            vis_safe[si] = vis_valid[si-1]*vis_valid[si]*vis_valid[si+1]
        visval_ok = np.sum(vis_safe, axis=0) >= max(np.sqrt(S),2)
        trajs = trajs[:,visval_ok]
        visibs = visibs[:,visval_ok]
        valids = valids[:,visval_ok]

        N = trajs.shape[1]

        # print('trajs', trajs.shape)
        if N > 32:
            # do FPS based on the trajectory shape
            trajs_ = np.minimum(np.maximum(trajs, np.zeros((2,), dtype=int)), np.array([W, H]) - 1) # S,2
            trajs_ = np.transpose(trajs_ - trajs_[0:1], [1,0,2]).reshape(N,S*2)
            inds = utils.misc.farthest_point_sample_py(trajs_, 32, deterministic=True)
            trajs = trajs[:,inds]
            visibs = visibs[:,inds]
            valids = valids[:,inds]
        
        if N < self.N_per:
            print('N=%d' % (N))
            return None

        # take the tid
        trajs = trajs[:,tid] # S,2
        visibs = visibs[:,tid] # S
        valids = valids[:,tid] # S

        # clamp to image bounds
        trajs = np.minimum(np.maximum(trajs, np.zeros((2,), dtype=int)), np.array([W, H]) - 1) # S,2
        
        return {
            'rgbs': rgbs,
            'trajs': trajs,
            'visibs': visibs,
            'valids': valids,
        }
    
    def __len__(self):
        # return 10
        return len(self.all_folder_names)
