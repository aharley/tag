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
import glob
import json
import imageio
import cv2
import re
from torchvision.transforms import ColorJitter, GaussianBlur
import pandas as pd
from scipy.interpolate import UnivariateSpline
from enum import Enum
from datasets.dataset import PointDataset

class BadjaDataset(PointDataset):
    def __init__(self,
                 dataset_location='/orion/u/aharley/badja2', 
                 use_augs=False,
                 S=8,
                 N=32,
                 strides=[1,2,3,4],
                 crop_size=(368, 496),
                 is_training=True):
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size, 
            use_augs=use_augs,
            is_training=is_training
        )
        print('loading badja dataset...')
        
        npzs = glob.glob('%s/complete_aa/*.npz' % self.dataset_location)
        npzs = sorted(npzs)
        # print('npzs', npzs)

        if self.is_training:
            npzs = npzs[-2:] # only use last two videos for training (dog running and gazelle)
        else: 
            npzs = npzs[:-2]

        df = pd.read_csv('%s/picks_and_coords.txt' % self.dataset_location, sep=' ', header=None)
        # print('df', df)
        track_names = df[0].tolist()
        pick_frames = np.array(df[1])
        # pick_frames gives: for each keypoint, the frame that serves as a fair init
        self.all_npzs = npzs
        
        self.all_npz_idxs = []
        self.all_animal_names = []
        self.all_kp_idxs = []
        self.all_full_idxs = []

        for ind in range(len(npzs)):
            o = np.load(npzs[ind])

            animal_name = o['animal_name']
            # self.animal_names.append(animal_name)
            trajs_g = o['trajs_g']
            valids_g = o['valids_g']
            S_local, N, D = trajs_g.shape

            for kp_id in range(N):
                short_name = '%s_%02d' % (animal_name, kp_id)
                txt_id = track_names.index(short_name)
                pick_id = pick_frames[txt_id]
                if pick_id >= 0:
                    for stride in strides:
                        full_idx = pick_id + np.arange(self.S) * stride
                        full_idx = [ij for ij in full_idx if ij < S_local]
                        valid = valids_g[full_idx,kp_id]
                        if len(full_idx) >= 8 and np.sum(valid) > 3:
                            self.all_npz_idxs.append(ind)
                            self.all_animal_names.append(animal_name)
                            self.all_kp_idxs.append(kp_id)
                            self.all_full_idxs.append(full_idx)
            
    def __len__(self):
        return len(self.all_full_idxs)

    def getitem_helper(self, index):

        npz_idx = self.all_npz_idxs[index]
        animal_name = self.all_animal_names[index]
        kp_id = self.all_kp_idxs[index]
        full_idx = self.all_full_idxs[index]
        o = np.load(self.all_npzs[npz_idx])

        animal_name = o['animal_name']
        trajs_g = o['trajs_g']
        valids_g = o['valids_g']
        S_local, N, D = trajs_g.shape

        traj = trajs_g[full_idx,kp_id]
        valid = valids_g[full_idx,kp_id]

        visibs = np.ones_like(valid)

        filenames = glob.glob('%s/videos/%s/*.png' % (self.dataset_location, animal_name)) + glob.glob('%s/videos/%s/*.jpg' % (self.dataset_location, animal_name))
        filenames = sorted(filenames)
        filenames = [filenames[fi] for fi in full_idx]

        rgb = cv2.imread(str(filenames[0]))[..., ::-1].copy()
        H, W = rgb.shape[:2]
        # print("H, W", H, W)
        sc = 1.0
        if H > 256:
            sc = 256 / H
            H_, W_ = int(H * sc), int(W * sc)
        rgbs = []
        for path in filenames:
            rgb = cv2.imread(str(path))[..., ::-1].copy()
            if sc < 1.0:
                rgb = cv2.resize(rgb, (W_, H_), interpolation=cv2.INTER_AREA)
            rgbs.append(rgb)
        rgbs = np.stack(rgbs, 0)
        # print('rgbs', rgbs.shape)

        if sc < 1.0:
            traj = traj * sc

        sample = {
            'rgbs': rgbs,
            'trajs': traj,
            'visibs': visibs,
        }
        return sample
    
        

    
