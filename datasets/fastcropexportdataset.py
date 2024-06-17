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

def get_multiscale_crops(images_, xywhs_, cH, cW, scales=[0.25,1.0]):
    B,C,H,W = images_.shape
    B2,D = xywhs_.shape
    assert(B==B2)
    assert(D==4)
    all_crops = []
    for si, sc in enumerate(scales):
        boxlist_ = utils.geom.get_boxlist_from_centroid_and_size(
            xywhs_[:,1],
            xywhs_[:,0],
            xywhs_[:,3].clamp(min=64)/sc,
            xywhs_[:,2].clamp(min=64)/sc,
        ).unsqueeze(1) # B,1,4
        crops_ = utils.geom.crop_and_resize(images_, boxlist_, cH, cW) # B,3,cH,cW
        crops = crops_.reshape(B,C,cH,cW)
        all_crops.append(crops)
    all_crops = torch.stack(all_crops, dim=1) # B,N,C,cH,cW
    return all_crops


def read_mp4(fn):
    try:
        print('reading', fn)
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
                 dataset_location='../datasets/alltrack_export',
                 version='au_trainA',
                 dsets=None,
                 S=32,
                 rand_frames=False,
                 crop_size=(384,512), 
                 horz_flip=False,
                 vert_flip=False,
                 time_flip=False,
                 use_augs=False,
                 is_training=True,
    ):
        print('loading export...')

        self.dataset_location = dataset_location
        self.S = S
        self.H, self.W = crop_size
        self.horz_flip = horz_flip
        self.vert_flip = vert_flip
        self.time_flip = time_flip
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

        # # self.all_idxs = [np.arange(self.S) for _ in range(len(folder_names))]
        # self.all_folder_names = folder_names

        # step through once and make sure all of the npzs are there
        self.all_idxs = []
        self.all_folder_names = []
        for fi, folder in enumerate(folder_names):

            # ffmpeg -v error -i file.avi -f null - 2>error.log
            # os.system('/usr/bin/ffmpeg -v error -i %s/rgb.mp4 -f null - 2>>error.log' % (folder))
            # os.system('/usr/bin/ffmpeg -v error -i %s/mask.mp4 -f null - 2>>error.log' % (folder))
            # out = os.popen('/usr/bin/ffmpeg -v error -i %s/mask.mp4 -f null - 2' % (folder)).read()
            out = os.popen('/usr/bin/ffmpeg -v error -i %s/mask.mp4 -f null - 2' % (folder)).read()

            print('out', out)

            if fi % 1000 == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
            
            # if os.path.isfile('%s/track.npz' % folder):
            #     self.all_idxs.append(np.arange(self.S))
            #     self.all_folder_names.append(folder)
            # else:
            #     print('missing track in %s' % folder)
        
        os.system('cat error.log')
        input()
                
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
        # full_idx = self.all_idxs[index]
        # print('full_idx', full_idx)
        # print('folder', folder)
        
        rgbs = read_mp4(folder  + '/rgb.mp4')
        masks = read_mp4(folder + '/mask.mp4')
        
        if len(masks)<self.S or len(rgbs)<self.S:
            print('mp4 rgb,mask len %d,%d in %s; returning fake' % (len(rgbs), len(masks), folder))
            fake_sample = {
                'crop_rgbs': torch.zeros((self.S,2,3,128,128), dtype=torch.float32), 
                'crop_masks': torch.zeros((self.S,2,1,128,128), dtype=torch.float32), 
                'crop_xys': torch.zeros((self.S,2,2,128,128), dtype=torch.float32), 
                # 'masks_g': np.zeros((self.S,1,self.H,self.W), dtype=torch.float32), 
                'track_g': torch.zeros((self.S,8), dtype=torch.float32),
                'xywhs_e0': torch.zeros((self.S,4), dtype=torch.float32),
                'dname': 'none',
                'folder': 'none',
                'index': 0,
                'step': 0,
                'dists': torch.zeros((self.S), dtype=torch.float32),
            }
            return fake_sample, False
        
        rgbs = np.stack(rgbs, axis=0) # S,H,W,3
        masks = np.stack(masks, axis=0) # S,H,W,3

        d = dict(np.load(folder + '/track.npz', allow_pickle=True))
        track = d['track_g']
        dname = str(d['dname'])
        # print('export dname', dname)
        
        # rgbs = rgbs[full_idx]
        # masks = masks[full_idx]
        # track = track[full_idx]

        if len(rgbs) > self.S:
            rgbs = rgbs[:self.S]
            masks = masks[:self.S]
            track = track[:self.S]
        
        rgbs = rgbs.transpose(0,3,1,2)
        rgbs = rgbs[:,::-1].copy() # BGR->RGB

        masks = masks.transpose(0,3,1,2)[:,0:1]
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


        return track, True
        # rgbs = torch.from_numpy(rgbs).float()
        # masks_g = torch.from_numpy(masks).float()
        # track_g = torch.from_numpy(track).float()
        # device = rgbs.device

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

        # track_g = track_g.unsqueeze(0)
        # rgbs = rgbs.unsqueeze(0)
        # masks_g = masks_g.unsqueeze(0)

        xywhs_e0 = torch.rand((S,4), dtype=torch.float32) * min(H,W)

        # init a random distance from gt,
        coeffs = torch.rand((S,1), dtype=torch.float32)
        coeffs[0] = 1.0 # make zeroth gt, so it's clear what to track
        xywhs_e0 = xywhs_e0*(1.0-coeffs) + xywhs_g*(coeffs)

        # # move closer to zero-vel
        # xywhs_e0 = xywhs_e0*0.5 + xywhs_g[0:1]*0.5
        
        # # move closer to zero-vel
        # xywhs_e0 = xywhs_e0*0.9 + xywhs_g[0:1]*0.1
        
        if self.time_flip: # increase the batchsize by time shuffling
            # note we do this first, because S is squeezed with B
            perm = np.sort(np.random.permutation(S-1)[:S-1])+1
            perm = np.concatenate([[0], perm], axis=0)

            rgbs_flip = rgbs[perm]
            masks_g_flip = masks_g[perm]
            track_g_flip = track_g[perm]
            xywhs_e0_flip = xywhs_e0[perm]
            
            rgbs = torch.cat([rgbs, rgbs_flip], dim=0)
            masks_g = torch.cat([masks_g, masks_g_flip], dim=0)
            track_g = torch.cat([track_g, track_g_flip], dim=0)
            xywhs_e0 = torch.cat([xywhs_e0, xywhs_e0_flip], dim=0)

            S = S*2
            
        if self.horz_flip: # increase the batchsize by horizontal flipping
            rgbs_flip = torch.flip(rgbs, [-1])
            masks_g_flip = torch.flip(masks_g, [-1])
            track_g_flip = track_g.clone()
            track_g_flip[:,0] = W-1 - track_g_flip[:,0]
            xywhs_e0_flip = xywhs_e0.clone()
            xywhs_e0_flip[:,0] = W-1 - xywhs_e0_flip[:,0]
            
            rgbs = torch.cat([rgbs, rgbs_flip], dim=0)
            masks_g = torch.cat([masks_g, masks_g_flip], dim=0)
            track_g = torch.cat([track_g, track_g_flip], dim=0)
            xywhs_e0 = torch.cat([xywhs_e0, xywhs_e0_flip], dim=0)

            S = S*2

        if self.vert_flip: # increase the batchsize by horizontal flipping
            rgbs_flip = torch.flip(rgbs, [-2])
            masks_g_flip = torch.flip(masks_g, [-2])
            track_g_flip = track_g.clone()
            track_g_flip[:,1] = H-1 - track_g_flip[:,1]
            xywhs_e0_flip = xywhs_e0.clone()
            xywhs_e0_flip[:,1] = H-1 - xywhs_e0_flip[:,1]
            
            rgbs = torch.cat([rgbs, rgbs_flip], dim=0)
            masks_g = torch.cat([masks_g, masks_g_flip], dim=0)
            track_g = torch.cat([track_g, track_g_flip], dim=0)
            xywhs_e0 = torch.cat([xywhs_e0, xywhs_e0_flip], dim=0)

            S = S*2


        xywhs_g = track_g[:,:4]
        
        # compute dists, just for vis help later
        dists = torch.norm(xywhs_e0 - xywhs_g, dim=-1) # S
            
        scales = [0.25,1.0]
        cH, cW = 128, 128
        N = len(scales)

        # rgbs_ = rgbs_ / 255.0
        # rgbs_ = (rgbs_ - self.mean.to(device))/self.std.to(device)
        # xys = utils.basic.meshgrid2d(S, H, W, norm=False, device=device, stack=True, on_chans=True)

        # print('rgbs', rgbs.shape)
        # print('xys', xys.shape)

        # ims = torch.cat([rgbs, xys, torch.ones_like(xys[:,0:1])], dim=1)
        # crop_ims = get_multiscale_crops(ims, xywhs_e0, cH, cW, scales)

        # crop_rgbs = crop_ims[:,:,:3]
        # crop_xys = crop_ims[:,:,3:5]
        # crop_masks = crop_ims[:,:,5:6]
        
        # crop_rgbs = get_multiscale_crops(rgbs, xywhs_e0, cH, cW, scales)
        # crop_xys = get_multiscale_crops(xys, xywhs_e0, cH, cW, scales)
        # crop_masks = get_multiscale_crops(torch.ones_like(xys[:,0:1]), xywhs_e0, cH, cW)

        # print('crop_rgbs', crop_rgbs.shape)
        # print('crop_xys', crop_xys.shape)
        # print('crop_masks', crop_masks.shape)
        
        # xys_g = track_g[:,:,0:2]
        # whs_g = track_g[:,:,2:4]
        # vis_g = track_g[:,:,4]
        # xys_valid = track_g[:,:,5]
        # whs_valid = track_g[:,:,6]
        # vis_valid = track_g[:,:,7]
        step = 0#np.zeros((1), dtype=np.int32)

        # if horz_flip:
        #     crop_rgbs = crop_rgbs.reshape(S//2,2,
        
        sample = {
            # 'crop_rgbs': crop_rgbs,
            # 'crop_xys': crop_xys,
            # 'crop_masks': crop_masks,
            # 'masks_g': masks,
            'track_g': track_g,
            'xywhs_e0': xywhs_e0,
            'dname': dname,
            'folder': folder.split('/')[-1],
            'index': index,
            'step': step,
            'dists': dists,
        }
        return sample, True

    def __len__(self):
        return len(self.all_folder_names)
