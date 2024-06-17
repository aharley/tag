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
# from datasets.dataset import MaskDataset
from datasets.dataset import PointDataset, mask2bbox
from pathlib import Path
import matplotlib.pyplot as plt


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

class AutotrajDataset(PointDataset):
    def __init__(self,
                 dataset_location='../AutoTraj',
                 version='av',
                 S=32, fullseq=True, chunk=None,
                 rand_frames=False,
                 crop_size=(384,512),
                 zooms=[1],
                 strides=[1,2],
                 use_augs=False,
                 is_training=True):
        super().__init__(
            dataset_location=dataset_location,
            S=S, fullseq=True, chunk=None,
            rand_frames=rand_frames,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training
        )
        print('loading autotraj...')
        
        self.dataset_location = Path(self.dataset_location) / version

        folder_names = sorted(list(self.dataset_location.glob('export/*/*/')))
        # folder_names = [fn for fn in folder_names if (fn.stem[-1] != '9' if is_training else fn.stem[-1] == '9')]
        # folder_names = [fn for fn in folder_names if (fn.stem[-1] != '9' if is_training else fn.stem[-1] == '9')]
        # self.folder_names = folder_names
        # print('found {:d} {} samples in {}'.format(len(self.folder_names), ('train' if is_training else 'test'), self.dataset_location))
        print('found {:d} {} folders in {}'.format(len(folder_names), ('train' if is_training else 'test'), self.dataset_location))
        
        # # step through once and make sure all of the npzs are there
        # new_folder_names = []
        # for fi, folder in enumerate(folder_names):
        #     if os.path.isfile('%s/trajs.npz' % folder):
        #         new_folder_names.append(folder)
        #     else:
        #         pass
        # folder_names = new_folder_names
        # print('filtered to %d valid folders' % len(folder_names))

        print('folder_names[0]', folder_names[0])

        if chunk is not None:
            assert(len(folder_names) > 100)
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            folder_names = chunkify(folder_names,100)[chunk]
            print('filtered to %d' % len(folder_names))

        self.all_info = []
        clip_step = S//2

        rgbs = read_mp4(str(folder_names[0] / 'rgb.mp4'))
        S_local = len(rgbs)
        
        for fi, folder in enumerate(folder_names):
            for stride in strides:
                for ii in range(0, S_local-self.S+1, clip_step):
                    full_idx = ii + np.arange(self.S)*stride
                    full_idx = [ij for ij in full_idx if ij < S_local]
                    if len(full_idx) < (((S_local - 1) // stride + 1 if self.S == -1 else self.S) if fullseq else 8): continue
                    for zoom in zooms:
                        # for timeflip in timeflips:
                        self.all_info.append([folder, full_idx, zoom])
                
        print('found {:d} {} samples in {}'.format(len(self.all_info), ('train' if is_training else 'test'), self.dataset_location))
    
    def getitem_helper(self, index):
        folder, full_idx, zoom = self.all_info[index]
        
        rgbs = read_mp4(str(folder / 'rgb.mp4'))
        masks = read_mp4(str(folder / 'mask.mp4'))
        amasks = read_mp4(str(folder / 'amask.mp4'))
        pmasks = read_mp4(str(folder / 'pmask.mp4'))
        
        if len(masks)==0 or len(amasks)==0 or len(pmasks)==0 or len(rgbs)==0:
            return None
        
        rgbs = np.stack(rgbs, axis=0) # S,H,W,3
        masks = np.stack(masks, axis=0) # S,H,W
        amasks = np.stack(amasks, axis=0) # S,H,W
        pmasks = np.stack(pmasks, axis=0) # S,H,W
        
        masks = (masks[:, :, :, 0] > 128).astype(np.float32)
        amasks = (amasks[:, :, :, 0] > 128).astype(np.float32)
        pmasks = (pmasks[:, :, :, 0] > 128).astype(np.float32)
        
        bboxes = np.stack([mask2bbox(mask) for mask in pmasks])
        whs = bboxes[:,2:4] - bboxes[:,0:2] # S,2
        # whs = whs, min=1, max=)

        # print('masks', masks.shape)
        # print('amasks', amasks.shape)

        d = dict(np.load(folder / 'trajs.npz', allow_pickle=True))
        xys = d['xys'] # S,2
        visibs = d['vis'] # S

        rgbs = rgbs[full_idx]
        masks = masks[full_idx]
        amasks = amasks[full_idx]
        
        xys = xys[full_idx]
        whs = whs[full_idx]
        visibs = visibs[full_idx]

        # if index % 2 == 0:
        #     # reverse the video
        #     rgbs = np.flip(rgbs, axis=0)
        #     masks = np.flip(masks, axis=0)
        #     xys = np.flip(xys, axis=0)
        #     whs = np.flip(whs, axis=0)
        #     visibs = np.flip(visibs, axis=0)

        # if (np.sum(visibs[:4])<4) or (np.sum(visibs) < np.sqrt(self.S)):
        #     return None

        # S = len(masks)
        # vis_g = np.ones((S))
        # for si, mask in enumerate(masks):
        #     if np.sum(mask) > 8:
        #         ys, xs = np.where(mask)
        #         x0, x1 = np.min(xs), np.max(xs)
        #         y0, y1 = np.min(ys), np.max(ys)
        #         crop = mask[y0:y1,x0:x1]
        #         fill = np.mean(crop)
        #         if fill < self.mask_fill_thr or (si==0 and fill < self.mask_fill_thr0):
        #             print('fill %.2f on frame %d' % (fill, si))
        #             return None
        #     else:
        #         vis_g[si] = 0
                
        # if np.sum(vis_g) < np.sqrt(S):
        #     print('vis %d/%d' % (np.sum(vis_g), S))
        #     return None

        S,H,W,C = rgbs.shape
        
        valids = np.ones_like(visibs)
        for si in range(S):
            # when a point moves far oob, don't supervise with it
            very_oob_inds = np.logical_or(
                np.logical_or(xys[si,0] < -64, xys[si,0] > W+64),
                np.logical_or(xys[si,1] < -64, xys[si,1] > H+64))
            valids[si,very_oob_inds] = 0
        
        if zoom > 1:
            xys, visibs, valids, rgbs, masks, amasks = utils.misc.data_zoom(zoom, xys, visibs, valids, rgbs, masks, masks2=amasks)
            _, H, W, _ = rgbs.shape

        masks0 = np.zeros_like(masks)
        for i in range(len(masks)):
            if valids[i]:
                xy = xys[i].round().astype(np.int32)
                x, y = xy[0], xy[1]
                x = x.clip(0,W-1)
                y = y.clip(0,H-1)
                masks0[i,y,x] = 1
            else:
                masks0[i] = 0.5
                
        # if oid > 0:
        full_masks = np.stack([masks0, masks, amasks], axis=-1)

        # all chans valid when points are valid
        masks_valid = np.zeros((S,3), dtype=np.float32)
        masks_valid[:,0] = valids
        masks_valid[:,1] = valids
        masks_valid[:,2] = valids

        rgbs = rgbs[:,:,:,::-1].copy() # BGR->RGB

        bboxes = np.concatenate([xys - whs/2.0, xys + whs/2.0], axis=-1)

        # if timeflip:
        #     rgbs = np.flip(rgbs, axis=0)
        #     bboxes = np.flip(bboxes, axis=0)
        #     full_masks = np.flip(full_masks, axis=0)
        #     masks_valid = np.flip(masks_valid, axis=0)
        #     xys = np.flip(xys, axis=0)
        #     visibs = np.flip(visibs, axis=0)
        #     valids = np.flip(valids, axis=0)

        sample = {
            'rgbs': rgbs,
            'bboxes': bboxes,
            'masks': full_masks,
            'masks_valid': masks_valid,
            'xys': xys,
            'visibs': visibs,
            'valids': valids,
        }
        return sample


    def __len__(self):
        return len(self.all_info)
