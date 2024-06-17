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
import cv2
import re
import sys
from pathlib import Path
from datasets.dataset import BBoxDataset
import pandas

class TknetDataset(BBoxDataset):
    def __init__(self,
                 dataset_location,
                 S=32, fullseq=False, chunk=None,
                 zooms=[1,2],
                 crop_size=(384,512),
                 use_augs=False,
                 is_training=True):
        print('loading tknet dataset...')
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training)
        
        self.root = Path(dataset_location)
        # self.seq_names = open(self.root / '{}_set.txt'.format('training' if is_training else 'testing')).read().splitlines()

        if is_training:
            self.folders = sorted(list(set(self.root.glob('*/frames/*/')) - set(self.root.glob('*/frames/[m-z]*/'))))
        else:
            self.folders = sorted(list(self.root.glob('*/frames/[m-z]*/')))

        print('found {:d} {} videos in {}'.format(len(self.folders), ('train' if is_training else 'test'), self.dataset_location))
        # print('seq_names', self.seq_names)

        if chunk is not None:
            assert(len(self.folders) > 100)
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.folders = chunkify(self.folders,100)[chunk]
            print('filtered to %d' % len(self.folders))
        

        # import ipdb; ipdb.set_trace()
        # # input()
        
        # self.anno_root = self.root / 'Tknet_Evaluation_Toolkit/annos'
        # print('found {:d} {} videos in {}'.format(len(self.seq_names), ('train' if is_training else 'test'), self.dataset_location))

        self.all_info = []

        stride = 30 # the gt here is only reliable every 30th frame
        
        clip_step = S//2
        
        for folder in self.folders:
            img_paths = sorted(list(folder.glob('*.jpg')))
            # S_local = len(img_paths)
            all_idx = np.arange(len(img_paths))

            all_idx = all_idx[::stride]
            
            # print('S_local', S_local)
            print('S_local', len(all_idx))
            
            if len(all_idx) > 8:

                anno_file = str(folder).replace('frames', 'anno') + '.txt'

                if os.path.isfile(anno_file):
                    # print('anno_file', anno_file)

                    bboxes = pandas.read_csv(anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
                    bboxes = bboxes[::stride]

                    if len(bboxes)==len(all_idx):
                        visibs = np.min(bboxes, axis=1) > 0
 
                        # print('bboxes', bboxes.shape)
                        all_idx = all_idx[visibs>0]
                        bboxes = bboxes[visibs>0]
                        
                        # print('bboxes', bboxes.shape)

                        # xywh -> xyxy
                        bboxes[..., 2:] += bboxes[..., :2]
                        xys = (bboxes[:, :2] + bboxes[:, 2:]) / 2
                        # print('xys', xys.shape)
                        xys = np.concatenate([xys[0:1], xys], axis=0)
                        # print('xys', xys.shape)
                        dists = np.linalg.norm(xys[1:] - xys[:-1], axis=-1)

                        # print('dists', dists)
                        d_med, d_max = np.median(dists), np.max(dists)
                        # print('dists mean, max', np.median(dists), np.max(dists))

                        # print('dists', dists.shape)
                        # # we will ONLY use frame0, because the visibility is ambiguous later
                        # all_idx_stride = all_idx[::stride]
                        
                        S_local = len(all_idx)
                        
                        for ii in range(0, max(S_local-self.S,1), clip_step):
                            local_idx = ii + np.arange(self.S)
                            local_idx = [ij for ij in local_idx if ij < S_local]
                            if len(local_idx) < (self.S if fullseq else 8): continue
                            
                            full_idx = all_idx[local_idx]
                            dists_here = dists[local_idx]
                            
                            if np.sum(dists_here) >= self.S and (d_max < d_med*8):
                                for zoom in zooms:
                                    self.all_info.append([folder, full_idx, zoom])
                                sys.stdout.write('.')
                                sys.stdout.flush()
                            else:
                                sys.stdout.write('m')
                                sys.stdout.flush()
        print("found {:d} samples in {}".format(len(self.all_info), self.dataset_location))
                        
    def __len__(self):
        return len(self.all_info)
    
    def getitem_helper(self, index):
        folder, full_idx, zoom = self.all_info[index]
        # print('folder', folder)
        # print('full_idx', full_idx)

        anno_file = str(folder).replace('frames', 'anno') + '.txt'

        bboxes = pandas.read_csv(anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        # # xywh -> xyxy
        # bboxes[..., 2:] += bboxes[..., :2]

        visibs = (bboxes[:, 2] > 0) & (bboxes[:, 3] > 0)

        xys = (bboxes[:, :2] + bboxes[:, 2:]) / 2
        # print('xys', xys.shape)
        xys = np.concatenate([xys[0:1], xys[1:]], axis=0)
        dists = np.linalg.norm(xys[1:] - xys[:-1], axis=-1)
        
        # img_paths = list(sorted(folder.glob('*.jpg')))
        # print('img_paths', img_paths)
        
        # img_paths = [img_paths[ii] for ii in full_idx]

        img_paths = ['%s/%d.jpg' % (folder, ii) for ii in full_idx]
        # print('img_paths idx', img_paths)
        
        bboxes = bboxes[full_idx]
        visibs = visibs[full_idx]

        rgb = cv2.imread(str(img_paths[0]))
        H, W = rgb.shape[:2]
        rgbs = []
        for path in img_paths:
            rgb = cv2.imread(str(path))[..., ::-1].copy()
            rgbs.append(rgb)
        
        rgbs = np.stack(rgbs) # S, C, H, W
        bboxes = np.stack(bboxes) # S, 4
        visibs = np.stack(visibs) # S
        
        # from xywh to xyxy
        bboxes[..., 2:] += bboxes[..., :2]

        rgbs, bboxes = utils.misc.data_pad_if_necessary_b(rgbs, bboxes, visibs)
        S,H,W,C = rgbs.shape
        if zoom > 1:
            whs = bboxes[:,2:4] - bboxes[:,0:2]
            whs = whs[visibs > 0.5]
            if np.mean(whs[:,0])*zoom >= W and np.mean(whs[:,1])*zoom >= H:
                return None
            bboxes, visibs, rgbs = utils.misc.data_zoom_bbox(zoom, bboxes, visibs, rgbs)
        safe = np.concatenate(([0], visibs[:-2] * visibs[1:-1] * visibs[2:], [0]))
        if np.sum(safe) < 2: return None
        
        sample = {
            'rgbs': rgbs,
            'bboxes': bboxes,
            'visibs': visibs, 
        }
        return sample
