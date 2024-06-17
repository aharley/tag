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
from pathlib import Path
from datasets.dataset import BBoxDataset


class LaSOTDataset(BBoxDataset):
    def __init__(self,
                 dataset_location='../LaSOT',
                 S=32, fullseq=False, chunk=None,
                 strides=[1,2],
                 zooms=[1,2],
                 crop_size=(384,512),
                 use_augs=False,
                 is_training=True):
        print('loading lasot dataset...')
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training)
        
        self.root = Path(dataset_location)
        self.seq_names = open(self.root / '{}_set.txt'.format('training' if is_training else 'testing')).read().splitlines()
        self.seq_names = sorted(self.seq_names)
        
        self.anno_root = self.root / 'LaSOT_Evaluation_Toolkit/annos'
        print('found {:d} {} videos in {}'.format(len(self.seq_names), ('train' if is_training else 'test'), self.dataset_location))

        clip_step = S//2 if is_training else S

        if chunk is not None:
            assert(len(self.seq_names) > 100)
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.seq_names = chunkify(self.seq_names,100)[chunk]
            print('filtered to %d' % len(self.seq_names))
        

        self.all_info = []
        
        for seq_name in self.seq_names:
            print('seq_name', seq_name)
            bboxes = np.array([line.split(',') for line in open(self.anno_root / f'{seq_name}.txt').read().splitlines()]).astype(int)
            absents = np.array(open(self.anno_root / 'absent' / f'{seq_name}.txt').read().splitlines()).astype(int)
            img_root = self.root / '{}/{}/img'.format(seq_name.split('-')[0], seq_name)
            img_paths = list(sorted(img_root.glob('*.jpg')))
            S_local = len(img_paths)

            xys = (bboxes[:, :2] + bboxes[:, 2:]) / 2
            dists = np.linalg.norm(xys[1:] - xys[:-1], axis=-1)
            dists = np.concatenate([[0], dists])

            for stride in strides:
                for ii in range(0, S_local, clip_step*stride):
                    full_idx = ii + np.arange(self.S)*stride
                    full_idx = [ij for ij in full_idx if ij < S_local]
                    dists_here = dists[full_idx]
                    visibs = 1-absents[full_idx]
                    safe = np.concatenate(([0], visibs[:-2] * visibs[1:-1] * visibs[2:], [0]))

                    if len(full_idx) < (((S_local - 1) // stride + 1 if self.S == -1 else self.S) if fullseq else 8):
                        continue
                    if np.mean(dists_here) < 4: continue
                    if np.sum(safe) < 3: continue

                    for zoom in zooms:
                        self.all_info.append([seq_name, full_idx, zoom])
                        sys.stdout.write('.')
                        sys.stdout.flush()
        print("found {:d} samples in {}".format(len(self.all_info), self.dataset_location))
                        
    def __len__(self):
        return len(self.all_info)
    
    def getitem_helper(self, index):
        seq_name, full_idx, zoom = self.all_info[index]
        
        bboxes = np.array([line.split(',') for line in open(self.anno_root / f'{seq_name}.txt').read().splitlines()]).astype(int)
        absents = np.array(open(self.anno_root / 'absent' / f'{seq_name}.txt').read().splitlines()).astype(int)
        img_root = self.root / '{}/{}/img'.format(seq_name.split('-')[0], seq_name)
        img_paths = list(sorted(img_root.glob('*.jpg')))

        img_paths = [img_paths[ii] for ii in full_idx]
        bboxes = bboxes[full_idx]
        absents = absents[full_idx]

        rgbs = [cv2.imread(str(path))[..., ::-1].copy() for path in img_paths]
        
        rgbs = np.stack(rgbs) # S, C, H, W
        bboxes = np.stack(bboxes)  # S, 4
        # from xywh to xyxy
        bboxes[..., 2:] += bboxes[..., :2]
        visibs = 1-absents

        rgbs, bboxes = utils.misc.data_pad_if_necessary_b(rgbs, bboxes, visibs)
        S,H,W,C = rgbs.shape
        if zoom > 1:
            whs = bboxes[:,2:4] - bboxes[:,0:2]
            whs = whs[visibs > 0.5]
            if np.mean(whs[:,0])*zoom >= W and np.mean(whs[:,1])*zoom >= H:
                return None
            bboxes, visibs, rgbs = utils.misc.data_zoom_bbox(zoom, bboxes, visibs, rgbs)
        safe = np.concatenate(([0], visibs[:-2] * visibs[1:-1] * visibs[2:], [0]))
        if np.sum(safe) < 2:
            return None
        
        sample = {
            'rgbs': rgbs,
            'visibs': visibs,
            'bboxes': bboxes,
        }
        return sample

# lasot test is validation, we can reuse the dataset
class LaSOTTestDataset(BBoxDataset):
    def __init__(self,
                 dataset_location='../LaSOT',
                 S=32,
                 crop_size=(256,256)):
        super().__init__(dataset_location=dataset_location,
                         S=S,
                         crop_size=crop_size,
                         use_augs=False,
                         is_training=False,
                         inference=True)
        self.root = Path(dataset_location)
        self.seq_names = open(self.root / '{}_set.txt'.format('testing')).read().splitlines()
        self.seq_names = sorted(self.seq_names)
        
        self.anno_root = self.root / 'LaSOT_Evaluation_Toolkit/annos'
        print('found {:d} {} videos in {}'.format(len(self.seq_names), ('test'), self.dataset_location))

        self.all_seq_names = []
        self.all_full_idx = []
        
        for i, seq_name in enumerate(self.seq_names):
            img_root = self.root / '{}/{}/img'.format(seq_name.split('-')[0], seq_name)
            img_paths = list(sorted(img_root.glob('*.jpg')))
            S_local = min(len(img_paths), S)

            full_idx = np.arange(0, S_local)

            self.all_seq_names.append(seq_name)
            self.all_full_idx.append(full_idx)
            sys.stdout.write('.')
            sys.stdout.flush()

    def __len__(self):
        return len(self.all_seq_names)
    
    def getitem_helper(self, index):
        seq_name = self.all_seq_names[index]
        full_idx = self.all_full_idx[index]
        
        bboxes = np.array([line.split(',') for line in open(self.anno_root / f'{seq_name}.txt').read().splitlines()]).astype(int)
        absents = np.array(open(self.anno_root / 'absent' / f'{seq_name}.txt').read().splitlines()).astype(int)
        img_root = self.root / '{}/{}/img'.format(seq_name.split('-')[0], seq_name)
        img_paths = list(sorted(img_root.glob('*.jpg')))

        img_paths = [img_paths[ii] for ii in full_idx]
        bboxes = bboxes[full_idx]
        absents = absents[full_idx]

        rgb0 = cv2.imread(str(img_paths[0]))[..., ::-1]
        H, W = rgb0.shape[:2]
        rgbs = [cv2.resize(cv2.imread(str(path))[..., ::-1], dsize=(self.crop_size[1], self.crop_size[0])) for path in img_paths]
        rgbs = np.stack(rgbs) # S, C, H, W
        bboxes = np.stack(bboxes).astype(np.float32)  # S, 4

        # from xywh to xyxy
        bboxes[..., 2:] += bboxes[..., :2]
        bboxes[..., 0] *= self.crop_size[1] / W
        bboxes[..., 2] *= self.crop_size[1] / W
        bboxes[..., 1] *= self.crop_size[0] / H
        bboxes[..., 3] *= self.crop_size[0] / H
        
        sample = {
            'rgbs': rgbs,
            'visibs': 1 - absents,  # S
            'bboxes': bboxes,
            'img_size': np.array([H, W]),
        }
        return sample

if __name__ == '__main__':
    ds = LaSOTTestDataset('/orion/u/yangyou/datasets/lasot', S=100000, crop_size=(128, 256))
    for d in ds:
        print(d)
        import pdb; pdb.set_trace()
