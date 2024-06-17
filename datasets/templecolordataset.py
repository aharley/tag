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
from datasets.dataset import PointDataset, BBoxDataset
from pathlib import Path

from datasets.dataset_utils import make_split


class TempleColorDataset(BBoxDataset):
    def __init__(
            self,
            dataset_location="../TempleColor128",
            S=32, fullseq=False, chunk=None,
            strides=[1, 2],
            rand_frames=False,
            crop_size=(384, 512), 
            use_augs=False,
            is_training=True,
    ):
        print("loading TempleColor128 dataset...")
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            rand_frames=rand_frames,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training,
        )

        self.root = Path(dataset_location) / 'seqs'
        self.gt_fns = list(self.root.glob("*/*_gt.txt"))
        # self.gt_fns = make_split(self.gt_fns, is_training, shuffle=True)
        print(
            "found {:d} videos in {} ".format(
                len(self.gt_fns), self.dataset_location
            )
        )

        self.all_info = []
        for fn in self.gt_fns:
            try:
                bboxes = np.loadtxt(fn).astype(int)
            except:
                bboxes = np.loadtxt(fn, delimiter=",").astype(int)
            visibs = (np.prod(bboxes[:, 2:], -1) > 16).astype(np.float32)
            S_local = bboxes.shape[0]
            print('S_local', S_local)
            for stride in strides:
                for ii in range(0, max(S_local - self.S * stride, 1), 8):
                    # print('bbox_areas[%d]' % ii, bbox_areas[ii])
                    full_idx = ii + np.arange(self.S) * stride
                    full_idx = [ij for ij in full_idx if ij < S_local]
                    if len(full_idx) < (((S_local - 1) // stride + 1 if self.S == -1 else self.S) if fullseq else 8):
                        continue
                    if np.sum(visibs[full_idx]) < 3: continue
                    self.all_info.append([fn, full_idx])
        print("found {:d} samples in {}".format(len(self.all_info), self.dataset_location))
        if chunk is not None:
            assert(len(self.all_info) > 100)
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.all_info = chunkify(self.all_info,100)[chunk]
            print('filtered to %d' % len(self.all_info))

    def getitem_helper(self, index):
        gt_fn, full_idx = self.all_info[index]
        
        try:
            bboxes = np.loadtxt(gt_fn).astype(int)
        except:
            bboxes = np.loadtxt(gt_fn, delimiter=",").astype(int)
        img_paths = list(
            sorted((gt_fn.parent / "img").glob("*.jpg"))
            + sorted((gt_fn.parent / "img").glob("*.png"))
        )

        if len(img_paths) != len(bboxes):  # make it robust
            return None
        # if len(img_paths) > len(bboxes):
        #     # print(f'{gt_fn.parts[-2]}: {len(bboxes)} != {len(img_paths)}')
        #     img_paths = img_paths[: len(bboxes)]

        bboxes = bboxes[full_idx]
        img_paths = [img_paths[i] for i in full_idx]

        visibs = (np.prod(bboxes[:, 2:], -1) > 16).astype(np.float32)

        rgbs = [cv2.imread(str(path))[..., ::-1].copy() for path in img_paths]

        rgbs = np.stack(rgbs)  # S, C, H, W
        bboxes = np.stack(bboxes)  # S, 4
        # from xywh to xyxy
        bboxes[..., 2:] += bboxes[..., :2]

        sample = {
            "rgbs": rgbs,
            "visibs": visibs,  # S
            "bboxes": bboxes,
        }
        return sample

    def __len__(self):
        return len(self.all_info)
