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
from icecream import ic

from datasets.dataset_utils import make_split


class MOT17Dataset(BBoxDataset):
    def __init__(
            self,
            dataset_location="/orion/group/MOT17",
            S=32, fullseq=False, chunk=None,
            strides=[1,2,3],
            zooms=[1,2],
            rand_frames=False,
            crop_size=(384, 512), 
            use_augs=False,
            is_training=True,
    ):
        print("loading mot17 dataset...")
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            rand_frames=rand_frames,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training,
        )

        # validation gt not provided
        self.root = Path(dataset_location)
        self.fns = sorted(list(self.root.glob("train/*FRCNN")))
        # self.fns = make_split(self.fns, is_training, shuffle=True)
        print("found {:d} videos in {}".format(len(self.fns), self.dataset_location))

        clip_step = S//2

        self.data = []
        self.all_info = []
        for fn in self.fns:
            gt_fn = fn / "gt/gt.txt"
            gt = np.loadtxt(gt_fn, delimiter=",").astype(np.float32)
            obj_ids = np.unique(gt[:, 1])
            for obj_id in obj_ids:
                gt_i = gt[gt[:, 1] == obj_id]
                visibles = gt_i[:, 8].astype(np.float32)
                S_local = gt_i.shape[0]
                for stride in strides:
                    for ii in range(0, max(S_local - self.S * stride, 1), clip_step*stride):
                        # print('bbox_areas[%d]' % ii, bbox_areas[ii])
                        full_idx = ii + np.arange(self.S) * stride
                        full_idx = [ij for ij in full_idx if ij < S_local]
                        if len(full_idx) < (((S_local - 1) // stride + 1 if self.S == -1 else self.S) if fullseq else 8):
                            continue
                        visibs = visibles[full_idx]
                        safe = np.concatenate(([0], visibs[:-2] * visibs[1:-1] * visibs[2:], [0]))
                        if np.sum(safe) < 2: continue
                        for zoom in zooms:
                            self.all_info.append([fn, gt_i[full_idx], zoom])
        print("found {:d} samples in {}".format(len(self.all_info), self.dataset_location))
        if chunk is not None:
            assert(len(self.all_info) > 100)
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.all_info = chunkify(self.all_info,100)[chunk]
            print('filtered to %d' % len(self.all_info))

    def getitem_helper(self, index):
        gt_fn, gt_info, zoom = self.all_info[index]

        img_ids = gt_info[:, 0]

        img_paths = [gt_fn / "img1/{:06d}.jpg".format(int(img_id)) for img_id in img_ids]

        visibs = gt_info[:, 8].astype(np.float32)
        bboxes = gt_info[:, 2:6].astype(np.float32)

        rgbs = [cv2.imread(str(path))[..., ::-1].copy() for path in img_paths]

        rgbs = np.stack(rgbs)  # S, C, H, W
        bboxes = np.stack(bboxes)  # S, 4
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
        if np.sum(safe) < 2:
            return None

        sample = {
            "rgbs": rgbs,
            "visibs": visibs, 
            "bboxes": bboxes,
        }
        return sample

    def __len__(self):
        return len(self.all_info)
