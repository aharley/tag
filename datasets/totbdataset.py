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


class TOTBDataset(BBoxDataset):
    def __init__(
            self,
            dataset_location="/orion/group/TOTB",
            S=32, fullseq=False, chunk=None,
            strides=[1, 2],
            rand_frames=False,
            crop_size=(384, 512), 
            use_augs=False,
            is_training=True,
    ):
        print("loading totb dataset...")
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            rand_frames=rand_frames,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training,
        )

        clip_step = S//2
        if not is_training:
            strides = [2]
            clip_step = S

        self.root = Path(dataset_location)
        self.gt_fns = sorted(list(self.root.glob("*/*/groundtruth.txt")))
        # self.gt_fns = make_split(self.gt_fns, is_training, shuffle=True)
        print("found {:d} videos in {}".format(len(self.gt_fns), self.dataset_location))

        self.all_info = []
        for fn in self.gt_fns:
            try:
                bboxes = np.loadtxt(fn).astype(int)
            except:
                bboxes = np.loadtxt(fn, delimiter=",").astype(int)
            full_occ_path = fn.parent / "full_occlusion.txt"
            out_of_view_path = fn.parent / "out_of_view.txt"
            full_occ = np.loadtxt(full_occ_path).astype(int)
            out_of_view = np.loadtxt(out_of_view_path).astype(int)
            visibs = np.logical_not(np.logical_or(full_occ, out_of_view)).astype(
                np.float32
            )
            S_local = bboxes.shape[0]
            print('S_local', S_local)
            for stride in strides:
                for ii in range(0, max(S_local - self.S * stride, 1), clip_step):
                    # print('bbox_areas[%d]' % ii, bbox_areas[ii])
                    full_idx = ii + np.arange(self.S) * stride
                    full_idx = [ij for ij in full_idx if ij < S_local]
                    if len(full_idx) < (((S_local - 1) // stride + 1 if self.S == -1 else self.S) if fullseq else 8):
                        continue
                    if np.sum(visibs[full_idx]) < 3: continue
                    # safe = np.concatenate(([0], visibs[:-2] * visibs[1:-1] * visibs[2:], [0]))
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
        full_occ_path = gt_fn.parent / "full_occlusion.txt"
        out_of_view_path = gt_fn.parent / "out_of_view.txt"
        meta_info_path = gt_fn.parent / "meta_info.txt"

        # print('img_paths[0]', img_paths[0])

        if len(img_paths) > len(bboxes):
            # print(f'{gt_fn.parts[-2]}: {len(bboxes)} != {len(img_paths)}')
            img_paths = img_paths[: len(bboxes)]

        full_occ = np.loadtxt(full_occ_path).astype(int)
        out_of_view = np.loadtxt(out_of_view_path).astype(int)
        visibs = np.logical_not(np.logical_or(full_occ, out_of_view)).astype(np.float32)

        bboxes = bboxes[full_idx]
        visibs = visibs[full_idx]
        img_paths = [img_paths[i] for i in full_idx]

        rgbs = [cv2.imread(str(path))[..., ::-1].copy() for path in img_paths]

        rgbs = np.stack(rgbs)  # S, C, H, W
        bboxes = np.stack(bboxes)  # S, 4
        bboxes[..., 2:] += bboxes[..., :2]

        sample = {
            "rgbs": rgbs,
            "visibs": visibs,  # S
            "bboxes": bboxes,
        }
        return sample

    def __len__(self):
        return len(self.all_info)
