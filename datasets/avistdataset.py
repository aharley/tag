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


class AVisTDataset(BBoxDataset):
    def __init__(
            self,
            dataset_location="/orion/group/avist",
            S=32, fullseq=False, chunk=None,
            strides=[1,2],
            zooms=[1,2],
            rand_frames=False,
            crop_size=(384, 512),
            use_augs=False,
            is_training=True,
    ):
        print("loading avist dataset...")
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            rand_frames=rand_frames,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training,
        )

        clip_step = S//2 if is_training else S
        if not is_training:
            strides = [1]

        self.root = Path(dataset_location)
        self.videos = sorted(list(self.root.glob("sequences/*")))
        print("found {:d} videos in {}".format(len(self.videos), self.dataset_location))

        self.all_info = []
        
        for video in self.videos:
            video_name = str(video).split('/')[-1]
            anno_path = self.root / "anno" / (video_name + ".txt")
            full_occ_path = self.root / "full_occlusion" / (video_name + "_full_occlusion.txt")
            out_of_view_path = self.root / "out_of_view" / (video_name + "_out_of_view.txt")
            try:
                bboxes = np.loadtxt(anno_path).astype(float)
            except:
                bboxes = np.loadtxt(anno_path, delimiter=",").astype(float)
            full_occ = np.loadtxt(full_occ_path, delimiter=",").astype(int)
            out_of_view = np.loadtxt(out_of_view_path, delimiter=",").astype(int)
            visibs = np.logical_not(np.logical_or(full_occ, out_of_view)).astype(np.float32)
            S_local = bboxes.shape[0]
            for stride in strides:
                for ii in range(0, max(S_local - self.S * stride, 1), clip_step*stride):
                    full_idx = ii + np.arange(self.S) * stride
                    full_idx = [ij for ij in full_idx if ij < S_local]
                    if len(full_idx) < (((S_local - 1) // stride + 1 if self.S == -1 else self.S) if fullseq else 8):
                        continue
                    # safe = np.concatenate(([0], visibs[:-2] * visibs[1:-1] * visibs[2:], [0]))
                    if np.sum(visibs[full_idx]) < 3: continue
                    for zoom in zooms:
                        self.all_info.append([video, full_idx, zoom])
        print("found {:d} samples in {}".format(len(self.all_info), self.dataset_location))
        if chunk is not None:
            assert(len(self.all_info) > 100)
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.all_info = chunkify(self.all_info,100)[chunk]
            print('filtered to %d' % len(self.all_info))

    def getitem_helper(self, index):
        video, full_idx, zoom = self.all_info[index]

        video_name = str(video).split('/')[-1]
        anno_path = self.root / "anno" / (video_name + ".txt")
        full_occ_path = self.root / "full_occlusion" / (video_name + "_full_occlusion.txt")
        out_of_view_path = self.root / "out_of_view" / (video_name + "_out_of_view.txt")
        try:
            bboxes = np.loadtxt(anno_path).astype(float)
        except:
            bboxes = np.loadtxt(anno_path, delimiter=",").astype(float)
        img_paths = list(
            sorted(video.glob("*.jpg"))
            + sorted(video.glob("*.png"))
        )

        if len(img_paths) > len(bboxes):
            # print(f'{gt_fn.parts[-2]}: {len(bboxes)} != {len(img_paths)}')
            img_paths = img_paths[: len(bboxes)]

        full_occ = np.loadtxt(full_occ_path, delimiter=",").astype(int)
        out_of_view = np.loadtxt(out_of_view_path, delimiter=",").astype(int)
        visibs = np.logical_not(np.logical_or(full_occ, out_of_view)).astype(np.float32)

        bboxes = bboxes[full_idx]
        visibs = visibs[full_idx]
        img_paths = [img_paths[i] for i in full_idx]

        rgbs = [cv2.imread(str(path))[..., ::-1].copy() for path in img_paths]

        rgbs = np.stack(rgbs)  # S, C, H, W
        bboxes = np.stack(bboxes)  # S, 4
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
            "rgbs": rgbs,
            "visibs": visibs,
            "bboxes": bboxes,
        }
        return sample

    def __len__(self):
        return len(self.all_info)
