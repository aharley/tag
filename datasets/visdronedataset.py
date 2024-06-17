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
import utils.misc
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


class VisDroneDataset(BBoxDataset):
    def __init__(
            self,
            dataset_location="/orion/group/visdrone",
            S=32, fullseq=False, chunk=None,
            strides=[1,2,3,4],
            zooms=[1,2,3,4],
            crop_size=(384, 512),
            use_augs=False,
            is_training=True,
    ):
        print("loading visdrone dataset...")
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training,
        )

        clip_step = S//2
        if not is_training:
            strides = [2]
            clip_step = S
        
        self.root = Path(dataset_location)
        split_dir = "VisDrone2019-SOT-train" if is_training else "VisDrone2019-SOT-val"
        self.gt_fns = sorted(list(self.root.glob(f"{split_dir}/annotations/*.txt")))
        self.attrs = sorted(list(self.root.glob(f"{split_dir}/attributes/*.txt")))

        print("found {:d} videos in {}".format(len(self.gt_fns), self.dataset_location))

        self.all_info = []
        for i in range(len(self.gt_fns)):
            fn = self.gt_fns[i]
            attr_path = self.attrs[i]
            try:
                attr = np.loadtxt(attr_path).astype(int)
            except:
                attr = np.loadtxt(attr_path, delimiter=",").astype(int)
            full_occ = attr[4]
            out_of_view = attr[7]
            visibs = np.logical_not(np.logical_or(full_occ, out_of_view)).astype(np.float32)
            # print('visibs', visibs)
            # can't make sense of these visibs.
            # the data seems to have everything visib anyway.
            try:
                bboxes = np.loadtxt(fn).astype(int)
            except:
                bboxes = np.loadtxt(fn, delimiter=",").astype(int)
            S_local = bboxes.shape[0]
            print('S_local', S_local)
            for stride in strides:
                for ii in range(0, S_local, clip_step*stride):
                    # print('bbox_areas[%d]' % ii, bbox_areas[ii])
                    full_idx = ii + np.arange(self.S) * stride
                    full_idx = [ij for ij in full_idx if ij < S_local]
                    if len(full_idx) < (((S_local - 1) // stride + 1 if self.S == -1 else self.S) if fullseq else 8):
                        continue
                    # if np.sum(visibs[full_idx]) < 3: continue

                    bboxes_here = bboxes[full_idx]
                    bboxes_here[..., 2:] += bboxes_here[..., :2]
                    xys = bboxes[:,0:2]*0.5 + bboxes[:,2:4]*0.5
                    mean_travel = np.mean(np.linalg.norm(xys[1:]-xys[:-1], axis=-1))
                    if mean_travel < 1: continue
                    
                    subset_name = str(fn).split("/")[-3]
                    video_name = str(fn).split("/")[-1][:-4]
                    # print(subset_name, video_name)
                    rgbs = sorted(list(self.root.glob(subset_name + "/sequences/" + video_name + "/*.jpg")))

                    for zoom in zooms:
                        self.all_info.append([fn, full_idx, rgbs, zoom])

            sys.stdout.write(".")
            sys.stdout.flush()
        print("found {:d} samples in {}".format(len(self.all_info), self.dataset_location))
        if chunk is not None:
            assert(len(self.all_info) > 100)
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.all_info = chunkify(self.all_info,100)[chunk]
            print('filtered to %d' % len(self.all_info))

    def getitem_helper(self, index):
        gt_fn, full_idx, img_paths, zoom = self.all_info[index]
        try:
            bboxes = np.loadtxt(gt_fn).astype(int)
        except:
            bboxes = np.loadtxt(gt_fn, delimiter=",").astype(int)

        visibs = np.ones(len(bboxes))

        # print('img_paths[0]', img_paths[0])

        if len(img_paths) > len(bboxes):
            # print(f'{gt_fn.parts[-2]}: {len(bboxes)} != {len(img_paths)}')
            img_paths = img_paths[: len(bboxes)]

        bboxes = bboxes[full_idx]
        visibs = visibs[full_idx]
        img_paths = [img_paths[i] for i in full_idx]
        # print(bboxes)
        # print(img_paths)

        rgbs = [cv2.imread(str(path))[..., ::-1].copy() for path in img_paths]

        rgbs = np.stack(rgbs)
        bboxes = np.stack(bboxes) 
        bboxes[..., 2:] += bboxes[..., :2]

        print('rgbs', rgbs.shape)

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
