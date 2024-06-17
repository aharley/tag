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


class BioDroneDataset(BBoxDataset):
    def __init__(
            self,
            dataset_location="/orion/group/biodrone",
            S=32, fullseq=False, chunk=None,
            strides=[1, 2],
            zooms=[1,2],
            rand_frames=False,
            crop_size=(384, 512),
            use_augs=False,
            is_training=True,
    ):
        print("loading biodrone dataset...")
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
        split_dir = "train" if is_training else "val"
        self.vid_dirs = sorted(list(self.root.glob(f"data/{split_dir}/frame_*")))

        print("found {:d} videos in {}".format(len(self.vid_dirs), self.dataset_location))

        self.all_info = []
        for i in range(len(self.vid_dirs)):
            video_num = str(self.vid_dirs[i]).split('/')[-1][-3:]
            bbox_path = self.root / os.path.join("attribute", "groundtruth", video_num + ".txt")
            occ_path = self.root / os.path.join("attribute", "occlusion", video_num + ".txt")
            absent_path = self.root / os.path.join("attribute", "absent", video_num + ".txt")
            try:
                occ = np.loadtxt(occ_path).astype(int)
            except:
                occ = np.loadtxt(occ_path, delimiter=",").astype(int)
            try:
                absent = np.loadtxt(absent_path).astype(int)
            except:
                absent = np.loadtxt(absent_path, delimiter=",").astype(int)
            try:
                bboxes = np.loadtxt(bbox_path).astype(int)
            except:
                bboxes = np.loadtxt(bbox_path, delimiter=",").astype(int)
            #print(occ.shape, absent.shape, bboxes.shape)
            visibs = np.logical_not(absent)
            S_local = bboxes.shape[0]
            for stride in strides:
                for ii in range(0, max(S_local-self.S*stride,1), clip_step*stride):
                    # print('bbox_areas[%d]' % ii, bbox_areas[ii])
                    full_idx = ii + np.arange(self.S)*stride
                    full_idx = [ij for ij in full_idx if ij < S_local]
                    if len(full_idx) < (((S_local - 1) // stride + 1 if self.S == -1 else self.S) if fullseq else 8):
                        continue
                    if np.sum(visibs[full_idx]) < 3: continue
                    for zoom in zooms:
                        self.all_info.append([i, video_num, full_idx, zoom])
            sys.stdout.write(".")
            sys.stdout.flush()
        print(
            "found {:d} samples in {}".format(
                len(self.all_info), self.dataset_location
            )
        )
        if chunk is not None:
            assert(len(self.all_info) > 100)
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.all_info = chunkify(self.all_info,100)[chunk]
            print('filtered to %d' % len(self.all_info))
        

    def getitem_helper(self, index):
        video_idx, video_num, full_idx, zoom = self.all_info[index]
        #print(video_num, full_idx)
        bbox_path = self.root / os.path.join("attribute", "groundtruth", video_num + ".txt")
        occ_path = self.root / os.path.join("attribute", "occlusion", video_num + ".txt")
        absent_path = self.root / os.path.join("attribute", "absent", video_num + ".txt")
        try:
            occ = np.loadtxt(occ_path).astype(int)
        except:
            occ = np.loadtxt(occ_path, delimiter=",").astype(int)
        try:
            absent = np.loadtxt(absent_path).astype(int)
        except:
            absent = np.loadtxt(absent_path, delimiter=",").astype(int)
        try:
            bboxes = np.loadtxt(bbox_path).astype(int)
        except:
            bboxes = np.loadtxt(bbox_path, delimiter=",").astype(int)
        #print("occ:", occ[full_idx])
        #print("absent:", absent[full_idx])
        video_dir = self.vid_dirs[video_idx]
        img_paths = sorted(list(video_dir.glob("*.jpg")))
        visibs = np.logical_not(absent)

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

        rgbs = np.stack(rgbs)  # S, C, H, W
        bboxes = np.stack(bboxes)  # S, 4
        bboxes[..., 2:] += bboxes[..., :2]
        #print("bboxes:", bboxes)

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
            "visibs": visibs,  # S
            "bboxes": bboxes,
        }
        return sample

    def __len__(self):
        return len(self.all_info)
