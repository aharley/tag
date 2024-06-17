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
import csv

# from datasets.dataset_utils import make_split


class MoCADataset(BBoxDataset):
    def __init__(
            self,
            dataset_location="/orion/group/MoCA",
            S=32, fullseq=False, chunk=None,
            strides=[1,2,3,4],
            zooms=[1,1.5,2],
            rand_frames=False,
            crop_size=(384, 512), 
            use_augs=False,
            is_training=True,
    ):
        print("loading moca dataset...")
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
            zooms = [1]
            
        self.root = Path(dataset_location)
        
        csv_file = open(os.path.join(self.dataset_location, "Annotations", "annotations.csv"), "r")
        csv_reader = csv.reader(csv_file, delimiter=",")
        row_count = 0
        self.gts = {}
        self.videos = []
        for row in csv_reader:
            if row_count >= 10:
                #print(row[1], row[1].split('/'))
                img_path = row[1]
                video_name = row[1].split('/')[1]
                if video_name not in self.videos:
                    self.videos.append(video_name)
                    self.gts[video_name] = {}
                    self.gts[video_name]["img_paths"] = []
                    self.gts[video_name]["bboxes"] = []
                bbox = json.loads(row[4])
                #print(dataset_location)
                full_path = os.path.join(dataset_location, "JPEGImages", img_path[1:])
                #print(full_path)
                self.gts[video_name]["img_paths"].append(full_path)
                self.gts[video_name]["bboxes"].append(bbox[1:])
                if bbox[0] != 2:
                    print("Error:", bbox)
            row_count += 1
        
        print(
            "found {:d} videos in {} (training={})".format(
                len(self.videos), self.dataset_location, is_training
            )
        )


        self.all_info = []
        for video in self.videos:
            bboxes = self.gts[video]["bboxes"]
            #bboxes = self.gts[video]["bboxes"]
            S_local = len(bboxes)
            # print('S_local', S_local)
            for stride in strides:
                for ii in range(0, S_local, clip_step):
                    # print('bbox_areas[%d]' % ii, bbox_areas[ii])
                    full_idx = ii + np.arange(self.S) * stride
                    full_idx = [ij for ij in full_idx if ij < S_local]
                    if len(full_idx) < (((S_local - 1) // stride + 1 if self.S == -1 else self.S) if fullseq else 8):
                        continue
                    for zoom in zooms:
                        self.all_info.append([video, full_idx, zoom])
        print(
            "found {:d} samples in {} (training={}); we made our own split".format(
                len(self.all_info), self.dataset_location, is_training
            )
        )
        if chunk is not None:
            assert(len(self.all_info) > 100)
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.all_info = chunkify(self.all_info,100)[chunk]
            print('filtered to %d clips' % len(self.all_info))
            # print('self.all_info', self.all_info)

    def getitem_helper(self, index):
        video, full_idx, zoom = self.all_info[index]
        img_paths = self.gts[video]["img_paths"]
        bboxes = np.array(self.gts[video]["bboxes"])
        visibs = np.ones((len(bboxes)))

        # print('img_paths[0]', img_paths[0])

        if len(img_paths) > len(bboxes):
            # print(f'{gt_fn.parts[-2]}: {len(bboxes)} != {len(img_paths)}')
            img_paths = img_paths[: len(bboxes)]

        bboxes = bboxes[full_idx]
        visibs = visibs[full_idx]
        img_paths = [img_paths[i] for i in full_idx]

        rgbs = [cv2.imread(str(path))[..., ::-1].copy() for path in img_paths]

        rgbs = np.stack(rgbs)  # S, C, H, W
        #bboxes = np.stack(bboxes)  # S, 4
        bboxes[..., 2:] += bboxes[..., :2]

        if self.is_training:
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
