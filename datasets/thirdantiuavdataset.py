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
from tqdm import tqdm

from datasets.dataset_utils import make_split


class ThirdAntiUAVDataset(BBoxDataset):
    def __init__(
        self,
            dataset_location="../UAV123",
            S=32, fullseq=False, chunk=None,
            rand_frames=False,
            crop_size=(384, 512), 
            strides=[1,2,3],
            zooms=[1,2],
            use_augs=False,
            is_training=True,
    ):
        print("loading third Anti UAV dataset...")
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            rand_frames=rand_frames,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training,
        )

        clip_step = S//2

        self.split = "train" if is_training else "val"
        self.root = Path(dataset_location)
        self.dataset_location = dataset_location
        # UAV provides both long-term and short-term tracking
        self.train_json = str(self.root / "data_files/extracted/d1eae953d1857b2003b05340b7a3c30aab6abcdbc107fce595779d7e7bd7d4db/train.json")
        self.val_json = str(self.root / "data_files/extracted/d1eae953d1857b2003b05340b7a3c30aab6abcdbc107fce595779d7e7bd7d4db/validation.json")
        if self.split == "train":
            #self.gts_raw = json.loads(open(self.train_json).read())
            self.videos = sorted(glob.glob(str(self.root) + "/data_files/extracted/*/train/*/000001.jpg"))
        elif self.split == "val":
            #self.gts_raw = json.loads(open(self.val_json).read())
            self.videos = sorted(glob.glob(str(self.root) + "/data_files/extracted/*/validation/*/000001.jpg"))
        self.video_paths = [video[:-10] for video in self.videos]
        self.videos = [video.split('/')[-2] for video in self.videos]
        self.gts = {}
        for path in self.video_paths:
            self.gts[path.split('/')[-2]] = json.loads(open(path + "IR_label.json").read())
            self.gts[path.split('/')[-2]]["image_paths"] = sorted(list(glob.glob(path + "*.jpg")))
        # print(self.gts["20190926_200510_1_3"].keys())
        #exit()
        print(
            "found {:d} videos in {}".format(
                len(self.videos), self.dataset_location
            )
        )
        """
        for video in tqdm(self.videos):
            indexes = [index for index, frame in enumerate(self.gts_raw["images"]) if frame["file_name"].startswith(video)]
            self.gts[video] = [self.gts_raw["images"][index] for index in indexes]
            for i in range(len(indexes)):
                self.gts[video][i].update(self.gts_raw["annotations"][indexes[i]])
        """

        self.all_info = []
        # self.video_names = []
        # self.full_idxs = []
        # self.inverts = [] # the data is infrared, so either flip should be OK

        for video in self.videos:
            bboxes = self.gts[video]["gt_rect"]
            try:
                ok = np.array(bboxes)
            except:
                # print(bboxes)
                print('some problem with', video)
                continue

            bbox0 = np.array(bboxes[0])
            x,y,w,h = bbox0

            img_paths = self.gts[video]["image_paths"]
            img_path0 = img_paths[0]
            rgb0 = cv2.imread(str(img_path0))[..., ::-1].copy()
            H, W, C = rgb0.shape
            if w/W < 0.02 or h/H < 0.02: # exclude targets that are <2% of image dim, because i can barely see them myself
                continue
            
            visibs = self.gts[video]["exist"]
            S_local = len(visibs)
            # print('S_local', S_local)
            for stride in strides:
                for ii in range(0, max(S_local - self.S*stride, 1), clip_step*stride):
                    full_idx = ii + np.arange(self.S)*stride
                    if len(full_idx) < (((S_local - 1) // stride + 1 if self.S == -1 else self.S) if fullseq else 8):
                        continue
                    for inv in [False, True]:
                        for zoom in zooms:
                            self.all_info.append([video, full_idx, inv, zoom])
        print("found {:d} samples in {}".format(len(self.all_info), self.dataset_location))
        if chunk is not None:
            assert(len(self.all_info) > 100)
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.all_info = chunkify(self.all_info,100)[chunk]
            print('filtered to %d' % len(self.all_info))

    def getitem_helper(self, index):
        video, full_idx, inv, zoom = self.all_info[index]
        
        bboxes = np.array(self.gts[video]["gt_rect"])
        visibs = np.array(self.gts[video]["exist"])
        img_paths = self.gts[video]["image_paths"]

        # assert len(img_paths) == len(bboxes)
        if bboxes.shape[0] > len(img_paths):
            bboxes = bboxes[: len(img_paths)]

        bboxes = bboxes[full_idx]
        visibs = visibs[full_idx]
        img_paths = [img_paths[i] for i in full_idx]

        rgbs = [cv2.imread(str(path))[..., ::-1].copy() for path in img_paths]

        rgbs = np.stack(rgbs)  # S, C, H, W
        bboxes = np.stack(bboxes)  # S, 4
        bboxes[np.any(np.isnan(bboxes), axis=1)] = 0
        # from xywh to xyxy
        bboxes[..., 2:] += bboxes[..., :2]

        if inv:
            rgbs = 255-rgbs

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
