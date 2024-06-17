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


class UAVDataset(BBoxDataset):
    def __init__(
        self,
        dataset_location="../UAV123",
        S=32,
        rand_frames=False,
        crop_size=(384, 512), 
        strides=[1,2,3],
        clip_step=8,
        use_augs=False,
        is_training=True,
    ):
        print("loading UAV dataset...")
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            rand_frames=rand_frames,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training,
        )

        if not is_training:
            strides = [2]
            clip_step = S

        self.root = Path(dataset_location)
        # UAV provides both long-term and short-term tracking
        self.gt_fns = sorted(list(self.root.glob("anno/UAV123/*.txt"))) + sorted(list(self.root.glob("anno/UAV20L/*.txt")))
        self.gt_fns = make_split(self.gt_fns, is_training, shuffle=True)
        print(
            "found {:d} videos in {}".format(
                len(self.gt_fns), self.dataset_location
            )
        )

        self.all_fns = []
        self.all_strides = []
        self.all_start_idxs = []
        for fn in self.gt_fns:
            for stride in strides:
                bboxes = np.loadtxt(fn, delimiter=",")
                sidx = 0
                eidx = len(bboxes) - 1
                for ii in range(
                    sidx,
                    max(eidx - self.S * stride + 1, sidx + 1),
                    min(self.S * stride // 2, clip_step),
                ):
                    self.all_fns.append(fn)
                    self.all_strides.append(stride)
                    self.all_start_idxs.append(ii)
        print(
            "found {:d} samples in {}".format(
                len(self.all_fns), self.dataset_location
            )
        )

    def getitem_helper(self, index):
        gt_fn = self.all_fns[index]
        stride = self.all_strides[index]
        start_ind = self.all_start_idxs[index]
        bboxes = np.loadtxt(gt_fn, delimiter=",")

        seq_name = gt_fn.stem
        offset = 0
        if "_" in gt_fn.stem and gt_fn.stem.split("_")[-1] != "s":
            seq_name = gt_fn.stem.split("_")[0]
            subseq_idx = int(gt_fn.stem.split("_")[1])
            for i in range(1, subseq_idx):
                offset += len(
                    np.loadtxt(gt_fn.parent / f"{seq_name}_{i}.txt", delimiter=",")
                )
        img_paths = list(
            sorted((self.root / "data_seq/UAV123" / seq_name).glob("*.jpg"))
            + sorted((self.root / "data_seq/UAV123" / seq_name).glob("*.png"))
        )[offset : offset + len(bboxes)]

        # assert len(img_paths) == len(bboxes)
        if len(bboxes) > len(img_paths):
            bboxes = bboxes[: len(img_paths)]

        visibs = 1.0 - np.any(np.isnan(bboxes), axis=1).astype(np.float32)

        def _pick_frames(ind):
            nonlocal bboxes, visibs, img_paths
            bboxes = bboxes[ind]
            visibs = visibs[ind]
            img_paths = [img_paths[ii] for ii in ind]

        num_frames = len(img_paths)
        # adam comment: i noticed some videos have an occlusion in the middle, so we can't use random starts
        ar = np.arange(num_frames)
        _pick_frames(ar[start_ind : start_ind + self.S * stride : stride])

        # we don't want first frame to be invisible
        if len(visibs) < 2 or (visibs[0] < 0.5 and visibs[1] < 0.5):
            return None

        rgbs = [cv2.imread(str(path))[..., ::-1].copy() for path in img_paths]

        rgbs = np.stack(rgbs)  # S, C, H, W
        bboxes = np.stack(bboxes)  # S, 4
        bboxes[np.any(np.isnan(bboxes), axis=1)] = 0
        # from xywh to xyxy
        bboxes[..., 2:] += bboxes[..., :2]

        sample = {
            "rgbs": rgbs,
            "visibs": visibs,  # S
            "bboxes": bboxes.astype(int),
        }
        return sample

    def __len__(self):
        return len(self.all_fns)
