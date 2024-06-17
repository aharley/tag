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


class Zef20BBoxDataset(BBoxDataset):
    def __init__(
            self,
            dataset_location="../3DZeF20",
            S=32, fullseq=False, chunk=None,
            strides=[1,2,3],
            crop_size=(384, 512), 
            use_augs=False,
            is_training=True,
    ):
        print("loading zef20 bbox dataset...")
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

        # validation gt not provided
        self.root = Path(dataset_location)
        gt_fns = sorted(list(self.root.glob("train/*/gt/gt.txt")))
        # gt_fns = make_split(gt_fns, is_training, shuffle=True)
        print("found {:d} videos in {}".format(len(gt_fns), self.dataset_location))

        self.all_fns = []
        self.all_ids = []
        self.all_strides = []
        self.all_views = []
        self.all_full_idx = []

        for gtf in gt_fns:
            gt = np.loadtxt(gtf, delimiter=",").astype(int)
            track_ids = np.unique(gt[:, 1])
            for track_id in track_ids:
                gt_here = gt[gt[:, 1] == track_id]
                frame_ids = gt_here[:, 0]
                img_paths = []
                for fid in range(np.min(frame_ids), np.max(frame_ids) + 1):
                    img_paths.append(gtf.parent.parent / "imgF" / f"{fid:06d}.jpg")

                S_local = len(img_paths)
                for stride in strides:
                    for ii in range(0, max(S_local - self.S * stride, 1), clip_step):
                        full_idx = ii + np.arange(self.S) * stride
                        full_idx = [ij for ij in full_idx if ij < S_local]
                        if len(full_idx) < (self.S if fullseq else 8): continue

                        for view in [0, 1]:
                            self.all_fns.append(gtf)
                            self.all_ids.append(track_id)
                            self.all_strides.append(stride)
                            self.all_views.append(view)
                            self.all_full_idx.append(full_idx)
        print(
            "found {:d} samples in {}".format(
                len(self.all_fns), self.dataset_location
            )
        )

    def getitem_helper(self, index):
        gt_fn = self.all_fns[index]
        track_id = self.all_ids[index]
        front_view = self.all_views[index]
        full_idx = self.all_full_idx[index]

        gt = np.loadtxt(gt_fn, delimiter=",").astype(int)
        gt = gt[gt[:, 1] == track_id]
        frame_ids = gt[:, 0]

        img_paths = []
        bboxes = []
        visibs = []
        for fid in range(np.min(frame_ids), np.max(frame_ids) + 1):
            img_paths.append(
                gt_fn.parent.parent
                / "img{}".format("F" if front_view > 0 else "T")
                / f"{fid:06d}.jpg"
            )
            if fid in frame_ids:
                bboxes.append(
                    gt[frame_ids == fid, 7 + 7 * front_view : 11 + 7 * front_view]
                )
                visibs.append(1 - gt[frame_ids == fid, 11 + 7 * front_view])
            else:
                bboxes.append(np.array([[0, 0, 0, 0]]))
                visibs.append(np.array([0]))
        bboxes = np.concatenate(bboxes)
        visibs = np.concatenate(visibs)

        img_paths = [img_paths[fi] for fi in full_idx]
        bboxes = bboxes[full_idx]
        visibs = visibs[full_idx]

        # # assert(False) # this dset needs work
        # if visibs[0] < 0.5:
        #     return None

        print('visibs', visibs)
        
        # occlusion flags are oddly aggressive, so let's just call it all visible
        visibs = np.ones_like(visibs)

        rgbs = [cv2.imread(str(path))[..., ::-1].copy() for path in img_paths]
        rgbs = np.stack(rgbs)  # S, C, H, W

        print('rgbs', rgbs.shape)

        # from xywh to xyxy
        bboxes[..., 2:] += bboxes[..., :2]

        sample = {
            "rgbs": rgbs,
            "visibs": visibs.astype(np.float32),  # S
            "bboxes": bboxes,
        }
        return sample

    def __len__(self):
        return len(self.all_fns)


class Zef20PointDataset(PointDataset):
    def __init__(
        self,
            dataset_location="../3DZeF20",
            S=32, fullseq=False, chunk=None,
            strides=[1,2,3],
            zooms=[1],
            crop_size=(384, 512), 
            use_augs=False,
            is_training=True,
    ):
        print("loading zef20 point dataset...")
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
            zooms = [2]

        # validation gt not provided
        self.root = Path(dataset_location)
        gt_fns = list(self.root.glob("train/*/gt/gt.txt"))
        gt_fns = make_split(gt_fns, is_training, shuffle=True)
        print(
            "found {:d} videos in {}".format(
                len(gt_fns), self.dataset_location
            )
        )

        self.all_fns = []
        self.all_ids = []
        self.all_strides = []
        self.all_zooms = []
        self.all_views = []
        self.all_full_idx = []

        for gtf in gt_fns:
            gt = np.loadtxt(gtf, delimiter=",").astype(int)
            track_ids = np.unique(gt[:, 1])
            for track_id in track_ids:
                gt_here = gt[gt[:, 1] == track_id]
                frame_ids = gt_here[:, 0]
                img_paths = []
                for fid in range(np.min(frame_ids), np.max(frame_ids) + 1):
                    img_paths.append(gtf.parent.parent / "imgF" / f"{fid:06d}.jpg")

                S_local = len(img_paths)
                for stride in strides:
                    for ii in range(0, max(S_local - self.S * stride, 1), clip_step):
                        full_idx = ii + np.arange(self.S) * stride
                        full_idx = [ij for ij in full_idx if ij < S_local]
                        if len(full_idx) < (self.S if fullseq else 8): continue
                        for view in [0, 1]:
                            for zoom in zooms:
                                self.all_fns.append(gtf)
                                self.all_ids.append(track_id)
                                self.all_strides.append(stride)
                                self.all_zooms.append(zoom)
                                self.all_views.append(view)
                                self.all_full_idx.append(full_idx)
        print(
            "found {:d} samples in {}".format(
                len(self.all_fns), self.dataset_location
            )
        )

    def getitem_helper(self, index):
        gt_fn = self.all_fns[index]
        track_id = self.all_ids[index]
        stride = self.all_strides[index]
        zoom = self.all_zooms[index]
        front_view = self.all_views[index]
        full_idx = self.all_full_idx[index]

        gt = np.loadtxt(gt_fn, delimiter=",").astype(int)
        gt = gt[gt[:, 1] == track_id]
        frame_ids = gt[:, 0]


        img_paths = []
        xys = []
        visibs = []
        for fid in range(np.min(frame_ids), np.max(frame_ids) + 1):
            img_paths.append(
                gt_fn.parent.parent
                / "img{}".format("F" if front_view > 0 else "T")
                / f"{fid:06d}.jpg"
            )
            if fid in frame_ids:
                xys.append(
                    gt[frame_ids == fid, 5 + 7 * front_view : 7 + 7 * front_view]
                )
                visibs.append(1 - gt[frame_ids == fid, 11 + 7 * front_view])
            else:
                xys.append(np.array([[0, 0]]))
                visibs.append(np.array([0]))
        xys = np.concatenate(xys)
        visibs = np.concatenate(visibs)

        img_paths = [img_paths[fi] for fi in full_idx]
        xys = xys[full_idx]
        visibs = visibs[full_idx]

        # if visibs[0] < 0.5:
        #     return None

        # # occlusion flags are oddly aggressive, so let's just call it all visible
        # visibs = np.ones_like(visibs)

        rgbs = [cv2.imread(str(path))[..., ::-1].copy() for path in img_paths]
        rgbs = np.stack(rgbs)  # S, C, H, W

        S, H, W, C = rgbs.shape

        if zoom > 1:
            valids = visibs[:]
            xys, visibs, valids, rgbs = utils.misc.data_zoom(zoom, xys, visibs, valids, rgbs)
        
        if np.sum(visibs) < 2:
            return None

        sample = {
            "rgbs": rgbs,
            "xys": xys,
            "visibs": visibs.astype(np.float32),  # S
        }
        return sample

    def __len__(self):
        return len(self.all_fns)
