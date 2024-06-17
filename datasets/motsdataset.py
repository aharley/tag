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
from datasets.dataset import PointDataset, BBoxDataset, MaskDataset, mask2bbox
from pathlib import Path
from icecream import ic
import pycocotools.mask as rletools

from datasets.dataset_utils import make_split


class MOTSDataset(MaskDataset):
    def __init__(
            self,
            dataset_location="/orion/group/MOTS",
            S=32, fullseq=True, chunk=None,
            strides=[1,2,3,4],
            zooms=[1,2,3,4],
            rand_frames=False,
            crop_size=(384, 512),
            use_augs=False,
            is_training=True,
    ):
        print("loading mots dataset...")
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            rand_frames=rand_frames,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training,
        )

        clip_step = S//2

        # validation gt not provided
        self.root = Path(dataset_location)
        self.fns = sorted(list(self.root.glob("train/*")))
        # self.fns = make_split(self.fns, is_training=is_training)
        print("found {:d} videos in {}".format(len(self.fns), self.dataset_location))

        self.data = []

        self.all_info = []
        # self.all_fns = []
        # self.all_gt = []
        # self.all_zooms = []
        for fn in self.fns:
            gt_fn = fn / "gt/gt.txt"
            gt = open(gt_fn).read().splitlines()
            gt = np.array([line.split(" ") for line in gt])
            obj_ids = np.unique(gt[:, 1].astype(int))
            for obj_id in obj_ids:
                if obj_id == 10000:
                    continue
                gt_i = gt[gt[:, 1].astype(int) == obj_id]
                frame_ids = gt_i[:, 0].astype(int)
                sidx = min(frame_ids)
                eidx = max(frame_ids) + 1
                for stride in strides:
                    for ii in range(sidx, eidx, clip_step):
                        # print('bbox_areas[%d]' % ii, bbox_areas[ii])
                        full_idx = ii + np.arange(self.S) * stride
                        full_idx = [ij for ij in full_idx if ij < eidx]
                        gt_padded = []
                        for idx in full_idx:
                            if idx in frame_ids:
                                gt_padded.append(gt_i[frame_ids == idx][0])
                            else:
                                gt_padded.append([idx, obj_id, 0, gt_i[0][3], gt_i[0][4], ""])
                        gt_padded = np.array(gt_padded)
                        if len(full_idx) == S: # zigzag doesn't make much sense here; always fullseq
                            for zoom in zooms:
                                self.all_info.append([fn, gt_padded, zoom])
                                # self.all_fns.append(fn)
                                # self.all_gt.append(gt_padded)
                                # self.all_zooms.append(zoom)

        print('found', len(self.all_info), 'samples')

        if chunk is not None:
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.all_info = chunkify(self.all_info,100)[chunk]
            print('filtered to %d all_info' % len(self.all_info))
            # print('self.all_info[:3]', self.all_info[:3])

    def getitem_helper(self, index):
        gt_fn, gt_info, zoom = self.all_info[index]

        img_ids = gt_info[:, 0].astype(int)
        img_paths = [gt_fn / "img1/{:06d}.jpg".format(img_id) for img_id in img_ids]
        
        masks = []
        for gt in gt_info:
            if gt[5] == "":
                mask = np.zeros((int(gt[3]), int(gt[4])))
            else:
                mask = rletools.decode({"size": [int(gt[3]), int(gt[4])], "counts": gt[5].encode('utf-8')})
            masks.append(mask)

        rgbs = []
        for path in img_paths:
            rgb = cv2.imread(str(path))[..., ::-1].copy()
            rgbs.append(rgb)

        rgbs = np.stack(rgbs)
        masks = np.stack(masks).astype(np.float32)
        S,H,W,C = rgbs.shape
        # print('H, W', H, W)

        if np.sum(masks) == 0:
            print('np.sum(masks)', np.sum(masks))
            return None

        mask_areas = (masks > 0).reshape(S,-1).sum(axis=1)
        mask_areas_norm = mask_areas / np.max(mask_areas)
        visibs = mask_areas_norm
        rgbs, masks = utils.misc.data_pad_if_necessary(rgbs, masks)
        S,H,W,C = rgbs.shape
        if zoom > 1:
            xys, _, valids, fills = utils.misc.data_get_traj_from_masks(masks)
            xys = utils.misc.data_replace_with_nearest(xys, valids)
            bboxes = np.stack([mask2bbox(mask) for mask in masks])
            whs = bboxes[:,2:4] - bboxes[:,0:2]
            whs = whs[visibs > 0.5]
            if np.mean(whs[:,0])*zoom >= W and np.mean(whs[:,1])*zoom >= H:
                return None
            xys, visibs, valids, rgbs, masks = utils.misc.data_zoom(zoom, xys, visibs, valids, rgbs, masks=masks)
        visibs = 1.0 * (visibs>0.5)
        safe = np.concatenate(([0], visibs[:-2] * visibs[1:-1] * visibs[2:], [0]))
        if np.sum(safe) < 2:
            return None
        
        sample = {
            "rgbs": rgbs,
            "masks": masks,
        }
        return sample

    def __len__(self):
        return len(self.all_info)
