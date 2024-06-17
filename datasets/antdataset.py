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
import utils.misc

from datasets.dataset_utils import make_split


# TODO: check if there is no test/validation split for this dataset
class AntDataset(BBoxDataset):
    def __init__(
            self,
            dataset_location="../ant",
            S=32, fullseq=False, chunk=None,
            strides=[1,2,3,4],
            zooms=[1,2,3],
            crop_size=(384, 512),
            use_augs=False,
            is_training=True,
    ):
        print("loading ant dataset...")
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size,
            is_training=is_training,
        )

        clip_step = S//2
        if not is_training:
            strides = [2]
            clip_step = S

        self.root = Path(dataset_location)
        self.gt_fns = sorted(list(self.root.glob("Ant_dataset/*/*/gt/gt.txt")))
        print("found {:d} videos in {}".format(len(self.gt_fns), self.dataset_location))
        # self.gt_fns = make_split(self.gt_fns, is_training, shuffle=True)

        self.all_info = []
        for gtf in self.gt_fns:
            gt = np.loadtxt(gtf, delimiter=",").astype(int)
            track_ids = np.unique(gt[:, 1])

            for tid in track_ids:
                gt_here = gt[gt[:, 1] == tid]
                frame_ids = gt_here[:, 0]

                img_paths = []
                for fid in range(np.min(frame_ids), np.max(frame_ids) + 1):
                    img_paths.append(gtf.parent.parent / "img" / f"{fid:06d}.jpg")

                S_local = len(img_paths)
                # print('S_local', S_local)
                for stride in strides:
                    for ii in range(0, max(S_local - self.S * stride, 1), clip_step*stride):
                        full_idx = ii + np.arange(self.S) * stride
                        full_idx = [ij for ij in full_idx if ij < S_local]
                        if len(full_idx) < (((S_local - 1) // stride + 1 if self.S == -1 else self.S) if fullseq else 4):
                            continue

                        for zoom in zooms:
                            self.all_info.append([gtf, tid, full_idx, zoom])
                        # self.all_fns.append(gtf)
                        # self.all_ids.append(tid)
                        # self.all_full_idx.append(full_idx)

        print(
            "found {:d} samples in {}".format(len(self.all_info), self.dataset_location)
        )
        
        if chunk is not None:
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.all_info = chunkify(self.all_info,100)[chunk]
            print('filtered to %d' % len(self.all_info))
        

    def __len__(self):
        return len(self.all_info)

    def getitem_helper(self, index):
        gt_fn, track_id, full_idx, zoom = self.all_info[index]

        gt = np.loadtxt(gt_fn, delimiter=",").astype(int)
        gt = gt[gt[:, 1] == track_id]
        frame_ids = gt[:, 0]

        img_paths = []
        bboxes = []
        visibs = []
        for fid in range(np.min(frame_ids), np.max(frame_ids) + 1):
            img_paths.append(gt_fn.parent.parent / "img" / f"{fid:06d}.jpg")
            if fid in frame_ids:
                bboxes.append(gt[frame_ids == fid, 2:6])
                visibs.append(gt[frame_ids == fid, -1])
            else:
                bboxes.append(np.array([[0, 0, 0, 0]]))
                visibs.append(np.array([0]))
        bboxes = np.concatenate(bboxes)
        visibs = np.concatenate(visibs)
        # print(bboxes.shape, visibs.shape)

        img_paths = [img_paths[fi] for fi in full_idx]
        bboxes = bboxes[full_idx]
        visibs = visibs[full_idx]

        rgbs = [cv2.imread(str(path))[..., ::-1].copy() for path in img_paths]

        rgbs = np.stack(rgbs)  # S, C, H, W
        # from xywh to xyxy
        bboxes[..., 2:] += bboxes[..., :2]

        visibs = visibs.astype(np.float32)

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


# class AntTestDataset(BBoxDataset):
#     def __init__(
#         self,
#         dataset_location="../ant",
#         S=32,
#         crop_size=(384, 512),
#     ):
#         print("loading ant dataset...")
#         super().__init__(dataset_location=dataset_location,
#                          S=S,
#                          crop_size=crop_size,
#                          is_training=False,
#                          inference=True)

#         self.root = Path(dataset_location)
#         self.gt_fns = sorted(list(self.root.glob("Ant_dataset/*/*/gt/gt.txt")))
#         print(
#             "found {:d} videos in {} (there is no train/test split)".format(
#                 len(self.gt_fns), self.dataset_location
#             )
#         )
#         self.gt_fns = make_split(self.gt_fns, False, shuffle=True)

#         self.all_fns = []
#         self.all_ids = []
#         self.all_full_idx = []
#         for gtf in self.gt_fns:
#             gt = np.loadtxt(gtf, delimiter=",").astype(int)
#             track_ids = np.unique(gt[:, 1])

#             for tid in track_ids:
#                 gt_here = gt[gt[:, 1] == tid]
#                 frame_ids = gt_here[:, 0]

#                 img_paths = []
#                 for fid in range(np.min(frame_ids), np.max(frame_ids) + 1):
#                     img_paths.append(gtf.parent.parent / "img" / f"{fid:06d}.jpg")

#                 S_local = min(len(img_paths), S)
#                 full_idx = np.arange(0, S_local)
                
#                 self.all_fns.append(gtf)
#                 self.all_ids.append(tid)
#                 self.all_full_idx.append(full_idx)

#         print(
#             "found {:d} samples in {}".format(len(self.all_fns), self.dataset_location)
#         )

#     def __len__(self):
#         return len(self.all_fns)

#     def getitem_helper(self, index):
#         gt_fn = self.all_fns[index]
#         track_id = self.all_ids[index]
#         full_idx = self.all_full_idx[index]

#         gt = np.loadtxt(gt_fn, delimiter=",").astype(int)
#         gt = gt[gt[:, 1] == track_id]
#         frame_ids = gt[:, 0]

#         img_paths = []
#         bboxes = []
#         visibs = []
#         for fid in range(np.min(frame_ids), np.max(frame_ids) + 1):
#             img_paths.append(gt_fn.parent.parent / "img" / f"{fid:06d}.jpg")
#             if fid in frame_ids:
#                 bboxes.append(gt[frame_ids == fid, 2:6])
#                 visibs.append(gt[frame_ids == fid, -1])
#             else:
#                 bboxes.append(np.array([[0, 0, 0, 0]]))
#                 visibs.append(np.array([0]))
#         bboxes = np.concatenate(bboxes)
#         visibs = np.concatenate(visibs)

#         img_paths = [img_paths[fi] for fi in full_idx]
#         bboxes = bboxes[full_idx]
#         visibs = visibs[full_idx]

#         rgbs = [cv2.imread(str(path))[..., ::-1].copy() for path in img_paths]

#         rgbs = np.stack(rgbs)  # S, C, H, W
#         # from xywh to xyxy
#         bboxes[..., 2:] += bboxes[..., :2]

#         sample = {
#             "rgbs": rgbs,
#             "visibs": visibs.astype(np.float32),  # S
#             "bboxes": bboxes,
#         }
#         return sample
