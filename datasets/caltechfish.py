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


class CaltechFishDataset(BBoxDataset):
    def __init__(
            self,
            dataset_location="/orion/group/CaltechFish",
            S=32, fullseq=False, chunk=None,
            strides=[1,2],
            zooms=[1,2],
            rand_frames=False,
            crop_size=(384, 512), 
            use_augs=False,
            is_training=True,
    ):
        print("loading caltechfish dataset...")
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
            clip_step = S
            zooms = [1]
            strides = [1]
        
        # validation gt is provided
        self.root = Path(dataset_location)
        print('root', self.root)
        gt_fns = sorted(list(self.root.glob("annotations/*/*/*.txt")))
        print("found {:d} videos in {}".format(len(gt_fns), self.dataset_location))
        if is_training:
            gt_fns = [gtf for gtf in gt_fns if 'val' not in gtf.parent.parent.name]
        else:
            gt_fns = [gtf for gtf in gt_fns if 'val' in gtf.parent.parent.name]
        print("using {:d} videos in {}".format(len(gt_fns), self.dataset_location))
        self.data = []
        for gtf in gt_fns:
            gt = np.loadtxt(gtf, delimiter=",").astype(int)
            obj_ids = np.unique(gt[:, 1])

            for obj_id in obj_ids:
                gt_i = gt[gt[:, 1] == obj_id]
                S_local = gt_i[:, 0].max() + 1
                print('S_local', S_local)
                for stride in strides:
                    for ii in range(0, max(S_local - self.S * stride, 1), clip_step*stride):
                        # print('bbox_areas[%d]' % ii, bbox_areas[ii])
                        full_idx = ii + np.arange(self.S) * stride
                        full_idx = [ij for ij in full_idx if ij < S_local]

                        if len(full_idx) < (((S_local - 1) // stride + 1 if self.S == -1 else self.S) if fullseq else 8):
                            continue
                        if not (ii in gt_i[:, 0]): continue
                        for zoom in zooms:
                            self.data.append(
                                {
                                    "obj_id": int(obj_id),
                                    # /*/annotations-tiny/scenex/seq01/seq01-001.txt
                                    "scene_id": gtf.parent.parent.name,
                                    "seq_id": gtf.parent.name,
                                    "stride": stride,
                                    "gt_fn": gtf,
                                    "full_idx": full_idx,
                                    "zoom": zoom,
                                }
                            )

        print(
            "found {:d} samples in {}".format(
                len(self.data), self.dataset_location
            )
        )
        if chunk is not None:
            assert(len(self.data) > 100)
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.data = chunkify(self.data,100)[chunk]
            print('filtered to %d' % len(self.data))
        

    def getitem_helper(self, index):
        data = self.data[index]
        scene_id, seq_id, obj_id, stride, gtf, full_idx, zoom = (
            data["scene_id"],
            data["seq_id"],
            data["obj_id"],
            data["stride"],
            data["gt_fn"],
            data["full_idx"],
            data["zoom"],
        )

        gt = np.loadtxt(gtf, delimiter=",").astype(int)
        gt = gt[gt[:, 1] == obj_id]
        frame_ids = gt[:, 0]

        img_paths = []
        bboxes = []
        visibs = []
        for fid in full_idx:
            img_paths.append(
                self.root
                / "frames/raw/{}/{}/{}.jpg".format(scene_id, seq_id, fid)
            )
            # print(img_paths[-1])
            if fid in frame_ids:
                bboxes.append(gt[frame_ids == fid, 2:6])
                visibs.append(np.ones((bboxes[-1].shape[0],)))
            else:
                bboxes.append(np.array([[0, 0, 0, 0]]))
                visibs.append(np.array([0]))

        bboxes = np.concatenate(bboxes)
        visibs = np.concatenate(visibs)

        rgbs = [cv2.imread(str(path))[..., ::-1].copy() for path in img_paths]

        rgbs = np.stack(rgbs) # S,H,W,C
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

    def __len__(self):
        return len(self.data)
