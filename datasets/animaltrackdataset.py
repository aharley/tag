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


def read_mp4(fn, scale=1.0):
    vidcap = cv2.VideoCapture(fn)
    frames = []
    while vidcap.isOpened():
        ret, frame = vidcap.read()
        if ret == False:
            break
        if not scale == 1.0:
            H, W = frame.shape[:2]
            H_, W_ = int(H * scale), int(W * scale)
            frame = cv2.resize(frame, (W_, H_), interpolation=cv2.INTER_AREA)
            frame = frame[..., ::-1]
        frames.append(frame)
    vidcap.release()
    return frames


class AnimalTrackDataset(BBoxDataset):
    def __init__(
            self,
            dataset_location="../AnimalTrack",
            S=32,
            fullseq=False,
            chunk=None,
            rand_frames=False,
            strides=[1,2,3,4],
            zooms=[1,1.5],
            crop_size=(384, 512),
            use_augs=False,
            is_training=True,
    ):
        print("loading AnimalTrack dataset...")
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            rand_frames=rand_frames,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training,
        )

        clip_step = S//2

        # test gt not provided
        self.root = Path(dataset_location)
        gt_fns = sorted(list(self.root.glob("gt_all/*.txt")))
        print("found {:d} videos in {}".format(len(gt_fns), self.dataset_location))
        # gt_fns = make_split(gt_fns, is_training, shuffle=True)

        self.all_info = []
        for gtf in gt_fns:
            gt = np.loadtxt(gtf, delimiter=",").astype(int)
            obj_ids = np.unique(gt[:, 1])
            for obj_id in obj_ids:
                gt_here = gt[gt[:, 1] == obj_id]
                frame_ids = gt_here[:, 0]

                bboxes = []
                visibs = []
                for fid in range(np.min(frame_ids), np.max(frame_ids) + 1):
                    if fid in frame_ids:
                        bboxes.append(gt_here[frame_ids == fid, 2:6])
                        visibs.append(np.ones((bboxes[-1].shape[0],)))
                    else:
                        bboxes.append(np.array([[0, 0, 0, 0]]))
                        visibs.append(np.array([0]))
                bboxes = np.concatenate(bboxes)
                visibs = np.concatenate(visibs)

                # print('S_local', len(bboxes))

                for stride in strides:
                    sidx = 0
                    eidx = np.max(frame_ids) - np.min(frame_ids) + 1
                    # print('eidx', eidx)
                    if eidx > self.S * stride:
                        for ii in range(sidx, max(eidx - self.S * stride + 1, sidx + 1), clip_step*stride):
                            boxes_here = bboxes[ii : ii + self.S * stride][::stride]
                            visibs_here = visibs[ii : ii + self.S * stride][::stride]
                            if np.sum(visibs_here) > 3:
                                dists = np.linalg.norm(boxes_here[1:, :2] - boxes_here[:-1, :2], axis=-1)
                                if np.mean(dists) > 2.0:
                                    for zoom in zooms:
                                        self.all_info.append(
                                            {
                                                "zoom": zoom,
                                                "obj_id": int(obj_id),
                                                "scene_id": gtf.stem[:-3],
                                                "stride": stride,
                                                "gt_fn": gtf,
                                                "start_idx": ii-sidx,
                                            }
                                        )
                                    # sys.stdout.write('.')
                                    # sys.stdout.flush()

        print("found {:d} samples in {}".format(len(self.all_info), self.dataset_location))
        if chunk is not None:
            assert(len(self.all_info) > 100)
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.all_info = chunkify(self.all_info,100)[chunk]
            print('filtered to %d' % len(self.all_info))

    def getitem_helper(self, index):
        data = self.all_info[index]
        scene_id, obj_id, stride, gtf, start_ind, zoom = (
            data["scene_id"],
            data["obj_id"],
            data["stride"],
            data["gt_fn"],
            data["start_idx"],
            data["zoom"],
        )

        gt = np.loadtxt(gtf, delimiter=",").astype(int)
        gt = gt[gt[:, 1] == obj_id]
        frame_ids = gt[:, 0]

        scale = 0.5
        fn = str(self.root / "videos_all/{}.mp4".format(scene_id))
        rgbs = read_mp4(fn, scale=scale)

        img_idxs = []
        bboxes = []
        visibs = []
        for fid in range(np.min(frame_ids), np.max(frame_ids) + 1):
            img_idxs.append(fid - 1)
            if fid in frame_ids:
                bboxes.append(gt[frame_ids == fid, 2:6])
                visibs.append(np.ones((bboxes[-1].shape[0],)))
            else:
                bboxes.append(np.array([[0, 0, 0, 0]]))
                visibs.append(np.array([0]))
        bboxes = np.concatenate(bboxes) * scale
        visibs = np.concatenate(visibs)

        def _pick_frames(ind):
            nonlocal bboxes, visibs, img_idxs
            bboxes = bboxes[ind]
            visibs = visibs[ind]
            img_idxs = [img_idxs[ii] for ii in ind]

        S = self.S * stride
        _pick_frames(np.arange(len(img_idxs))[start_ind : start_ind + S : stride])

        # print('len(img_idxs)', len(img_idxs))
        # if len(img_idxs) < 8:
        #     return None

        # if visibs.shape[0] < 2 or visibs[0] == 0 or visibs[1] == 0:
        #     return None

        rgbs = np.stack(rgbs)[np.stack(img_idxs)]  # S,H,W,C
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
        if np.sum(safe) < 2: return None
        
        sample = {
            "rgbs": rgbs,
            "visibs": visibs,
            "bboxes": bboxes,
        }
        return sample

    def __len__(self):
        return len(self.all_info)
