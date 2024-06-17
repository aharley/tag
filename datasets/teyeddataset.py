from collections import defaultdict
import glob
import os
from re import L
import cv2
import time
import json
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

from datasets.dataset_utils import make_split


# this dataset is huge
class TeyedPointDataset(PointDataset):
    def __init__(
            self,
            dataset_location="../teyed",
            S=32, fullseq=False, chunk=None,
            strides=[1], # only one stride
            zooms=[1.5],
            crop_size=(384, 512),
            use_augs=False,
            is_training=True,
    ):
        print("loading teyed point dataset...")
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training,
        )

        # the data is highly redundant, so we use a large clip stride
        # if is_training:
        #     clip_step = S*32
        # else:
        # clip_step = S*64 
            
        # validation gt not provided
        self.root = Path(dataset_location) / "TEyeDSSingleFiles"
        print(self.root)
        # TODO: replace Dikablis with other datasets
        video_names = sorted(list(set([path.stem.split(".mp4")[0] for path in self.root.glob("Dikablis/ANNOTATIONS/*.txt")])))
        video_names = make_split(video_names, is_training, shuffle=True)
        print("found {:d} videos in {} (we made our own train/test split)".format(len(video_names), self.dataset_location))

        if chunk is not None:
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            video_names = chunkify(video_names,100)[chunk]
            print('filtered to %d video_names' % len(video_names))
            print('video_names', video_names)

        self.all_video_names = []
        self.all_anno_types = []
        self.all_full_idx = []
        self.all_lm_idx = []
        self.all_zooms = []

        # For now, only use the eyelid
        for anno_type in ["lid_lm_2D"]:
            for video_name in video_names[:]:
                if (
                    video_name == "DikablisSS_10_1" and anno_type == "iris_lm_2D"
                ):  # this annotation is corrupted
                    continue
                annos = open(
                    next(
                        self.root.glob(
                            "*/ANNOTATIONS/"
                            + (video_name + ".mp4" + anno_type + ".txt")
                        )
                    )
                ).readlines()
                valids = open(
                    next(
                        self.root.glob(
                            "*/ANNOTATIONS/"
                            + (
                                video_name
                                + ".mp4"
                                + "validity_"
                                + anno_type.split("_")[0]
                                + ".txt"
                            )
                        )
                    )
                ).readlines()
                valid_dict = {}
                for l in valids[1:]:
                    _info = l.split(";")
                    frame_id = int(_info[0])
                    valid = int(_info[1])
                    valid_dict[frame_id] = True if valid > 0 else False

                frame_ids = []
                for l in annos[1:]:
                    _info = l.split(";")
                    frame_id = int(_info[0])
                    if valid_dict[frame_id]:
                        frame_ids.append(frame_id)
                    _landmarks = _info[2:-1]

                if len(frame_ids) < self.S: # don't allow zigzag
                    continue
                start = np.min(frame_ids)
                end = np.max(frame_ids) + 1
                for stride in strides:
                    # for ii in range(start, max(end - self.S * stride, start + 1), clip_step) # wide clip stride, to reduce data redundancy
                    # ii = 0
                    for lm_idx in range(34 if anno_type == "lid_lm_2D" else 8):
                        # for ii in range(start+lm_idx*self.S, max(end - self.S * stride, start + 1), clip_step):

                        # the data is highly redundant so let's just take one clip per kp per vid
                        ii = start+lm_idx*self.S
                        full_idx = ii + np.arange(self.S) * stride
                        full_idx = [ij for ij in full_idx if ij < end]
                        if len(full_idx) == self.S:
                            for zoom in zooms:
                                self.all_video_names.append(video_name)
                                self.all_anno_types.append(anno_type)
                                self.all_full_idx.append(full_idx)
                                self.all_lm_idx.append(lm_idx)
                                self.all_zooms.append(zoom)
                                # sys.stdout.write('.')
                            # sys.stdout.flush()

        print(
            "found {:d} samples in {}".format(
                len(self.all_video_names), self.dataset_location
            )
        )

    def getitem_helper(self, index):
        video_name = self.all_video_names[index]
        anno_type = self.all_anno_types[index]
        full_idx = self.all_full_idx[index]
        lm_idx = self.all_lm_idx[index]
        zoom = self.all_zooms[index]

        annos = open(
            next(
                self.root.glob(
                    "*/ANNOTATIONS/" + (video_name + ".mp4" + anno_type + ".txt")
                )
            )
        ).readlines()
        valids = open(
            next(
                self.root.glob(
                    "*/ANNOTATIONS/"
                    + (
                        video_name
                        + ".mp4"
                        + "validity_"
                        + anno_type.split("_")[0]
                        + ".txt"
                    )
                )
            )
        ).readlines()

        # valids = open(self.root / video_name.split('_')[0] / 'ANNOTATIONS' / (video_name + '.mp4' + 'validity_' + anno_type + '.txt')).readlines()
        valid_dict = {}
        for l in valids[1:]:
            _info = l.split(";")
            frame_id = int(_info[0])
            valid = int(_info[1])
            valid_dict[frame_id] = True if valid > 0 else False

        # annos = open(self.root / video_name.split('_')[0] / 'ANNOTATIONS' / (video_name + '.mp4' + anno_type + '.txt')).readlines()
        annos_dict = {}
        for l in annos[1:]:
            _info = l.split(";")
            frame_id = int(_info[0])
            _landmarks = _info[2:-1]

            x = float(_landmarks[2 * lm_idx])
            y = float(_landmarks[2 * lm_idx + 1])

            annos_dict[frame_id] = [x, y]

        visibs = []
        xys = []
        for idx in full_idx:
            if idx not in annos_dict or valid_dict[idx] == False:
                visibs.append(0)
                xys.append(np.zeros((2,), dtype=np.float32))
            else:
                visibs.append(1)
                xys.append(np.array(annos_dict[idx], dtype=np.float32))

        vidcap = cv2.VideoCapture(
            str(next(self.root.glob("*/VIDEOS/" + (video_name + ".mp4"))))
        )
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, min(full_idx) - 1)
        rgbs = []
        for frame in range(min(full_idx) - 1, max(full_idx)):
            success, image = vidcap.read()
            if frame + 1 in full_idx:
                rgbs.append(image[..., ::-1].copy())

        rgbs = np.stack(rgbs)
        xys = np.stack(xys)
        visibs = np.array(visibs)

        S,H,W,C = rgbs.shape

        # if np.sum(visibs)==S:
        #     return None

        travel = np.sum(np.linalg.norm(xys[1:] - xys[:-1], axis=-1))
        # print('travel/max(H,W)', travel/max(H,W))
        if travel/max(H,W) < 0.1:
            print('travel', travel/max(H,W))
            return None

        if visibs[0] == 0 or visibs[-1] == 0:
            return None

        valids = visibs.copy()
        if zoom > 1:
            xys, visibs, valids, rgbs = utils.misc.data_zoom(zoom, xys, visibs, valids, rgbs)
            _, H, W, _ = rgbs.shape

        sample = {
            "rgbs": rgbs,
            "xys": xys,
            "visibs": visibs,
        }
        return sample

    def __len__(self):
        return len(self.all_video_names)


class TeyedSegDataset(MaskDataset):
    def __init__(
            self,
            dataset_location="../teyed",
            S=32, fullseq=False, chunk=None,
            strides=[1], # one stride
            zooms=[1.5],
            crop_size=(384, 512), 
            use_augs=False,
            is_training=True,
    ):
        print("loading teyed mask dataset...")
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training,
        )

        # # the data is highly redundant, so we use a large clip stride
        # if is_training:
        #     clip_step = S*32
        # else:
        #     clip_step = S*64 
            
        # validation gt not provided
        self.root = Path(dataset_location) / "TEyeDSSingleFiles"
        print(self.root)
        # TODO: replace Dikablis with other datasets
        video_names = list(
            set(
                [
                    path.stem.split(".mp4")[0]
                    for path in self.root.glob("Dikablis/ANNOTATIONS/*.txt")
                ]
            )
        )
        video_names = make_split(video_names, is_training, shuffle=True)
        print(
            "found {:d} videos in {} (we made our own train/test split)".format(
                len(video_names), self.dataset_location
            )
        )

        if chunk is not None:
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            video_names = chunkify(video_names,100)[chunk]
            print('filtered to %d video_names' % len(video_names))
            print('video_names', video_names)

        self.all_video_names = []
        self.all_anno_types = []
        self.all_full_idx = []
        self.all_lm_idx = []
        self.all_zooms = []

        for anno_type in ["iris_lm_2D", "pupil_lm_2D", "lid_lm_2D"]:
            for video_name in video_names[:]:
                if (
                    video_name == "DikablisSS_10_1" and anno_type == "iris_lm_2D"
                ):  # this annotation is corrupted
                    continue
                annos = open(
                    next(
                        self.root.glob(
                            "*/ANNOTATIONS/"
                            + (video_name + ".mp4" + anno_type + ".txt")
                        )
                    )
                ).readlines()
                valids = open(
                    next(
                        self.root.glob(
                            "*/ANNOTATIONS/"
                            + (
                                video_name
                                + ".mp4"
                                + "validity_"
                                + anno_type.split("_")[0]
                                + ".txt"
                            )
                        )
                    )
                ).readlines()
                valid_dict = {}
                for l in valids[1:]:
                    _info = l.split(";")
                    frame_id = int(_info[0])
                    valid = int(_info[1])
                    valid_dict[frame_id] = True if valid > 0 else False

                frame_ids = []
                for l in annos[1:]:
                    _info = l.split(";")
                    frame_id = int(_info[0])
                    if valid_dict[frame_id]:
                        frame_ids.append(frame_id)

                if len(frame_ids) < self.S: # don't allow zigzag
                    continue
                start = np.min(frame_ids)
                end = np.max(frame_ids) + 1
                for stride in strides:
                    for lm_idx in range(34 if anno_type == "lid_lm_2D" else 8):
                        # the data is highly redundant so let's just take one clip per kp per vid
                        ii = start+lm_idx*self.S
                        full_idx = ii + np.arange(self.S) * stride
                        full_idx = [ij for ij in full_idx if ij < end]
                        if len(full_idx) == self.S:
                            for zoom in zooms:
                                self.all_video_names.append(video_name)
                                self.all_anno_types.append(anno_type)
                                self.all_full_idx.append(full_idx)
                                self.all_lm_idx.append(lm_idx)
                                self.all_zooms.append(zoom)
                            # sys.stdout.write('.')
                            # sys.stdout.flush()

        print(
            "found {:d} samples in {}".format(
                len(self.all_video_names), self.dataset_location
            )
        )

    def getitem_helper(self, index):
        video_name = self.all_video_names[index]
        anno_type = self.all_anno_types[index]
        full_idx = self.all_full_idx[index]
        lm_idx = self.all_lm_idx[index]
        zoom = self.all_zooms[index]

        valids = open(
            next(
                self.root.glob(
                    "*/ANNOTATIONS/"
                    + (
                        video_name
                        + ".mp4"
                        + "validity_"
                        + anno_type.split("_")[0]
                        + ".txt"
                    )
                )
            )
        ).readlines()

        # valids = open(self.root / video_name.split('_')[0] / 'ANNOTATIONS' / (video_name + '.mp4' + 'validity_' + anno_type + '.txt')).readlines()
        valid_dict = {}
        for l in valids[1:]:
            _info = l.split(";")
            frame_id = int(_info[0])
            valid = int(_info[1])
            valid_dict[frame_id] = True if valid > 0 else False

        # annos = open(self.root / video_name.split('_')[0] / 'ANNOTATIONS' / (video_name + '.mp4' + anno_type + '.txt')).readlines()
        segs_dict = {}
        segcap = cv2.VideoCapture(
            str(
                next(
                    self.root.glob(
                        "*/ANNOTATIONS/"
                        + (
                            video_name
                            + ".mp4"
                            + anno_type.replace("lm", "seg")
                            + ".mp4"
                        )
                    )
                )
            )
        )
        segcap.set(cv2.CAP_PROP_POS_FRAMES, min(full_idx) - 1)
        for frame in range(min(full_idx) - 1, max(full_idx)):
            success, image = segcap.read()
            if frame + 1 in full_idx:
                segs_dict[frame + 1] = image[..., 0] / 255.0

        vidcap = cv2.VideoCapture(
            str(next(self.root.glob("*/VIDEOS/" + (video_name + ".mp4"))))
        )
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, min(full_idx) - 1)
        rgbs = []
        for frame in range(min(full_idx) - 1, max(full_idx)):
            success, image = vidcap.read()
            if frame + 1 in full_idx:
                rgbs.append(image[..., ::-1].copy())

        visibs = []
        segs = []
        for idx in full_idx:
            if idx not in segs_dict or valid_dict[idx] == False:
                visibs.append(0)
                segs.append(np.zeros_like(rgbs[0][:, :, 0]).astype(np.float32))
            else:
                visibs.append(1)
                segs.append(np.array(segs_dict[idx], dtype=np.float32))

        rgbs = np.stack(rgbs)
        segs = np.stack(segs)
        visibs = np.array(visibs)

        masks = (segs > 0.5).astype(np.float32)

        if visibs[0] == 0 or visibs[-1] == 0:
            return None

        if zoom > 1:
            xys, _, valids, fills = utils.misc.data_get_traj_from_masks(masks)
            xys = utils.misc.data_replace_with_nearest(xys, valids)
            
            bboxes = np.stack([mask2bbox(mask) for mask in masks])
            whs = bboxes[:,2:4] - bboxes[:,0:2]
            whs = whs[visibs > 0.5]
            # print('mean wh', np.mean(whs[:,0]), np.mean(whs[:,1]))
            # if np.mean(whs[:,0]) >= W/2 and np.mean(whs[:,1]) >= H/2:
            #     # print('would reject')
            #     # big already
            #     return None
            xys, visibs, valids, rgbs, masks = utils.misc.data_zoom(zoom, xys, visibs, valids, rgbs, masks=masks)

        S,H,W,C = rgbs.shape
        # print('mask sums', masks.reshape(S,-1).sum(axis=1))
        # masks = np.stack([(vis_segs == tid + 1).astype(np.float32), full_segs[..., tid].astype(np.float32), occ_segs.astype(np.float32)], -1)
        masks = np.stack([masks*0, masks*0, masks], axis=-1)
        
        # chans 1,2 valid
        S = len(full_idx)
        masks_valid = np.zeros((S,3), dtype=np.float32)
        masks_valid[:,2] = 1

        sample = {
            "rgbs": rgbs,
            "visibs": visibs,
            "masks": masks,
            'masks_valid': masks_valid,
        }
        return sample

    def __len__(self):
        return len(self.all_video_names)
