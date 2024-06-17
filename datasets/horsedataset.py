import torch
import numpy as np
import os
import scipy.ndimage
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import random
import glob
import json
import imageio
import cv2
from datasets.dataset import PointDataset
import pickle
import utils.misc

from datasets.dataset_utils import make_split

# https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_animal_keypoint.html#horse-10


class HorseDataset(PointDataset):
    def __init__(
            self,
            dataset_location,
            S=32, fullseq=False, chunk=None,
            strides=[1,2,3],
            zooms=[1,1.5,2],
            crop_size=None,
            use_augs=False,
            is_training=True,
    ):
        print("loading horse dataset...")
        super().__init__(
            dataset_location=dataset_location,
            S=S, 
            strides=strides,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training,
        )

        clip_step = S//2
        if not is_training:
            strides = [2]
            clip_step = S

        self.dataset_location = dataset_location
        self.S = S
        self.anno_path = os.path.join(self.dataset_location, "seq_annotation.pkl")
        with open(self.anno_path, "rb") as f:
            self.annotation = pickle.load(f)

        self.video_names = list(self.annotation.keys())
        print(f"found {len(self.annotation)} unique videos in {dataset_location}")
        self.video_names = make_split(self.video_names, is_training, shuffle=True)

        self.all_info = []

        rgb = None

        for video_name in self.video_names:
            video = self.annotation[video_name]
            S_local = len(video)
            print('S_local', S_local)
            for stride in strides:
                for ii in range(0, S_local, clip_step*stride):
                    full_idx = ii + np.arange(self.S) * stride
                    full_idx = [ij for ij in full_idx if ij < S_local]
                    if len(full_idx) < (((S_local - 1) // stride + 1 if self.S == -1 else self.S) if fullseq else 8):
                        continue

                    samples = [video[idx] for idx in full_idx]
                    visibs = []
                    trajs = []
                    for sample in samples:
                        visibs.append(np.squeeze(sample["keypoints_visible"], 0))
                        trajs.append(np.squeeze(sample["keypoints"], 0))
                    visibs = np.stack(visibs)
                    trajs = np.stack(trajs)

                    if rgb is None:
                        img_path = samples[0]["img_path"]
                        img_path = self.dataset_location + img_path
                        rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                        H, W, C = rgb.shape

                    for si in range(len(full_idx)):
                        # avoid 2px edge, since these are not really visible (according to adam)
                        oob_inds = np.logical_or(
                            np.logical_or(
                                trajs[si, :, 0] < 2, trajs[si, :, 0] >= W-2
                            ),
                            np.logical_or(
                                trajs[si, :, 1] < 2, trajs[si, :, 1] >= H-2
                            ),
                        )
                        visibs[si, oob_inds] = 0

                    vis_ok = visibs.sum(0) > 4

                    S, N, _ = trajs.shape

                    for ni in range(N):
                        if vis_ok[ni]:
                            for zoom in zooms:
                                self.all_info.append([video_name, full_idx, ni, zoom])

        print(f"found {len(self.all_info)} samples in {dataset_location}")

        if chunk is not None:
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.all_info = chunkify(self.all_info,100)[chunk]
            print('filtered to %d all_info' % len(self.all_info))
            # print('self.all_info', self.all_info)
        

    def getitem_helper(self, index):
        # print('index', index)
        video_name, full_idx, ni, zoom = self.all_info[index]

        video = self.annotation[video_name]
        samples = [video[idx] for idx in full_idx]


        rgbs = []
        trajs = []
        visibs = []
        for sample in samples:
            # print('sample', sample)
            img_path = sample["img_path"]
            img_path = self.dataset_location + img_path
            rgbs.append(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
            trajs.append(np.squeeze(sample["keypoints"], 0))
            visibs.append(np.squeeze(sample["keypoints_visible"], 0))

        rgbs = np.stack(rgbs, axis=0)
        trajs = np.stack(trajs, axis=0)
        visibs = np.stack(visibs, axis=0)

        S, H, W, C = rgbs.shape
        S, N, D = trajs.shape

        for si in range(S):
            # avoid 2px edge, since these are not really visible (according to adam)
            oob_inds = np.logical_or(
                np.logical_or(
                    trajs[si, :, 0] < 2, trajs[si, :, 0] >= W-2
                ),
                np.logical_or(
                    trajs[si, :, 1] < 2, trajs[si, :, 1] >= H-2
                ),
            )
            visibs[si, oob_inds] = 0
        
        xys = trajs[:, ni]
        visibs = visibs[:, ni]

        xys = utils.misc.data_replace_with_nearest(xys, visibs)
        valids = visibs[:]
        if zoom > 1:
            xys, visibs, valids, rgbs, _ = utils.misc.data_zoom(zoom, xys, visibs, valids, rgbs)
            S,H,W,C = rgbs.shape
        visibs = 1.0 * (visibs>0.5)
        safe = np.concatenate(([0], visibs[:-2] * visibs[1:-1] * visibs[2:], [0]))
        if np.sum(safe) < 2: return None
            
        d = {
            "rgbs": rgbs.astype(np.uint8),  # S, H, W, C
            "xys": xys.astype(np.int64),  # S, 2
            "visibs": visibs.astype(np.float32),  # S
        }
        return d

    def __len__(self):
        return len(self.all_info)
