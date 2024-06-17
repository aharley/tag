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
import gc
from datasets.dataset_utils import make_split

# import psutil
# import os
# def memory_usage():
#     process = psutil.Process(os.getpid())
#     memory_info = process.memory_info()
#     return memory_info.rss  # in bytes


class PacePointDataset(PointDataset):
    def __init__(
            self,
            dataset_location="/orion/group/bop/colspa_tracking",
            S=32, fullseq=False, chunk=None,
            strides=[1,2,3], # note the data is only 360x640, so zooming can't go far
            zooms=[1,1.5],
            crop_size=(384,512), 
            use_augs=False,
            is_training=True,
    ):
        print("loading PACE point dataset...")
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training,
        )

        clip_step = S//2 if is_training else S
        
        # validation gt not provided, so we split the dataset ourselves
        self.root = Path(dataset_location) / 'train_pbr'
        scenes = sorted(list(self.root.glob("*/")))
        # scenes = make_split(scenes, is_training, shuffle=True)
        print("found {:d} videos in {}".format(len(scenes), self.root))

        if chunk is not None:
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            scenes = chunkify(scenes,100)[chunk]
            print('filtered to %d scenes' % len(scenes))
            print('scenes', scenes)

        if not self.is_training:
            strides = [2]

        self.all_scenes = []
        self.all_ids = []
        self.all_strides = []
        self.all_nocs = []
        self.all_full_idx = []
        self.all_xys = []
        self.all_zooms = []

        min_area_th = 16
        keep_per_object = 16

        for scene_root in scenes[:]:
            scene_gt = json.load(open(scene_root / "scene_gt.json", "r"))
            track_ids = [gt['obj_id'] for gt in scene_gt['0']]
            S_local = len(scene_gt) # all seqs are 100 frames
            # print('S_local', S_local)
            for i, track_id in enumerate(track_ids):
                for stride in strides:
                    for ii in range(2, max(S_local - self.S * stride, 3), clip_step):
                        mask0 = cv2.imread(str(scene_root / "mask_visib" / f"{ii:06d}_{i:06d}.png"), cv2.IMREAD_GRAYSCALE) > 0
                        if mask0.sum() < min_area_th:
                            continue
                        nocs0 = cv2.imread(str(scene_root / "rgb_nocs" / f"{ii:06d}.png"))[..., ::-1] / 255.0
                        
                        full_idx = ii + np.arange(self.S) * stride
                        full_idx = [ij for ij in full_idx if ij < S_local]
                        if len(full_idx) == self.S: # always fullseq
                            # dp not choose points on edge, they are unstable (jumping visiblity)
                            mask0_erosion = cv2.erode(mask0.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=3)
                            if mask0_erosion.sum() < min_area_th:
                                continue
                            valid_idxs = np.where(mask0_erosion)
                            
                            xys = np.array([valid_idxs[1], valid_idxs[0]]).T
                            xys = xys[np.random.choice(len(xys), (keep_per_object,), replace=False)]
                            for xy in xys:
                                for zoom in zooms:
                                    self.all_scenes.append(scene_root)
                                    self.all_ids.append(track_id)
                                    self.all_strides.append(stride)
                                    self.all_nocs.append(np.array(nocs0[xy[1], xy[0]].tolist()))  # do a copy in order to free nocs0
                                    self.all_full_idx.append(full_idx)
                                    self.all_xys.append(xy)
                                    self.all_zooms.append(zoom)
                                sys.stdout.write('.')
                                sys.stdout.flush()
        print(
            "found {:d} samples in {}".format(
                len(self.all_xys), self.root
            )
        )

    def getitem_helper(self, index):
        scene_root = self.all_scenes[index]
        track_id = self.all_ids[index]
        full_idx = self.all_full_idx[index]
        xy = self.all_xys[index].astype(np.float32)
        nocs0 = self.all_nocs[index]
        zoom = self.all_zooms[index]
        
        scene_gt = json.load(open(scene_root / "scene_gt.json", "r"))
        xys = np.zeros((len(full_idx), 2), dtype=np.float32)
        visibs = np.zeros((len(full_idx),), dtype=np.float32)
        img_paths = []
        cnt = -1
        for fid in range(np.min(full_idx), np.max(full_idx) + 1):
            # break if xy is out of image
            if xy[0] < 0 or xy[0] >= 639 or xy[1] < 0 or xy[1] >= 359:
                break
            flow = np.load(scene_root / "flow" / f"{fid:06d}.npy")  # flow in xy format
            delta = np.stack(scipy.ndimage.map_coordinates(flow, [[xy[1], xy[1]], [xy[0], xy[0]], [0, 1]], order=2, mode='nearest'), -1)
            
            if fid in full_idx:
                cnt += 1
                img_paths.append(scene_root / "rgb" / f"{fid:06d}.jpg")
                obj_ids = [gt['obj_id'] for gt in scene_gt[str(fid)]]
                if track_id not in obj_ids:
                    xy += delta
                    continue
                obj_idx = obj_ids.index(track_id)
                mask = cv2.imread(str(scene_root / "mask_visib" / f"{fid:06d}_{obj_idx:06d}.png"), cv2.IMREAD_GRAYSCALE) > 0
                if not mask[int(xy[1]), int(xy[0])]:
                    xy += delta
                    continue
                nocs = cv2.imread(str(scene_root / "rgb_nocs" / f"{fid:06d}.png"))[..., ::-1] / 255.0
                if np.linalg.norm(nocs[int(xy[1]), int(xy[0])] - nocs0) > 0.3:
                    xy += delta
                    continue
                xys[cnt] = xy
                visibs[cnt] = 1.0
            xy += delta

        S = len(img_paths)

        # discard samples where we only got a subseq
        if S < self.S:
            print('S', S)
            return None

        rgbs = [cv2.imread(str(path))[..., ::-1].copy() for path in img_paths]
        rgbs = np.stack(rgbs)  # S, C, H, W
        # print('rgbs', rgbs.shape)
        
        visibs = visibs.astype(np.float32)[:S] # S
        xys = xys[:S]
        # note: OOB and invis points are NOT valid in pacepoint,
        # so we do not create a valids tensor here

        travel = np.sum(visibs[1:] * np.linalg.norm(xys[1:] - xys[:-1], axis=-1))
        H, W = rgbs[0,0].shape
        # print('travel/max(H,W)', travel/max(H,W))
        if travel/max(H,W) < 0.05:
            print('travel', travel/max(H,W))
            return None

        xys = utils.misc.data_replace_with_nearest(xys, visibs)
        valids = visibs[:]
        if zoom > 1:
            xys, visibs, valids, rgbs = utils.misc.data_zoom(zoom, xys, visibs, valids, rgbs)
            S,H,W,C = rgbs.shape
        visibs = 1.0 * (visibs>0.5)
        safe = np.concatenate(([0], visibs[:-2] * visibs[1:-1] * visibs[2:], [0]))
        if np.sum(safe) < 2: return None

        sample = {
            "rgbs": rgbs,
            "visibs": visibs, 
            "xys": xys,
        }
        return sample

    def __len__(self):
        return len(self.all_xys)
   
