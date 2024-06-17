from numpy import random
import torch
import numpy as np
import os
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import random
from torch._C import dtype, set_flush_denormal
import utils.basic
import utils.improc
import glob
import cv2
from torchvision.transforms import ColorJitter, GaussianBlur
from datasets.dataset import PointDataset
import sys
import csv
from pathlib import Path
import json

from datasets.dataset_utils import make_split


class SurgicalHandsDataset(PointDataset):
    def __init__(
            self,
            dataset_location="/orion/group/surgicalhands",
            use_augs=False,
            S=8, fullseq=False, chunk=None,
            strides=[1, 2],
            crop_size=(368, 496),
            is_training=True,
    ):
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size, 
            use_augs=use_augs,
            is_training=is_training,
        )
        print("loading surgical hands dataset...")

        clip_step = S//2
        if not is_training:
            strides = [1]
            clip_step = S

        self.S = S

        self.root = Path(dataset_location)
        self.gt_path = self.root / "annotations.json"
        self.annotations = json.loads(open(str(self.gt_path)).read())
        self.use_augs = use_augs
        self.sequences = []

        self.videos = sorted(self.annotations.keys())
        print("found %d unique videos in %s" % (len(self.videos), dataset_location))
        #print(self.videos)
        
        self.videos = make_split(self.videos, is_training, shuffle=True, train_ratio=0.8)

        self.gt = {}
        print("loading annotations...")
        
        self.rgb_paths = []
        self.full_idxs = []
        self.video_names = []
        self.tids = []
        self.sides = []

        ## load trajectories
        print("loading trajectories...")
        
        for video in self.videos:
            gts = self.annotations[video]
            vid_anno = sorted(gts['annotations'], key = lambda d: d['id'])
            frame_anno = sorted(gts['images'], key = lambda d: d['id'])
            total_frames = len(frame_anno)
            frameid2idx = {}
            for i in range(total_frames):
                frameid2idx[frame_anno[i]["id"]] = i
            left_bbox = np.zeros((total_frames, 4))
            right_bbox = np.zeros((total_frames, 4))
            left_point_trajs = np.zeros((total_frames, 21, 3))
            right_point_trajs = np.zeros((total_frames, 21, 3))
            frame_paths = [self.root / "images" / img_anno["video_dir"] / img_anno["file_name"] for img_anno in frame_anno]
            for hand_anno in vid_anno:
                idx = frameid2idx[hand_anno["image_id"]]
                if hand_anno['category_id'] == 'left':
                    left_bbox[idx] = np.array(hand_anno['bbox'])
                    left_point_trajs[idx] = np.array(hand_anno['keypoints']).reshape(21, 3)
                elif hand_anno['category_id'] == 'right':
                    right_bbox[idx] = np.array(hand_anno['bbox'])
                    right_point_trajs[idx] = np.array(hand_anno['keypoints']).reshape(21, 3)
            self.gt[video] = {}
            self.gt[video]["left_bbox"] = left_bbox
            self.gt[video]["right_bbox"] = right_bbox
            self.gt[video]["left_trajs"] = left_point_trajs
            self.gt[video]["right_trajs"] = right_point_trajs
            S_local = total_frames
            N_local = 21
            for stride in strides:
                for ii in range(0, max(S_local - self.S * stride + 1, 1), clip_step*stride):
                    for side in ["left", "right"]:
                        if side == "left":
                            bboxes = left_bbox
                            trajs = left_point_trajs
                        elif side == "right":
                            bboxes = right_bbox
                            trajs = right_point_trajs
                        for ni in range(N_local):
                            full_idx = ii + np.arange(self.S)*stride
                            full_idx = [ij for ij in full_idx if ij < S_local]

                            if len(full_idx) < (((S_local - 1) // stride + 1 if self.S == -1 else self.S) if fullseq else 8):
                                continue

                            
                            if np.sum(trajs[full_idx, ni, 2] == 0) == 0:
                                traj = trajs[full_idx,ni,:2]
                                visibs = (trajs[full_idx,ni,2]-1).astype(np.float32)
                                print('visibs', visibs)
                                dist = np.linalg.norm(traj[1:] - traj[:-1], axis=-1)
                                # print('traj %d' % ni, traj, traj.shape)
                                # print('visibs %d' % ni, visibs, visibs.shape)
                                # print('dist %d' % ni, dist, dist.shape)
                                if visibs[0] and np.sum(visibs) > 2 and np.max(dist) < 128:
                                    self.rgb_paths.append([frame_paths[i] for i in full_idx])
                                    self.full_idxs.append(full_idx)
                                    self.video_names.append(video)
                                    self.tids.append(ni)
                                    self.sides.append(side)
            #print(gts.keys())
        
        print("done")

        print("collected %d clips in %s" % (len(self.rgb_paths), dataset_location))

    def getitem_helper(self, index):
        rgb_paths = self.rgb_paths[index]
        full_idx = self.full_idxs[index]
        video_name = self.video_names[index]
        tid = self.tids[index]
        side = self.sides[index]
        # print('video_name', video_name)
        
        if side == "left":
            trajs_all = self.gt[video_name]["left_trajs"]
        elif side == "right":
            trajs_all = self.gt[video_name]["right_trajs"]

        xys = trajs_all[full_idx, tid, :2].astype(np.float32)
        visibs = (trajs_all[full_idx, tid, 2] - 1).astype(np.float32)
        S_video, D = xys.shape
        assert D == 2
        # print('trajs', trajs, trajs.shape)

        dist = np.linalg.norm(xys[1:] - xys[:-1], axis=-1)
        # print('dist', dist)

        rgbs = []
        for rgb_path in rgb_paths:
            if not os.path.isfile(rgb_path):
                print('missing image:', rgb_path)
                return None
            
            with Image.open(rgb_path) as im:
                rgbs.append(np.array(im)[:, :, :3])

        rgbs = np.stack(rgbs, axis=0)  # S,H,W,3

        H, W, C = rgbs[0].shape
        assert C == 3

        sample = {
            "rgbs": rgbs,
            "xys": xys,
            "visibs": visibs,
        }
        return sample

    def __len__(self):
        return len(self.rgb_paths)
