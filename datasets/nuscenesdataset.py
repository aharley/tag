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
import math
from torchvision.transforms import ColorJitter, GaussianBlur
from datasets.dataset import PointDataset, BBoxDataset
from pathlib import Path
from datasets.dataset_utils import make_split


class NuScenesDataset(BBoxDataset):
    def __init__(
            self,
            dataset_location="/orion/group/nuscenes",
            S=32, fullseq=False, chunk=None,
            strides=[1,2],
            zooms=[1,2],
            rand_frames=False,
            crop_size=(384, 512),
            use_augs=False,
            is_training=True,
    ):
        print("loading nuscenes dataset...")
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
            strides = [2]
            clip_step = S

        img_anno_path = os.path.join(dataset_location, "v1.0-trainval", "image_annotations.json")
        instances_path = os.path.join(dataset_location, "v1.0-trainval", "instance.json")
        start_time = time.time()
        img_anno = json.loads(open(img_anno_path).read())
        print("Image annotations loaded, took %.3f seconds" % (time.time() - start_time))
        #print(img_anno[0])
        start_time = time.time()
        self.bbox_tracks = {}
        for anno in img_anno:
            obj_id = anno["instance_token"]
            cam_id = anno["filename"].split("__")[1]
            track_id = obj_id + "_" + cam_id
            if track_id not in self.bbox_tracks.keys():
                self.bbox_tracks[track_id] = []
                #print(track_id)
            vis = int(anno['visibility_token'])
            vis = 0.2 if vis == 1 else (0.2 + vis / 5.0)
            # 1 0-40 2 40-60 3 60-80 4 80-100
            frame = {
                'bbox': anno['bbox_corners'],
                'vis': vis,
                'filename': anno['filename']
            }
            self.bbox_tracks[track_id].append(frame)
            
        for obj in self.bbox_tracks.keys():
            self.bbox_tracks[obj] = sorted(self.bbox_tracks[obj], key=lambda d: d['filename']) 
        print("Bbox track dict constructed, took %.3f seconds" % (time.time() - start_time))
        print("found {:d} bbox tracks in {}".format(len(self.bbox_tracks.keys()), self.dataset_location))

        #exit()
        self.all_info = []
        for track_id in self.bbox_tracks.keys():
            #print(occ.shape, absent.shape, bboxes.shape)
            bboxes = np.array([frame["bbox"] for frame in self.bbox_tracks[track_id]])
            visibs = np.array([frame["vis"] for frame in self.bbox_tracks[track_id]])
            S_local = bboxes.shape[0] # 4 to 150
            print('S_local', S_local)
            # if fullseq and S_local < self.S: continue
            for stride in strides:
                for ii in range(0, S_local, clip_step*stride):
                    # print('bbox_areas[%d]' % ii, bbox_areas[ii])
                    full_idx = ii + np.arange(self.S)*stride
                    full_idx = [ij for ij in full_idx if ij < S_local]
                    if len(full_idx) < (((S_local - 1) // stride + 1 if self.S == -1 else self.S) if fullseq else 8):
                        continue
                    if np.sum(visibs[full_idx]) < 3: continue
                    # for flip in [False, True]: # 
                    for zoom in zooms:
                        self.all_info.append([track_id, full_idx, zoom])
            sys.stdout.write(".")
            sys.stdout.flush()
        print("found {:d} samples in {}".format(len(self.all_info), self.dataset_location))
        if chunk is not None:
            assert(len(self.all_info) > 100)
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.all_info = chunkify(self.all_info,100)[chunk]
            print('filtered to %d' % len(self.all_info))

    def getitem_helper(self, index):
        track_id, full_idx, zoom = self.all_info[index]
        
        bboxes = np.array([frame["bbox"] for frame in self.bbox_tracks[track_id]])
        visibs = np.array([frame["vis"] for frame in self.bbox_tracks[track_id]])
        img_paths = [os.path.join(self.dataset_location, frame["filename"]) for frame in self.bbox_tracks[track_id]]

        # print('img_paths[0]', img_paths[0])

        if len(img_paths) > len(bboxes):
            # print(f'{gt_fn.parts[-2]}: {len(bboxes)} != {len(img_paths)}')
            img_paths = img_paths[: len(bboxes)]

        bboxes = bboxes[full_idx]
        visibs = visibs[full_idx]
        img_paths = [img_paths[i] for i in full_idx]
        # print(full_idx)
        # print(img_paths)
        # print(bboxes)
        # print(img_paths)

        rgbs = [cv2.imread(str(path))[..., ::-1].copy() for path in img_paths]

        rgbs = np.stack(rgbs)  # S,H,W,3
        bboxes = np.stack(bboxes)  # S,4; xyxy

        # print('rgbs', rgbs.shape)
        # print("bboxes:", bboxes, bboxes.shape)
        S, H, W, _ = rgbs.shape
            
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
