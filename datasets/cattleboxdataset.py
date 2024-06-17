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


class CattleBoxDataset(BBoxDataset):
    def __init__(
            self,
            dataset_location="/orion/group/CattleEyeView",
            S=32, fullseq=False, chunk=None,
            strides=[1,2],
            rand_frames=False,
            crop_size=(384, 512), 
            use_augs=False,
            is_training=True,
    ):
        print("loading CattleEyeView box dataset...")
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            rand_frames=rand_frames,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training,
        )

        clip_step = S//2
        # flips = [False, True]
        # if not is_training:
        #     strides = [2]
        #     clip_step = S
        #     flips = [False]

        #self.dataset_location = dataset_location

        filename = "coco_track_train.json" if is_training else "coco_track_test.json"
        gt_file = os.path.join(self.dataset_location, "annotation", "pose_COCO", filename)
        json_file = open(gt_file).read()
        gt_json = json.loads(json_file)

        #print(gt_json["annotations"][0])
        self.tracks = {}
        for anno in gt_json["annotations"]:
            if len(anno["bbox"]) != 4:
                print("Error bbox:", anno["bbox"])
            if type(anno["bbox_head"]) is list:
                if len(anno["bbox_head"]) != 4:
                    print("Error head bbox:", anno["bbox_head"])
                track_id = str(anno['video_id']) + "_" + str(anno['instance_id']) + "_head"
                if track_id not in self.tracks.keys():
                    #print("Head track:", track_id)
                    self.tracks[track_id] = []
                frame_anno = {
                    "filename": anno["file_name"],
                    "bbox": anno["bbox_head"],
                    "vis": anno["visibility"]
                }
                self.tracks[track_id].append(frame_anno)
            if anno["occluded"] or anno["truncated"] or anno["visibility"] != 1:
                print("Not fully visible")
                print(anno)
            track_id = str(anno['video_id']) + "_" + str(anno['instance_id'])
            if track_id not in self.tracks.keys():
                #print(track_id)
                self.tracks[track_id] = []
            frame_anno = {
                "filename": anno["file_name"],
                "bbox": anno["bbox"],
                "vis": anno["visibility"]
            }
            filenames = [frame["filename"] for frame in self.tracks[track_id]]
            self.tracks[track_id].append(frame_anno)

        print("found {:d} bbox tracks in {}".format(len(self.tracks.keys()), self.dataset_location))

        #exit()
        self.all_info = []
        for track_id in self.tracks.keys():
            #print(occ.shape, absent.shape, bboxes.shape)
            bboxes = np.array([frame["bbox"] for frame in self.tracks[track_id]])
            visibs = np.array([frame["vis"] for frame in self.tracks[track_id]])
            #print(track_id)
            #print(bboxes)
            #print(visibs)
            S_local = bboxes.shape[0] # 4 to 150
            # print('S_local', S_local)
            for stride in strides:
                for ii in range(0, max(S_local-self.S*stride,1), clip_step*stride):
                    # print('bbox_areas[%d]' % ii, bbox_areas[ii])
                    full_idx = ii + np.arange(self.S)*stride
                    full_idx = [ij for ij in full_idx if ij < S_local]
                    visibs_here = visibs[full_idx]
                    if len(full_idx) == S and np.sum(visibs_here) > 3:
                        # for flip in flips: # inflate the dataset with vert flips
                        self.all_info.append([track_id, full_idx])
            sys.stdout.write(".")
            sys.stdout.flush()
        print(
            "found {:d} samples in {}".format(
                len(self.all_info), self.dataset_location
            )
        )
        if chunk is not None:
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.all_info = chunkify(self.all_info,100)[chunk]
            print('filtered to %d' % len(self.all_info))
            # print('self.all_info', self.all_info)
        

    def getitem_helper(self, index):
        # track_id, full_idx, flip = self.all_info[index]
        track_id, full_idx = self.all_info[index]
        
        bboxes = np.array([frame["bbox"] for frame in self.tracks[track_id]])
        visibs = np.array([frame["vis"] for frame in self.tracks[track_id]])
        img_paths = [self.dataset_location + "/images/" + frame["filename"] for frame in self.tracks[track_id]]

        # print('img_paths[0]', img_paths[0])

        if len(img_paths) > len(bboxes):
            # print(f'{gt_fn.parts[-2]}: {len(bboxes)} != {len(img_paths)}')
            img_paths = img_paths[: len(bboxes)]

        bboxes = bboxes[full_idx]
        visibs = visibs[full_idx]
        img_paths = [img_paths[i] for i in full_idx]
        
        rgbs = [cv2.imread(str(path))[..., ::-1].copy() for path in img_paths]

        rgbs = np.stack(rgbs)  # S,H,W,3
        bboxes = np.stack(bboxes)  # S,4
        bboxes[..., 2:] += bboxes[..., :2]
        _, H, W, C = rgbs.shape

        # if flip:
        #     assert(C==3)
        #     rgbs = np.flip(rgbs, axis=1)
        #     tmp = bboxes[:,3].copy()
        #     bboxes[:,3] = bboxes[:,1]
        #     bboxes[:,1] = tmp

        visibs = 1.0 * (visibs>0.5)
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
        return len(self.all_info)
