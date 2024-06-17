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
import glob
import json
import imageio
import cv2
import re
import sys
from torchvision.transforms import ColorJitter, GaussianBlur
import pandas as pd
from scipy.interpolate import UnivariateSpline
from enum import Enum
from datasets.dataset import PointDataset
import utils.misc

class CattlePointDataset(PointDataset):
    def __init__(
            self,
            dataset_location="/orion/group/CattleEyeView",
            S=32, fullseq=False, chunk=None,
            strides=[1,2],
            zooms=[1,2],
            rand_frames=False,
            crop_size=(384, 512), 
            use_augs=False,
            is_training=True):
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size, 
            use_augs=use_augs,
            is_training=is_training
        )
        print('loading CattleEyeView point dataset...')

        clip_step = S//2
        flips = [False, True]
        if not is_training:
            strides = [2]
            clip_step = S
            flips = [False]

        filename = "coco_track_train.json" if is_training else "coco_track_test.json"
        gt_file = os.path.join(self.dataset_location, "annotation", "pose_COCO", filename)
        json_file = open(gt_file).read()
        gt_json = json.loads(json_file)
        self.tracks = {}
        for anno in gt_json["annotations"]:
            #print(len(anno["keypoints"]))
            num_pts = len(anno["keypoints"]) // 3
            for pt_id in range(num_pts):
                x, y, flag = anno["keypoints"][pt_id*3 : pt_id*3+3]
                track_id = str(anno['video_id']) + "_" + str(anno['instance_id']) + "_" + str(pt_id)
                if flag == 0:
                    continue
                vis = flag - 1
                if track_id not in self.tracks.keys():
                    #print(track_id)
                    self.tracks[track_id] = []
                frame_anno = {
                    "filename": anno["file_name"],
                    "coords": [x, y],
                    "vis": vis
                }
                filenames = [frame["filename"] for frame in self.tracks[track_id]]
                self.tracks[track_id].append(frame_anno)

        print("found {:d} point tracks in {}".format(len(self.tracks.keys()), self.dataset_location))


        self.all_info = []
        # self.all_full_idx = []
        # self.all_track_ids = []
        # self.all_zooms = []
        for track_id in self.tracks.keys():
            #print(occ.shape, absent.shape, bboxes.shape)
            coords = np.array([frame["coords"] for frame in self.tracks[track_id]])
            visibs = np.array([frame["vis"] for frame in self.tracks[track_id]])
            #print(track_id)
            #print(bboxes)
            #print(visibs)
            S_local = coords.shape[0]
            # print('S_local', S_local)
            for stride in strides:
                for ii in range(0, S_local, clip_step*stride):
                    # print('bbox_areas[%d]' % ii, bbox_areas[ii])
                    full_idx = ii + np.arange(self.S)*stride
                    full_idx = [ij for ij in full_idx if ij < S_local]
                    visibs_here = visibs[full_idx]
                    # zigzag seems to retrace the track in a very boring way
                    if len(full_idx) == S and np.sum(visibs_here) > 3:
                        for flip in flips: # inflate the dataset with vert flips
                            for zoom in zooms: # inflate the dataset with zooms
                                self.all_info.append([track_id, full_idx, flip, zoom])
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

        track_id, full_idx, flip, zoom = self.all_info[index]
        
        coords = np.array([frame["coords"] for frame in self.tracks[track_id]])
        visibs = np.array([frame["vis"] for frame in self.tracks[track_id]])
        img_paths = [self.dataset_location + "/images/" + frame["filename"] for frame in self.tracks[track_id]]

        # print('img_paths[0]', img_paths[0])

        if len(img_paths) > len(coords):
            # print(f'{gt_fn.parts[-2]}: {len(bboxes)} != {len(img_paths)}')
            img_paths = img_paths[: len(coords)]

        coords = coords[full_idx]
        visibs = visibs[full_idx]
        img_paths = [img_paths[i] for i in full_idx]
        #print(full_idx)
        #print(img_paths)
        # print(bboxes)
        # print(img_paths)

        rgbs = [cv2.imread(str(path))[..., ::-1].copy() for path in img_paths]

        rgbs = np.stack(rgbs) # S,H,W,3
        xys = np.stack(coords) # S,2

        S = xys.shape[0]
        
        xys = utils.misc.data_replace_with_nearest(xys, visibs)
        valids = np.ones_like(visibs)
        
        if zoom > 1:
            xys, visibs, valids, rgbs, _ = utils.misc.data_zoom(zoom, xys, visibs, valids, rgbs)

        visibs = 1.0 * (visibs>0.5)
        safe = visibs*0
        for si in range(1,S-1):
            safe[si] = visibs[si-1]*visibs[si]*visibs[si+1]
            
        if np.sum(safe) < 2:
            print('safe', safe)
            return None

        if flip:
            _, H, W, C = rgbs.shape
            assert(C==3)
            rgbs = np.flip(rgbs, axis=1)
            xys[:,1] = H-1 - xys[:,1]
        
        sample = {
            'rgbs': rgbs,
            'xys': xys,
            'visibs': visibs,
        }
        return sample
    
        
            
    def __len__(self):
        return len(self.all_info)

    
