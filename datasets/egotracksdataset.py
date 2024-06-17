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
from tqdm import tqdm
from datasets.dataset_utils import make_split


class EgoTracksDataset(BBoxDataset):
    def __init__(
            self,
            dataset_location="/orion/group/egotracks/v2",
            S=32, fullseq=False, chunk=None,
            strides=[1, 2],
            zooms=[1, 2],
            rand_frames=False,
            crop_size=(384, 512),
            use_augs=False,
            is_training=True,
    ):
        print("loading egotracks dataset...")
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

        # validation gt not provided
        self.root = Path(dataset_location)
        self.anno = json.load(open(self.root / 'egotracks/egotracks_{}.json'.format('train' if is_training else 'val')))
        self.videos = list(self.anno['videos'])
        self.videos = make_split(self.videos, is_training, shuffle=True, sort_first=False)
        print("found {:d} videos in {}".format(len(self.videos), self.dataset_location))

        if chunk is not None:
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.videos = chunkify(self.videos,100)[chunk]
            print('filtered to %d videos' % len(self.videos))
        
        self.all_clips = []
        self.all_frame_idxs = []
        self.all_bboxes = []
        self.all_visibs = []
        self.all_zooms = []
        for video in self.videos[:]:
            for clip in video['clips']:
                if not clip['annotation_complete']:
                    continue
                clip_fn = self.root / 'clips' / (clip['clip_uid'] + '.mp4')
                cap = cv2.VideoCapture(str(clip_fn))
                
                tracks = clip['annotations'][0]
                for key, track in tracks['query_sets'].items():
                    if not track['is_valid'] or not 'lt_track' in track:
                        continue
                    
                    if not np.all(['exported_clip_frame_number' in item for item in track['lt_track']]):
                        continue
                    
                    frame_numbers = np.array([item['exported_clip_frame_number'] for item in track['lt_track']])
                    sorted_idx = np.argsort(frame_numbers)
                    S_local = len(frame_numbers)
                    print('S_local', S_local)
                        
                    for stride in strides:
                        # egotracks has internal stride of 6
                        for start_idx in range(0, S_local-self.S*stride, clip_step*stride):
                            start_frame_idx = frame_numbers[sorted_idx[start_idx]]
                            bboxes = np.zeros((self.S, 4))
                            visibs = np.zeros(self.S)
                            frame_idxs = start_frame_idx + np.arange(self.S) * stride * 6

                            # fill up the data
                            for idx in range(start_idx, start_idx + self.S * stride, stride):
                                frame_idx = frame_numbers[sorted_idx[idx]]
                                # print(frame_idx, start_frame_idx, stride, start_idx, idx, self.S, self.S * stride, self.S * stride * 6)
                                cont_idx = (frame_idx - start_frame_idx) // (stride * 6)
                                if cont_idx >= self.S: # fullseq
                                    break
                                item = track['lt_track'][sorted_idx[idx]]
                                x, y, w, h = item['x'], item['y'], item['width'], item['height']
                                x, y, w, h = int(x), int(y), int(w), int(h)
                                bboxes[cont_idx] = np.array([x, y, x + w, y + h])
                                visibs[cont_idx] = 1.0
                            if np.sum(visibs) > 4:
                                for zoom in zooms:
                                    self.all_clips.append(clip_fn)
                                    self.all_frame_idxs.append(frame_idxs)
                                    self.all_bboxes.append(bboxes)
                                    self.all_visibs.append(visibs)
                                    self.all_zooms.append(zoom)
                                    # sys.stdout.write('.')
                                    # sys.stdout.write('.')
                                    # sys.stdout.flush()
                            
        print(
            "found {:d} samples in {}".format(
                len(self.all_clips), self.dataset_location
            )
        )

    def getitem_helper(self, index):
        clip_fn = self.all_clips[index]
        frame_idxs = self.all_frame_idxs[index]
        bboxes = self.all_bboxes[index]
        visibs = self.all_visibs[index]
        zoom = self.all_zooms[index]
        
        rgbs = []
        cap = cv2.VideoCapture(str(clip_fn))
        for frame_idx in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print('frame not found')
                return None
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgbs.append(frame)

        rgbs = np.stack(rgbs)
        bboxes = np.stack(bboxes)

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
        return len(self.all_clips)
