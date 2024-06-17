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


class DanceTrackDataset(BBoxDataset):
    def __init__(self,
                 dataset_location='../DanceTrack',
                 S=32, fullseq=False, chunk=None,
                 crop_size=(384,512), 
                 strides=[1,2],
                 zooms=[1,2],
                 use_augs=False,
                 is_training=True):
        print('loading dance track dataset...')
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size, 
            use_augs=use_augs,
            is_training=is_training)
        
        self.root = Path(dataset_location) / '{}'.format('train' if is_training else 'val')
        gt_fns = sorted(list(self.root.glob('dancetrack*/gt/gt.txt')))
        print('found {:d} videos in {} (training={})'.format(len(gt_fns), self.dataset_location, is_training))

        clip_step = S//2
        if not is_training:
            clip_step = S

        self.all_info = []

        for gtf in gt_fns:
            gt = np.loadtxt(gtf, delimiter=',').astype(int)
            obj_ids = np.unique(gt[:, 1])
            scene_id = gtf.parts[-3]
            for obj_id in obj_ids:
                gt_here = gt[gt[:, 1] == obj_id]
                frame_ids = gt_here[:, 0]
                start = np.min(frame_ids)
                end = np.max(frame_ids) + 1
                img_paths = []
                bboxes = []
                visibs = []
                for fid in range(start, end):
                    img_paths.append(self.root / '{}/img1/{:08d}.jpg'.format(scene_id, fid))
                    # print(img_paths[-1])
                    if fid in frame_ids:
                        bboxes.append(gt_here[frame_ids == fid, 2:6])
                        visibs.append(np.ones((bboxes[-1].shape[0],)))
                    else:
                        bboxes.append(np.array([[0, 0, 0, 0]]))
                        visibs.append(np.array([0]))
                bboxes = np.concatenate(bboxes)
                visibs = np.concatenate(visibs)
                # print('S_local', len(bboxes))
                # print('S_local', len(bboxes), 'visibs', visibs)
                S_local = len(bboxes)
                for stride in strides:
                    for ii in range(start, max(end - self.S*stride, start + 1), clip_step*stride):
                        
                        full_idx = ii + np.arange(self.S)*stride
                        full_idx = [ij - start for ij in full_idx if ij < end]

                        if len(full_idx) < (((S_local - 1) // stride + 1 if self.S == -1 else self.S) if fullseq else 8):
                            continue
                        # safe = np.concatenate(([0], visibs[:-2] * visibs[1:-1] * visibs[2:], [0]))
                        if np.sum(visibs[full_idx]) < 3: continue

                        for zoom in zooms:
                            self.all_info.append([gtf, int(obj_id), full_idx, scene_id, zoom])
                            # self.gtfs.append(gtf)
                            # self.obj_ids.append(int(obj_id))
                            # self.full_idxs.append(full_idx)
                            # self.scene_ids.append(scene_id)
            sys.stdout.write('.')
            sys.stdout.flush()
        print('found {:d} samples in {}'.format(len(self.all_info), self.dataset_location))
        if chunk is not None:
            assert(len(self.all_info) > 100)
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.all_info = chunkify(self.all_info,100)[chunk]
            print('filtered to %d' % len(self.all_info))
        

    def getitem_helper(self, index):
        gtf, obj_id, full_idx, scene_id, zoom = self.all_info[index]
        
        gt = np.loadtxt(gtf, delimiter=',').astype(int)
        gt = gt[gt[:, 1] == obj_id]
        frame_ids = gt[:, 0]
        
        img_paths = []
        bboxes = []
        visibs = []
        for fid in range(np.min(frame_ids), np.max(frame_ids) + 1):
            img_paths.append(self.root / '{}/img1/{:08d}.jpg'.format(scene_id, fid))
            # print(img_paths[-1])
            if fid in frame_ids:
                bboxes.append(gt[frame_ids == fid, 2:6])
                visibs.append(np.ones((bboxes[-1].shape[0],)))
            else:
                bboxes.append(np.array([[0, 0, 0, 0]]))
                visibs.append(np.array([0]))
        bboxes = np.concatenate(bboxes)
        visibs = np.concatenate(visibs)
        
        img_paths = [img_paths[i] for i in full_idx]
        bboxes = bboxes[full_idx]
        visibs = visibs[full_idx]

        rgbs = [cv2.imread(str(path))[..., ::-1].copy() for path in img_paths]
        
        rgbs = np.stack(rgbs) # S, C, H, W
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
            'rgbs': rgbs,
            'visibs': visibs,
            'bboxes': bboxes,
        }
        return sample


    def __len__(self):
        return len(self.all_info)
