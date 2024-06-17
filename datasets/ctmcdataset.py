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
import utils.misc
import glob
import json
import imageio
import cv2
import re
import sys
from torchvision.transforms import ColorJitter, GaussianBlur
from datasets.dataset import PointDataset, BBoxDataset, MaskDataset, mask2bbox, bbox2mask
from pathlib import Path
from icecream import ic

from datasets.dataset_utils import make_split


class CTMCDataset(BBoxDataset):
    def __init__(
            self,
            dataset_location="../CTMC",
            S=32,
            fullseq=False,
            chunk=None,
            crop_size=(384, 512), 
            strides=[2,4],
            zooms=[1,1.5,2],
            use_augs=False,
            is_training=True,
    ):
        print("loading CTMC dataset...")
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            fullseq=fullseq,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training,
        )
        
        if not is_training:
            strides = [2]
            zooms = [1]
            clip_step = S
        else:
            clip_step = S // 2
        
        self.root = Path(dataset_location) / 'train'
        self.seq_names = sorted([gtf.parts[-3] for gtf in self.root.glob("*/gt/gt.txt")])
        self.seq_names = make_split(self.seq_names, is_training, shuffle=True)
        print("found {:d} videos in {}".format(len(self.seq_names), self.dataset_location))

        self.all_info = []
        for si, seq_name in enumerate(self.seq_names):
            gt = np.loadtxt(self.root / seq_name / 'gt/gt.txt', delimiter=",").astype(int)
            self.process_video(gt, strides, zooms, clip_step, seq_name)

            # sys.stdout.write(".")
            sys.stdout.write("%d " % si)
            sys.stdout.flush()
            
        if chunk is not None:
            assert(len(self.all_info) > 100)
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.all_info = chunkify(self.all_info,100)[chunk]
            print('filtered to %d' % len(self.all_info))
                                
        print('found {:d} samples in {}'.format(len(self.all_info), self.dataset_location))
    
    def get_bbox_visibs(self, gt, full_idx, tid):
        bboxes = []
        visibs = []
        for i in full_idx:
            gt_here = gt[gt[:, 0] == i]
            if tid in gt_here[:, 1]:
                bbox = gt_here[gt_here[:, 1] == tid, 2:6]
                visib = np.ones((bbox.shape[0],))
            else:
                bbox = np.zeros((1, 4))
                visib = np.zeros((1,))
            bboxes.append(bbox)
            visibs.append(visib)
        bboxes = np.concatenate(bboxes)
        visibs = np.concatenate(visibs)
        return bboxes, visibs
    
    def get_track_ids(self, gt):
        track_ids = []
        for i in range(1, max(gt[:, 0]) + 1):
            track_ids.append(np.unique(gt[gt[:, 0] == i, 1]))
        return track_ids

    def process_video(self, gt, strides, zooms, clip_step, seq_name):
        if max(gt[:, 0]) < 4:
            return
        
        track_ids = self.get_track_ids(gt)
        
        for stride in strides:
            self.extract_clips(gt, stride, zooms, track_ids, clip_step, seq_name)
    
    def extract_clips(self, gt, stride, zooms, track_ids, clip_step, seq_name):
        S_local = max(gt[:, 0])
        # note, ctmc has 1-indexed frames
        for start_idx in range(1, S_local + 1, clip_step * stride):
            full_idx = start_idx + np.arange(self.S) * stride
            full_idx = full_idx[full_idx < S_local + 1]
            
            if len(full_idx) < (((S_local - 1) // stride + 1 if self.S == -1 else self.S) if self.fullseq else 8):
                continue
            
            valid_ids = np.unique(np.concatenate([track_ids[i - 1] for i in full_idx]))
            
            for tid in valid_ids:
                bboxes_here, visibs = self.get_bbox_visibs(gt, full_idx, tid)

                if np.sum(visibs) < self.S:
                    continue
                
                # if non visible, the dist will still be large, since bbox become zero
                dists = np.linalg.norm(bboxes_here[1:, :2] - bboxes_here[:-1, :2], axis=-1)
                if np.mean(dists) < 1.:
                    continue
                
                for zoom in zooms:
                    self.all_info.append((gt, tid, stride, full_idx, zoom, seq_name))
                    

    def getitem_helper(self, index):
        gt, track_id, stride, full_idx, zoom, seq_name = self.all_info[index]
        
        bboxes, visibs = self.get_bbox_visibs(gt, full_idx, track_id)
        image_paths = [self.root / "{}/img1/{:06d}.jpg".format(seq_name, fid) for fid in full_idx]
        # from xywh to xyxy
        bboxes[..., 2:] += bboxes[..., :2]

        rgbs = []
        for path in image_paths:
            rgb = cv2.imread(str(path))[..., ::-1].copy()
            rgbs.append(rgb)
        rgbs = np.stack(rgbs)
        
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
            print('safe', safe)
            return None
        
        sample = {
            'rgbs': rgbs,
            'visibs': visibs,  # S
            'bboxes': bboxes,
        }
        return sample

    def __len__(self):
        return len(self.all_info)


class CTMCMaskDataset(MaskDataset):
    def __init__(
            self,
            dataset_location="../CTMC",
            S=32, fullseq=False, chunk=None,
            crop_size=(384, 512), 
            strides=[2,4],
            zooms=[1,1.5,2],
            use_augs=False,
            is_training=True,
    ):
        print("loading CTMC mask dataset...")
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            fullseq=fullseq,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training,
        )

        if not is_training:
            strides = [2]
            zooms = [1]
            clip_step = S
        else:
            clip_step = S//2
        
        if not is_training:
            strides = [2]
            zooms = [1]
            clip_step = S
        else:
            clip_step = S // 2
        
        self.root = Path(dataset_location) / 'train'
        self.seq_names = sorted([gtf.parts[-3] for gtf in self.root.glob("*/gt/gt.txt")])
        self.seq_names = make_split(self.seq_names, is_training, shuffle=True)
        print("found {:d} videos in {}".format(len(self.seq_names), self.dataset_location))

        self.all_info = []
        for si, seq_name in enumerate(self.seq_names):
            gt = np.loadtxt(self.root / seq_name / 'gt/gt.txt', delimiter=",").astype(int)
            self.process_video(gt, strides, zooms, clip_step, seq_name)

            sys.stdout.write("%d " % si)
            sys.stdout.flush()

        print('found {:d} samples in {}'.format(len(self.all_info), self.dataset_location))
        
        if chunk is not None:
            assert(len(self.all_info) > 100)
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.all_info = chunkify(self.all_info,100)[chunk]
            print('filtered to %d' % len(self.all_info))
                                
        

    def get_bbox_visibs(self, gt, full_idx, tid):
        bboxes = []
        visibs = []
        for i in full_idx:
            gt_here = gt[gt[:, 0] == i]
            if tid in gt_here[:, 1]:
                bbox = gt_here[gt_here[:, 1] == tid, 2:6]
                visib = np.ones((bbox.shape[0],))
            else:
                bbox = np.zeros((1, 4))
                visib = np.zeros((1,))
            bboxes.append(bbox)
            visibs.append(visib)
        bboxes = np.concatenate(bboxes)
        visibs = np.concatenate(visibs)
        return bboxes, visibs
    
    def get_track_ids(self, gt):
        track_ids = []
        for i in range(1, max(gt[:, 0]) + 1):
            track_ids.append(np.unique(gt[gt[:, 0] == i, 1]))
        return track_ids

    def process_video(self, gt, strides, zooms, clip_step, seq_name):
        if max(gt[:, 0]) < 4:
            return
        
        track_ids = self.get_track_ids(gt)
        
        for stride in strides:
            self.extract_clips(gt, stride, zooms, track_ids, clip_step, seq_name)
    
    def extract_clips(self, gt, stride, zooms, track_ids, clip_step, seq_name):
        S_local = max(gt[:, 0])
        # note, ctmc has 1-indexed frames
        for start_idx in range(1, S_local + 1, clip_step * stride):
            full_idx = start_idx + np.arange(self.S) * stride
            full_idx = full_idx[full_idx < S_local + 1]
            
            if len(full_idx) < (((S_local - 1) // stride + 1 if self.S == -1 else self.S) if self.fullseq else 4):
                continue
            
            valid_ids = np.unique(np.concatenate([track_ids[i - 1] for i in full_idx]))
            
            for tid in valid_ids:
                bboxes_here, visibs = self.get_bbox_visibs(gt, full_idx, tid)
                
                if np.sum(visibs) < self.S:
                    continue
                
                # if non visible, the dist will still be large, since bbox become zero
                dists = np.linalg.norm(bboxes_here[1:, :2] - bboxes_here[:-1, :2], axis=-1)
                if np.mean(dists) < 1.:
                    continue
                
                for zoom in zooms:
                    self.all_info.append((gt, tid, stride, full_idx, zoom, seq_name))
                    
    def getitem_helper(self, index):
        gt, track_id, stride, full_idx, zoom, seq_name = self.all_info[index]
        
        bboxes, visibs = self.get_bbox_visibs(gt, full_idx, track_id)
        image_paths = [self.root / "{}/img1/{:06d}.jpg".format(seq_name, fid) for fid in full_idx]
        # from xywh to xyxy
        bboxes[..., 2:] += bboxes[..., :2]

        rgbs = []
        for path in image_paths:
            rgb = cv2.imread(str(path))[..., ::-1].copy()
            rgbs.append(rgb)
        rgbs = np.stack(rgbs)
        S,H,W,C = rgbs.shape
        
        masks = [bbox2mask(bbox, W, H) for bbox in bboxes]
        masks = np.stack(masks) * 0.5

        if np.sum(masks) == 0:
            print('np.sum(masks)', np.sum(masks))
            return None

        # padding and zooming
        mask_areas = (masks > 0).reshape(masks.shape[0],-1).sum(axis=1)
        mask_areas_norm = mask_areas / np.max(mask_areas)
        visibs = mask_areas_norm
        
        rgbs, masks = utils.misc.data_pad_if_necessary(rgbs, masks)
        if zoom > 1:
            xys, _, valids, fills = utils.misc.data_get_traj_from_masks(masks)
            xys = utils.misc.data_replace_with_nearest(xys, valids)
        
            bboxes = np.stack([mask2bbox(mask) for mask in masks])
            whs = bboxes[:,2:4] - bboxes[:,0:2]
            whs = whs[visibs > 0.5]
            if np.mean(whs[:,0])*zoom >= W and np.mean(whs[:,1])*zoom >= H:
                return None
            xys, visibs, valids, rgbs, masks = utils.misc.data_zoom(zoom, xys, visibs, valids, rgbs, masks=masks)
        visibs = 1.0 * (visibs>0.5)
        safe = np.concatenate(([0], visibs[:-2] * visibs[1:-1] * visibs[2:], [0]))
        if np.sum(safe) < 2:
            return None

        mask_cnts = np.stack([mask.sum() for mask in masks])
        if mask_cnts.max() <= 64:
            print('burst: max_cnts', mask_cnts.max())
            return None

        bboxes = [mask2bbox(mask) for mask in masks]
        bboxes = np.stack(bboxes, axis=0)

        for i in range(1, len(bboxes)):
            xy_prev = (bboxes[i - 1, :2] + bboxes[i - 1, 2:]) / 2
            xy = (bboxes[i, :2] + bboxes[i, 2:]) / 2
            dist = np.linalg.norm(xy - xy_prev)
            if np.sum(masks[i]) > 0 and np.sum(masks[i - 1]) > 0:
                if dist > 64:
                    print('large motion detected in {}'.format(image_paths[i]))
                    return None

        sample = {
            "rgbs": rgbs,
            "masks": masks,
        }
        return sample

    def __len__(self):
        return len(self.all_info)
