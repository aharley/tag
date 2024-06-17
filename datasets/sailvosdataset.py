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
import glob
import json
import imageio
import cv2
import re
import sys
from torchvision.transforms import ColorJitter, GaussianBlur
from datasets.dataset import MaskDataset, mask2bbox
from icecream import ic
import matplotlib
from pathlib import Path
import matplotlib.pyplot as plt

from datasets.dataset_utils import make_split
import utils.misc

class SAILVOSDataset(MaskDataset):
    def __init__(self,
                 dataset_location='../SAIL-VOS',
                 S=32, fullseq=False, chunk=None,
                 crop_size=(384,512), 
                 strides=[1,2],
                 zooms=[1,2],
                 use_augs=False,
                 is_training=True,
    ):

        print('loading SAIL-VOS dataset...')
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size, 
            use_augs=use_augs,
            is_training=is_training
        )
        self.dataset_location = os.path.join(dataset_location)
        # only list dirs
        self.video_names = sorted([path for path in os.listdir(self.dataset_location) if os.path.isdir(os.path.join(self.dataset_location, path))])
        self.video_names = make_split(self.video_names, is_training=is_training)
        print('found {:d} videos in {}'.format(len(self.video_names), self.dataset_location))
        if not is_training:
            self.video_names = self.video_names[:2]
        
        # self.video_names = [vid for vid in self.video_names if ('Ped' in vid)]
        
        if chunk is not None:
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.video_names = chunkify(self.video_names,100)[chunk]
            print('filtered to %d video_names' % len(self.video_names))
            print('self.video_names', self.video_names)

        self.all_video_names = []
        self.all_tids = []
        self.all_zooms = []
        self.all_full_idx = []

        clip_step = S//2
        if not is_training:
            strides = [1]
            zooms = [1]
        
        for video_name in self.video_names:
            video_dir = '%s/%s/images' % (self.dataset_location, video_name)
            frames = sorted(os.listdir(video_dir))
            S_local = len(frames)

            vis_seg0 = np.load(os.path.join(self.dataset_location, video_name, 'visible', '%06d.npy' % (0)))
            vis_seg1 = np.load(os.path.join(self.dataset_location, video_name, 'visible', '%06d.npy' % (S_local//4)))
            vis_seg2 = np.load(os.path.join(self.dataset_location, video_name, 'visible', '%06d.npy' % (S_local//2)))
            vis_seg3 = np.load(os.path.join(self.dataset_location, video_name, 'visible', '%06d.npy' % (3*S_local//4)))
            vis_seg4 = np.load(os.path.join(self.dataset_location, video_name, 'visible', '%06d.npy' % (S_local-1)))

            tids = []
            for subname in sorted(os.listdir(os.path.join(self.dataset_location, video_name))):
                # if subname not in ['images', 'visible']:

                # many of the "objects" seem weird and untrackable
                # so let's stick with humans
                if 'Ped' in subname: 
                    tid = int(subname[:4])
                    for vs in [vis_seg0, vis_seg1, vis_seg2, vis_seg3, vis_seg4]:
                        mask = vs==tid
                        if np.sum(mask) > 128:
                            tids.append(tid)
            tids = np.unique(np.array(tids))

            vis_segs = []
            for t in range(S_local):
                vis_seg = np.load(os.path.join(self.dataset_location, video_name, 'visible', f'{t:06d}.npy'))
                vis_segs.append(vis_seg)
            vis_segs = np.stack(vis_segs)
            
            for stride in strides:
                for ii in range(0, S_local, clip_step*stride):
                    full_idx = ii + np.arange(self.S) * stride
                    full_idx = [ij for ij in full_idx if ij < S_local]
                    if len(full_idx) == self.S: # always fullseq

                        vis_segs_here = vis_segs[full_idx]
                        
                        all_masks = []
                        all_tids = []
                        for tid in tids:
                            masks = [(seg == tid).astype(np.float32) for seg in vis_segs_here]
                            masks = np.stack(masks, axis=0)

                            mask_sums = np.sum(masks.reshape(S,-1), axis=1)

                            if np.mean(mask_sums) < 64:
                                continue
                            
                            if np.min(mask_sums[mask_sums>0]) < 128:
                                continue

                            mask_areas = (masks > 0).reshape(S,-1).sum(axis=1)
                            mask_areas_norm = mask_areas / np.max(mask_areas)
                            visibs = mask_areas_norm
                            if np.sum(visibs>0.5) < 8:
                                continue
                            
                            # bboxes = np.stack([mask2bbox(mask) for mask in masks])
                            # whs = bboxes[:,2:4] - bboxes[:,0:2]
                            
                            # # help eliminate cuts and glitches
                            # ious = []
                            # for si in range(1,S):
                            #     mask0 = masks[si-1]
                            #     mask1 = masks[si]
                            #     if np.sum(mask0) > 0 and np.sum(mask1) > 0:
                            #         inter = np.sum(mask0*mask1)
                            #         union = np.sum(np.clip(mask0+mask1, 0, 1))
                            #         ious.append(inter/(1e-4+union))
                            # ious = np.stack(ious)
                            # if np.min(ious) == 0:
                            #     continue
                            
                            # _, _, _, fills = utils.misc.data_get_traj_from_masks(masks)
                            # if np.sum(fills) < S//4:
                            #     continue
                            all_masks.append(masks)
                            all_tids.append(tid)
                        print('found %d ok masks' % len(all_masks))
                        
                        # for ni in range(len(all_masks)):
                        for tid in all_tids:
                            for zoom in zooms:
                                self.all_video_names.append(video_name)
                                self.all_full_idx.append(full_idx)
                                self.all_tids.append(tid)
                                self.all_zooms.append(zoom)
                        sys.stdout.write('.')
                        sys.stdout.flush()
        print('\nfound {:d} samples in {}'.format(len(self.all_video_names), self.dataset_location))
        
    def __len__(self):
        return len(self.all_video_names)
        
    def getitem_helper(self, index):
        video_name = self.all_video_names[index]
        full_idx = self.all_full_idx[index]
        tid = self.all_tids[index]
        zoom = self.all_zooms[index]

        S = len(full_idx)

        # let's constrain ourselves to the things that are visible in the middle frame
        vis_seg = np.load(os.path.join(self.dataset_location, video_name, 'visible', '%06d.npy' % (full_idx[S//2])))
        
        # print('vis_seg0', vis_seg0.shape)
        # return None
              
        # # mask = vis_seg0 == tid
        # # ys, xs = np.where(mask)
        # # x0, x1 = np.min(xs), np.max(xs)
        # # y0, y1 = np.min(ys), np.max(ys)
        # # crop = mask[y0:y1, x0:x1]
        # # fill = np.mean(crop)
        # # print('mean', np.mean(mask), 'fill', fill)

        # tids = []
        # for subname in sorted(os.listdir(os.path.join(self.dataset_location, video_name))):
        #     # if subname not in ['images', 'visible']:
        #     if 'Ped' in subname: 
        #         tid = int(subname[:4])
        #         mask = vis_seg==tid
        #         # print('np.sum(mask)', np.sum(mask))
        #         if np.sum(mask) > 32:
        #             tids.append(tid)

        vis_segs = []
        for t in full_idx:
            vis_seg = np.load(os.path.join(self.dataset_location, video_name, 'visible', f'{t:06d}.npy'))
            vis_segs.append(vis_seg)
        vis_segs = np.stack(vis_segs)
        
        # S,H,W,C = rgbs.shape

        # all_masks = []
        # all_tids = []
        # for oid in tids:
        masks = [(seg == tid).astype(np.float32) for seg in vis_segs]
        masks = np.stack(masks, axis=0)

        mask_sums = np.sum(masks.reshape(S,-1), axis=1)
        # print('mask_sums', mask_sums)
        
        # # if it's present, be present at some decent size
        # if np.min(mask_sums[mask_sums>0]) < 128:
        #     continue
        
        
        rgbs = []
        for t in full_idx:
            rgb = cv2.imread(os.path.join(self.dataset_location, video_name, 'images', f'{t:06d}.bmp'))[..., ::-1]
            rgbs.append(rgb)
        rgbs = np.stack(rgbs, axis=0)

        full_masks = []
        for t in full_idx:
            paths = list((Path(self.dataset_location) / video_name).glob(f'{tid:04d}*/{t:06d}.png'))
            if len(paths) == 0:
                frame_mask = np.zeros((rgbs.shape[1], rgbs.shape[2]), dtype=np.uint8)
            else:
                frame_mask = cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE)
                if frame_mask is not None:
                    frame_mask = (frame_mask > 0).astype(np.uint8)
                else:
                    frame_mask = np.zeros((rgbs.shape[1], rgbs.shape[2]), dtype=np.uint8)
            full_masks.append(frame_mask)
        full_masks = np.stack(full_masks)

        masks = masks.astype(np.float32)
        full_masks = full_masks.astype(np.float32)
        
        mask_areas = (masks > 0).reshape(S,-1).sum(axis=1)
        mask_areas_norm = mask_areas / np.max(mask_areas)
        visibs = mask_areas_norm
        rgbs, masks, full_masks = utils.misc.data_pad_if_necessary(rgbs, masks, full_masks)
        S,H,W,C = rgbs.shape
        if zoom > 1:
            xys, _, valids, fills = utils.misc.data_get_traj_from_masks(masks)
            xys = utils.misc.data_replace_with_nearest(xys, valids)
            bboxes = np.stack([mask2bbox(mask) for mask in masks])
            whs = bboxes[:,2:4] - bboxes[:,0:2]
            whs = whs[visibs > 0.5]
            if np.mean(whs[:,0])*zoom >= W and np.mean(whs[:,1])*zoom >= H:
                return None
            xys, visibs, valids, rgbs, masks, full_masks = utils.misc.data_zoom(zoom, xys, visibs, valids, rgbs, masks=masks, masks2=full_masks)
        visibs = 1.0 * (visibs>0.5)
        safe = np.concatenate(([0], visibs[:-2] * visibs[1:-1] * visibs[2:], [0]))
        if np.sum(safe) < 2:
            return None

        masks = np.stack([masks, masks, full_masks], -1)

        # chans 1,2 valid
        S = len(full_idx)
        masks_valid = np.zeros((S,3), dtype=np.float32)
        masks_valid[:,1] = 1
        masks_valid[:,2] = 1
        
        sample = {
            'rgbs': rgbs,
            'masks': masks,
            'masks_valid': masks_valid,
            # 'visibs': visibs,
            # 'valids': valids,
        }
        return sample
        
        # vis_segs_pxl_cnt = []
        # for k in tids:
        #     vis_segs_pxl_cnt.append(np.sum(vis_segs == k))
        # vis_segs_pxl_cnt = np.stack(vis_segs_pxl_cnt, -1)
        
        # occ_segs = []
        # for i, t in enumerate(full_idx):
        #     obj_mask = full_segs[i, ..., tids.index(tid)] == 1
        #     obj_full_region = np.sum(obj_mask)
        #     occ_seg = np.zeros_like(obj_mask, dtype=np.float32)
        #     region_masks = vis_segs[i][obj_mask]
        #     for occ_id in np.unique(region_masks):
        #         if occ_id == 0 or occ_id == tid:
        #             continue
        #         overlap_area = np.sum(region_masks == occ_id)
        #         # if we have a large overlap and the candidate is visible.
        #         if overlap_area > 32:
        #             occ_seg[full_segs[i, ..., tids.index(occ_id)] == 1] = 1.
        #         elif overlap_area > 1:
        #             occ_seg[full_segs[i, ..., tids.index(occ_id)] == 1] = np.maximum(0.5, occ_seg[full_segs[i, ..., tids.index(occ_id)] == 1])
        #     occ_segs.append(occ_seg)
        # occ_segs = np.stack(occ_segs)
        
        # # masks = np.stack([(vis_segs == tid).astype(np.float32),
        # #                   full_segs[..., tids.index(tid)].astype(np.float32),
        # #                   occ_segs.astype(np.float32)], -1)
        # masks = np.stack([(vis_segs == tid).astype(np.float32),
        #                   (vis_segs == tid).astype(np.float32),
        #                   full_segs[..., tids.index(tid)].astype(np.float32)], -1)
        # }
