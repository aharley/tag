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


class BEE23Dataset(BBoxDataset):
    def __init__(self,
                 dataset_location='../BEE23',
                 S=32, fullseq=False, chunk=None,
                 rand_frames=False,
                 crop_size=(384,512), 
                 strides=[1,2,3,4],
                 zooms=[1,2],
                 use_augs=False,
                 is_training=True):
        print('loading BEE23 dataset...')
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            rand_frames=rand_frames,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training)

        clip_step = S//2
        if not is_training:
            strides = [1]
            clip_step = S
        
        # validation gt not provided
        self.root = Path(dataset_location) / 'bee/{}'.format('train' if is_training else 'test')
        gt_fns = sorted(list(self.root.glob('BEE*/gt/gt.txt')))
        print('found {:d} videos in {}'.format(len(gt_fns), self.dataset_location))

        self.data = []
        for gtf in gt_fns:
            gt = np.loadtxt(gtf, delimiter=',').astype(int)
            obj_ids = np.unique(gt[:, 1])
            for obj_id in obj_ids:
                scene_id = gtf.parts[-3]
                obj_id = int(obj_id)
                gt_i = gt[gt[:, 1] == obj_id]
                frame_ids = gt_i[:, 0]

                bboxes = []
                visibs = []
                for fid in range(np.min(frame_ids), np.max(frame_ids) + 1):
                    if fid in frame_ids:
                        bboxes.append(gt_i[frame_ids == fid, 2:6])
                        visibs.append(np.ones((bboxes[-1].shape[0],)))
                    else:
                        bboxes.append(np.array([[0, 0, 0, 0]]))
                        visibs.append(np.array([0]))
                bboxes = np.concatenate(bboxes)
                visibs = np.concatenate(visibs)
                S_local = len(bboxes)
                
                for stride in strides:
                    sidx = 0
                    eidx = np.max(frame_ids) - np.min(frame_ids) + 1
                    for ii in range(sidx, max(eidx - self.S * stride + 1, sidx + 1), clip_step*stride):
                        start_ind = ii - sidx

                        full_idx = start_ind + np.arange(self.S)*stride
                        full_idx = [ij for ij in full_idx if ij < S_local]
                        
                        if len(full_idx) < 8:
                            continue
                        
                        bboxes_here = bboxes[full_idx]
                        visibs_here = visibs[full_idx]

                        if visibs_here[0]==0 or visibs_here[1]==0:
                            continue

                        mots = []
                        for i in range(1, len(bboxes_here)):
                            if visibs_here[i] > 0 and visibs_here[i-1] > 0:
                                xy_prev = (bboxes_here[i-1, :2] + bboxes_here[i-1, 2:]) / 2
                                xy = (bboxes_here[i, :2] + bboxes_here[i, 2:]) / 2
                                dist = np.linalg.norm(xy - xy_prev)
                                mots.append(dist)
                        mots = np.stack(mots)

                        if np.mean(mots) < 1.0:
                            # print('no motion detected in {}'.format(img_paths[i]))
                            # return None
                            continue

                        if np.max(mots) > 32:
                            # motion discontinuity
                            continue

                        for zoom in zooms:
                            self.data.append({
                                'obj_id': obj_id,
                                'scene_id': scene_id,
                                'stride': stride,
                                'gt_fn': gtf,
                                'start_ind': start_ind,
                                'zoom': zoom,
                            })
        print('found {:d} samples in {}'.format(len(self.data), self.dataset_location))

        if chunk is not None:
            assert(len(self.data) > 100)
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.data = chunkify(self.data,100)[chunk]
            print('filtered to %d' % len(self.data))

        
    def getitem_helper(self, index):
        data = self.data[index]
        scene_id, obj_id, stride, gtf, start_ind, zoom = data['scene_id'], data['obj_id'], data['stride'], data['gt_fn'], data['start_ind'], data['zoom']
        
        gt = np.loadtxt(gtf, delimiter=',').astype(int)
        gt_i = gt[gt[:, 1] == obj_id]
        frame_ids = gt_i[:, 0]
        
        img_paths = []
        bboxes = []
        visibs = []
        for fid in range(np.min(frame_ids), np.max(frame_ids) + 1):
            # BEE23 has two different folder structures
            img_path = self.root / '{}/images/{:06d}.jpg'.format(scene_id, fid)
            if not img_path.exists():
                img_path = self.root / '{}/img1/{:06d}.jpg'.format(scene_id, fid)
            img_paths.append(img_path)
            # print(img_paths[-1])
            if fid in frame_ids:
                bboxes.append(gt_i[frame_ids == fid, 2:6])
                visibs.append(np.ones((bboxes[-1].shape[0],)))
            else:
                bboxes.append(np.array([[0, 0, 0, 0]]))
                visibs.append(np.array([0]))
        bboxes = np.concatenate(bboxes)
        visibs = np.concatenate(visibs)
        
        def _pick_frames(ind):
            nonlocal bboxes, visibs, img_paths
            bboxes = bboxes[ind]
            visibs = visibs[ind]
            img_paths = [img_paths[ii] for ii in ind]
        
        S = self.S * stride
        _pick_frames(np.arange(len(img_paths))[start_ind:start_ind+S:stride])
        
        # if len(img_paths) < 8:
        #     print('len(img_paths)', len(img_paths))
        #     return None

        # if visibs[0]==0 or visibs[1]==0:
        #     print('visibs', visibs)
        #     return None

        # mots = []
        # for i in range(1, len(img_paths)):
        #     xy_prev = (bboxes[i-1, :2] + bboxes[i-1, 2:]) / 2
        #     xy = (bboxes[i, :2] + bboxes[i, 2:]) / 2
        #     dist = np.linalg.norm(xy - xy_prev)
        #     if visibs[i] > 0 and visibs[i-1] > 0:
        #         if dist > 32:
        #             print('large motion detected in {}'.format(img_paths[i]))
        #             return None
        #         else:
        #             mots.append(dist)
                    
        # # print('np.mean(mots)', np.mean(mots))
        # if np.mean(mots) < 4.0:
        #     # print('no motion detected in {}'.format(img_paths[i]))
        #     return None

        rgbs = [cv2.imread(str(path))[..., ::-1].copy() for path in img_paths]
        rgbs = np.stack(rgbs) # S, C, H, W
        # from xywh to xyxy
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
        return len(self.data)
