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
from datasets.dataset import PointDataset, BBoxDataset, bbox2mask
from datasets.dataset_utils import make_split
from pathlib import Path


def read_annotation_file(annotation_path):
    annotations = []
    with open(annotation_path, "r") as file:
        for line in file:
            data = line.strip().split(" ")
            frame_num = int(data[0])
            obj_id = int(data[1])
            if obj_id < 0:
                continue
            truncated = int(data[3])
            occluded = int(data[4])
            bbox = [float(x) for x in data[6:10]]  # Format: [left, top, right, bottom]
            annotations.append((frame_num, obj_id, truncated, occluded, bbox))
    return annotations


class KittiDataset(BBoxDataset):
    def __init__(
        self,
        dataset_location="../kitti",
        S=32, fullseq=False, chunk=None,
        crop_size=(384, 512), 
        strides=[1,2,3,4],
        zooms=[1,1.5],
        use_augs=False,
        is_training=True,
    ):
        print("loading kitti dataset...")
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training,
            fullseq=fullseq,
        )
        
        if not is_training:
            strides = [1]
            zooms = [1]
            clip_step = S
        else:
            clip_step = S // 2

        self.root = Path(dataset_location) / 'training'
        self.seq_names = sorted([seq.parts[-1] for seq in self.root.glob("image_02/*")])
        # self.seq_names = make_split(self.seq_names, is_training, shuffle=True)
        print("found {:d} sequences in {}".format(len(self.seq_names), self.dataset_location))

        self.all_info = []
        for seq_name in self.seq_names[:]:
            gt = self.read_annotation_file(self.root / 'label_02' / f'{seq_name}.txt')
            self.process_video(gt, strides, zooms, clip_step, seq_name)

            sys.stdout.write(".")
            sys.stdout.flush()

        if chunk is not None:
            assert(len(self.all_info) > 100)
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.all_info = chunkify(self.all_info, 100)[chunk]
            print('filtered to %d' % len(self.all_info))

        print('found {:d} samples in {}'.format(len(self.all_info), self.dataset_location))
    
    def read_annotation_file(self, file_path):
        data = np.loadtxt(file_path, delimiter=" ", dtype={
            'names': ('frame', 'track_id', 'type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom', '3D_height', '3D_width', '3D_length', '3D_x', '3D_y', '3D_z', 'rotation_y'),
            'formats': ('i4', 'i4', 'U10', 'f4', 'i4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4')
        })
        return data
    
    def get_bbox_visibs(self, gt, full_idx, tid):
        bboxes = []
        visibs = []
        for i in full_idx:
            # Extract annotations for the current frame
            gt_here = gt[gt['frame'] == i]
            
            if tid in gt_here['track_id']:
                
                gt_here_tid = gt_here[gt_here['track_id'] == tid]
                
                bbox = np.array([gt_here_tid['bbox_left'][0], gt_here_tid['bbox_top'][0], gt_here_tid['bbox_right'][0], gt_here_tid['bbox_bottom'][0]], dtype=np.float32)
                # visib = 0.9*gt_here_tid['truncated'][0] * (1.0 - gt_here_tid['occluded'][0])

                trunc = gt_here_tid['truncated'][0]
                occ = gt_here_tid['occluded'][0]
                if occ >= 0:
                    # visib = max(trunc - occ/3.0, 0)
                    # visib = max(max(trunc+0.5,1.0) - occ/3.0, 0)
                    visib = 1.0 - min(max(occ-1,0),3)/3.0
                else:
                    visib = 0.0
                # print('gt_here_tid[occluded]', gt_here_tid['occluded'])
            else:
                bbox = np.array([0, 0, 0, 0], dtype=np.float32)
                visib = 0.0
            bboxes.append(bbox)
            visibs.append(visib)
        
        # Concatenate all bounding boxes and visibility arrays
        bboxes = np.stack(bboxes)
        visibs = np.stack(visibs)
        
        return bboxes, visibs
    
    def get_track_ids(self, gt):
        track_ids = []
        for i in range(0, max(gt['frame']) + 1):
            gt_here = gt[gt['frame'] == i]
            track_ids.append(np.array(list(set(gt_here['track_id']))))
        return track_ids

    def process_video(self, gt, strides, zooms, clip_step, seq_name):
        # kitti has 0-indexed frames
        if max(gt['frame']) + 1 < 4:
            return
        
        track_ids = self.get_track_ids(gt)
        
        for stride in strides:
            self.extract_clips(gt, stride, zooms, track_ids, clip_step, seq_name)
    
    def extract_clips(self, gt, stride, zooms, track_ids, clip_step, seq_name):
        S_local = max(gt['frame']) + 1
        print('S_local', S_local)
        for start_idx in range(0, S_local, clip_step * stride):
            full_idx = start_idx + np.arange(self.S) * stride
            full_idx = full_idx[full_idx < S_local]
            
            if len(full_idx) < (((S_local - 1) // stride + 1 if self.S == -1 else self.S) if self.fullseq else 4):
                continue
            
            valid_ids = np.unique(np.concatenate([track_ids[i] for i in full_idx]))
            for tid in valid_ids:

                gt_here = gt[gt['frame'] == start_idx]
                gt_here_tid = gt_here[gt_here['track_id'] == tid]
                # print('gt_here_tid', gt_here_tid)
                label = gt_here_tid['type']
                if len(label)==0: continue
                if label[0]=='DontCare': continue # junk
                if label[0]=='Pedestrian': continue # seems like shifty labels
                # print('label', label)
                
                bboxes_here, visibs = self.get_bbox_visibs(gt, full_idx, tid)
                safe = np.concatenate(([0], visibs[:-2] * visibs[1:-1] * visibs[2:], [0]))
                if np.sum(safe) < 2: continue
                
                dists = np.linalg.norm(bboxes_here[1:, :2] - bboxes_here[:-1, :2], axis=1)
                if np.mean(dists) < 2.: continue
            
                for zoom in zooms:
                    self.all_info.append((gt, tid, stride, full_idx, zoom, seq_name))
                
    def getitem_helper(self, index):
        gt, track_id, stride, full_idx, zoom, seq_name = self.all_info[index]
        bboxes, visibs = self.get_bbox_visibs(gt, full_idx, track_id)
        image_paths = [self.root / f"image_02/{seq_name}/{frame_idx:06d}.png" for frame_idx in full_idx]
        
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
        if np.sum(safe) < 2: return None
        
        S,H,W,C = rgbs.shape

        # the visibs are a little unreliable
        # so we'll compute masks and then adjust
        masks = np.stack([bbox2mask(bbox, W, H) for bbox in bboxes])
        # print('masks', masks.sahpe)

        # print('bboxes', bboxes)

        # S = sample['masks'].shape[0]
        # mask_areas = (masks > 0).reshape(S,-1).sum(axis=1) # S
        # mask_areas_norm = mask_areas / np.max(mask_areas) # S
        # visibs *= np.clip(mask_areas_norm, 0.5, 1)

        # in kitti,
        # the real issue is horizontal truncation
        # so we will eliminate the H dim in the area calc
        mask_areas = (masks.reshape(S,H,W) > 0).sum(axis=1) # S,H
        mask_areas = (mask_areas > 0).sum(axis=1) # S
        mask_areas_norm = mask_areas / np.max(mask_areas)
        # visibs *= np.clip(mask_areas_norm, 0.5, 1)
        visibs = mask_areas_norm

        sample = {
            'rgbs': rgbs,
            'visibs': visibs,
            'bboxes': bboxes,
        }
        return sample

    def __len__(self):
        return len(self.all_info)
