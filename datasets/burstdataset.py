from typing import Optional, List, Tuple, Union, Dict, Any
import json
import numpy as np
import pycocotools.mask as cocomask
import time
from numpy import random
from numpy.core.numeric import full
import torch
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
from datasets.dataset import MaskDataset, mask2bbox
from icecream import ic
from pathlib import Path


def intify_track_ids(video_dict: Dict[str, Any]):
    video_dict["track_category_ids"] = {
        int(track_id): category_id for track_id, category_id in video_dict["track_category_ids"].items()
    }

    for t in range(len(video_dict["segmentations"])):
        video_dict["segmentations"][t] = {
            int(track_id): seg
            for track_id, seg in video_dict["segmentations"][t].items()
        }

    return video_dict


def rle_ann_to_mask(rle: str, image_size: Tuple[int, int]) -> np.ndarray:
    return cocomask.decode({
        "size": image_size,
        "counts": rle.encode("utf-8")
    }).astype(bool)


def mask_to_rle_ann(mask: np.ndarray) -> Dict[str, Any]:
    assert mask.ndim == 2, f"Mask must be a 2-D array, but got array of shape {mask.shape}"
    rle = cocomask.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


class BurstDataset(MaskDataset):
    def __init__(self,
                 dataset_location='../burst',
                 S=32,
                 crop_size=(384, 512), 
                 strides=[1, 2, 3, 4],
                 zooms=[1,1.5,2],
                 fullseq=False,
                 chunk=None,
                 use_augs=False,
                 is_training=True):

        print('loading burst dataset...')
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            fullseq=fullseq,
            crop_size=crop_size, 
            use_augs=use_augs,
            is_training=is_training
        )

        self.split = 'train' if is_training else 'val'
        self.annotations = json.load(open(Path(dataset_location) / ('train/train.json' if is_training else 'val/all_classes.json'), 'r'))
        self.videos = [intify_track_ids(video) for video in self.annotations["sequences"]]
        print(f'found {len(self.videos)} videos in {self.dataset_location}')
        
        if not is_training:
            strides = [1]
            zooms = [1]
            clip_step = S
        else:
            clip_step = S // 2
            
        self.all_info = []
        for video in self.videos:
            self.process_video(video, strides, zooms, clip_step)
            sys.stdout.write(".")
            sys.stdout.flush()

        print(f'found {len(self.all_info)} samples in {self.dataset_location}')
        
        if chunk is not None:
            assert(len(self.all_info) > 100)
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.all_info = chunkify(self.all_info,100)[chunk]
            print('filtered to %d' % len(self.all_info))

    def process_video(self, video, strides, zooms, clip_step):
        segmentations = video["segmentations"]
        if len(segmentations) < 4:
            return

        track_ids = self.get_track_ids(segmentations)
        
        for stride in strides:
            self.extract_clips(video, stride, zooms, track_ids, clip_step)

    def get_track_ids(self, segmentations):
        track_ids = []
        for seg in segmentations:
            track_ids.append(np.array(list(seg.keys())))
        return track_ids

    def extract_clips(self, video, stride, zooms, track_ids, clip_step):
        S_local = len(video['segmentations'])
        for start_idx in range(0, S_local, clip_step * stride):
            full_idx = start_idx + np.arange(self.S) * stride
            full_idx = full_idx[full_idx < S_local]
            
            if len(full_idx) < (((S_local - 1) // stride + 1 if self.S == -1 else self.S) if self.fullseq else 4):
                continue
            
            valid_ids = np.unique(np.concatenate([track_ids[i] for i in full_idx]))
            
            for tid in valid_ids:
                for zoom in zooms:
                    self.all_info.append((video, tid, stride, full_idx, zoom))

    def __len__(self):
        return len(self.all_info)

    def getitem_helper(self, index):
        video, track_id, stride, full_idx, zoom = self.all_info[index]

        h, w = video["height"], video["width"]
        segmentations = video["segmentations"]
        base_dir = Path(self.dataset_location).parent / 'tao' / 'frames' / self.split / video["dataset"] / video["seq_name"]
        image_paths = [base_dir / img_path for img_path in video["annotated_image_paths"]]
        
        segs = []
        for i in range(len(segmentations)):
            if track_id in segmentations[i]:
                segs.append(segmentations[i][track_id]['rle'])
            else:
                segs.append(None)
        
        image_paths = [image_paths[i] for i in full_idx]
        segs = [segs[i] for i in full_idx]

        rgbs = []
        for path in image_paths:
            rgb = cv2.imread(str(path))[..., ::-1].copy()
            rgbs.append(rgb)

        masks = [rle_ann_to_mask(seg, (h, w)) if seg is not None else np.zeros((h, w), dtype=bool) for seg in segs]
        
        rgbs = np.stack(rgbs)
        masks = np.stack(masks).astype(np.float32)
        S,H,W,C = rgbs.shape
        
        # padding and zooming
        mask_areas = (masks > 0).reshape(masks.shape[0],-1).sum(axis=1)
        mask_areas_norm = mask_areas / np.max(mask_areas)
        visibs = mask_areas_norm

        if self.is_training:
            rgbs, masks = utils.misc.data_pad_if_necessary(rgbs, masks)
        if zoom > 1:
            xys, _, valids, fills = utils.misc.data_get_traj_from_masks(masks)
            xys = utils.misc.data_replace_with_nearest(xys, valids)
        
            bboxes = np.stack([mask2bbox(mask) for mask in masks])
            whs = bboxes[:,2:4] - bboxes[:,0:2]
            whs = whs[visibs > 0.5]
            if np.mean(whs[:,0])*zoom >= W and np.mean(whs[:,1])*zoom >= H:
                print('unwilling to zoom')
                return None
            xys, visibs, valids, rgbs, masks = utils.misc.data_zoom(zoom, xys, visibs, valids, rgbs, masks=masks)
        visibs = 1.0 * (visibs>0.5)
        safe = np.concatenate(([0], visibs[:-2] * visibs[1:-1] * visibs[2:], [0]))
        if np.sum(safe) < 2:
            print('safe', safe)
            return None

        mask_cnts = np.stack([mask.sum() for mask in masks])
        if mask_cnts.max() <= 64:
            print('burst: max_cnts', mask_cnts.max())
            return None

        bboxes = [mask2bbox(mask) for mask in masks]
        bboxes = np.stack(bboxes, axis=0)

        # for i in range(1, len(bboxes)):
        #     xy_prev = (bboxes[i - 1, :2] + bboxes[i - 1, 2:]) / 2
        #     xy = (bboxes[i, :2] + bboxes[i, 2:]) / 2
        #     dist = np.linalg.norm(xy - xy_prev)
        #     if np.sum(masks[i]) > 0 and np.sum(masks[i - 1]) > 0:
        #         if dist > 64:
        #             print('large motion detected in {}'.format(image_paths[i]))
        #             return None
        # print('mask sums', masks.reshape(S,-1).sum(axis=1))

        sample = {
            'rgbs': rgbs,
            'masks': masks,
        }

        return sample



if __name__ == '__main__':
    print(rle_ann_to_mask('', (480, 640)).dtype)
