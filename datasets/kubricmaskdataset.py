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
from datasets.dataset import MaskDataset, mask2bbox
from icecream import ic
import matplotlib
from pathlib import Path
import matplotlib.pyplot as plt

from datasets.dataset_utils import make_split

# np.random.seed(125)
# torch.multiprocessing.set_sharing_strategy('file_system')

def segm_rgb_to_ids_kubric(segm_rgb):  # , num_inst=None):
    '''
    :param segm_rgb (*, 3) array of RGB values.
    :return segm_ids (*, 1) array of 1-based instance IDs (0 = background).
    '''
    # We assume that hues are distributed across the range [0, 1] for instances in the image, ranked
    # by their integer ID. Check kubric plotting.hls_palette() for more details.
    hsv = matplotlib.colors.rgb_to_hsv(segm_rgb)
    to_rank = hsv[..., 0]  # + hsv[..., 2] * 1e-5
    unique_hues = np.sort(np.unique(to_rank))
    hue_start = 0.01
    assert np.isclose(unique_hues[0], 0.0, rtol=1e-3, atol=1e-3), str(unique_hues)

    # commented this because object ID 0 may not be always visible in every frame:
    # assert np.isclose(unique_hues[1], hue_start, rtol=1e-3, atol=1e-3), str(unique_hues)

    # The smallest jump inbetween subsequent hues determines the highest instance ID that is VALO,
    # which is <= the total number of instances. Skip the very first hue, which is always 0 and
    # corresponds to background.
    hue_steps = np.array([unique_hues[i] - unique_hues[i - 1] for i in range(2, len(unique_hues))])

    # For this sanity check to work, we must never have more than ~95 instances per scene.
    assert np.all(hue_steps >= 1e-2), str(hue_steps)

    # IMPORTANT NOTE: The current VALO set may be a strict SUBSET of the original VALO set (recorded
    # in the metadata), because we already applied frame subsampling here! In practice, this
    # sometimes causes big (i.e. integer multiple) jumps in hue_steps.
    # NEW: Ignore outliers the smart way.
    adjacent_steps = hue_steps[hue_steps <= np.min(hue_steps) * 1.5]
    hue_step = np.mean(adjacent_steps)

    # The jump from background to first instance is a special case, so ensure even distribution.
    nice_rank = to_rank.copy()
    nice_rank[nice_rank >= hue_start] += hue_step - hue_start
    ids_approx = (nice_rank / hue_step)

    segm_ids = np.round(ids_approx)[..., None].astype(np.int32)  # (T, H, W, 1).
    return segm_ids


class KubricRandomDataset(MaskDataset):
    def __init__(self,
                 dataset_location='../kubric',
                 S=32, fullseq=False, chunk=None, 
                 crop_size=(384,512), 
                 strides=[1,2,3,4],
                 zooms=[1,1.5,2],
                 use_augs=False,
                 is_training=True):

        print('loading kubric_random dataset...')
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size, 
            use_augs=use_augs,
            is_training=is_training
        )

        clip_step = S//2
        
        self.dataset_location = os.path.join(dataset_location, 'train' if is_training else 'val')
        self.video_names = sorted(os.listdir(self.dataset_location))
        print('found {:d} videos in {}'.format(len(self.video_names), self.dataset_location))

        if chunk is not None:
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.video_names = chunkify(self.video_names,100)[chunk]
            print('filtered to %d video_names' % len(self.video_names))
            print('self.video_names', self.video_names)

        self.all_info = []

        # self.all_video_names = []
        # self.all_tids = []
        # self.all_full_idx = []

        for video_name in self.video_names[:]:
            metadata = json.load(next(Path(os.path.join(self.dataset_location, video_name)).glob('*.json')).open())
            
            K = metadata['scene']['num_valo_instances']
            S_local = metadata['scene']['num_frames']
            print('S_local', S_local)
            for stride in strides:
                for ii in range(0, S_local, clip_step*stride):
                    full_idx = ii + np.arange(self.S)*stride
                    full_idx = [ij for ij in full_idx if ij < S_local]

                    if len(full_idx) < (((S_local - 1) // stride + 1 if self.S == -1 else self.S) if fullseq else 8):
                        continue

                    for tid in range(K):
                        for zoom in zooms:
                            self.all_info.append([video_name, full_idx, tid, zoom])
                    sys.stdout.write('.')
                    sys.stdout.flush()

        print('found %d samples in %s' % (len(self.all_info), dataset_location))
        # if chunk is not None:
        #     def chunkify(lst,n):
        #         return [lst[i::n] for i in range(n)]
        #     self.all_info = chunkify(self.all_info,100)[chunk]
        #     print('filtered to %d all_info' % len(self.all_info))
        #     print('self.all_info[:3]', self.all_info[:3])
        
    def __len__(self):
        return len(self.all_info)
        
    def getitem_helper(self, index):
        # print('fetching index', index)
        video_name, full_idx, tid, zoom = self.all_info[index]

        video_dir = os.path.join(self.dataset_location, video_name, 'frames')
        if not os.path.exists(video_dir):
            video_dir = os.path.join(self.dataset_location, video_name, 'frames_p0_v0')
        metadata = json.load(next(Path(os.path.join(self.dataset_location, video_name)).glob('*.json')).open())
        K = metadata['scene']['num_valo_instances']

        vis_segs = []
        for t in full_idx:
            vis_segs.append(plt.imread(os.path.join(video_dir, f'segmentation_{t:05d}.png'))[..., :3])
        vis_segs = np.stack(vis_segs)
        vis_segs = segm_rgb_to_ids_kubric(vis_segs)[..., 0]  # vis segs are 1-indexed

        # print('full_idx', full_idx)
        # print('K', K, 'tid', tid)
        
        if np.sum(vis_segs == tid + 1) < 64:
            print('np.sum(masks)', np.sum(vis_segs == tid + 1))
            return None
        
        rgbs = []
        for t in full_idx:
            rgbs.append(plt.imread(os.path.join(video_dir, f'rgba_{t:05d}.png'))[..., :3])
        rgbs = (np.stack(rgbs) * 255).astype(np.uint8) # S,360,480

        # print('rgbs', rgbs.shape)
        
        
        full_segs = []
        for t in full_idx:
            frame_segs = []
            for k in range(K):
                seg = (plt.imread(os.path.join(video_dir, f'divided_segmentation_{k:03d}_{t:05d}.png'))[..., :3].sum(-1) > 0.1).astype(np.uint8)
                frame_segs.append(seg)
            full_segs.append(np.stack(frame_segs, -1))
        full_segs = np.stack(full_segs)
        
        
        vis_segs_pxl_cnt = []
        for k in range(K):
            vis_segs_pxl_cnt.append(np.sum(vis_segs == k + 1))
        vis_segs_pxl_cnt = np.stack(vis_segs_pxl_cnt, -1)
        
        occ_segs = []
        for i, t in enumerate(full_idx):
            obj_mask = full_segs[i, ..., tid] == 1
            obj_full_region = np.sum(obj_mask)
            occ_seg = np.zeros_like(obj_mask, dtype=np.float32)
            region_masks = vis_segs[i][obj_mask]
            for occ_id in np.unique(region_masks):
                if occ_id == 0 or occ_id == tid + 1:
                    continue
                overlap_area = np.sum(region_masks == occ_id)
                # if we have a large overlap and the candidate is visible.
                # if overlap_area / obj_full_region > 0.05:
                # elif overlap_area / obj_full_region > 0.01:
                if overlap_area > 32:
                    occ_seg[full_segs[i, ..., occ_id - 1] == 1] = 1.
                elif overlap_area > 1:
                    occ_seg[full_segs[i, ..., occ_id - 1] == 1] = np.maximum(0.5, occ_seg[full_segs[i, ..., occ_id - 1] == 1])
            occ_segs.append(occ_seg)
        occ_segs = np.stack(occ_segs)


        masks0 = (vis_segs == tid + 1).astype(np.float32)
        masks1 = full_segs[..., tid].astype(np.float32)

        sim0 = np.mean(masks0[0:1]==masks0[1:])
        sim1 = np.mean(masks0==masks1)
        if sim0 > 0.99 and sim1 > 0.9999:
            print('sim0', sim0, 'sim1', sim1)
            return None
        
        # # for this data,
        # # maybe it will be better to inflate via some random over-sampling subseq,
        # # rather than zigzag
        # if S < self.S:
        #     ara = np.arange(S, self.S)
        #     ara = (np.arange(self.S)/(self.S-1)*(S-1)).round().astype(np.int32)
        #     rgbs = rgbs[ara]
        #     masks = masks[ara]
        #     masks_valid = masks_valid[ara]
        
        # if zoom > 1:
        #     xys, visibs, valids, rgbs, masks = utils.misc.data_zoom(zoom, xys, visibs, valids, rgbs, masks)
        #     _, H, W, _ = rgbs.shape

        S,H,W,C = rgbs.shape
        assert(C==3)

        if np.sum(masks0) < 32:
            print('np.sum(masks0)', np.sum(masks0))
            return None

        mask_areas = (masks0 > 0).reshape(S,-1).sum(axis=1)
        mask_areas_norm = mask_areas / np.max(mask_areas)
        visibs = mask_areas_norm
        rgbs, masks0, masks1 = utils.misc.data_pad_if_necessary(rgbs, masks0, masks1)
        S,H,W,C = rgbs.shape
        if zoom > 1:
            xys, _, valids, fills = utils.misc.data_get_traj_from_masks(masks0)
            xys = utils.misc.data_replace_with_nearest(xys, valids)
            bboxes = np.stack([mask2bbox(mask) for mask in masks0])
            whs = bboxes[:,2:4] - bboxes[:,0:2]
            whs = whs[visibs > 0.5]
            # print('whs', whs, 'mean whs', np.mean(whs, axis=0), 'W', W, 'H', H)
            if np.mean(whs[:,0])*zoom >= W and np.mean(whs[:,1])*zoom >= H:
                return None
            xys, visibs, valids, rgbs, masks0, masks1 = utils.misc.data_zoom(zoom, xys, visibs, valids, rgbs, masks=masks0, masks2=masks1)
        visibs = 1.0 * (visibs>0.5)
        safe = np.concatenate(([0], visibs[:-2] * visibs[1:-1] * visibs[2:], [0]))
        if np.sum(safe) < 2: return None

        masks = np.stack([masks0, masks0, masks1], -1)
        # chans 1,2 valid
        S = len(full_idx)
        masks_valid = np.zeros((S,3), dtype=np.float32)
        masks_valid[:,1] = 1
        masks_valid[:,2] = 1
        
        sample = {
            'rgbs': rgbs,
            'masks': masks,
            'masks_valid': masks_valid,
        }
        
        return sample


class KubricContainersDataset(KubricRandomDataset):
    def __init__(self,
                 dataset_location='../kubric',
                 S=32, fullseq=False, chunk=None,
                 crop_size=(384,512), 
                 strides=[1,2,3,4],
                 zooms=[1,1.25,1.5,2,3],
                 use_augs=False,
                 is_training=True):

        print('loading kubric_containers dataset...')
        MaskDataset.__init__(
            self,
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size, 
            use_augs=use_augs,
            is_training=is_training
        )
        self.dataset_location = os.path.join(dataset_location, 'kubbench_v3')
        self.video_names = sorted(os.listdir(self.dataset_location))
        # self.video_names = make_split(self.video_names, is_training=is_training)
        print('found {:d} videos in {}'.format(len(self.video_names), self.dataset_location))

        clip_step = S//4 # very unique data

        # if chunk is not None:
        #     def chunkify(lst,n):
        #         return [lst[i::n] for i in range(n)]
        #     self.video_names = chunkify(self.video_names,100)[chunk]
        #     print('filtered to %d video_names' % len(self.video_names))
        #     print('self.video_names', self.video_names)

        self.all_info = []

        for video_name in self.video_names[:]:
            metadata = json.load(next(Path(os.path.join(self.dataset_location, video_name)).glob('*.json')).open())
            
            K = metadata['scene']['num_valo_instances']
            S_local = metadata['scene']['num_frames']
            print('S_local', S_local)
            for stride in strides:
                for ii in range(0, S_local, clip_step*stride):
                    full_idx = ii + np.arange(self.S)*stride
                    full_idx = [ij for ij in full_idx if ij < S_local]

                    if len(full_idx) < (((S_local - 1) // stride + 1 if self.S == -1 else self.S) if fullseq else 8):
                        continue

                    for tid in range(K):
                        for zoom in zooms:
                            self.all_info.append([video_name, full_idx, tid, zoom])
                    sys.stdout.write('.')
                    sys.stdout.flush()
        print('found', len(self.all_info), 'samples')

        if chunk is not None:
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.all_info = chunkify(self.all_info,100)[chunk]
            print('filtered to %d all_info' % len(self.all_info))
            print('self.all_info[:3]', self.all_info[:3])
        
