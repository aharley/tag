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

# np.random.seed(125)
# torch.multiprocessing.set_sharing_strategy('file_system')


class Bl30kDataset(MaskDataset):
    def __init__(
            self,
            dataset_location='../bl30k/BL30K',
            S=32, fullseq=False, chunk=None,
            strides=[1,2],
            zooms=[1],
            rand_frames=False,
            crop_size=(384,512),
            use_augs=False,
            is_training=True,
    ):

        print('loading bl30k dataset...')
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            rand_frames=rand_frames,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training
        )

        clip_step = S*2 # highly redundant data
        if not is_training:
            strides = [2]
            zooms = [1]

        folder_names = []
        folder_names = sorted([folder.split('/')[-1] for folder in glob.glob('%s/JPEGImages/*' % (self.dataset_location))])
        folder_names = [fn for fn in folder_names if (fn[-1] != '9' if is_training else fn[-1] == '9')]
        print('found {:d} {} folders in {}'.format(len(folder_names), ('train' if is_training else 'test'), dataset_location))

        if chunk is not None:
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            folder_names = chunkify(folder_names,100)[chunk]
            print('filtered to %d folder_names' % len(folder_names))
            print('folder_names', folder_names)

        seg_folder = '%s/Annotations/%s' % (self.dataset_location, folder_names[0])
        seg_filenames = glob.glob('%s/*.png' % seg_folder)
        S_local = len(seg_filenames)
        
        self.all_folder_names = []
        self.all_idxs = []
        self.all_obj_ids = []
        self.all_zooms = []
        
        self.N_per = 3
        for fi, folder in enumerate(folder_names):
            print('S_local', S_local)
            for stride in strides:
                for ii in range(0, S_local, clip_step*stride):
                    full_idx = ii + np.arange(self.S)*stride
                    full_idx = [ij for ij in full_idx if ij < S_local]
                    if len(full_idx) == self.S:
                        for ni in range(self.N_per):
                            for zoom in zooms:
                                self.all_folder_names.append(folder)
                                self.all_idxs.append(full_idx)
                                self.all_obj_ids.append(ni)
                                self.all_zooms.append(zoom)

        print('found {:d} {} samples in {}'.format(len(self.all_folder_names), ('train' if is_training else 'test'), self.dataset_location))
            

    def getitem_helper(self, index):
        folder = self.all_folder_names[index]
        full_idx = self.all_idxs[index]
        obj_id = self.all_obj_ids[index]
        zoom = self.all_zooms[index]
        # print('full_idx', full_idx)
        
        rgb_folder = '%s/JPEGImages/%s' % (self.dataset_location, folder)
        seg_folder = '%s/Annotations/%s' % (self.dataset_location, folder)

        rgb_filenames = glob.glob('%s/*.jpg' % rgb_folder)
        seg_filenames = glob.glob('%s/*.png' % seg_folder)

        rgb_filenames = sorted(rgb_filenames)
        seg_filenames = sorted(seg_filenames)

        S_local = len(rgb_filenames)

        rgb_filenames = [rgb_filenames[idx] for idx in full_idx]
        seg_filenames = [seg_filenames[idx] for idx in full_idx]

        S = len(rgb_filenames)

        rgbs = []
        for fn in rgb_filenames:
            im = cv2.imread(fn)[..., ::-1]
            im = np.array(im)
            rgbs.append(np.array(im)[:, :, :3])
                
        segs = []
        for fn in seg_filenames:
            im = cv2.imread(fn)[..., ::-1]
            im = np.array(im)
            if im.ndim==2:
                print('blk: seg ndim is 2 instead of 3')
                return None
            segs.append(np.array(im)[:, :, :3])

        segA = np.stack(segs, axis=0).reshape(-1, 3)
        valid = segA.sum(-1) > 0 # don't count bkg
        obj_ids = np.unique(segA[valid], axis=0) # NSeg x 3
        # print('obj_ids', obj_ids)

        if obj_id >= len(obj_ids):
            print('obj_id unavailable')
            return None

        all_masks = []
        for oid in obj_ids:
            obj_color = obj_ids[obj_id]
            masks = [(np.sum((seg == obj_color).astype(int), -1) == 3).astype(np.float32) for seg in segs]
            masks = np.stack(masks, axis=0)

            all_masks.append(masks)

        # we prefer masks that are moving
        all_diffs = []
        for oid in range(len(all_masks)):
            masks = all_masks[oid]
            diff = 0
            for si in range(1,S):
                diff += np.sum(np.abs(masks[si]-masks[si-1]))
            # print('total diff', diff)
            all_diffs.append(diff)
        all_diffs = np.stack(all_diffs)
        inds = np.argsort(-all_diffs)
        all_masks = [all_masks[ii] for ii in inds]

        masks = all_masks[obj_id]
            
        rgbs = np.stack(rgbs)
        masks = np.stack(masks)
        S = masks.shape[0]

        xys, _, valids, fills = utils.misc.data_get_traj_from_masks(masks)
        xys = utils.misc.data_replace_with_nearest(xys, valids)
        visibs = valids[:]

        vis_ok = np.sum(visibs > 0.5)
        if vis_ok < 4:
            print('vis_ok', vis_ok)
            return None
         
        if zoom > 1:
            xys, visibs, valids, rgbs, masks = utils.misc.data_zoom(zoom, xys, visibs, valids, rgbs, masks=masks)

            vis_ok = np.sum(visibs > 0.5)
            if vis_ok < 4:
                print('vis_ok after zoom', vis_ok)
                return None
        
        sample = {
            'rgbs': rgbs, 
            'masks': masks,
        }
        return sample

    def __len__(self):
        return len(self.all_folder_names)
