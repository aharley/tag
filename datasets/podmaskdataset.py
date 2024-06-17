from numpy import random
import torch
import numpy as np
import os
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import random
from torch._C import dtype, set_flush_denormal
import utils.misc
import glob
import cv2
from torchvision.transforms import ColorJitter, GaussianBlur
from datasets.dataset import MaskDataset, mask2bbox

class PodMaskDataset(MaskDataset):
    def __init__(self,
                 dataset_location,
                 use_augs=False,
                 S=8, fullseq=False, chunk=None,
                 N=32,
                 strides=[1,2,3],
                 zooms=[1,2],
                 crop_size=(368, 496),
                 is_training=True):
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training
        )
        print('loading pointodyssey dataset...')

        if self.is_training:
            dset = 'train'
            self.N_per = 4
        else:
            dset = 'val'
            self.N_per = 1
            strides = [1]
            
        self.S = S
        self.N = N
        self.base_strides = strides
        self.num_strides = len(strides)
        self.crop_size = crop_size

        self.use_augs = use_augs
        self.traj_paths = []
        self.subdirs = []
        self.sequences = []

        # self.mask_fill_thr = 0.05

        self.subdirs.append(os.path.join(dataset_location, dset))

        for subdir in self.subdirs:
            for seq in glob.glob(os.path.join(subdir, "*")):
                if os.path.isdir(seq):
                    seq_name = seq.split('/')[-1]
                    self.sequences.append(seq)

        self.sequences = sorted(self.sequences)
        print('found %d unique videos in %s (dset=%s)' % (len(self.sequences), dataset_location, dset))

        if chunk is not None:
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.sequences = chunkify(self.sequences,100)[chunk]
            print('filtered to %d sequences' % len(self.sequences))
            print('self.sequences', self.sequences)
        
        self.rgb_paths = []
        self.mask_paths = []
        self.full_idxs = []
        self.annotation_paths = []
        self.tids = []
        self.strides = []
        self.zooms = []

        H,W = 540,960

        clip_step = S//2
        
        ## load trajectories
        print('loading trajectories...')
        for seq in self.sequences:
            rgb_path = os.path.join(seq, 'rgbs')
            annotations_path = os.path.join(seq, 'anno.npz')
            if os.path.isfile(annotations_path):
                for stride in strides:
                    S_local = len(os.listdir(rgb_path))
                    for ii in range(0,max(S_local-self.S*stride+1, 1), clip_step*stride):
                        for ni in range(self.N_per):
                            full_idx = ii + np.arange(self.S)*stride
                            full_idx = [ij for ij in full_idx if ij < S_local]
                            if len(full_idx)==self.S: # always fullseq
                                for zoom in zooms:
                                    self.rgb_paths.append([os.path.join(seq, 'rgbs', 'rgb_%05d.jpg' % idx) for idx in full_idx])
                                    self.mask_paths.append([os.path.join(seq, 'masks', 'mask_%05d.png' % idx) for idx in full_idx])
                                    self.annotation_paths.append(os.path.join(seq, 'anno.npz'))
                                    self.full_idxs.append(full_idx)
                                    self.tids.append(ni)
                                    self.strides.append(stride)
                                    self.zooms.append(zoom)

        print('collected %d clips of length %d in %s (dset=%s)' % (
            len(self.rgb_paths), self.S, dataset_location, dset))


    def getitem_helper(self, index):

        rgb_paths = self.rgb_paths[index]
        mask_paths = self.mask_paths[index]
        full_idx = self.full_idxs[index]
        obj_id = self.tids[index]
        stride = self.strides[index]
        zoom = self.zooms[index]
        stride_ind = self.base_strides.index(stride)

        rgbs = []
        for rgb_path in rgb_paths:
            with Image.open(rgb_path) as im:
                rgbs.append(np.array(im)[:, :, :3])
        rgbs = np.stack(rgbs, axis=0) # S,H,W,3

        S,H,W,C = rgbs.shape
        
        assert(S==self.S)
        # print('rgbs', rgbs.shape)

        segs = []
        for mask_path in mask_paths:
            with Image.open(mask_path) as im:
                mask = np.array(im)
                if np.sum(mask==0) > 128:
                    # fill holes caused by fog/smoke
                    mask_filled = cv2.medianBlur(mask, 7)
                    mask[mask==0] = mask_filled[mask==0]
                segs.append(mask) # H,W
        segs = np.stack(segs, axis=0) # S,H,W
        # print('segs', segs.shape)

        segA = segs.reshape(-1)
        obj_ids = np.unique(segA) # NSeg

        all_masks = []
        for oid in obj_ids:
            masks = [(seg == oid).astype(np.float32) for seg in segs]
            masks = np.stack(masks, axis=0)
            # print('mask sum', np.sum(masks))
            if np.sum(masks) < 8*S:
                continue
            bboxes = np.stack([mask2bbox(mask) for mask in masks])
            whs = bboxes[:,2:4] - bboxes[:,0:2]
            # print('whs', whs)
            if np.max(whs[:,0]) > W/2 or np.max(whs[:,1]) > H/2:
                continue
            _, _, _, fills = utils.misc.data_get_traj_from_masks(masks)
            # print('sum(fills)', np.sum(fills))
            if np.sum(fills) < S//4:
                # print('low sum(fills):', np.sum(fills))
                continue
            all_masks.append(masks)
        # print('found %d complete masks' % len(all_masks))

        if obj_id >= len(all_masks):
            print('obj_id unavailable')
            return None
        
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

        # print('taking id', obj_id)
        masks = all_masks[obj_id]
        # print('mask sums', np.sum(masks.reshape(S,-1), axis=1))

        rgbs = np.stack(rgbs, axis=0)
        masks = np.stack(masks, axis=0)

        mask_areas = (masks > 0).reshape(S,-1).sum(axis=1)
        mask_areas_norm = mask_areas / np.max(mask_areas)
        visibs = mask_areas_norm
        
        rgbs, masks = utils.misc.data_pad_if_necessary(rgbs, masks)
        S,H,W,C = rgbs.shape
        
        if zoom > 1:
            xys, _, valids, fills = utils.misc.data_get_traj_from_masks(masks)
            xys = utils.misc.data_replace_with_nearest(xys, valids)
            
            bboxes = np.stack([mask2bbox(mask) for mask in masks])
            whs = bboxes[:,2:4] - bboxes[:,0:2]
            whs = whs[visibs > 0.5]
            # print('mean wh', np.mean(whs[:,0]), np.mean(whs[:,1]))
            if np.mean(whs[:,0]) >= W/2 and np.mean(whs[:,1]) >= H/2:
                # print('would reject')
                # big already
                return None
            xys, visibs, valids, rgbs, masks = utils.misc.data_zoom(zoom, xys, visibs, valids, rgbs, masks=masks)

        if np.sum(visibs>0.5) < 3:
            print('np.sum(visibs>0.5)', np.sum(visibs>0.5))
            return None
        
        visibs = 1.0 * (visibs>0.5)
        safe = visibs*0
        for si in range(1,S-1):
            safe[si] = visibs[si-1]*visibs[si]*visibs[si+1]
            
        if np.sum(safe) < 2:
            print('safe', safe)
            return None
        
        sample = {
            'rgbs': rgbs,
            'masks': masks,
        }
        return sample

    def __len__(self):
        return len(self.rgb_paths)
