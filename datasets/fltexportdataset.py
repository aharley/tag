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
# from detectron2.structures.masks import polygons_to_bitmask

# import utils.py
# import utils.basic
import utils.geom
# import utils.improc

import glob
import json

import imageio
import cv2
import re

from torchvision.transforms import ColorJitter, GaussianBlur


# np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')

class FltexportDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir='../', version='ad', dset='t', use_augs=False, N=0, S=4, crop_size=(368, 496), force_inb=True, force_double_inb=False, force_all_inb=False, force_twice_vis=False, force_thrice_vis=False, force_late_vis=True, force_last_vis=False, force_all_vis=False, rand_frames=False):
        self.dataset_location = '%s/%s' % (root_dir, version)

        self.S = S
        self.N = N

        self.use_augs = use_augs
        self.rand_frames = rand_frames
        
        # self.mp4_paths = []
        # self.npz_paths = []
        # self.mask_paths = []
        # # self.heavy_paths = []
        # self.flow_f_paths = []
        # self.flow_b_paths = []

        self.force_inb = force_inb
        self.force_double_inb = force_double_inb
        self.force_all_inb = force_all_inb
        self.force_twice_vis = force_twice_vis
        self.force_thrice_vis = force_thrice_vis
        self.force_late_vis = force_late_vis
        self.force_last_vis = force_last_vis
        self.force_all_vis = force_all_vis

        dataset_location = self.dataset_location + '/export'

        # if dset=='t':
        #     folder_names = [folder.split('/')[-1] for folder in glob.glob(os.path.join(self.dataset_location, "0[0-7]*"))]
        # elif dset=='v':
        #     folder_names = [folder.split('/')[-1] for folder in glob.glob(os.path.join(self.dataset_location, "0[8-9]*"))]
        # else:
        #     assert(False) # t or v

        if dset=='t':
            folder_names = []
            head_names = [folder.split('/')[-1] for folder in glob.glob(os.path.join(dataset_location, "*"))]
            for head in head_names:
                folder_names += ['%s/%s' % (head, folder.split('/')[-1]) for folder in glob.glob(os.path.join(dataset_location, head, "*"))]
            folder_names = [fn for fn in folder_names if (not fn[-1]=='9')]
        elif dset=='v':
            folder_names = []
            head_names = [folder.split('/')[-1] for folder in glob.glob(os.path.join(dataset_location, "*"))]
            for head in head_names:
                folder_names += ['%s/%s' % (head, folder.split('/')[-1]) for folder in glob.glob(os.path.join(dataset_location, head, "*"))]
            # folder_names = [fn if (fn[-1]=='9') for fn in folder_names]
            # print('val foldernames', folder_names)
            # print('fn[0]', folder_names[0])
            # print('fn[0][-1]', folder_names[0][-1])
            folder_names = [fn for fn in folder_names if (fn[-1]=='9')]
            # folder_names = [fn if fn[-1]=='9']
        else:
            assert(False) # valset not really ready in this version
        
        folder_names = sorted(folder_names)
        # print('folder_names', folder_names)
        self.folder_names = folder_names
        print('found %d %s samples in %s' % (len(self.folder_names), dset, self.dataset_location))

        # photometric augmentation
        # self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        # self.photo_aug = ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.125/3.14)
        # self.blur = GaussianBlur(5, sigma=(0.1, 2.0))

        self.photo_aug = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.25/3.14)
        self.blur_aug = GaussianBlur(11, sigma=(0.1, 2.0))
        
        self.blur_aug_prob = 0.5
        self.color_aug_prob = 0.5

        # occlusion augmentation
        self.eraser_aug_prob = 0.9
        self.eraser_bounds = [2, 100]
        self.eraser_max = 10

        # occlusion augmentation
        self.replace_aug_prob = 0.9
        self.replace_bounds = [2, 100]
        self.replace_max = 20

        # spatial augmentations
        self.pad_bounds = [0, 100]
        self.crop_size = crop_size
        self.resize_lim = [0.25, 2.0] # sample resizes from here
        self.resize_delta = 0.2
        self.max_crop_offset = 100
        
        # self.crop_size = crop_size
        # self.min_scale = -0.1 # 2^this
        # self.max_scale = 1.0 # 2^this
        # self.spatial_aug_prob = 0.8
        # self.stretch_prob = 0.8
        # self.max_stretch = 0.2
        
        self.do_flip = True
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.5
        self.max_crop_offset = 10
        
    def __getitem__(self, index):
        gotit = False
        # failed_once = False
        fail_count = 0

        sample, gotit = self.getitem_helper(index)#, failed_once=failed_once)
        
        if not gotit:
            # print('warning: sampling failed')
            # fake sample, so we can still collate
            sample = {
                'rgbs': torch.zeros((self.S, 3, self.crop_size[0], self.crop_size[1]), dtype=torch.uint8),
                'masks_g': torch.zeros((self.S, 1, self.crop_size[0], self.crop_size[1]), dtype=torch.float32),
                'xys_g': torch.zeros((self.S, 2), dtype=torch.float32),
                'whs_g': torch.zeros((self.S, 2), dtype=torch.float32),
                'vis_g': torch.zeros((self.S), dtype=torch.float32),
                'xys_valid': torch.zeros((self.S), dtype=torch.float32),
                'whs_valid': torch.zeros((self.S), dtype=torch.float32),
                'vis_valid': torch.zeros((self.S), dtype=torch.float32),
            }
        # print('samp rgbs', sample['rgbs'].shape)
        return sample, gotit
    
    def getitem_helper(self, index):
        gotit = False
        while not gotit:

            folder = self.folder_names[index]
            # print('folder', folder)

            # import cv2
            vidcap = cv2.VideoCapture('%s/export/%s/rgb.mp4' % (self.dataset_location, folder))

            rgbs = []
            while(vidcap.isOpened()):
                ret, frame = vidcap.read()
                if ret == False:
                    break
                rgbs.append(frame)
            # cv2.destroyAllWindows()
            vidcap.release()
            # import ipdb; ipdb.set_trace()

            if len(rgbs) < self.S:
                print('len(rgbs)', len(rgbs))
                return None, False
            
            assert(len(rgbs)>=self.S)

            S = len(rgbs)

            vidcap = cv2.VideoCapture('%s/export/%s/mask.mp4' % (self.dataset_location, folder))
            masks = []
            while(vidcap.isOpened()):
                ret, frame = vidcap.read()
                if ret == False:
                    break
                masks.append(frame)
            # cv2.destroyAllWindows()
            vidcap.release()
            
            rgbs = np.stack(rgbs, axis=0) # S,H,W,3
            masks = np.stack(masks, axis=0) # S,H,W,3
            masks = masks[:,:,:,0] # take one chan and eliminate this dim

            npz_f = '%s/export/%s/trajs.npz' % (self.dataset_location, folder)
            d = dict(np.load(npz_f, allow_pickle=True))
            trajs = d['trajs'] # S, N, 2
            visibles = d['vis'] # S, N

            # print('rgbs', rgbs.shape)
            # print('masks', masks.shape)
            # print('trajs', trajs.shape)
            # print('visibles', visibles.shape)

            rgbs, masks, trajs, visibles = self.just_crop(rgbs, masks, trajs, visibles)

            S = self.S
            S_here = len(rgbs)
            assert(S_here >= S)

            N = trajs.shape[1]
            if N==0:
                print('flt N==0')
                return None, False

            if S_here>self.S*2 and (np.random.rand() < 0.5):
                if S_here>self.S*3 and (np.random.rand() < 0.5):
                    if S_here>self.S*4 and (np.random.rand() < 0.5):
                        # print('taking ::4')
                        rgbs = rgbs[::4]
                        masks = masks[::4]
                        trajs = trajs[::4]
                        visibles = visibles[::4]
                    else:
                        # print('taking ::3')
                        rgbs = rgbs[::3]
                        masks = masks[::3]
                        trajs = trajs[::3]
                        visibles = visibles[::3]
                else:
                    # print('taking ::2')
                    rgbs = rgbs[::2]
                    masks = masks[::2]
                    trajs = trajs[::2]
                    visibles = visibles[::2]
            else:
                # print('taking ::1')
                pass

            if np.sum(visibles[1:])==0:
                print('flt np.sum(visibles[1:])', np.sum(visibles[1:]))
                return None, False

            S_here = len(rgbs)

            rgbs = torch.from_numpy(np.stack(rgbs, 0)).permute(0, 3, 1, 2) # S, C, H, W
            masks = torch.from_numpy(np.stack(masks, 0)).unsqueeze(1) # S, 1, H, W
            trajs = torch.from_numpy(trajs).squeeze(1) # S,2
            visibles = torch.from_numpy(visibles).squeeze(1) # S

            # make it B,S,...
            rgbs = rgbs[:S].unsqueeze(0)
            masks = masks[:S].unsqueeze(0)
            trajs = trajs[:S].unsqueeze(0)
            visibles = visibles[:S].unsqueeze(0)
            # B = 1

            B, S, C, H, W = rgbs.shape

            # print('rgbs', rgbs.shape)
            # print('masks', masks.shape)

            # masks are currently 255 at gt, and 128 at ignore
            masks_pos = (masks > 191.0).float()
            masks_ign = (masks > 64.0).float()
            masks = (masks_pos + masks_ign)/2.0 
            
            masks_ = masks_pos.reshape(B*S,1,H,W).clone()
            for bs in range(B*S):
                if torch.sum(masks_[bs])==0:
                    masks_[bs,0,0,0] = 1.0

            boxes_ = utils.geom.get_box2d_from_mask(masks_)

            ys_, xs_ = utils.geom.get_centroid_from_box2d(boxes_)
            hs_, ws_ = utils.geom.get_size_from_box2d(boxes_)

            xys_g = torch.stack([xs_, ys_], dim=-1).reshape(B,S,2)
            whs_g = torch.stack([ws_, hs_], dim=-1).reshape(B,S,2)

            # actually use the trajs as gt xy
            xys_g = trajs.clone()
            # and use the zeroth wh
            whs_g = whs_g[:,0:1].repeat(1,S,1)
            
            # wh_add = torch.from_numpy(np.random.uniform(wh_add_min, wh_add_max, (1,1,2))).float()#.to(device)
            # whs_g = whs_g + wh_add
            whs_g = whs_g.clamp(min=16) # don't be smaller than this please

            vis_g = visibles.clone()
            xys_valid = torch.ones_like(vis_g)
            whs_valid = torch.zeros_like(vis_g)
            whs_valid[:1] = 1.0
            vis_valid = torch.ones_like(vis_g)

            sample = {}
            sample['rgbs'] = rgbs.squeeze(0)
            sample['masks_g'] = masks.squeeze(0)
            sample['xys_g'] = xys_g.squeeze(0)
            sample['whs_g'] = whs_g.squeeze(0)
            sample['vis_g'] = vis_g.squeeze(0)
            sample['xys_valid'] = xys_valid.squeeze(0)
            sample['whs_valid'] = whs_valid.squeeze(0)
            sample['vis_valid'] = vis_valid.squeeze(0)

            return sample, True
            
    def just_crop(self, rgbs, masks, trajs, visibles):
        '''
        Input:
            rgbs --- list np.array (S, H, W, 3)
            trajs --- np.array (S, N, 2)
            visibles --- np.array (S, N)
        '''

        S = len(rgbs)
        H, W, C = rgbs[0].shape
        _, N, D = trajs.shape

        # def unstack(a, axis=0):
        #     return [np.squeeze(e, axis) for e in np.split(a, a.shape[axis], axis=axis)]
        # rgbs = unstack(rgbs, axis=0)
        # masks = unstack(masks, axis=0)

        # if self.crop_size[0] < H and self.crop_size[1] < W:
        # crop
        # y0 = np.random.randint(0, H - self.crop_size[0] + 1)
        # x0 = np.random.randint(0, W - self.crop_size[1] + 1)

        H_new = H
        W_new = W
        
        mid_x = trajs[0,0,0]
        mid_y = trajs[0,0,1]
        x0 = int(mid_x - self.crop_size[1]//2)
        y0 = int(mid_y - self.crop_size[0]//2)

        if H_new==self.crop_size[0]:
            y0 = 0
        else:
            y0 = min(max(0, y0), H_new - self.crop_size[0] - 1)

        if W_new==self.crop_size[1]:
            x0 = 0
        else:
            x0 = min(max(0, x0), W_new - self.crop_size[1] - 1)
            
        rgbs = [rgb[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]] for rgb in rgbs]
        masks = [mask[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]] for mask in masks]
        
        # masks = [mask[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]] for mask in masks]
        trajs[:,:,0] -= x0
        trajs[:,:,1] -= y0

        vis0 = visibles[0,:] >= 0
        if self.force_double_inb:
            inbound0 = (trajs[0,:,0] >= 0) & (trajs[0,:,0] <= self.crop_size[1]-1) & (trajs[0,:,1] >= 0) & (trajs[0,:,1] <= self.crop_size[0]-1)
            inbound1 = (trajs[1,:,0] >= 0) & (trajs[1,:,0] <= self.crop_size[1]-1) & (trajs[1,:,1] >= 0) & (trajs[1,:,1] <= self.crop_size[0]-1)
            inbound = inbound0 & inbound1 & vis0
        elif self.force_all_inb:
            inbound = (trajs[0,:,0] >= 0) & (trajs[0,:,0] <= self.crop_size[1]-1) & (trajs[0,:,1] >= 0) & (trajs[0,:,1] <= self.crop_size[0]-1)
            for s in range(1,S):
                inboundi = (trajs[s,:,0] >= 0) & (trajs[s,:,0] <= self.crop_size[1]-1) & (trajs[s,:,1] >= 0) & (trajs[s,:,1] <= self.crop_size[0]-1)
                inbound = inbound & inboundi & vis0
        else:
            inbound = (trajs[0,:,0] >= 0) & (trajs[0,:,0] <= self.crop_size[1]-1) & (trajs[0,:,1] >= 0) & (trajs[0,:,1] <= self.crop_size[0]-1) & vis0

        trajs = trajs[:,inbound]
        visibles = visibles[:,inbound]
        
        # mark oob points as invisible
        for s in range(S):
            oob_inds = np.logical_or(np.logical_or(trajs[s,:,0] < 0, trajs[s,:,0] > self.crop_size[1]-1), np.logical_or(trajs[s,:,1] < 0, trajs[s,:,1] > self.crop_size[0]-1))
            visibles[s,oob_inds] = 0
            
        rgbs = np.stack(rgbs, axis=0)
        masks = np.stack(masks, axis=0)

        return rgbs, masks, trajs, visibles

    def add_photometric_augs(self, rgbs, trajs, visibles, eraser=True, replace=False):
        T, N, _ = trajs.shape

        # print('trajs', trajs.shape)
        # print('len(rgbs)', len(rgbs))
        
        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert(S==T)

        # rgbs = [0.1*rgb.astype(np.float32) for rgb in rgbs]
        
        if eraser:
            ############ eraser transform (per image after the first) ############
            rgbs = [rgb.astype(np.float32) for rgb in rgbs]
            for i in range(1, S):
                if np.random.rand() < self.eraser_aug_prob:
                    # mean_color = np.mean(rgbs[i].reshape(-1, 3), axis=0)
                    for _ in range(np.random.randint(1, self.eraser_max+1)): # number of times to occlude
                        # mean_color = np.mean(rgbs[i].reshape(-1, 3), axis=0)
                        
                        xc = np.random.randint(0, W)
                        yc = np.random.randint(0, H)
                        dx = np.random.randint(self.eraser_bounds[0], self.eraser_bounds[1])
                        dy = np.random.randint(self.eraser_bounds[0], self.eraser_bounds[1])
                        x0 = np.clip(xc - dx/2, 0, W-1).round().astype(np.int32)
                        x1 = np.clip(xc + dx/2, 0, W-1).round().astype(np.int32)
                        y0 = np.clip(yc - dy/2, 0, H-1).round().astype(np.int32)
                        y1 = np.clip(yc + dy/2, 0, H-1).round().astype(np.int32)
                        # print(x0, x1, y0, y1)
                        mean_color = np.mean(rgbs[i][y0:y1, x0:x1, :].reshape(-1,3), axis=0)
                        rgbs[i][y0:y1, x0:x1, :] = mean_color

                        occ_inds = np.logical_and(np.logical_and(trajs[i,:,0] >= x0, trajs[i,:,0] < x1), np.logical_and(trajs[i,:,1] >= y0, trajs[i,:,1] < y1))
                        visibles[i, occ_inds] = 0
            rgbs = [rgb.astype(np.uint8) for rgb in rgbs]

        if replace:

            rgbs_alt = [np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs]
            rgbs_alt = [np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs_alt]
            
            ############ replace transform (per image after the first) ############
            rgbs = [rgb.astype(np.float32) for rgb in rgbs]
            rgbs_alt = [rgb.astype(np.float32) for rgb in rgbs_alt]
            for i in range(1, S):
                if np.random.rand() < self.replace_aug_prob:
                    for _ in range(np.random.randint(1, self.replace_max+1)): # number of times to occlude
                        xc = np.random.randint(0, W)
                        yc = np.random.randint(0, H)
                        dx = np.random.randint(self.replace_bounds[0], self.replace_bounds[1])
                        dy = np.random.randint(self.replace_bounds[0], self.replace_bounds[1])
                        x0 = np.clip(xc - dx/2, 0, W-1).round().astype(np.int32)
                        x1 = np.clip(xc + dx/2, 0, W-1).round().astype(np.int32)
                        y0 = np.clip(yc - dy/2, 0, H-1).round().astype(np.int32)
                        y1 = np.clip(yc + dy/2, 0, H-1).round().astype(np.int32)

                        wid = x1-x0
                        hei = y1-y0
                        y00 = np.random.randint(0, H-hei)
                        x00 = np.random.randint(0, W-wid)
                        fr = np.random.randint(0, S)
                        rep = rgbs_alt[fr][y00:y00+hei, x00:x00+wid, :]
                        rgbs[i][y0:y1, x0:x1, :] = rep
                                                
                        # print(x0, x1, y0, y1)
                        # mean_color = np.mean(rgbs[i][y0:y1, x0:x1, :].reshape(-1,3), axis=0)
                        # rgbs[i][y0:y1, x0:x1, :] = mean_color

                        
                        # mean_color = np.mean(rgbs[i][y0:y1, x0:x1, :].reshape(-1,3), axis=0)
                        # rgbs[i][y0:y1, x0:x1, :] = mean_color
                        
                        occ_inds = np.logical_and(np.logical_and(trajs[i,:,0] >= x0, trajs[i,:,0] < x1), np.logical_and(trajs[i,:,1] >= y0, trajs[i,:,1] < y1))
                        visibles[i, occ_inds] = 0
            rgbs = [rgb.astype(np.uint8) for rgb in rgbs]
            

        ############ photometric augmentation ############
        if np.random.rand() < self.color_aug_prob:
            # random per-frame amount of aug
            rgbs = [np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs]

        if np.random.rand() < self.blur_aug_prob:
            # random per-frame amount of blur
            rgbs = [np.array(self.blur_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs]

        return rgbs, trajs, visibles

    

    def __len__(self):
        return len(self.folder_names)
