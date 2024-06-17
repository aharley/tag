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
import utils.geom
import glob
import json
import imageio
import cv2
import re
from torchvision.transforms import ColorJitter, GaussianBlur
from datasets.dataset import MaskDataset
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import albumentations as A
from datasets.dataset import augment_video

# np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')


def mask2bbox(mask):
    if mask.ndim == 3:
        mask = mask[..., 0]
    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.array((0, 0, 10, 10), dtype=int)
    tl = np.array([np.min(xs), np.min(ys)])
    br = np.array([np.max(xs), np.max(ys)]) + 1
    return np.concatenate([tl, br])

def bbox2mask(bbox, w, h):
    mask = np.zeros((h, w), dtype=np.float32)
    if (bbox[3] - bbox[1]) * (bbox[2] - bbox[0]) <= 0:
        return mask
    bbox = bbox.astype(int)
    mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1.
    return mask


def read_mp4(fn):
    # videodata = skvideo.io.vread(fn)
    # print('vid', videodata.shape)
    # return videodata
        
    try:
        vidcap = cv2.VideoCapture(fn)
        frames = []
        if not vidcap.isOpened():
            print('some problem with file', fn)
            return []
        while(vidcap.isOpened()):
            try:
                ret, frame = vidcap.read()
            except:
                print('some problem with file', fn)
                vidcap.release()
                return frames
            if ret == False:
                break
            frames.append(frame)
        vidcap.release()
        return frames
    except:
        print('some problem with file', fn)
        return []

class ExportDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_location='../datasets/alltrack_export',
                 version='au_trainA',
                 dsets=None,
                 dsets_exclude=None,
                 S=32,
                 rand_frames=False,
                 shuffle=True,
                 data_shape=(384,512), 
                 target_shape=(384,512),
                 prompt_stride=1,
                 mask_stride=2,
                 quick=False,
                 random_anchor=True,
                 use_augs=False,
                 is_training=True,
    ):
        print('loading export...')

        self.dataset_location = dataset_location
        self.S = S
        self.H, self.W = data_shape
        self.cH, self.cW = target_shape
        self.use_augs = use_augs
        self.random_anchor = random_anchor

        self.prompt_stride = prompt_stride
        self.mask_stride = mask_stride

        assert(self.cH % 32 == 0)
        assert(self.cW % 32 == 0)
        
        self.dataset_location = Path(self.dataset_location) / version

        dataset_names = self.dataset_location.glob('*/')
        self.dataset_names = [fn.stem for fn in dataset_names]

        self.mean = torch.as_tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1).float()
        self.std = torch.as_tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1).float()

        # self.color_augmenter = A.ReplayCompose([
        #     A.GaussNoise(p=0.1),
        #     A.OneOf([
        #         A.MotionBlur(),
        #         A.MedianBlur(),
        #         A.Blur(),
        #     ], p=0.5),
        #     A.OneOf([
        #         A.CLAHE(clip_limit=2),
        #         A.Sharpen(),
        #         A.Emboss(),
        #     ], p=0.5),
        #     A.RGBShift(),
        #     A.RandomBrightnessContrast(),
        #     A.RandomGamma(),
        #     A.HueSaturationValue(),
        #     A.ImageCompression(),
        # ], p=0.8)

        self.color_augmenter = A.ReplayCompose([
            A.GaussNoise(p=0.1),
            A.OneOf([
                A.MotionBlur(),
                A.MedianBlur(),
                A.Blur(),
            ], p=0.5),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
            ], p=0.5),
            A.RGBShift(),
            A.RandomBrightnessContrast(),
            A.HueSaturationValue(),
            A.ImageCompression(),
        ], p=0.9)
        
        self.color_augmenter_nodestroy = A.ReplayCompose([
            A.RGBShift(),
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
            A.HueSaturationValue(),
        ], p=0.9)
        
        
        print('dataset_names', self.dataset_names)

        folder_names = self.dataset_location.glob('*/*/*/')
        # folder_names = [fn for fn in folder_names if (fn.stem[-1] != '9' if is_training else fn.stem[-1] == '9')]
        
        folder_names = [str(fn) for fn in folder_names]

        # folder_names = sorted(list(folder_names))
        if shuffle:
            random.shuffle(folder_names)
        else:
            folder_names = sorted(list(folder_names))
            
        # print('found {:d} {} folders in {}'.format(len(folder_names), ('train' if is_training else 'test'), self.dataset_location))
        print('found {:d} {} folders in {}'.format(len(folder_names), version, self.dataset_location))
        # print('found {:d} samples in {}'.format(len(self.all_folder_names), self.dataset_location))

        # print('folder_names', folder_names)
        if dsets is not None:
            print('dsets', dsets)
            new_folder_names = []
            for fn in folder_names:
                for dset in dsets:
                    if dset in fn:
                        new_folder_names.append(fn)
                        break
            folder_names = new_folder_names
            print('filtered to %d folders' % len(folder_names))

        # print('folder_names', folder_names)
        if dsets_exclude is not None:
            print('dsets_exclude', dsets_exclude)
            new_folder_names = []
            for fn in folder_names:
                keep = True
                for dset in dsets_exclude:
                    if dset in fn:
                        keep = False
                        break
                if keep:
                    new_folder_names.append(fn)
            folder_names = new_folder_names
            print('filtered to %d folders' % len(folder_names))
            
        if quick:
            # print('taking random 101 folders')
            # local_random = random.Random(0)
            # local_random.shuffle(folder_names)
            folder_names = sorted(folder_names)
            folder_names = folder_names[:101]
            print('folder_names', folder_names)

        self.all_folder_names = folder_names

        # # step through once and make sure all of the npzs are there
        # print('stepping through...')
        # self.all_folder_names = []
        # for fi, folder in enumerate(folder_names):
        #     # print('%d/%d; folder %s' % (fi, len(folder_names), folder))
        #     print('%d/%d' % (fi, len(folder_names)))
        #     if os.path.isfile('%s/track.npy' % folder):
        #         self.all_folder_names.append(folder)
        #     else:
        #         print('missing track in %s' % folder)
        #         pass
        # print('ok done stepping.')

        # load one and make sure it matches
        print('reading as test sample:', self.all_folder_names[0])
        rgbs = read_mp4(self.all_folder_names[0] + '/rgb.mp4')
        print('test sample len and image shape:', len(rgbs), rgbs[0].shape)
        S_local = len(rgbs)
        assert(self.S<=S_local)
        H, W, C = rgbs[0].shape
        assert(self.H==H)
        assert(self.W==W)

        # self.all_folder_names = sorted(self.all_folder_names)[:200]
                
        # if S_local==S:
        #     self.all_idxs = [np.arange(self.S) for _ in range(len(folder_names))]
        #     self.all_folder_names = folder_names
        # else:
        #     for fi, folder in enumerate(folder_names):
        #         if os.path.isfile(str(folder / 'track.npz')): 
        #             for ii in range(0,S_local-self.S+1,8):
        #                 full_idx = ii + np.arange(self.S)
        #                 self.all_folder_names.append(folder)
        #                 self.all_idxs.append(full_idx)
        #                 sys.stdout.write('.')
        #                 sys.stdout.flush()
        # print('found {:d} {} samples in {}'.format(len(self.all_folder_names), ('train' if is_training else 'test'), self.dataset_location))
        print('found {:d} {} samples in {}'.format(len(self.all_folder_names), version, self.dataset_location))
        # print('found {:d} samples in {}'.format(len(self.all_folder_names), self.dataset_location))
        
    def __color_augment__(self, rgbs):
        augment_video(self.color_augmenter, image=rgbs)
        
    def __color_augment_nodestroy__(self, rgbs):
        augment_video(self.color_augmenter_nodestroy, image=rgbs)

    def getitem_helper(self, index):

        folder = self.all_folder_names[index]
        # print('folder', folder)

        cH, cW = self.cH, self.cW
        fH, fW = self.cH//self.mask_stride, self.cW//self.mask_stride

        mC = 5 # xy, modal mask, amodal mask, lt, rb

        fake_prompt = torch.zeros((self.S,1,cH,cW), dtype=torch.float32)
        fake_prompt[0] = 1
        fake_sample = {
            'rgbs': torch.zeros((self.S, 3, cH, cW), dtype=torch.float32), 
            'masks_g': torch.zeros((self.S, mC, fH, fW), dtype=torch.float32), 
            'track_g': torch.zeros((self.S, 13), dtype=torch.float32), 
            'prompts_g': fake_prompt, 
            'dname': 'none',
            'step': torch.zeros((), dtype=torch.int32), 
        }

        try: 
            rgbs = read_mp4(folder  + '/rgb.mp4')
            masks = read_mp4(folder + '/mask.mp4')
        except:
            print('some exception with folder', folder)
            return fake_sample
            
        if len(rgbs)<self.S or len(masks)<self.S:
            print('some problem with folder', folder)
            return fake_sample
        
        rgbs = np.stack(rgbs, axis=0) # S,H,W,3
        masks = np.stack(masks, axis=0) # S,H,W,3
        
        try: 
            # d = dict(np.load(folder + '/track.npz', allow_pickle=True))
            d = np.load(folder + '/track.npy', allow_pickle=True).item()
        except: # zipfile.BadZipfile:
            d = None
        if d is None:
            print('some problem with folder', folder)
            return fake_sample
            # index = np.random.randint(len(self.all_folder_names))

        track_g = d['track_g'] # S,13
        dname = str(d['dname'])
        # print('export dname', dname)

        S1 = len(rgbs)
        S2 = len(masks)
        S3 = track_g.shape[0]
        if not (S1 == S2) or not (S1 == S3):
            print('some problem with folder', folder)
            return fake_sample
        
        assert(not np.any(np.isnan(rgbs)))
        assert(not np.any(np.isnan(masks)))
        assert(not np.any(np.isnan(track_g)))

        if rgbs.shape[0] > self.S:
            # we want to take a valid subseq
            # we'll compute a viable anchor first,
            # so that the subseq can include the anchor
            masks_valid = track_g[:,10:] # S,3
            if np.sum(masks_valid[:,0]) > 0: # point dataset
                vis_valid = track_g[:,6] * track_g[:,7]
                # req 3 in a row, for disambig
                # safe_g = vis_valid*0
                # for si in range(1,self.S-1):
                #     safe_g[si] = vis_valid[si-1]*vis_valid[si]*vis_valid[si+1]

                # if np.sum(safe_g)>0:
                #     anchor_inds = np.nonzero(safe_g.reshape(-1)>0.5)[0]
                # else:
                # print('taking anchor from vis_valid instead of safe_g')
                if np.sum(vis_valid)==0:
                    print('no vis_valid in folder', folder)
                    return fake_sample
                anchor_inds = np.nonzero(vis_valid.reshape(-1)>0.5)[0]
            else:
                # take any vis>0.5
                vis_g = track_g[:,6]
                anchor_inds = np.nonzero(vis_g.reshape(-1)>0.5)[0]

            # print('anchor_inds', anchor_inds)
            if len(anchor_inds)==0:
                print('no anchors1 in folder', folder)
                return fake_sample
            
            if self.random_anchor:
                # if np.random.rand() < 0.1:
                #     anchor_ind = anchor_inds[0]
                # else:
                anchor_ind = anchor_inds[np.random.randint(len(anchor_inds))]
            else:
                anchor_ind = anchor_inds[0]
            
            if self.use_augs:
                max_choice = S1//self.S
                astride = np.random.randint(max_choice)+1
            else:
                astride = 1
                
            s0 = max(anchor_ind-int(self.S//(astride/2.0)),0)
            s1 = s0+self.S*astride
            s1 = min(s1, S1)
            s0 = s1-self.S*astride
            ara = np.arange(s0,s1, astride)
                
            nearest = np.argmin(np.abs(ara - anchor_ind))
            ara[nearest] = anchor_ind

            rgbs = rgbs[ara]
            masks = masks[ara]
            track_g = track_g[ara]

        masks_valid = track_g[:,10:] # S,3
        if np.sum(masks_valid[:,0]) > 0: # point dataset
            vis_valid = track_g[:,6] * track_g[:,7]
            # req 3 in a row, for disambig
            # safe_g = vis_valid*0
            # for si in range(1,self.S-1):
            #     safe_g[si] = vis_valid[si-1]*vis_valid[si]*vis_valid[si+1]

            # if np.sum(safe_g)>0:
            #     anchor_inds = np.nonzero(safe_g.reshape(-1)>0.5)[0]
            # else:
            # print('taking anchor from vis_valid')
            # assert(np.sum(vis_valid)>0)
            if np.sum(vis_valid)==0:
                print('no vis_valid in folder', folder)
                return fake_sample
            anchor_inds = np.nonzero(vis_valid.reshape(-1)>0.5)[0]
        else:
            # take any vis>0.5
            vis_g = track_g[:,6]
            anchor_inds = np.nonzero(vis_g.reshape(-1)>0.5)[0]

        # print('anchor_inds', anchor_inds)
        if len(anchor_inds)==0:
            print('no anchors1 in folder', folder)
            return fake_sample
        if self.random_anchor:
            if np.random.rand() < 0.1:
                anchor_ind = anchor_inds[0]
            else:
                anchor_ind = anchor_inds[np.random.randint(len(anchor_inds))]
        else:
            anchor_ind = anchor_inds[0]
                
        assert(len(rgbs)==self.S)
        assert(len(masks)==self.S)
        assert(track_g.shape[0]==self.S)

        if self.use_augs:
            if np.random.rand() < 0.8:
                # spatial resizing augs

                # note we can't downsample too much, or else there is no crop
                h_change = 16*np.random.randint(-3,6)
                w_change = 16*np.random.randint(-3,6)

                S,H,W,C = rgbs.shape
                assert(C==3)

                new_H, new_W = H+h_change, W+w_change
                # print('targeting', new_H, new_W)

                new_H = max(new_H, cH)
                new_W = max(new_W, cW)

                sc_x = new_W/W
                sc_y = new_H/H

                rgbs = np.stack([cv2.resize(rgb, (new_W, new_H), interpolation=cv2.INTER_AREA) for rgb in rgbs], axis=0)
                masks = np.stack([cv2.resize(mask, (new_W, new_H), interpolation=cv2.INTER_NEAREST) for mask in masks], axis=0)

                sc_xy = np.array([sc_x, sc_y]).reshape(1,2)
                track_g[:,0:2] *= sc_xy
                track_g[:,2:4] *= sc_xy
                track_g[:,4:6] *= sc_xy

                S,H,W,C = rgbs.shape
                # print('resized rgbs',  rgbs.shape)

        rgbs = rgbs.transpose(0,3,1,2)
        rgbs = rgbs[:,::-1].copy() # BGR->RGB

        masks = masks.transpose(0,3,1,2)
        masks = masks[:,::-1].copy() # BGR->RGB

        masks_pos = (masks > (256-32)) * 1.0
        masks_ign = (masks > (128-32)) * 1.0

        masks = (masks_pos + masks_ign)/2.0


        # print('folder, rgbs', folder, rgbs.shape)
        # print('rgbs', rgbs.shape)
        # print('masks', masks.shape)
        # print('track_g', track_g.shape)
        # print('dname', dname)
        # print('folder', folder)
        # print('index', index)
        # print('rgbs, masks, track', rgbs.shape, masks.shape, track_g.shape)
        
        if self.use_augs:

            if np.random.rand() < 0.1:
                # totally some random non-anchor frames
                drops = np.random.randint(9)
                for dr in range(drops):
                    ii = np.random.randint(self.S)
                    if not ii==anchor_ind:
                        rgbs[ii] *= 0
                        track_g[ii,6] = 0 # set vis to 0
                
            
            if np.random.rand() < 0.8:
                # the augmenter wants channels last
                rgbs = np.transpose(rgbs, [0,2,3,1])

                if dname=='bioparticle' or np.random.rand() < 0.5:
                    # the signal is very fragile
                    self.__color_augment_nodestroy__(rgbs)
                else:
                    self.__color_augment__(rgbs)
                # self.__color_augment_nodestroy__(rgbs)
                    
                rgbs = np.transpose(rgbs, [0,3,1,2])

            if np.random.rand() < 0.2:
                # on some random non-anchor frames, apply extra augs
                rgbs = np.transpose(rgbs, [0,2,3,1])
                extras = np.random.randint(5)
                for ex in range(extras):
                    ii = np.random.randint(self.S)
                    if not ii==anchor_ind:
                        rgbs_ = rgbs[ii:ii+1]
                        self.__color_augment__(rgbs_)
                        rgbs[ii:ii+1] = rgbs_
                rgbs = np.transpose(rgbs, [0,3,1,2])

        S = len(rgbs)

        rgbs = torch.from_numpy(rgbs).float() # S,3,H,W
        masks = torch.from_numpy(masks).float() # S,3,H,W
        track_g = torch.from_numpy(track_g).float() # S,13
        device = rgbs.device

        S,C,H,W = rgbs.shape

        # assert(cH==self.H)
        # assert(cW==self.W)

        # if cH < self.H or cW < self.W:

        # if False:
        if np.random.rand() < 0.5:
            # zero-vel crop on anchor, with some random offset
            xys_g = track_g[:,0:2]
            lts_g = track_g[:,2:4]
            rbs_g = track_g[:,4:6]
            vis_g = track_g[:,6]
            xy_mid = xys_g[anchor_ind].round().long() # 2
            xmid, ymid = xy_mid[0], xy_mid[1]
            xmid = xmid + torch.randint(-cW//4,cW//4,(1,),device=device)
            ymid = ymid + torch.randint(-cH//4,cH//4,(1,),device=device)
            xmid = torch.clamp(xmid, cW//2, W-cW//2)
            ymid = torch.clamp(ymid, cH//2, H-cH//2)
            x0, x1 = torch.clamp(xmid-cW//2, 0), torch.clamp(xmid+cW//2, 0, W)
            y0, y1 = torch.clamp(ymid-cH//2, 0), torch.clamp(ymid+cH//2, 0, H)
            assert(x1-x0==cW)
            assert(y1-y0==cH)

            # print('new bbox', x0, y0, x1, y1)
            offset = torch.stack([x0, y0], dim=0).reshape(1,2)
            # print('new bbox', x0, y0, x1, y1, 'offset', offset)
            rgbs_local = rgbs[:,:,y0:y1,x0:x1]
            masks_local = masks[:,:,y0:y1,x0:x1]
            xys_g_local = xys_g - offset
            lts_g_local = lts_g - offset
            rbs_g_local = rbs_g - offset
            
        else:
            # zero-vel crop on anchor, with shifting offset
            xys_g = track_g[:,0:2]
            lts_g = track_g[:,2:4]
            rbs_g = track_g[:,4:6]
            vis_g = track_g[:,6]
            xy_mid = xys_g[anchor_ind].round().long() # 2
            xmid, ymid = xy_mid[0:1], xy_mid[1:2]
            xmid = xmid + torch.randint(-cW//8,cW//8,(S,),device=device)
            ymid = ymid + torch.randint(-cH//8,cH//8,(S,),device=device)
            xmid = torch.clamp(xmid, cW//2, W-cW//2)
            ymid = torch.clamp(ymid, cH//2, H-cH//2)
            
            # smooth out
            xmid_, ymid_ = xmid.clone(), ymid.clone()
            for _ in range(32):
                for si in range(1,S-1):
                    xmid_[si] = (xmid[si-1] + xmid[si] + xmid[si+1])/3.0
                    ymid_[si] = (ymid[si-1] + ymid[si] + ymid[si+1])/3.0
            xmid, ymid = xmid_.clone(), ymid_.clone()

            # print('x0', x0.shape)
            # print('x1', x1.shape)
            # print('y0', y0.shape)
            # print('y1', y1.shape)
            
            x0, x1 = torch.clamp(xmid-cW//2, 0), torch.clamp(xmid+cW//2, 0, W)
            y0, y1 = torch.clamp(ymid-cH//2, 0), torch.clamp(ymid+cH//2, 0, H)
            
            assert(torch.all(x1-x0==cW))
            assert(torch.all(y1-y0==cH))

            # print('new bbox', x0, y0, x1, y1)
            offset = torch.stack([x0, y0], dim=1) # S,2
            # print('new bbox', x0, y0, x1, y1, 'offset', offset)

            # print('xy0 xy1', x0.shape, y0.shape, x1.shape, y1.shape, 'offset', offset.shape, 'xys_g_local', xys_g.shape)

            rgbs_local = []
            masks_local = []
            for si in range(S):
                rgbs_local.append(rgbs[si,:,y0[si]:y1[si],x0[si]:x1[si]])
                masks_local.append(masks[si,:,y0[si]:y1[si],x0[si]:x1[si]])
            rgbs_local = torch.stack(rgbs_local, dim=0)
            masks_local = torch.stack(masks_local, dim=0)
                
            xys_g_local = xys_g - offset
            lts_g_local = lts_g - offset
            rbs_g_local = rbs_g - offset
            

        # revise the data now that we cropped: 
        # going outside borders means invis
        xs_val0 = xys_g_local[:,0] >= 0
        xs_val1 = xys_g_local[:,0] <= cW-1
        ys_val0 = xys_g_local[:,1] >= 0
        ys_val1 = xys_g_local[:,1] <= cH-1
        vis_g_local = vis_g * xs_val0.float() * xs_val1.float() * ys_val0.float() * ys_val1.float()
        # clamp the xy for fair regression
        xys_g_local[:,0] = xys_g_local[:,0].clamp(0,cW-1)
        xys_g_local[:,1] = xys_g_local[:,1].clamp(0,cH-1)
        lts_g_local[:,0] = lts_g_local[:,0].clamp(0,cW-1)
        lts_g_local[:,1] = lts_g_local[:,1].clamp(0,cH-1)
        rbs_g_local[:,0] = rbs_g_local[:,0].clamp(0,cW-1)
        rbs_g_local[:,1] = rbs_g_local[:,1].clamp(0,cH-1)

        rgbs = rgbs_local
        masks = masks_local
        track_g[:,0:2] = xys_g_local
        track_g[:,2:4] = lts_g_local
        track_g[:,4:6] = rbs_g_local
        track_g[:,6] = vis_g_local

        # # overwrite chan0 using cropped coords
        # valid_mask = xys_valid_local.bool() * masks_valid_local[:,0].bool()
        # seq_indices = torch.where(valid_mask)
        # x_indices = xys_g_local[seq_indices, 0].round().long()
        # y_indices = xys_g_local[seq_indices, 1].round().long()
        # masks_g_local[seq_indices, 0] *= 0
        # masks_g_local[seq_indices, 0, y_indices, x_indices] = 1

        # # always at least put the clamped coord into mask2
        # x_indices = xys_g_local[..., 0].round().long()
        # y_indices = xys_g_local[..., 1].round().long()
        # masks_g_local[torch.arange(xys_g_local.size(0)), 2, y_indices, x_indices] = 1

        H, W = cH, cW

        # we have a bias in the data where many trajs are locked at center
        # measure dist from center
        x0, y0 = track_g[:,2], track_g[:,3]
        x1, y1 = track_g[:,4], track_g[:,5]
        xc, yc = (x0+x1)/2.0, (y0+y1)/2.0
        xg, yg = track_g[:,0], track_g[:,1]
        # xyc0 = np.stack([xc,yc], axis=1).reshape(-1,2)
        # xyc1 = track_g[:,:2]
        x_dist0 = torch.min(torch.abs(xg-cW//2))
        x_dist1 = torch.min(torch.abs(xc-cW//2))
        y_dist0 = torch.min(torch.abs(yg-cH//2))
        y_dist1 = torch.min(torch.abs(yc-cH//2))
        if np.random.rand() < 0.25 and ((x_dist0<4 and y_dist0<4) or (x_dist1<4 and y_dist1<4)):
            # print('dists:', x_dist0, y_dist0, x_dist1, y_dist1)
            # skipping this bc too easy
            return fake_sample
        
        x_dist0 = torch.mean(torch.abs(xg-cW//2))
        x_dist1 = torch.mean(torch.abs(xc-cW//2))
        y_dist0 = torch.mean(torch.abs(yg-cH//2))
        y_dist1 = torch.mean(torch.abs(yc-cH//2))
        if np.random.rand() < 0.5 and ((x_dist0<8 and y_dist0<8) or (x_dist1<8 and y_dist1<8)):
            # print('dists:', x_dist0, y_dist0, x_dist1, y_dist1)
            # skipping this bc too easy
            return fake_sample

        
            
        
        # xy_ref = np.array((cH//2,cW//2).reshape(1,2))
        # dist0 = np.linalg.norm(xyc0-xy_ref, axis=1)
        # dist1 = np.linalg.norm(xyc1-xy_ref, axis=1)
        # print('dist0', dist0, 'dist1', dist1)
        # if np.random.rand() < 0.5 and (dist0 < 1 or dist1 < 1):
        #     # skipping this bc too easy
        #     return fake_sample


        # add channels for corners
        masks = torch.cat([masks, 0*masks[:,:2]], dim=1) # S,mC,H,W

        # completely rewrite mask0
        # we will only use this channel in point datasets
        # also write in mask3,mask4
        xys_valid = track_g[:,7]
        whs_valid = track_g[:,8]
        for sj in range(S):
            masks[sj,0] *= 0
            xc, yc = track_g[sj,0].long(), track_g[sj,1].long()
            x0, y0 = track_g[sj,2].long(), track_g[sj,3].long()
            x1, y1 = track_g[sj,4].long(), track_g[sj,5].long()
            if xys_valid[sj]>0:
                masks[sj,0,yc,xc] = 1
            masks[sj,3,y0,x0] = 1
            masks[sj,4,y1,x1] = 1

        # we need to add info on when to supervise corners
        masks_valid = torch.zeros((S,mC), dtype=torch.float32)
        masks_valid[:,:3] = track_g[:,10:13]
        if torch.sum(masks_valid[:,0]) > 0: # point dataset
            if dname in ['bioparticle','cattlepoint','horse','autotraj']: # size is reliable
                masks_valid[:,3] = xys_valid
                masks_valid[:,4] = xys_valid
            else:
                # supervise the anchor step and its neighbors, and let the rest go free
                ara = np.arange(max(anchor_ind-2,0), min(anchor_ind+2,S))
                masks_valid[ara,3] = xys_valid[ara]
                masks_valid[ara,4] = xys_valid[ara]
        else:
            # supervise every step that has a nonzero modal mask
            # (note this includes box datasets, since those put 0.5 masks)
            # also note i don't totally trust bboxes_valid, since some datasets were exported before this strat was decided
            for sj in range(S):
                if torch.sum(masks[sj,1] > 0) > 1:
                    masks_valid[sj,3] = 1
                    masks_valid[sj,4] = 1
        # since we want supervision in the form of masks,
        # let's immediately set invalid masks as ignores
        masks_valid_ = masks_valid.reshape(S,mC,1,1).repeat(1,1,H,W)
        masks[masks_valid==0] = 0.5
        
        # fill out annotations where we can
        for sj in range(S):
            mask0 = masks[sj,0] # point
            mask1 = masks[sj,1] # modal mask
            mask2 = masks[sj,2] # amodal mask

            # put mask1 into mask2 when it's valid
            if masks_valid[sj,1]==1:
                if masks_valid[sj,0]==0:
                    # note mask0 is special bc it's amodal point tracking,
                    # so we only use mask1 when the point mask is invalid
                    mask0[mask1 < 0.5] = 0
                mask2[mask1 > 0.5] = 1

            if torch.max(mask0) < 1 and masks_valid[sj,3]>0:# or torch.sum(mask1==1)>0):
                # print('torch.max(mask0)', torch.max(mask0), 'masks_valid[sj,3]', masks_valid[sj,3])
                # ask for box-center pos
                x0, y0 = track_g[sj,2], track_g[sj,3]
                x1, y1 = track_g[sj,4], track_g[sj,5]
                xc, yc = (x0+x1)/2.0, (y0+y1)/2.0
                mask0[yc.long(),xc.long()] = 1
                
            # always put mask0 into mask2,
            # since mask0 often includes useful cropped coords,
            # and both of these masks should be amodal
            mask2[mask0 > 0.5] = mask0[mask0 > 0.5]
                
            if masks_valid[sj,2]==1:
                mask0[mask2 < 0.5] = 0
                mask1[mask2 < 0.5] = 0
                
            masks[sj,:3] = torch.stack([mask0, mask1, mask2])
        masks_usable = torch.ones_like(masks_valid)
        # masks_valid[:] = 1 # now all masks are valid i think < no, i want to keep this as-is

        # fill in some amodal supervision heuristically
        # this is a bit expensive, and i don't need it on every sample:
        if np.random.rand() < 0.01:
            if torch.sum(masks_valid[:,2])==0:
                # when we do not have amodal supervision,
                # we still want some negatives,
                # so we will dilate the mask a lot,
                # and use the exterior as negative
                masks1_fat = utils.improc.dilate2d((masks[:,1:2] > 0.5).float(), times=16).clamp(0,1)
                masks2 = masks[:,2:3]
                masks1_fat = ((masks1_fat>0.5).float() + (masks2>0.5).float()).clamp(0,1)
                masks2[masks1_fat==0] = 0
                masks[:,2:3] = masks2
                
        if masks_valid[anchor_ind,0]==1:
            prompt = masks[anchor_ind,0] # H,W
        elif masks_valid[anchor_ind,1]==1:
            prompt = masks[anchor_ind,1] # H,W
        else:
            assert(masks_valid[anchor_ind,2]==1)
            prompt = masks[anchor_ind,2] # H,W
        prompt = (prompt > 0.1).float()
        prompt_sum = (prompt==1).float().sum()
        
        if prompt_sum==0:
            print('somehow sum(prompt)==0 in folder %s; returning fake' % folder)
            return fake_sample

        if np.random.rand() < 0.8: # replace perfect prompt with an imperfect one
            if np.random.rand() < 0.5:
                # boxify
                if prompt_sum > 16:
                    prompt_bak = prompt.clone()
                    # erode & dilate, to get rid of stragglers before we boxify
                    prompt = utils.improc.dilate2d(utils.improc.erode2d(prompt.reshape(1,1,H,W)))[0,0]
                    if torch.sum(prompt)==0: # undo
                        prompt = prompt_bak
                prompt = torch.from_numpy(bbox2mask(mask2bbox(prompt.numpy()), cW, cH))

            prompt_bak = prompt.clone()
                
            # either erode or dilate
            times = np.random.randint(4)+1
            if np.random.rand() < 0.5:
                prompt = utils.improc.erode2d(prompt.reshape(1,1,H,W), times=times)[0,0]
            else:
                prompt = utils.improc.dilate2d(prompt.reshape(1,1,H,W), times=times)[0,0]
                
            if torch.sum(prompt)==0: # undo
                prompt = prompt_bak

        # if we are in a mask dataset,
        # with some small prob,
        # use a point prompt instead of a mask prompt,
        # so that the model learns to segment at the mask level during point pred.
        # and note this requires also changing the point supervision, to no longer have a 1 at the box center
        mask = masks[anchor_ind,1]
        if np.random.rand() < 0.1 and torch.sum(masks_valid[:,0])==0 and torch.sum(mask>0.5) > 8: # mask we can convert into a point prompt
            ys, xs = np.where(mask)
            ind = np.random.permutation(len(xs))[0]
            y, x = ys[ind], xs[ind]
            prompt = torch.zeros_like(prompt)
            prompt[y,x] = 1

            times = np.random.randint(4)+1
            prompt = utils.improc.dilate2d(prompt.reshape(1,1,H,W), times=times)[0,0]

            # rewrite mask annotations
            for sj in range(S):
                mask0 = masks[sj,0] # point mask
                mask1 = masks[sj,1] # modal mask
                mask3 = masks[sj,3] # tl mask
                mask4 = masks[sj,4] # br mask
                
                mask0[mask1<0.5] = 0
                mask0[mask1==1] = 0.5
                
                if sj==anchor_ind:
                    mask0[y,x] = 1

                # also corners should be 0 outside the mask
                mask3[mask1<0.5] = 0
                mask3[mask1==1] = 0.5
                mask4[mask1<0.5] = 0
                mask4[mask1==1] = 0.5

                masks[sj,0] = mask0
                masks[sj,3] = mask3
                masks[sj,4] = mask4
                
            track_g[:,7] = 0 # xys invalid
            track_g[:,8] = 0 # ltrbs invalid
                
        prompts = torch.zeros((S,1,H,W), dtype=torch.float32)
        prompts[anchor_ind,0] = prompt

        
        # if np.random.rand() < 0.8: # replace perfect prompt with an imperfect one

        
        # prompt_py = prompt_bak[0,0].numpy() # H,W
        # if torch.sum((masks[si]==1).float()) > 16 and np.random.rand() < 0.1:
        #     # we have an actual mask available
        #     # we want to prompt using a point inside the mask
        #     ys, xs = np.where(prompt_py)
        #     # print('ys, xs', len(ys), len(xs)) 
        #     if len(ys):
        #         ii = np.random.randint(len(ys))
        #         y, x = ys[ii], xs[ii]
        #         prompt_py *= 0
        #         prompt_py[y,x] = 1

        #         w = 1
        #         h = 1
        #         if np.random.rand() < 0.5:
        #             prompt_py = cv2.dilate(prompt_py.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(np.float32)
        #             w += 1
        #             h += 1
        #         if np.random.rand() < 0.5:
        #             prompt_py = cv2.dilate(prompt_py.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(np.float32)
        #             w += 1
        #             h += 1

        #         # xywhs_g = track_g[:,:,:4]
        #         # xys_g = track_g[:,:,0:2]
        #         # whs_g = track_g[:,:,2:4]
        #         # vis_g = track_g[:,:,4]
        #         # xys_valid = track_g[:,:,5] # B,S
        #         # whs_valid = track_g[:,:,6] # B,S
        #         # vis_valid = track_g[:,:,7] # B,S

        #         track_g[si,0] = x * prompt_stride
        #         track_g[si,1] = y * prompt_stride
        #         track_g[si,2] = w * prompt_stride
        #         track_g[si,3] = h * prompt_stride

        #         # disable xy and wh supervision beyond the prompt step
        #         track_g[:,5] = 0 # xy_valid
        #         track_g[si,5] = 1 # xy_valid
        #         track_g[:,6] = 0 # wh_valid
        #         track_g[si,6] = 1 # wh_valid
                
        #         prompt = torch.from_numpy(prompt_py)
        #         prompts[si,0] = prompt
            
            

        # # focus the supervision onto a region around the GT
        # masks_any = (masks>0.1).float()
        # valid_masks = torch.ones_like(masks)
        # valid_masks_narrow = utils.improc.dilate2d(masks_any.reshape(S*3,1,fH,fW), times=8).reshape(S,3,fH,fW).clamp(0,1)
        # for si in range(S):
        #     for mi in range(3):
        #         if torch.sum(masks_any[si,mi])>0:
        #             # print('narrowing', si, mi)
        #             valid_masks[si,mi] = valid_masks_narrow[si,mi]
        #         # else:
        #         #     print('not narrowing', si, mi)
        # # masks = masks.reshape(-1)
        # # valid_masks = valid_masks.reshape(-1)
        # masks[valid_masks==0] = 0.5

        rgbs = rgbs / 255.0
        rgbs = (rgbs - self.mean)/self.std

        step = torch.zeros((), dtype=torch.int32)

        # # make masks 2chan (partial and full)
        # masks = masks[:,:2]

        # print('rgbs_local', rgbs_local.shape)
        # print('masks_g_local', masks_g_local.shape)
        # print('prompts_local', prompts_local.shape)
        # print('prompts_local', prompts_local.shape)
        # print('cH, cW', cH, cW)

        if self.prompt_stride > 1:
            prompts = F.max_pool2d(prompts, kernel_size=self.prompt_stride, stride=self.prompt_stride)
            prompts = prompts.reshape(S,1,cH//self.prompt_stride,cW//self.prompt_stride)

        if self.mask_stride > 1:
            masks = F.max_pool2d(masks, kernel_size=self.mask_stride, stride=self.mask_stride) # preserve positives this way
            masks = masks.reshape(S,mC,cH//self.mask_stride,cW//self.mask_stride)

        # add small ignore region for all point and corner masks
        for mi in [0,3,4]:
            masksm = masks[:,mi:mi+1]
            # turn any existing 1 into a a wider 0.5
            masksm_fat = utils.improc.dilate2d((masksm > 0.5).float(), times=1).clamp(0,1) * 0.5
            # keep the original 0.5s
            masksm_fat[masksm==0.5] = 0.5
            # keep the original 1s
            masksm_fat[masksm==1] = 1
            # masksm[masksm_fat==1] += 0.5
            # masksm = masksm.clamp(0,1)
            masks[:,mi:mi+1] = masksm_fat


        sample = {
            'rgbs': rgbs,
            'masks_g': masks,
            # 'masks_valid': masks_valid,
            'prompts_g': prompts,
            'track_g': track_g,
            # 'xywhs_e0': xywhs_e0,
            'dname': dname,
            # 'folder': folder.split('/')[-1],
            # 'index': index,
            'step': step,
        }
        return sample
        
    def __getitem__(self, index):
        gotit = False
        fails = 0
        while not gotit and fails < 4:
            samp = self.getitem_helper(index)
            if samp['dname'] != 'none':
                gotit = True
            else:
                fails += 1
                index = np.random.randint(len(self.all_folder_names))
        if fails > 2:
            print('note: sampling failed %d times' % fails)
        return samp

    def __len__(self):
        return len(self.all_folder_names)


# unit test
if __name__ == '__main__':
    ds = ExportDataset('/orion/u/aharley/datasets/alltrack_export', version='pq_trainA', use_augs=True, S=32, final_stride=4, prompt_stride=2, quick=False, random_anchor=True)
    for d in ds:
        print(d['rgbs'].shape)
        break
