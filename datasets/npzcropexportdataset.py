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
import zipfile
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
# np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')
# import torchvision.transforms.functional as F
from einops import rearrange, repeat

# def get_crop(im, boxlist_pt, PH, PW):
#     crops = []
#     for i in range(im.shape[0]):
#         x1, y1, x2, y2 = boxlist_pt[i].squeeze().int().tolist()
#         h, w = im[i].shape[-2:]
#         # Pad if we have negative values, or the bottom right coordinate goes beyond the image.
#         pad_left = max(-x1, 0)
#         pad_right = max(x2 - w, 0)
#         pad_top = max(-y1, 0)
#         pad_bottom = max(y2 - h, 0)
#         pad = [pad_left, pad_top, pad_right, pad_bottom]
#         curr_im = im[i]
#         if any(p > 0 for p in pad):
#             # print(f"{pad=}, ({x1=}, {y1=}, {x2=}, {y2=}), ({w=}, {h=})")
#             curr_im = F.pad(curr_im, pad, padding_mode="constant", fill=0)
#             x1 += pad_left
#             x2 += pad_left
#             y1 += pad_top
#             y2 += pad_top
#             # print(f"{x1=}, {y1=}, {x2=}, {y2=}, {i=}, {boxlist_pt.shape=}, {curr_im.shape=}, {im[i].shape=}")
#         crops.append(F.resized_crop(curr_im, top=y1, left=x1, height=x2 - x1, width=y2 - y1, size=(PH, PW), interpolation=0))
#         # crops.append(F.resized_crop(curr_im, top=y1, left=x1, height=x2 - x1, width=y2 - y1, size=(PH, PW)))
#     return crops


def get_multiscale_crops(images_, xywhs_, cH, cW, scales=[0.25,1.0]):
    B,C,H,W = images_.shape
    B2,D = xywhs_.shape
    assert(B==B2)
    assert(D==4)
    all_crops = []
    for si, sc in enumerate(scales):
        boxlist_ = utils.geom.get_boxlist_from_centroid_and_size(
            xywhs_[:,1],
            xywhs_[:,0],
            xywhs_[:,3].clamp(min=64)/sc,
            xywhs_[:,2].clamp(min=64)/sc,
        ).unsqueeze(1) # B,1,4
        # crops_ = get_crop(images_, boxlist_, cH, cW)
        # crops = torch.stack(crops_, dim=0)
        crops_ = utils.geom.crop_and_resize(images_, boxlist_, cH, cW) # B,3,cH,cW
        crops = crops_.reshape(B,C,cH,cW)
        all_crops.append(crops)
    all_crops = torch.stack(all_crops, dim=1) # B,N,C,cH,cW
    return all_crops


def read_mp4(fn):
    try:
        vidcap = cv2.VideoCapture(fn)
        frames = []
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


def check_file_status(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        return False
    
    try:
        # Try to open the file in exclusive mode (depending on the OS)
        with open(file_path, 'a', os.O_EXCL) as file:
            return True
    except OSError as e:
        # If an OSError is raised, it could be due to the file being used by another process
        return False
    
    
class ExportDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_location='../datasets/alltrack_export',
                 version='au_trainA',
                 dsets=None,
                 use_estimates=False,
                 S=32,
                 rand_frames=False,
                 crop_size=(384,384), 
                 horz_flip=False,
                 vert_flip=False,
                 time_flip=False,
                 use_augs=False,
                 is_training=True,
    ):
        print('loading npzcrop export...')

        self.dataset_location = dataset_location
        self.S = S
        self.H, self.W = crop_size
        self.horz_flip = horz_flip
        self.vert_flip = vert_flip
        self.time_flip = time_flip
        self.dataset_location = Path(self.dataset_location) / version
        self.use_estimates = use_estimates

        self.mean = torch.as_tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1).float()
        self.std = torch.as_tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1).float()

        dataset_names = self.dataset_location.glob('*/')
        self.dataset_names = [fn.stem for fn in dataset_names]

        print('dataset_names', self.dataset_names)

        folder_names = self.dataset_location.glob('*/*/*/')
        # folder_names = [fn for fn in folder_names if (fn.stem[-1] != '9' if is_training else fn.stem[-1] == '9')]
        
        folder_names = [str(fn) for fn in folder_names]

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

        self.all_folder_names = []
        self.all_idxs = []

        # self.all_folder_names = folder_names

        # step through once and make sure all of the npzs are there
        self.all_idxs = []
        self.all_folder_names = []
        for fi, folder in enumerate(folder_names):
            if os.path.isfile('%s/track.npz' % folder):
                self.all_idxs.append(np.arange(self.S))
                self.all_folder_names.append(folder)
            else:
                print('missing track in %s' % folder)
        
        print('found {:d} {} samples in {}'.format(len(self.all_folder_names), version, self.dataset_location))
        
    def __getitem__(self, index):
        folder = self.all_folder_names[index]
        # print('folder', folder)
        
        scales = [0.25,1.0]
        N = len(scales)
        cH, cW = 128, 128
        stride = 32
        sH, sW = cH//stride, cW//stride
        final_stride = 4
        fH, fW = cH//final_stride, cW//final_stride

        fake_sample = {
            'crop_rgbs': torch.zeros((self.S, N, 3, cH, cW), dtype=torch.float32), 
            'crop_masks': torch.zeros((self.S, 1, fH, fW), dtype=torch.float32), 
            'crop_xys': torch.zeros((self.S, N, 2, sH, sW), dtype=torch.float32), 
            'crop_valids': torch.zeros((self.S, N, 1, sH, sW), dtype=torch.float32), 
            'track_g': torch.zeros((self.S, 8), dtype=torch.float32), 
            'xywhs_e': torch.zeros((self.S, 4), dtype=torch.float32), 
            'dname': 'none',
            'folder': 'none',
            'step': torch.zeros((), dtype=torch.int32), 
        }
        
        try: 
            d = dict(np.load(folder + '/track.npz', allow_pickle=True))
        except: # i've seen zipfile and zlib
            d = None
        if d is None:
            print('some problem with folder', folder)
            return fake_sample

        rgbs = d['rgbs']
        masks = d['masks_g']
        track = d['track_g']
        dname = str(d['dname'])
        
        if len(rgbs) > self.S:
            rgbs = rgbs[:self.S]
            masks = masks[:self.S]
            track = track[:self.S]

        masks_pos = (masks > 192.0) * 1.0
        masks_ign = (masks > 64.0) * 1.0
        masks = (masks_pos + masks_ign)/2.0
        
        assert(not np.any(np.isnan(rgbs)))
        assert(not np.any(np.isnan(masks)))
        assert(not np.any(np.isnan(track)))

        rgbs = torch.from_numpy(rgbs).float().unsqueeze(0)
        masks = torch.from_numpy(masks).float().unsqueeze(0)
        track_g = torch.from_numpy(track).float().unsqueeze(0)
        device = rgbs.device

        B,S,C,H,W = rgbs.shape
        assert(self.H==H)
        assert(self.W==W)

        xywhs_g = track_g[:,:,:4]

        # touching any border means invalid
        xys_valid = track_g[:,:,5]
        xs_val0 = xywhs_g[:,:,0] >= 1
        xs_val1 = xywhs_g[:,:,0] <= W-2
        ys_val0 = xywhs_g[:,:,1] >= 1
        ys_val1 = xywhs_g[:,:,1] <= H-2
        xys_valid = xys_valid * (xs_val0.float() * xs_val1.float() * ys_val0.float() * ys_val1.float())
        track_g[:,:,5] = xys_valid

        # init an estimate
        xs_e = torch.rand((S), dtype=torch.float32)*(W-1)
        ys_e = torch.rand((S), dtype=torch.float32)*(H-1)
        ws_e = torch.rand((S), dtype=torch.float32)*min(H-1,W-1)+1
        hs_e = torch.rand((S), dtype=torch.float32)*min(H-1,W-1)+1
        xywhs_e = torch.stack([xs_e, ys_e, ws_e, hs_e], dim=1).unsqueeze(0) # 1,S,4

        # move the whole traj a random amount toward gt
        coeff = torch.rand((1), dtype=torch.float32) # 0 to 1
        xywhs_e = xywhs_e*coeff + xywhs_g*(1-coeff)
        # xywhs_e = xywhs_g.clone()

        xys_ = utils.basic.meshgrid2d(B*S, H, W, norm=False, device=device, stack=True, on_chans=True)
        xyvs_ = torch.cat([xys_, torch.ones_like(xys_[:,0:1])], dim=1) 
        rgbs_ = rgbs.reshape(B*S,3,H,W)
        masks_ = masks.reshape(B*S,1,H,W)

        step = torch.zeros((), dtype=torch.int32)

        # since we use distributed sampler, each process should see different subsets,
        # and within each epoch, we are not going to read the same data that is concurrently written by the model
        # we should not be reading a currently writing file, but still check
        if self.use_estimates:
            pth = '%s/xywhs_e.npz' % folder
            if check_file_status(pth):
                try:
                    data = np.load(pth, allow_pickle=True)
                    prev_xywhs_e = data['xywhs_e']
                    prev_loss = data['loss']
                    prev_step = data['step']
                    # print('found an xywh: step %d, loss %.1f' % (prev_step, prev_loss))
                    if prev_loss > 1.0 and prev_step < 32:
                        # print('> using it')
                        xywhs_e = torch.from_numpy(prev_xywhs_e).float().reshape(1,S,4)
                        step = torch.from_numpy(prev_step).reshape([])
                    else: 
                        print('> discarding an xywh: dname %s, step %d, loss %.1f' % (dname, prev_step, prev_loss))
                        # print('> discarding it')
                        # delete the file
                        os.remove(pth)
                except:
                    print('> some problem with', pth, '; discarding it')
                    # delete the file
                    os.remove(pth)
            else:
                # print('%s not found' % pth)
                pass

        # clip to bounds
        xywhs_e[:,:,0] = xywhs_e[:,:,0].clamp(min=0,max=W-1)
        xywhs_e[:,:,1] = xywhs_e[:,:,1].clamp(min=0,max=H-1)
        xywhs_e[:,:,2:] = xywhs_e[:,:,2:].clamp(min=1)

        # always lock time0
        xywhs_e[:,0] = xywhs_g[:,0]

        xywhs_e_ = xywhs_e.reshape(B*S,4)

        rgbmxyvs_ = torch.cat([rgbs_, masks_, xyvs_], dim=1)
        crop_rgbmxyvs_ = get_multiscale_crops(rgbmxyvs_, xywhs_e_, cH, cW, scales=scales).reshape(B*S*N,7,cH,cW)
        crop_rgbs_ = crop_rgbmxyvs_[:,:3]
        crop_masks_ = crop_rgbmxyvs_[:,3:4]
        crop_xyvs_ = crop_rgbmxyvs_[:,4:7]

        center_xys_ = xywhs_e_[:,:2].reshape(B,S,1,2).repeat(1,1,N,1).reshape(B*S*N,2,1,1)
        crop_xyvs_[:,:2] = crop_xyvs_[:,:2] - center_xys_
        crop_xyvs_[:,2:3] = (crop_xyvs_[:,2:3] > 0.5).float()
        crop_xyvs_[:,:2] = crop_xyvs_[:,:2] * crop_xyvs_[:,2:3]

        wid = xywhs_e[:,:,2].clamp(min=64)/scales[0]
        hei = xywhs_e[:,:,3].clamp(min=64)/scales[0]
        # get the bounds of our widest zoom level
        x0 = xywhs_e[:,:,0]-wid/2
        y0 = xywhs_e[:,:,1]-hei/2
        x1 = xywhs_e[:,:,0]+wid/2
        y1 = xywhs_e[:,:,1]+hei/2
        sx = wid/cW
        sy = hei/cH
        sc = torch.stack([sx,sy], dim=-1).reshape(1,S,2)

        vis_g = track_g[:,:,4]
        for si in range(S):
            xg = xywhs_g[0,si,0]
            yg = xywhs_g[0,si,1]
            # if the point is outside the widest zoom, call it invis
            if xg < x0[0,si] or xg > x1[0,si] or yg < y0[0,si] or yg > y1[0,si]:
                vis_g[0,si] = 0
        track_g[:,:,4] = vis_g

        crop_xyv_ims = crop_xyvs_.reshape(B,S,N,3,cH,cW)[0]
        crop_xys = crop_xyv_ims[:,:,:2] # S,N,2,cH,cW
        crop_valids = crop_xyv_ims[:,:,2:3] # S,N,1,cH,cW

        crop_xys_ = crop_xys.reshape(S*N,2,cH,cW)
        crop_valids_ = crop_valids.reshape(S*N,1,cH,cW)

        crop_xys_ = rearrange(crop_xys_, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=stride, p2=stride) # S*N,sH*sW,patches,2
        crop_valids_ = rearrange(crop_valids_, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=stride, p2=stride)

        # average up the valid xys inside
        crop_xys_ = utils.basic.reduce_masked_mean(crop_xys_, crop_valids_.repeat(1,1,1,2), dim=-2) # S*N,sH*sW,2
        crop_valids_ = crop_valids_.max(dim=-2)[0] # S*N,sH*sW,1
        crop_valids_ = (crop_valids_ > 0).float()

        # apply mean/std
        crop_rgbs_ = crop_rgbs_ / 255.0
        crop_rgbs_ = (crop_rgbs_ - self.mean)/self.std

        crop_rgbs = crop_rgbs_.reshape(S,N,3,cH,cW)
        crop_masks = crop_masks_.reshape(S,N,1,cH,cW)
        crop_xys = crop_xys_.reshape(S,N,2,sH,sW)
        crop_valids = crop_valids_.reshape(S,N,1,sH,sW)

        xywhs_e = xywhs_e.reshape(S,4)
        track_g = track_g.reshape(S,8)

        crop_masks = crop_masks[:,-1] # last one, which is scale 1.0; S,1,cH,cW
        crop_masks = F.max_pool2d(crop_masks,
                                  kernel_size=final_stride,
                                  stride=final_stride).reshape(S,1,fH,fW)

        sample = {
            'crop_rgbs': crop_rgbs,
            'crop_masks': crop_masks,
            'crop_xys': crop_xys,
            'crop_valids': crop_valids,
            'track_g': track_g,
            'xywhs_e': xywhs_e,
            'dname': dname,
            'folder': folder,
            'step': step, 
        }
        return sample

    def __len__(self):
        return len(self.all_folder_names)


