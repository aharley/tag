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
from datasets.dataset import MaskDataset
from datasets.dataset_utils import make_split
from tqdm import tqdm

# np.random.seed(125)
# torch.multiprocessing.set_sharing_strategy('file_system')

class BEDLAMSEGDataset(MaskDataset):
    def __init__(self,
                 dataset_location='/orion/group/BEDLAM/',
                 S=32, fullseq=False, chunk=None,
                 crop_size=(384, 512),
                 strides=[2], # stride1 often too easy; stride3 sometimes too hard
                 zooms=[1,2],
                 use_augs=False,
                 is_training=True,
    ):
        print('loading BEDLAM dataset...')
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training
        )
        self.dataset_location = os.path.join(dataset_location, 'be_imagedata_download')
        self.annots_location = os.path.join(dataset_location, 'BEDLAM/data_processing/bedlam_data/processed_labels')

        annot_paths = sorted(os.listdir(os.path.join(self.dataset_location, self.annots_location)))
        print('len(annot_paths)', len(annot_paths))

        if chunk is not None:
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            annot_paths = chunkify(annot_paths,100)[chunk]
            print('filtered to %d' % len(annot_paths))
        else:
            print('chunk', chunk)

        self.all_rgb_names = []
        self.all_mask_names1 = []
        self.all_mask_names2 = []
        self.all_mask_names3 = []
        self.all_annots = []
        self.all_zooms = []
        clip_step = S//2
        H, W = 720, 1280
        
        for annot_path in annot_paths:
            spl = annot_path.split('_')
            dir_here = '_'.join(spl[:-3])
            seq_here = '_'.join(spl[-3:-1])
            try:
                body_id = int(spl[-1].split('.')[0])
            except Exception as e:
                print('spl', spl)
                print('e', e) 
                continue

            di = dict(np.load('%s/%s' % (self.annots_location, annot_path)))
            img_names = di['imgname']
            img_names = [img.split('/')[1] for img in img_names]
            img_names = [img.split('.')[0] for img in img_names]

            rgb_names = []
            mask_names1 = []
            mask_names2 = []
            mask_names3 = []
            broken = False
            for img in img_names:
                rgb_name = '%s/%s/png/%s/%s.png' % (self.dataset_location, dir_here, seq_here, img)
                mask_name1 = '%s/%s/masks/%s/%s_%02d_body.png' % (self.dataset_location, dir_here, seq_here, img, body_id)
                mask_name2 = '%s/%s/masks/%s/%s_%02d_clothing.png' % (self.dataset_location, dir_here, seq_here, img, body_id)
                mask_name3 = '%s/%s/masks/%s/%s_env.png' % (self.dataset_location, dir_here, seq_here, img)

                if not os.path.isfile(mask_name1) or not os.path.isfile(mask_name2) or not os.path.isfile(mask_name3):
                    broken = True
                
                rgb_names.append(rgb_name)
                mask_names1.append(mask_name1)
                mask_names2.append(mask_name2)
                mask_names3.append(mask_name3)

            if broken:
                continue
            
            S_local = len(mask_names1)

            for stride in strides:
                for ii in range(0, S_local, clip_step*stride):
                    full_idx = ii + np.arange(self.S) * stride
                    full_idx = [ij for ij in full_idx if ij < S_local]
                    if len(full_idx) < S: continue # fullseq

                    rgb_names_here = [rgb_names[fi] for fi in full_idx]
                    mask_names1_here = [mask_names1[fi] for fi in full_idx]
                    mask_names2_here = [mask_names2[fi] for fi in full_idx]
                    mask_names3_here = [mask_names3[fi] for fi in full_idx]

                    for zoom in zooms:
                        self.all_rgb_names.append(rgb_names_here)
                        self.all_mask_names1.append(mask_names1_here)
                        self.all_mask_names2.append(mask_names2_here)
                        self.all_mask_names3.append(mask_names3_here)
                        self.all_zooms.append(zoom)
                    sys.stdout.write('.')
                    sys.stdout.flush()
        print('loaded {} samples'.format(len(self.all_rgb_names)))

    def __len__(self):
        return len(self.all_rgb_names)

    def getitem_helper(self, index):

        video_frames = self.all_rgb_names[index]
        mask_names1 = self.all_mask_names1[index]
        mask_names2 = self.all_mask_names2[index]
        mask_names3 = self.all_mask_names3[index]
        zoom = self.all_zooms[index]

        # print('video_frames', video_frames, len(video_frames))
        # print('mask_names', mask_names, len(mask_names))
        
        # video_frames = self.all_video_names[index]
        # mask_names = self.all_mask_names[index]
        # zoom = self.all_zooms[index] 

        rgb = cv2.imread(video_frames[0])
        rgbs = []
        for path in video_frames:
            rgb = cv2.imread(str(path))[..., ::-1].copy()
            rgbs.append(rgb)
        rgbs = np.stack(rgbs)
        S,H,W,C = rgbs.shape
        # print('rgbs', rgbs.shape)
            
        masks1 = []
        masks2 = []
        masks3 = []
        visibs = []
        for si, (path1,path2,path3) in enumerate(zip(mask_names1, mask_names2, mask_names3)):
            # if os.path.exists(path1):
            mask1 = cv2.imread(str(path1), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            # else:
            #     mask1 = np.zeros((H,W), dtype=np.float32)
            # if os.path.exists(path2):
            mask2 = cv2.imread(str(path2), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            # else:
            #     mask2 = np.zeros((H,W), dtype=np.float32)

            # if os.path.exists(path2):
            mask3 = cv2.imread(str(path3), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            # else:
            #     mask3 = np.zeros((H,W), dtype=np.float32)

            mask1 = (mask1 > 0).astype(np.float32)
            mask2 = (mask2 > 0).astype(np.float32)
            mask3 = (mask3 > 0).astype(np.float32)
            masks1.append(mask1)
            masks2.append(mask2)
            masks3.append(mask3)

        
        masks1 = np.stack(masks1) # body
        masks2 = np.stack(masks2) # clothing
        masks3 = np.stack(masks3) # env
        # masks = ((masks1+masks2)>0).astype(np.float32)

        # print('masks1', np.sum(masks1))
        # print('masks2', np.sum(masks2))
        # print('masks3', np.sum(masks3))

        # # print('xys', xys, xys.shape)
        # travel = np.sum(np.linalg.norm(xys[1:] - xys[:-1], axis=-1))
        # # print('travel', travel)


        # the masks in bedlam are pretty great
        # we can get noisy body part masks by taking the best conn-comp in the _body mask,
        # > the noiise here has to do with overlapping joints 
        # and also get full-obj masks by combining the body/clothing/env maps (getting proper ignores)
        
        # masksX = np.zeros_like(masks2)
        # for si in range(len(masks1)):

        #     if masks1[si,y,x] > 0:
        #         num_labels, labels = cv2.connectedComponents(masks1[si].astype(np.uint8))
        #         best = labels[y,x]
        #         # # print('labels', labels.shape)
        #         # dists = np.zeros((num_labels))
        #         # for ni in range(num_labels):
        #         #     bin_mask = labels==ni
        #         #     ys_, xs_ = np.where(bin_mask)
        #         #     xy_ = np.stack([xs_, ys_], axis=-1).mean(axis=0)
        #         #     # print('xy', xy)
        #         #     # print('xy_', xy_)
        #         #     xy = xys[si].round().astype(np.int32)
        #         #     # xy_ = xys_
        #         #     # y_, x_ = ys_.mean(), xs_.mean()
        #         #     dist = np.linalg.norm(xy_-xy)
        #         #     dists[ni] = dist
        #         # best = np.argmin(dists)
        #         bin_mask = labels==best

        #         masksX[si] = bin_mask.astype(np.float32)
        #     else:
        #         masksX[si] = 0.5


        # # compute safe(r) part mask sup
        # # > note this is not perfect bc sometimes the smaller conncomps still contain >1 limb
        # masks_any = (np.sum(masksX, axis=0) > 0).astype(np.float32)
        # part_masks = 0.5*np.ones_like(masksX)
        # for si in range(S):
        #     mask_neg = 1-masks_any
        #     mask = 0.5*np.ones_like(masksX[si])
        #     mask[mask_neg>0] = 0
        #     mask[masks1[si]==0] = 0
        #     part_masks[si] = mask
        # # use the smallest 3 masks as hard sup
        # maskX_sums = np.sum(masksX.reshape(S,-1), axis=1)
        # maskX_sums[maskX_sums==0] = H*W # don't use the empties
        # inds = np.argsort(maskX_sums)
        # for ii in range(3):
        #     part_masks[inds[ii]] = masksX[inds[ii]]
        
        # compute safe amodal mask sup
        modal_masks = ((masks1+masks2)>0).astype(np.float32)
        masks_any = (np.sum(masks1+masks2, axis=0) > 0).astype(np.float32)
        amodal_masks = 0.5*np.ones_like(masks1)
        for si in range(S):
            mask_pos = masks1[si] + masks2[si] # body+clothing 
            # mask_neg = masks3[si] # env

            # ignore any component that overlaps with pos,
            # so that we correctly ignore occluders here
            mask_ign = 0*mask_pos
            num_labels, labels = cv2.connectedComponents((1-masks3[si]).astype(np.uint8))
            for ni in range(num_labels):
                mask = (labels==ni)
                if np.sum(mask_pos*mask) > 0:
                    mask_ign += mask
                    
            mask = 0*mask_pos
            mask[mask_ign>0] = 0.5
            mask[mask_pos>0] = 1
            # mask = 0.5*np.ones_like(masks1[si])
            # mask[mask_pos>0] = 1
            # mask[mask_neg>0] = 0
            amodal_masks[si] = mask

        # full_masks = np.stack([masks0, masks, masks], axis=-1)
        full_masks = np.stack([modal_masks, modal_masks, amodal_masks], axis=-1)

        # all chans valid when points are valid!
        # (this is maybe the only dataset where this happens)
        masks_valid = np.zeros((S,3), dtype=np.float32)
        masks_valid[:,0] = 0
        masks_valid[:,1] = 1
        masks_valid[:,2] = 1

        mask_areas = (modal_masks > 0).reshape(S,-1).sum(axis=1)
        # print('mask_areas', mask_areas)
        mask_areas_norm = mask_areas / np.max(mask_areas)
        visibs = mask_areas_norm
        valids = np.ones_like(visibs)
        # print('visibs', visibs)

        if zoom > 1:
            xys, _, valids, fills = utils.misc.data_get_traj_from_masks(modal_masks)
            xys = utils.misc.data_replace_with_nearest(xys, valids)
            xys, visibs, valids, rgbs, full_masks = utils.misc.data_zoom(zoom, xys, visibs, valids, rgbs, full_masks)
        S, H, W, _ = rgbs.shape
        
        visibs = 1.0 * (visibs>0.5)
        safe = np.concatenate(([0], visibs[:-2] * visibs[1:-1] * visibs[2:], [0]))
        if np.sum(safe) < 2:
            print('pre-zoom safe', safe)
            return None
        
        sample = {
            'rgbs': rgbs,
            'masks': full_masks,
            'masks_valid': masks_valid,
        }
        return sample
