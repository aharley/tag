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
from datasets.dataset import MaskDataset, mask2bbox, get_crop_around_bbox
from icecream import ic
from pycocotools.coco import COCO
import albumentations as A

class COCODataset(MaskDataset):
    def __init__(self,
                 dataset_location='../coco',
                 S=32, fullseq=True, chunk=None,
                 crop_size=(384,512),
                 use_augs=False,
                 is_training=True,
    ):
        print('loading COCO dataset...')
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size, 
            use_augs=use_augs,
            is_training=is_training
        )
        self.dataset_location = os.path.join(dataset_location, 'train2017' if is_training else 'val2017')
        self.coco = COCO(os.path.join(dataset_location, 'annotations/instances_{}.json'.format('train2017' if is_training else 'val2017')))
        self.image_ids = list(self.coco.imgs.keys())
        print('loaded image_ids')
        self.affine_transform = A.Compose([
            A.Affine(scale=(0.3, 0.5), translate_percent=(-0.3, 0.3), keep_ratio=True, p=1.)],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=[]))
        print(f"found {len(self.image_ids)} samples in {dataset_location}")

        if chunk is not None:
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.image_ids = chunkify(self.image_ids,100)[chunk]
            print('filtered to %d image_ids' % len(self.image_ids))
            # print('self.image_ids', self.image_ids)
        
        self.all_image_ids = []
        self.all_ann_ids = []
        for img_id in self.image_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id,iscrowd=False)
            anns = self.coco.loadAnns(ann_ids)

            if len(anns) > 0:

                ann_ids_ok = []
                ann_areas = []

                for ann_id, ann in enumerate(anns):
                    mask = self.coco.annToMask(ann)
                    mask_sum = np.sum(mask)

                    # print('mask', mask.shape)
                    H, W = mask.shape

                    bbox = mask2bbox(mask)
                    # print('bbox', bbox)

                    x0, y0, x1, y1 = bbox

                    wid = x1-x0
                    hei = y1-y0

                    # don't occupy the full frame please, and don't be too tiny
                    if wid > W*0.8 or hei > H*0.8 or wid<8 or hei<8: continue
                    if mask_sum < 256: continue
                    fill_amount = np.mean(mask[y0:y1,x0:x1])
                    if fill_amount > 0.7: continue # too boxy
                    if fill_amount < 0.3: continue # weird to indicate via box

                    ann_ids_ok.append(ann_id)
                    ann_areas.append(mask_sum)

                if len(ann_ids_ok):
                    inds = np.argsort(np.array(ann_areas))
                    ann_id = ann_ids_ok[inds[-1]]

                    self.all_image_ids.append(img_id)
                    self.all_ann_ids.append(ann_id)

                    sys.stdout.write('.')
                    sys.stdout.flush()

                                
        print(f"found {len(self.all_image_ids)} total samples in {dataset_location}")

    def getitem_helper(self, index):
        img_id = self.all_image_ids[index]
        ann_id = self.all_ann_ids[index]
        
        # load img
        img_path = os.path.join(self.dataset_location, self.coco.loadImgs(img_id)[0]['file_name'])
        rgb = cv2.imread(img_path)[..., ::-1]

        H, W = rgb.shape[:2]
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        ann = anns[ann_id]
        mask = self.coco.annToMask(ann)
        mask_sum = np.sum(mask)

        bbox = mask2bbox(mask)
        ratio = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
        
        wildness = np.random.uniform(0.0, 0.2)
        ratios = np.full(self.S, ratio)
        
        visibs = np.full(self.S, 1)
        rgbs = np.stack([rgb for _ in range(self.S)])
        bboxes = np.stack([bbox for _ in range(self.S)])
        masks = np.stack([mask for _ in range(self.S)])

        # augmenting further seems unnecessary
        # def augment(augmenter, **kwargs):
        #     keys = kwargs.keys()
        #     for i in range(len(next(iter(kwargs.values())))):
        #         data = augmenter(**{
        #             key: kwargs[key][i] if key not in ['bboxes', 'keypoints'] else [kwargs[key][i]] for key in keys
        #         })
        #         for key in keys:
        #             if key == 'bboxes':
        #                 if len(data[key]) == 0:
        #                     kwargs[key][i] = np.array([0, 0, 1, 1])
        #                 else:
        #                     kwargs[key][i] = np.array(data[key]).reshape(4)
        #             elif key == 'keypoints':
        #                 kwargs[key][i] = np.array(data[key]).reshape(2)
        #             else:
        #                 kwargs[key][i] = data[key]
        # bboxes_raw = bboxes.copy()
        # augment(self.affine_transform, image=np.zeros_like(rgbs), bboxes=bboxes)
        # bboxes[:, 2:] = np.maximum(bboxes[:, 2:], bboxes[:, :2] + 1)

        bboxes_crop = get_crop_around_bbox(bboxes, rgbs.shape[-2], rgbs.shape[-3], ratios, min_keep=1.)
        bboxes_crop = wildness * bboxes_crop + (1 - wildness) * bboxes_crop[0]
        bboxes_crop = bboxes_crop.astype(int)
        bboxes_crop[:, 2:] = np.maximum(bboxes_crop[:, 2:], bboxes_crop[:, :2] + 1)

        # expand the crops, so that we do not zoom in a crazy amount  
        H, W, C = rgbs[0].shape
        bboxes_crop[:,:2] = np.maximum(bboxes_crop[:,:2]-512, np.array([0,0]))
        bboxes_crop[:,2:] = np.minimum(bboxes_crop[:,2:]+512, np.array([W, H]) - 1)
        
        # rgbs_cr = []
        # masks_cr = []

        rgbs_cr = np.zeros((self.S, self.crop_size[0], self.crop_size[1], 3))
        masks_cr = np.zeros((self.S, self.crop_size[0], self.crop_size[1]))
        coords_cr = np.zeros((self.S, self.crop_size[0], self.crop_size[1], 2))

        def meshgrid2d(Y, X):
            grid_y = np.linspace(0.0, Y-1, Y)
            grid_y = np.reshape(grid_y, [Y, 1])
            grid_y = np.tile(grid_y, [1, X])

            grid_x = np.linspace(0.0, X-1, X)
            grid_x = np.reshape(grid_x, [1, X])
            grid_x = np.tile(grid_x, [Y, 1])

            # outputs are Y x X
            return grid_y, grid_x

        ys, xs = meshgrid2d(H,W)
        coords = np.stack([xs,ys], axis=-1)
        # print('rgbs', rgbs.shape)
        # print('coords', coords.shape)
        
        # for i, bbox_crop in enumerate(bboxes_crop):
        for i, bbox_crop in enumerate(bboxes_crop):
            # if i==0:
            crop_resize = A.Compose(
                [
                    A.Crop(*bbox_crop),
                    A.Affine(rotate=(-15,15), scale=(0.75, 1.5), translate_percent=(-0.5, 0.5), keep_ratio=True, p=1.),
                    # A.Affine(rotate=(-5,5), scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), keep_ratio=True, p=1.),
                    A.Resize(self.crop_size[0], self.crop_size[1], interpolation=cv2.INTER_LINEAR)],
            )# bbox_params=A.BboxParams(format='pascal_voc', min_area=1, min_visibility=0.0, label_fields=[]))
            # print('crop_resize', crop_resize)
            

            # else:
            #     crop_resize = A.Compose(
            #         [
            #             A.Crop(*data['bboxes'][0]),
            #             # A.Affine(rotate=(-15,15), scale=(0.8, 2.0), translate_percent=(-0.1, 0.1), keep_ratio=True, p=1.),
            #             A.Affine(rotate=(-5,5), scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), keep_ratio=True, p=1.),
            #             A.Resize(self.crop_size[0], self.crop_size[1], interpolation=cv2.INTER_LINEAR)],
            #         bbox_params=A.BboxParams(format='pascal_voc', min_area=1, min_visibility=0.0, label_fields=[]))

            # data = crop_resize(image=rgbs[i], mask=masks[i], bboxes=[bboxes[i]])
            # if len(data['bboxes']) == 0:  # this box is invalid, just change turn off visib
            #     visibs[i] = 0
            #     bboxes[i] = mask2bbox(data['mask'])
            # else:
            #     bboxes[i] = data['bboxes'][0]
            data = crop_resize(image=np.concatenate([rgbs[i], coords], axis=-1), mask=masks[i])
            # print('data image', data['image'].shape)
            rgbs_cr[i] = data['image'][:,:,:3]
            coords_cr[i] = data['image'][:,:,3:]
            # masks_cr[i], coords_cr[i] = data['mask']
            masks_cr[i] = data['mask']
            # masks_cr[i] = data['mask']

            # else:
            #     crop_resize = A.Compose(
            #         [
            #             # A.Crop(*bboxes[i-1]),
            #             # A.Affine(rotate=(-5,5), scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), keep_ratio=True, p=1.), # 
            #             A.Affine(rotate=(-1,1), scale=(0.99, 1.01), translate_percent=(-0.01, 0.01), keep_ratio=True, p=1.),
            #             A.Resize(self.crop_size[0], self.crop_size[1], interpolation=cv2.INTER_LINEAR)],
            #     )#bbox_params=A.BboxParams(format='pascal_voc', min_area=1, min_visibility=0.0, label_fields=[]))
            #     data = crop_resize(image=rgbs_cr[i-1], mask=masks_cr[i-1])#, bboxes=[bboxes[i-1]])
            #     # if len(data['bboxes']) == 0:  # this box is invalid, just change turn off visib
            #     #     visibs[i] = 0
            #     #     bboxes[i] = mask2bbox(data['mask'])
            #     # else:
            #     #     bboxes[i] = data['bboxes'][0]
            #     rgbs_cr[i] = data['image']
            #     masks_cr[i] = data['mask']
            #     print('mask sum', data['mask'].sum())

                
                
        rgbs = rgbs_cr.copy()
        masks = masks_cr.copy()
        coords = coords_cr.copy()


        # mask0 = masks[0]
        # x0, y0, x1, y1 = bboxes[0]
        # fill_amount = np.mean(mask0[y0:y1,x0:x1])
        # if fill_amount < 0.2:
        #     return None
        # if masks[0].sum() < 64 or masks[1].sum() < 64:
        #     return None
        # if visibs[0] == 0 or visibs[1] == 0:
        #     return None

        S,H,W,C = rgbs.shape
        # print('H, W', H, W)

        if np.sum(masks) == 0:
            print('np.sum(masks)', np.sum(masks))
            return None

        # cumulative warping is bad, but what i can do is:
        # pick the smallest or largest element first,
        # and then sort according to similarity

        # mask_sums = np.sum(masks.reshape(S,-1), axis=1)
        # inds = np.argsort(mask_sums)
        # rgbs = rgbs[inds]
        # masks = masks[inds]
        # coords = coords[inds]

        # W_, H_ = W//4, H//4
        # masks_small = np.stack([cv2.resize(mask, (W_, H_), interpolation=cv2.INTER_AREA) for mask in masks], axis=0)
        # print('masks', masks.shape)
        # print('masks_small', masks_small.shape)

        for si in range(1,S):
            # mask0 = masks_small[si-1]
            # mask0_ = mask0.reshape(1,H_*W_)
            # masks1 = masks_small[si:].reshape(-1,H_*W_)
            coord0_ = coords[si-1].reshape(1,H*W*2)
            coords_ = coords[si:].reshape(-1,H*W*2)
            
            dists = np.abs(coord0_-coords_).sum(axis=1)
            ind = np.argmin(dists)

            bak_rgb = rgbs[si].copy()
            bak_mask = masks[si].copy()
            bak_coord = coords[si].copy()
            rgbs[si] = rgbs[si+ind]
            masks[si] = masks[si+ind]
            coords[si] = coords[si+ind]
            rgbs[si+ind] = bak_rgb
            masks[si+ind] = bak_mask
            coords[si+ind] = bak_coord
            

        mask_areas = (masks > 0).reshape(S,-1).sum(axis=1)
        mask_areas_norm = mask_areas / np.max(mask_areas)
        visibs = mask_areas_norm
         
        rgbs, masks = utils.misc.data_pad_if_necessary(rgbs, masks)
        S,H,W,C = rgbs.shape

        # # if zoom > 1:
        # xys, _, valids, fills = utils.misc.data_get_traj_from_masks(masks)
        # xys = utils.misc.data_replace_with_nearest(xys, valids)
        # zoom = 2
        # # bboxes = np.stack([mask2bbox(mask) for mask in masks])
        # # whs = bboxes[:,2:4] - bboxes[:,0:2]
        # # whs = whs[visibs > 0.5]
        # # # print('mean wh', np.mean(whs[:,0]), np.mean(whs[:,1]))
        # # if np.mean(whs[:,0])*zoom >= W and np.mean(whs[:,1])*zoom >= H:
        # #     # print('would reject')
        # #     # big already
        # #     return None
        # xys, visibs, valids, rgbs, masks = utils.misc.data_zoom(zoom, xys, visibs, valids, rgbs, masks=masks)

        visibs = (visibs > 0.5).astype(np.float32)
        safe = np.concatenate(([0], visibs[:-2] * visibs[1:-1] * visibs[2:], [0]))
        if np.sum(safe) < 2: return None
        
        return {
            'rgbs': rgbs,
            'masks': masks,
            # 'bboxes': bboxes,
            # 'visibs': visibs,
        }
    
    def __len__(self):
        return len(self.all_image_ids)
