import torch
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import albumentations as A
import numpy as np
import albumentations.augmentations.functional as F
from functools import partial
from icecream import ic
import cv2
from visdom import Visdom
# vis = Visdom(port=12345)
from albumentations.augmentations.geometric.transforms import PadIfNeeded, Affine, HorizontalFlip
# from PIL import Image
import utils.misc

# bbox rule:
# (x_min, y_min, x_max, y_max), with (x_max, y_max) excluded.
# this represents the top-left corner of corresponding indexed pixels
# e.g., a bounding box with (0, 0, 800, 800) takes the region between top-left of 0th pixel and the top-left of 800th pixel (aka bottom-right of 799th pixel)
# to get the cropped image, we just need to do img[0:800, 0:800] and we are done.

def get_crop_around_bbox(bbox, max_w, max_h, ratios, min_keep=0.6):
    x0, y0, x1, y1 = [bbox[..., i] for i in range(4)]
    w, h = x1 - x0, y1 - y0
    
    tl_max = np.stack([x1 - w * min_keep, y1 - h * min_keep], -1)
    tl = np.random.randint(np.zeros_like(tl_max), np.maximum(tl_max, 1))  # N x 2
    br_min = np.stack([x0 + w * min_keep, y0 + h * min_keep], -1)
    br_min = np.minimum(np.maximum(tl + min_keep * np.stack([w, h], -1), br_min), np.array([max_w, max_h]))
    br = np.random.randint(br_min, np.broadcast_to(np.array([max_w + 1, max_h + 1]), br_min.shape))
    w_new, h_new = np.split(br - tl, 2, -1)
    w_new, h_new = w_new[..., 0], h_new[..., 0]
    mask = w_new / h_new < ratios
    br[mask, 1] = tl[mask, 1] + h_new[mask]
    mask = w_new / h_new > ratios
    br[mask, 0] = tl[mask, 0] + w_new[mask]
    tl[:,0] = tl[:,0].clip(0,max_w-1) 
    tl[:,1] = tl[:,1].clip(0,max_h-1)
    br[:,0] = br[:,0].clip(1,max_w) 
    br[:,1] = br[:,1].clip(1,max_h)
    return np.concatenate([tl, br], -1)

# augment video sequences with the same augmenter, only support single bbox now
def augment_video(augmenter, **kwargs):
    assert isinstance(augmenter, A.ReplayCompose)
    keys = kwargs.keys()
    for i in range(len(next(iter(kwargs.values())))):
        data = augmenter(**{
            key: kwargs[key][i] if key not in ['bboxes', 'keypoints'] else [kwargs[key][i]] for key in keys
        })
        if i == 0:
            augmenter = partial(A.ReplayCompose.replay, data['replay'])
        for key in keys:
            if key == 'bboxes':
                kwargs[key][i] = np.array(data[key]).reshape(4)
            elif key == 'keypoints':
                kwargs[key][i] = np.array(data[key]).reshape(2)
            else:
                kwargs[key][i] = data[key]
    
def normalize_bbox(bboxes, max_w, max_h, denormalize=False):
    bboxes_shape = bboxes.shape
    if not denormalize:
        bboxes = (bboxes.reshape(*bboxes_shape[:-1], 2, 2) / np.array([max_w, max_h])).reshape(*bboxes_shape[:-1], 4)
    else:
        bboxes = (bboxes.reshape(*bboxes_shape[:-1], 2, 2) * np.array([max_w, max_h])).reshape(*bboxes_shape[:-1], 4)
    return bboxes
    

def mask2bbox(mask):
    assert(mask.ndim==2)
    # if mask.ndim == 3:
    #     mask = mask[..., 0]
    ys, xs = np.where(mask > 0.4)
    if ys.size == 0 or xs.size==0:
        return np.array((0, 0, 0, 0), dtype=int)
    lt = np.array([np.min(xs), np.min(ys)])
    rb = np.array([np.max(xs), np.max(ys)]) + 1
    return np.concatenate([lt, rb])

def bbox2mask(bbox, w, h):
    mask = np.zeros((h, w), dtype=np.float32)
    if (bbox[3] - bbox[1]) * (bbox[2] - bbox[0]) <= 0:
        return mask
    bbox = bbox.astype(int)
    mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1.
    return mask

    
class TrackingDataset(Dataset):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        for key, val in kwargs.items():
            setattr(self, key, val)
        self.color_augmenter = A.ReplayCompose([
            A.GaussNoise(p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.RGBShift(r_shift_limit=105, g_shift_limit=45, b_shift_limit=40, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.5),
            A.RandomGamma(gamma_limit=(80,120), p=0.5),
            A.HueSaturationValue(hue_shift_limit=172, sat_shift_limit=20, val_shift_limit=27, p=0.3),
            A.ImageCompression(quality_lower=4, quality_upper=100, p=0.4),
        ], p=0.8)
        self.hole_augmenter = A.ReplayCompose([
            A.CoarseDropout(max_height=0.1, max_width=0.1, mask_fill_value=0, p=0.25)
        ])
        self.spatial_augmenter = A.ReplayCompose([
            HorizontalFlip(p=0.5),
            Affine(scale=(0.25, 0.9), keep_ratio=True, p=1.)],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=[]),
            keypoint_params=A.KeypointParams(format='xy', label_fields=[], remove_invisible=False)
        )

    # def __augment__(self, sample):
    #     raise NotImplementedError()
        
    # def __spatial_augment__(self, sample):
    #     raise NotImplementedError()
    
    # # borrowed from https://github.com/shanice-l/gdrnpp_bop2022/blob/f3ca18632f4b68c15ab3306119c364a0497282a7/det/yolox/data/datasets/mosaicdetection.py#L198
    # def __color_augment__(self, sample):
    #     augment_video(self.color_augmenter, image=sample['rgbs'])
    
    def getitem_helper(self, index):
        raise NotImplementedError()
    
    def __pre_processing__(self, sample):
        pass
    
    def __post_processing__(self, sample):
        pass
    
    def __zigzag__(self, sample):

        S_here = len(sample['rgbs'])
        while S_here < self.S:
            for key, val in sample.items():
                sample[key] = np.concatenate([sample[key], np.flip(sample[key], axis=0)], axis=0)
            S_here *= 2
        if S_here > self.S:
            for key, val in sample.items():
                sample[key] = sample[key][:self.S]
        S_here = len(sample['rgbs'])
        return sample

        
        new_sample = {'rgbs': []}
        cur_index = 0
        direction = 1

        while len(new_sample['rgbs']) < self.S:
            # steps to go
            num_steps = np.random.randint(low=1, high=min(len(sample['rgbs']), 5) + 1)
            num_steps = min(num_steps, self.S - len(new_sample['rgbs']))

            # whether to change direction
            if np.random.rand() < 0.3:
                direction *= -1

            # we start at the current location (inclusive) and add frames sequentially
            for _ in range(num_steps):
                for key, val in sample.items():
                    if key not in new_sample:
                        new_sample[key] = []
                    new_sample[key].append(val[cur_index])

                if np.random.rand() > 0.2:
                    cur_index += direction

                # if this got too small (means we were at 0, and now at -1), let's bounce back the direction
                # and set cur_index to 1
                if cur_index <= -1:
                    direction = 1
                    cur_index = 1

                # if this got too large (means we were at S-1, and now at S), let's bounce back the direction
                # and set cur_index to S-2
                if cur_index == len(sample['rgbs']):
                    direction = -1
                    cur_index = len(sample['rgbs']) - 2

        for key, val in new_sample.items():
            sample[key] = np.stack(new_sample[key])
        return sample

    def get_fake_sample(self):
        S = self.S
        H, W = self.crop_size
        fake_sample = {
            'rgbs': np.zeros((S,3,H,W), dtype=np.uint8),
            'masks_g': np.zeros((S,3,H,W), dtype=np.float32),
            'masks_valid': np.zeros((S,3), dtype=np.float32),
            'xys_g': np.zeros((S,2), dtype=np.int32),
            'bboxes_g': np.zeros((S,2), dtype=np.int32), # this used to be 10s, and i don't know why
            'vis_g': np.zeros((S), dtype=np.float32),
            'xys_valid': np.zeros((S), dtype=np.float32),
            'bboxes_valid': np.zeros((S), dtype=np.float32),
            'vis_valid': np.zeros((S), dtype=np.float32),
        }
        return fake_sample
            
    def __getitem__(self, index):
        sample = self.getitem_helper(index)
        # print('sample', sample)
        if sample is None:
            print('sample is None!')
            return self.get_fake_sample()
        
        # if hasattr(self, 'inference') and self.inference:
        #     self.__pre_processing__(sample)
        #     self.__post_processing__(sample)
        #     return sample

        if len(sample['rgbs']) < self.S:
            # print('applying zigzag to inflate S')
            self.__zigzag__(sample)
            
        elif len(sample['rgbs']) > self.S:
            raise ValueError('len %d > %d with idx %d' % (len(sample['rgbs']), self.S, index))
        
        # print('init rgbs', sample['rgbs'].shape, sample['rgbs'].dtype)
        self.__pre_processing__(sample)

        # print('pre rgbs', sample['rgbs'].shape, sample['rgbs'].dtype)
        # print('pre masks', sample['masks'].shape, sample['masks'].sum(), sample['masks'].dtype)
        # sample['masks'].sum() == 0:
        
        self.__just_resize_then_centercrop__(sample)

        # print('resized rgbs', sample['rgbs'].shape, sample['rgbs'].dtype)
        
        if sample['masks'].sum() == 0:
            print('somehow all masks are empty!')
            return self.get_fake_sample()

        # print('rgbs before post', sample['rgbs'].shape)
        
        self.__post_processing__(sample)

        # print('rgbs after post', sample['rgbs'].shape)
        
        # common postproc: when something is unavailable, use the nearest available data
        invalid_idx = np.where(sample['xys_valid']==0)[0]
        valid_idx = np.where(sample['xys_valid']==1)[0]
        if valid_idx.size == 0:
            print('no valid idx based on xy')
            sample = self.get_fake_sample()
        else:
            for idx in invalid_idx:
                nearest = valid_idx[np.argmin(np.abs(valid_idx - idx))]
                # print('replacing xy %d with %d' % (idx, nearest))
                sample['xys_g'][idx] = sample['xys_g'][nearest]

        invalid_idx = np.where(sample['bboxes_valid']==0)[0]
        valid_idx = np.where(sample['bboxes_valid']==1)[0]
        if valid_idx.size == 0:
            print('no valid idx based on bbox')
            sample = self.get_fake_sample()
        else:
            for idx in invalid_idx:
                nearest = valid_idx[np.argmin(np.abs(valid_idx - idx))]
                # print('replacing wh %d with %d' % (idx, nearest))
                sample['bboxes_g'][idx] = sample['bboxes_g'][nearest]
                
        # print('postproc rgbs', sample['rgbs'].shape, sample['rgbs'].dtype)
        # print('masks_g', sample['masks_g'].shape, sample['masks_g'].dtype)
        # print('xys_g', sample['xys_g'].shape, sample['xys_g'].dtype)
        # print('bboxes_g', sample['bboxes_g'].shape, sample['bboxes_g'].dtype)
        # print('vis_g', sample['vis_g'].shape, sample['vis_g'].dtype)
        # print('xys_valid', sample['xys_valid'].shape, sample['xys_valid'].dtype)
        # print('bboxes_valid', sample['bboxes_valid'].shape, sample['bboxes_valid'].dtype)
        # print('vis_valid', sample['vis_valid'].shape, sample['vis_valid'].dtype)

        # if sample['vis_g'][0] <= 0 or (sample['vis_g'] > 0).sum() < max(2,int(np.sqrt(self.S))):  # if first frame is not visible, just resample
        if (sample['vis_g'] > 0).sum() < 2: #max(2,int(np.sqrt(self.S))):  # if we don't have a few visible frames, just resample
            print('vis_g.sum()', sample['vis_g'].sum())
            # print('vis_g', sample['vis_g'])
            sample = self.get_fake_sample()
        
        return sample
    
    def __len__(self):
        raise NotImplementedError()

def clip_bboxes(rgbs, bboxes):
    S,H,W,C = rgbs.shape
    bboxes[:,0] = bboxes[:,0].clip(min=0,max=W-1)
    bboxes[:,1] = bboxes[:,1].clip(min=0,max=H-1)
    bboxes[:,2] = bboxes[:,2].clip(min=1,max=W)
    bboxes[:,3] = bboxes[:,3].clip(min=1,max=H)
    return bboxes

import matplotlib.pyplot as plt
class BBoxDataset(TrackingDataset):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    # def __spatial_augment__(self, sample):
    #     rgbs, masks, bboxes = sample['rgbs'], sample['masks'], sample['bboxes']  # len S
    #     bboxes[:, 2:] = np.maximum(bboxes[:, 2:], bboxes[:, :2] + 1)
        
    #     crop_size = self.crop_size
    #     ratio = crop_size[1] / crop_size[0]
    #     wildness = np.random.uniform(0, 0.2)
    #     ratios = np.random.uniform(ratio * .75, ratio * 1.33, len(bboxes))
    #     ratios = wildness * ratios + (1. - wildness) * np.full_like(ratios, ratio)
        
    #     augment_video(self.spatial_augmenter, image=rgbs, mask=masks, bboxes=bboxes, keypoints=np.zeros((len(rgbs), 2)))
    #     bboxes[:, 2:] = np.maximum(bboxes[:, 2:], bboxes[:, :2] + 1)

    #     bboxes_crop = get_crop_around_bbox(bboxes, rgbs.shape[-2], rgbs.shape[-3], ratios)
    #     bboxes_crop = wildness * bboxes_crop + (1 - wildness) * bboxes_crop[0]
    #     bboxes_crop = bboxes_crop.astype(int)
    #     bboxes_crop[:, 2:] = np.maximum(bboxes_crop[:, 2:], bboxes_crop[:, :2] + 1)
        
    #     rgbs_cr = []
    #     masks_cr = []
    #     for i, bbox_crop in enumerate(bboxes_crop):
    #         crop_resize = A.Compose([
    #                 A.Crop(*bbox_crop), 
    #                 A.Resize(*crop_size)], 
    #             bbox_params=A.BboxParams(format='pascal_voc', min_area=1, min_visibility=0., label_fields=[]))
            
    #         data = crop_resize(image=rgbs[i], mask=masks[i], bboxes=[bboxes[i]])
    #         if len(data['bboxes']) == 0:  # this box is invalid, just change turn off visib
    #             sample['visibs'][i] = 0
    #             bboxes[i] = mask2bbox(data['mask'])
    #         else:
    #             bboxes[i] = data['bboxes'][0]
    #         rgbs_cr.append(data['image'])
    #         masks_cr.append(data['mask'])
            
    #     invalid_idx = np.where([np.sum(mask) == 0 for mask in masks_cr])[0]
    #     valid_idx = np.array(list(set(range(len(masks_cr))) - set(invalid_idx)))
    #     if valid_idx.size > 0:  # if no valid index, we will resample later because visib is all 0
    #         for idx in invalid_idx:
    #             nearest = valid_idx[np.argmin(np.abs(valid_idx - idx))]
    #             bboxes[idx] = bboxes[nearest]
            
    #     sample['rgbs'] = np.stack(rgbs_cr)
    #     sample['masks'] = np.stack(masks_cr)

    def __just_resize_then_centercrop__(self, sample):
        rgbs, masks, bboxes = sample['rgbs'], sample['masks'], sample['bboxes'] 
        # bboxes[:, 2:] = np.maximum(bboxes[:, 2:], bboxes[:, :2] + 1)
        # print('in masks', sample['masks'].shape)
        # print('in masks', masks.sum())

        S,H,W,C = rgbs.shape
        # print('input rgbs', rgbs.shape)
        # print('input masks', masks.shape)
        # print('crop_size', self.crop_size)

        target_H, target_W = self.crop_size
        scale = max(target_H / H, target_W / W)
        new_H, new_W = int(np.ceil(H * scale)), int(np.ceil(W * scale))
        # print('target', target_H, target_W, 'scale', scale, 'new HW', new_H, new_W)
        
        # Resize images and masks
        resized_rgbs = [cv2.resize(rgb, (new_W, new_H), interpolation=cv2.INTER_AREA) for rgb in rgbs]
        resized_masks = [cv2.resize(mask, (new_W, new_H), interpolation=cv2.INTER_NEAREST) for mask in masks]
        resized_rgbs = np.array(resized_rgbs)
        resized_masks = np.array(resized_masks)

        # print('resized masks', resized_masks.sum())
        # print('resized_rgbs', resized_rgbs.shape)
        # print('resized_masks', resized_masks.shape)

        # Scale bounding boxes
        scaled_bboxes = np.round(bboxes * scale).astype(int)

        def crop_image_and_bbox(image, bbox, crop_H, crop_W, x_start, y_start):
            xmin, ymin, xmax, ymax = bbox
            # x_center, y_center = (xmin + xmax) // 2, (ymin + ymax) // 2
            # x_start = max(0, x_center - crop_W // 2)
            # y_start = max(0, y_center - crop_H // 2)

            # # Adjust to prevent cropping beyond the image boundaries
            # if x_start + crop_W > new_W:
            #     x_start = new_W - crop_W
            # if y_start + crop_H > new_H:
            #     y_start = new_H - crop_H

            # Update the bounding box coordinates relative to the crop
            new_xmin = max(0, xmin - x_start)
            new_ymin = max(0, ymin - y_start)
            new_xmax = min(crop_W, xmax - x_start)
            new_ymax = min(crop_H, ymax - y_start)
            
            return image[y_start:y_start + crop_H, x_start:x_start + crop_W], [new_xmin, new_ymin, new_xmax, new_ymax]

        # get coords for cropping with, guided by the boxes
        x_starts = []
        y_starts = []
        for si in range(S):
            xmin, ymin, xmax, ymax = scaled_bboxes[si]
            x_center, y_center = (xmin + xmax) // 2, (ymin + ymax) // 2
            x_start = max(0, x_center - target_W // 2)
            y_start = max(0, y_center - target_H // 2)
            # adjust to prevent cropping beyond the image boundaries
            if x_start + target_W > new_W:
                x_start = new_W - target_W
            if y_start + target_H > new_H:
                y_start = new_H - target_H
            x_starts.append(x_start)
            y_starts.append(y_start)
            
        # smooth out (in floating point)
        for _ in range(S*2):
            for ii in range(1,S-1):
                x_starts[ii] = (x_starts[ii-1] + x_starts[ii] + x_starts[ii+1])/3.0
                y_starts[ii] = (y_starts[ii-1] + y_starts[ii] + y_starts[ii+1])/3.0
        x_starts = [int(x) for x in x_starts]
        y_starts = [int(y) for y in y_starts]
        # print('x_starts', x_starts)
        
        # apply cropping and adjust bounding boxes
        cropped_rgbs = []
        cropped_masks = []
        cropped_bboxes = []
        for s in range(S):
            cropped_rgb, new_bbox = crop_image_and_bbox(resized_rgbs[s], scaled_bboxes[s], target_H, target_W, x_starts[s], y_starts[s])
            cropped_mask, _ = crop_image_and_bbox(resized_masks[s], scaled_bboxes[s], target_H, target_W, x_starts[s], y_starts[s])
            cropped_rgbs.append(cropped_rgb)
            cropped_masks.append(cropped_mask)
            cropped_bboxes.append(new_bbox)

        # print('cropped_rgb', cropped_rgb.shape)
        # print('cropped_mask', cropped_mask.shape)
        # print('cropped masks', np.array(cropped_masks).sum())
            
        sample['rgbs'] = np.array(cropped_rgbs)
        sample['masks'] = np.array(cropped_masks)#[..., None]
        sample['bboxes'] = np.array(cropped_bboxes)

        mask_areas = (np.array(cropped_masks)[...,1] > 0).reshape(S,-1).sum(axis=1)
        mask_areas_new = mask_areas / np.max(mask_areas)
        sample['visibs'] = np.minimum(sample['visibs'], np.clip(mask_areas_new,0.1,1))

        # sample['visibs'] *= mask_

        # print('out rgbs', sample['rgbs'].shape)
        # print('out masks', sample['masks'].shape)
        # print('output rgbs', np.array(cropped_rgbs).shape)
        # print('output masks', np.array(cropped_masks).shape)
        return sample
    
    # def __augment__(self, sample):
    #     self.__color_augment__(sample)
    #     # self.__spatial_augment__(sample)
    #     # pre_area = (sample['masks'] > 0).sum(-1).sum(-1)
    #     augment_video(self.hole_augmenter, image=sample['rgbs'], mask=sample['masks'])
        
    #     # post_area = (sample['masks'] > 0).sum(-1).sum(-1)
    #     # sample['visibs'] = np.clip(post_area / np.prod(sample['bboxes'][..., 2:] - sample['bboxes'][..., :2], -1), 0, 1)
    #     # sample['visibs'][(post_area < pre_area / 2) | (post_area < 4)] = 0.

    #     S = sample['masks'].shape[0]
    #     mask_areas = (sample['masks'] > 0).reshape(S,-1).sum(axis=1)
    #     mask_areas_norm = mask_areas / np.max(mask_areas)
    #     print('mask_areas_norm', mask_areas_norm)
    #     # sample['visibs'] = (np.sum(sample['masks'][..., 0], (-1, -2)) > 8).astype(np.float32)
    #     sample['visibs'] = mask_areas_norm
        
    
    def __pre_processing__(self, sample):
        
        if 'masks' not in sample:
            sample['masks'] = np.stack([bbox2mask(bbox, sample['rgbs'].shape[-2], sample['rgbs'].shape[-3]) for bbox in sample['bboxes']]) * 0.5
            sample['masks'] = sample['masks'][..., None]

        S,H,W,mC = sample['masks'].shape
        if mC==1:
            sample['masks'] = np.repeat(sample['masks'], 3, axis=-1)
            masks_valid = np.zeros((S,3), dtype=np.float32)
            # only chan1 valid
            masks_valid[:,1] = 1
            sample['masks_valid'] = masks_valid
        else:
            assert(mC==3) 
            assert('masks_valid' in sample)
        # print('masks endpre', sample['masks'].shape)
        
        if 'visibs' not in sample:
            S = sample['masks'].shape[0]
            mask_areas = (sample['masks'][...,1] > 0).reshape(S,-1).sum(axis=1)
            mask_areas_norm = mask_areas / np.max(mask_areas)
            sample['visibs'] = mask_areas_norm
            # sample['visibs'] = np.ones((sample['masks'].shape[0],), dtype=np.float32)
            
        sample['bboxes'] = clip_bboxes(sample['rgbs'], sample['bboxes'])
    
    def __post_processing__(self, sample):
        sample['rgbs'] = np.moveaxis(sample['rgbs'], -1, 1).astype(np.uint8) # S,C,H,W
        sample['masks_g'] = np.moveaxis(sample['masks'].astype(np.float32), -1, 1) # S,C,H,W

        # S = sample['masks'].shape[0]
        # if 'visibs' not in sample:
        #     sample['visibs'] = np.ones((S), dtype=np.float32)
        # mask_areas = (sample['masks'] > 0).reshape(S,-1).sum(axis=1) # S
        # mask_areas_norm = mask_areas / np.max(mask_areas) # S
        # sample['visibs'] *= np.clip(mask_areas_norm, 0.5, 1)
            
        sample['vis_g'] = sample['visibs'].astype(np.float32)
        # print('visibs', sample['visibs'])

        if 'valids' in sample:
            sample['xys_valid'] = sample['valids'].astype(np.float32)
            sample['bboxes_valid'] = sample['valids'].astype(np.float32)
        else:
            sample['xys_valid'] = (sample['visibs'] > 0).astype(np.float32)
            sample['bboxes_valid'] = (sample['visibs'] > 0).astype(np.float32)
        
        sample['bboxes_g'] = (sample['bboxes']).astype(np.int32)
        sample['xys_g'] = ((sample['bboxes'][:, 2:] + sample['bboxes'][:, :2]) // 2 + 0.5).astype(np.int32)
        sample['vis_valid'] = np.ones_like(sample['vis_g']).astype(np.float32)

        # don't need these anymore
        del sample['bboxes']
        del sample['masks']
        # del sample['trajs']
        del sample['visibs']
        
        
class MaskDataset(BBoxDataset):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    def __pre_processing__(self, sample):
        if sample['masks'].ndim == 3:
            sample['masks'] = sample['masks'][..., None]
        # masks is S,H,W,mC
        # print('masks pre', sample['masks'].shape)
        S,H,W,mC = sample['masks'].shape
        if mC==1:
            sample['masks'] = np.repeat(sample['masks'], 3, axis=-1)
            masks_valid = np.zeros((S,3), dtype=np.float32)
            # only chan1 valid
            masks_valid[:,1] = 1
            sample['masks_valid'] = masks_valid
        else:
            assert(mC==3) 
            assert('masks_valid' in sample)
        # print('masks endpre', sample['masks'].shape)

        if 'bboxes' not in sample:
            if np.sum(sample['masks_valid'][:,1])>0:
                bboxes = np.stack([mask2bbox(mask[:,:,1]) for mask in sample['masks']])
            else: # must be mask2
                bboxes = np.stack([mask2bbox(mask[:,:,2]) for mask in sample['masks']])
            # print('bboxes', bboxes, bboxes.shape)
            # print('sum zero', (np.sum(bboxes, axis=1)==0).astype(np.float32))
            bboxes_valid = 1-(np.sum(bboxes, axis=1)==0).astype(np.float32)
            # print('bboxes_valid', bboxes_valid, 'sum', np.sum(bboxes_valid), 'S', S)
            if np.sum(bboxes_valid) < S:
                bboxes = utils.misc.data_replace_with_nearest(bboxes, bboxes_valid)
            sample['bboxes'] = bboxes
            sample['bboxes_valid'] = bboxes_valid

        # write visibs into sample, disregarding existing visibs (which are really never here)
        if np.sum(sample['masks_valid'][:,1])>0:
            mask_areas = sample['masks'][:,:,:,1].reshape(S,-1).sum(axis=1)
        else:
            mask_areas = sample['masks'][:,:,:,2].reshape(S,-1).sum(axis=1)
        mask_areas_norm = mask_areas / np.max(mask_areas)
        sample['visibs'] = mask_areas_norm
            
        sample['bboxes'] = clip_bboxes(sample['rgbs'], sample['bboxes'])
    
    def __post_processing__(self, sample):
        # print('masks post', sample['masks'].shape)
        S,H,W,mC = sample['masks'].shape
        # compute visibs using current masks
        if np.sum(sample['masks_valid'][:,1])>0:
            mask_areas = sample['masks'][:,:,:,1].reshape(S,-1).sum(axis=1)
        else:
            mask_areas = sample['masks'][:,:,:,2].reshape(S,-1).sum(axis=1)
        mask_areas_norm = mask_areas / np.max(mask_areas)
        sample['visibs'] = mask_areas_norm

        # give me xys at the center-of-mass of each mask
        # note that right now we don't use xys/xys_valid as supervision, we only use masks
        
        if np.sum(sample['masks_valid'][:,1])>0:
            xys, _, xys_valid, _ = utils.misc.data_get_traj_from_masks(sample['masks'][...,1]) # use objmask
        else:
            xys, _, xys_valid, _ = utils.misc.data_get_traj_from_masks(sample['masks'][...,2]) # use amodal mask
            
        if np.sum(xys_valid) < S:
            xys = utils.misc.data_replace_with_nearest(xys, xys_valid)
        # xys_valid = np.zeros_like(xys[:,0])
        # choice = np.random.choice(np.nonzero(sample['visibs']>0.5)[0])
        # xys_valid[choice] = 1 # only say one is valid
        sample['xys_g'] = xys
        sample['xys_valid'] = xys_valid
                
        sample['rgbs'] = np.moveaxis(sample['rgbs'], -1, 1).astype(np.uint8)
        # print(sample['masks'].shape)
        sample['masks_g'] = np.moveaxis(sample['masks'].astype(np.float32), -1, 1)
        sample['vis_g'] = sample['visibs'].astype(np.float32)
        sample['bboxes_g'] = sample['bboxes'].round().astype(np.int32)
        sample['bboxes_valid'] = (sample['visibs'] > 0).astype(np.float32)
        sample['vis_valid'] = np.ones_like(sample['vis_g']).astype(np.float32)

        # # masks_valid = np.zeros((S,3), dtype=np.float32)
        # # # make masks 3chan if not already
        # # mC = sample['masks_g'].shape[1]
        # if mC==1:
        #     sample['masks_g'] = sample['masks_g'].repeat(1,3,1,1)

        # don't need these anymore
        del sample['bboxes']
        del sample['masks']
        # del sample['trajs']
        del sample['visibs']
    
    # def __augment__(self, sample):
    #     self.__color_augment__(sample)
    #     # self.__spatial_augment__(sample)
    #     pre_area = (sample['masks'] > 0).sum(-1).sum(-1)
    #     augment_video(self.hole_augmenter, image=sample['rgbs'], mask=sample['masks'])
        
    #     # post_area = (sample['masks'] > 0).sum(-1).sum(-1)
    #     # sample['visibs'][(post_area < pre_area / 2) | (post_area < 4)] = 0.  # do not recalcuate visib since we know it
        

class PointDataset(TrackingDataset):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    def __pre_processing__(self, sample):

        assert 'xys' in sample
        assert 'visibs' in sample
        
        S,H,W,C = sample['rgbs'].shape

        if 'valids' not in sample:
            sample['valids'] = sample['visibs']
        
        if 'bboxes' not in sample:
            # put a box around each point
            sz = np.random.randint(2, 4, (2,)).astype(np.int32)            
            bboxes = np.concatenate([sample['xys']-sz, sample['xys']+sz], axis=-1) 
            bboxes[:,0] = bboxes[:,0].clip(min=0,max=W-1)
            bboxes[:,1] = bboxes[:,1].clip(min=0,max=H-1)
            bboxes[:,2] = bboxes[:,2].clip(min=1,max=W)
            bboxes[:,3] = bboxes[:,3].clip(min=1,max=H)
            sample['bboxes'] = bboxes

        if 'masks' not in sample:
            # write a mask with a single 1 at each valid (clamped) xy, in chan0
            masks = np.zeros_like(sample['rgbs']).astype(np.float32)
            masks_valid = np.zeros((S,3), dtype=np.float32)
            for i in range(len(masks)):
                if sample['valids'][i]:
                    xy = sample['xys'][i].astype(np.int32)
                    x, y = xy[0], xy[1]
                    x = x.clip(0,W-1)
                    y = y.clip(0,H-1)
                    masks[i,y,x,0] = 1
                    masks_valid[i,0] = 1 # only zeroth chan mask valid
                else:
                    masks[i,:,:,0] = 0.5
            # set chan1-2 to ignore
            masks[...,1] = 0.5
            masks[...,2] = 0.5

            # masks_valid = np.ones_like(sample['valids'])

            # print('prepmasks', masks.shape)
            sample['masks'] = masks
            sample['masks_valid'] = masks_valid
        else:
            assert('masks_valid' in sample) # when we have masks, we should also have masks_valid

        # print('masks', sample['masks'].shape)
        sample['bboxes'] = clip_bboxes(sample['rgbs'], sample['bboxes'])
        
    def __just_resize_then_centercrop__(self, sample):
        rgbs, masks, bboxes, xys, visibs = sample['rgbs'], sample['masks'], sample['bboxes'], sample['xys'] , sample['visibs'] 
        # bboxes[:, 2:] = np.maximum(bboxes[:, 2:], bboxes[:, :2] + 1)

        # print('xys before crop', xys)

        S,H,W,C = rgbs.shape
        # print('input rgbs', rgbs.shape)
        # print('input masks', masks.shape)
        # print('crop_size', self.crop_size)

        target_H, target_W = self.crop_size
        scale = max(target_H / H, target_W / W)
        new_H, new_W = int(np.ceil(H * scale)), int(np.ceil(W * scale))
        
        # Resize images and masks

        resized_rgbs = [cv2.resize(rgb, (new_W, new_H), interpolation=cv2.INTER_AREA) for rgb in rgbs]
        resized_masks = [cv2.resize(mask, (new_W, new_H), interpolation=cv2.INTER_NEAREST) for mask in masks]
        resized_rgbs = np.array(resized_rgbs)
        resized_masks = np.array(resized_masks)

        # print('resized_rgbs', resized_rgbs.shape)
        # print('resized_masks', resized_masks.shape)

        # Scale bounding boxes
        scaled_bboxes = np.round(bboxes * scale).astype(int)
        scaled_xys = np.round(xys * scale).astype(int)

        # print('scaled_xys', scaled_xys)

        def crop_image_and_bbox(image, bbox, pt, crop_H, crop_W, x_start, y_start):
            xmin, ymin, xmax, ymax = bbox
            # x_center, y_center = (xmin + xmax) // 2, (ymin + ymax) // 2
            # x_start = max(0, x_center - crop_W // 2)
            # y_start = max(0, y_center - crop_H // 2)

            # # Adjust to prevent cropping beyond the image boundaries
            # if x_start + crop_W > new_W:
            #     x_start = new_W - crop_W
            # if y_start + crop_H > new_H:
            #     y_start = new_H - crop_H

            # Update the bounding box coordinates relative to the crop
            new_xmin = max(0, xmin - x_start)
            new_ymin = max(0, ymin - y_start)
            new_xmax = min(crop_W, xmax - x_start)
            new_ymax = min(crop_H, ymax - y_start)

            pt_x = max(0, pt[0] - x_start)
            pt_y = max(0, pt[1] - y_start)
            
            return image[y_start:y_start + crop_H, x_start:x_start + crop_W], [new_xmin, new_ymin, new_xmax, new_ymax], [pt_x, pt_y]

        # get coords for cropping with, guided by the xys
        x_starts = []
        y_starts = []
        for si in range(S):
            x_center, y_center = scaled_xys[si]
            x_start = int(max(0, x_center - target_W // 2))
            y_start = int(max(0, y_center - target_H // 2))
            # adjust to prevent cropping beyond the image boundaries
            if x_start + target_W > new_W:
                x_start = new_W - target_W
            if y_start + target_H > new_H:
                y_start = new_H - target_H
            x_starts.append(x_start)
            y_starts.append(y_start)
        # print('x_starts', x_starts)
            
        # smooth out (in floating point)
        for _ in range(S*2):
            for ii in range(1,S-1):
                x_starts[ii] = (x_starts[ii-1] + x_starts[ii] + x_starts[ii+1])/3.0
                y_starts[ii] = (y_starts[ii-1] + y_starts[ii] + y_starts[ii+1])/3.0
        x_starts = [int(x) for x in x_starts]
        y_starts = [int(y) for y in y_starts]
        # print('x_starts smooth', x_starts)
        
        # apply cropping and adjust bounding boxes
        cropped_rgbs = []
        cropped_masks = []
        cropped_bboxes = []
        cropped_xys = []
        for s in range(S):
            cropped_rgb, new_bbox, new_pt = crop_image_and_bbox(resized_rgbs[s], scaled_bboxes[s], scaled_xys[s], target_H, target_W, x_starts[s], y_starts[s])
            # print('cropped_rgb', cropped_rgb.shape)
            cropped_mask, _, _ = crop_image_and_bbox(resized_masks[s], scaled_bboxes[s], scaled_xys[s], target_H, target_W, x_starts[s], y_starts[s])
            # print('cropped_mask', cropped_mask.shape)
            cropped_rgbs.append(cropped_rgb)
            cropped_masks.append(cropped_mask)
            cropped_bboxes.append(new_bbox)
            cropped_xys.append(new_pt)

        for si in range(S):
            xy = cropped_xys[si]
            if xy[0] < 0 or xy[1] < 0 or xy[0] > new_W-1 or xy[1] > new_H-1:
                sample['visibs'][si] *= 0

        sample['rgbs'] = np.array(cropped_rgbs)
        sample['masks'] = np.array(cropped_masks)#[..., None]
        sample['bboxes'] = np.array(cropped_bboxes)
        sample['xys'] = np.array(cropped_xys)

        # mask_areas = (np.array(cropped_masks)[...,1] > 0).reshape(S,-1).sum(axis=1)
        # mask_areas_new = mask_areas / np.max(mask_areas)
        # sample['visibs'] = np.minimum(sample['visibs'], np.clip(mask_areas_new,0.1,1))
        
        # print('output rgbs', np.array(cropped_rgbs).shape)
        # print('crop masks', np.array(cropped_masks)[...,None].shape)
        # print('crop visibs', sample['visibs'])
        # print('cropped_xys', cropped_xys)
        return sample
    
    
    def __post_processing__(self, sample):

        # rewrite the boxes to be small in the final resolution
        S,H,W,C = sample['rgbs'].shape
        
        # for si in range(S):
        #     xy = sample['xys'][si]
        #     if xy[0] < 0 or xy[1] < 0 or xy[0] > W-1 or xy[1] > H-1:
        #         sample['visibs'][si] *= 0

        # sample['masks'] = np.moveaxis(sample['rgbs'], -1, 1).astype(np.uint8)
        
        # print('postmasks', sample['masks'].shape)
            
            
        # re-write the chan0 masks to be 1 at the clamped xy.
        # this is necessary bc previous values may have been lost while resizing
        masks0 = np.zeros_like(sample['rgbs'][:,:,:,0]).astype(np.float32)
        # print('masks', masks.shape)
        for i in range(S):
            if sample['valids'][i]:
                xy = sample['xys'][i].round().astype(np.int32)
                x, y = xy[0], xy[1]
                x = x.clip(0,W-1)
                y = y.clip(0,H-1)
                masks0[i,y,x] = 1
            else:
                masks0[i] = 0.5 # make the whole frame gray
        # masks = np.stack([masks0, *masks0, 0*masks0]
        sample['masks'][:,:,:,0] = masks0

        sample['rgbs'] = np.moveaxis(sample['rgbs'], -1, 1).astype(np.uint8)
        sample['masks_g'] = np.moveaxis(sample['masks'], -1, 1).astype(np.float32)
        # sample['masks_g'] = sample['masks'].astype(np.float32)
        sample['vis_g'] = sample['visibs'].astype(np.float32)

        sample['xys_g'] = sample['xys'].round().astype(np.int32)
        
        if 'valids' in sample:
            sample['xys_valid'] = sample['valids'].astype(np.float32)
            sample['vis_valid'] = sample['valids'].astype(np.float32)
        else:
            # xy and vis are only valid when visible
            sample['xys_valid'] = sample['visibs'].astype(np.float32)
            sample['vis_valid'] = sample['visibs'].astype(np.float32)

        sample['bboxes_g'] = sample['bboxes'].round().astype(np.int32)
        sample['bboxes_valid'] = np.ones_like(sample['visibs']).astype(np.float32)


        # don't need these anymore
        del sample['xys']
        del sample['bboxes']
        del sample['masks']
        del sample['visibs']
