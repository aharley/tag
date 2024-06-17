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
from icecream import ic

# np.random.seed(125)
# torch.multiprocessing.set_sharing_strategy('file_system')

class DAVISDataset(MaskDataset):
    def __init__(self,
                 dataset_location='../DAVIS',
                 S=32,
                 crop_size=(384,512),
                 fullseq=False,
                 chunk=None,
                 strides=[1,2,3,4], # high quality dataset
                 clip_step=2, # very high quality dataset
                 use_augs=False,
                 is_training=True):

        print('loading DAVIS dataset...')
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size, 
            use_augs=use_augs,
            is_training=is_training
        )
        self.dataset_location = os.path.join(dataset_location)
        split = "train" if is_training else "val"
        self.video_names = open(os.path.join(self.dataset_location, f"ImageSets/2017/{split}.txt")).read().splitlines()
        print('found {:d} videos in {}'.format(len(self.video_names), self.dataset_location))
        self.all_video_names = []
        self.all_tids = []
        self.all_full_idx = []

        for video_name in self.video_names:
            # print(video_name)
            # sys.stdout.write(video_name)
            
            video_dir = os.path.join(self.dataset_location, 'JPEGImages', '480p', video_name)
            annotation_dir = os.path.join(self.dataset_location, 'Annotations', '480p', video_name)
            frames = sorted(os.listdir(video_dir))
            S_local = len(frames)

            frame0 = frames[0]
            seg0 = cv2.imread(os.path.join(annotation_dir, frame0.replace('.jpg', '.png')), cv2.IMREAD_GRAYSCALE)
            
            valid_segs = np.array([v for v in np.unique(seg0.reshape(-1)) if v > 0]) # NSeg
            
            for stride in strides:
                for ii in range(0, max(S_local-self.S*stride,1), clip_step):
                    full_idx = ii + np.arange(self.S)*stride
                    full_idx = [ij for ij in full_idx if ij < S_local]

                    if len(full_idx) < (((S_local - 1) // stride + 1 if self.S == -1 else self.S) if fullseq else 8):
                        continue
                    
                    for tid in valid_segs:
                        self.all_video_names.append(video_name)
                        self.all_full_idx.append(full_idx)
                        self.all_tids.append(tid)
                        sys.stdout.write('.')
                        sys.stdout.flush()
        
    def __len__(self):
        return len(self.all_video_names)
        
    def getitem_helper(self, index):
        video = self.all_video_names[index]
        full_idx = self.all_full_idx[index]
        tid = self.all_tids[index]

        video_dir = os.path.join(self.dataset_location, 'JPEGImages', '480p', video)
        annotation_dir = os.path.join(self.dataset_location, 'Annotations', '480p', video)

        frames = sorted(os.listdir(video_dir))

        frames = [frames[idx] for idx in full_idx]

        image_paths = [os.path.join(video_dir, fn) for fn in frames]
        seg_paths = [os.path.join(annotation_dir, fn.replace('.jpg', '.png')) for fn in frames]
        rgb = cv2.imread(str(image_paths[0]))
        rgbs = []
        for path in image_paths:
            rgb = cv2.imread(str(path))[..., ::-1].copy()
            rgbs.append(rgb)
        segs = []
        for path in seg_paths:
            seg = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            segs.append(seg)
        masks = [(seg == tid).astype(np.float32) for seg in segs]

        if np.mean(masks[0]) < 0.001:
            print('mask0_mean', np.mean(masks[0]))
            return None
        
        sample = {
            'rgbs': np.stack(rgbs),
            'masks': np.stack(masks),
        }
        
        return sample


class DAVISTestDataset(MaskDataset):
    def __init__(self,
                 dataset_location='../DAVIS',
                 S=32,
                 crop_size=(384,512)):

        super().__init__(dataset_location=dataset_location,
                         S=S,
                         crop_size=crop_size,
                         use_augs=False,
                         is_training=False,
                         inference=True)
        self.dataset_location = os.path.join(dataset_location)
        split = "val"
        self.video_names = open(os.path.join(self.dataset_location, f"ImageSets/2017/{split}.txt")).read().splitlines()
        print('found {:d} videos in {}'.format(len(self.video_names), self.dataset_location))
        self.all_video_names = []
        self.all_tids = []
        self.all_full_idx = []

        for video_name in self.video_names:
            video_dir = os.path.join(self.dataset_location, 'JPEGImages', '480p', video_name)
            annotation_dir = os.path.join(self.dataset_location, 'Annotations', '480p', video_name)
            frames = sorted(os.listdir(video_dir))
            S_local = min(len(frames), S)

            frame0 = frames[0]
            seg0 = cv2.imread(os.path.join(annotation_dir, frame0.replace('.jpg', '.png')), cv2.IMREAD_GRAYSCALE)
            
            valid_segs = np.array([v for v in np.unique(seg0.reshape(-1)) if v > 0]) # NSeg

            full_idx = np.arange(0, S_local)
            for tid in valid_segs:
                self.all_video_names.append(video_name)
                self.all_full_idx.append(full_idx)
                self.all_tids.append(tid)
                sys.stdout.write('.')
                sys.stdout.flush()
        
    def __len__(self):
        return len(self.all_video_names)
        
    def getitem_helper(self, index):
        video = self.all_video_names[index]
        full_idx = self.all_full_idx[index]
        tid = self.all_tids[index]

        video_dir = os.path.join(self.dataset_location, 'JPEGImages', '480p', video)
        annotation_dir = os.path.join(self.dataset_location, 'Annotations', '480p', video)

        frames = sorted(os.listdir(video_dir))

        frames = [frames[idx] for idx in full_idx]

        image_paths = [os.path.join(video_dir, fn) for fn in frames]
        seg_paths = [os.path.join(annotation_dir, fn.replace('.jpg', '.png')) for fn in frames]
        rgb = cv2.imread(str(image_paths[0]))
        rgbs = []
        for path in image_paths:
            rgb = cv2.imread(str(path))[..., ::-1].copy()
            rgbs.append(rgb)
        segs = []
        for path in seg_paths:
            seg = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            segs.append(seg)
        masks = [(seg == tid).astype(np.float32) for seg in segs]
        sample = {
            'rgbs': np.stack(rgbs),
            'masks': np.stack(masks),
        }
        
        return sample
