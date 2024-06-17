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
from datasets.dataset import PointDataset
from pathlib import Path

# from scenedetect.scene_detector import SceneDetector
# from PIL import Image


class PoseTrackDataset(PointDataset):
    def __init__(self,
                 dataset_location='../datasets/posetrack',
                 S=32, fullseq=False, chunk=None,
                 strides=[1,2],
                 zooms=[1.25,2,3,4],
                 crop_size=(384,512),
                 use_augs=False,
                 is_training=True):
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training
        )
        print('loading posetrack dataset...')

        clip_step = S//2 if is_training else S
        if not is_training:
            strides = [1]
            zooms = [2]
        
        self.dataset_location = Path(self.dataset_location)
        self.gt_fns = sorted(list(self.dataset_location.glob('posetrack_mot/mot/{}/*/gt/gt_kpts.txt'.format('train' if is_training else 'val'))))

        print('found {:d} {} videos in {}'.format(len(self.gt_fns), ('train' if is_training else 'val'), self.dataset_location))

        if chunk is not None:
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.gt_fns = chunkify(self.gt_fns,100)[chunk]
            print('filtered to %d videos' % len(self.gt_fns))

        # self.all_fns = []
        # self.all_tids = []
        # self.all_kps = []
        # self.all_starts = []
        # self.all_strides = []

        kps = np.concatenate([np.arange(3), np.arange(5, 17)])

        self.all_full_idx = []
        self.all_track_idx = []
        self.all_kp_idx = []
        self.all_gt_fn = []
        self.all_zooms = []
        
        for gt_fn in self.gt_fns:
            gt = np.loadtxt(gt_fn, delimiter=',').astype(int)
            image_info = json.load(open(gt_fn.parent.parent / 'image_info.json'))
            track_ids = np.unique(gt[:, 1])

            for track_id in track_ids:

                # for kp_idx in kps:
                for kp_idx in kps:
                    gt_here = gt[(gt[:, 1] == track_id) & (gt[:, 2+3*kp_idx+2] > 0)]
                    frame_ids = gt_here[:, 0]

                    if len(frame_ids) < (self.S if fullseq else 8): continue

                    # img_paths = []
                    trajs = []
                    visibs = []
                    # # in rare cases, not all frame indexs in image_info
                    for fid in range(np.min(frame_ids), np.max(frame_ids) + 1):
                        # img_path = image_info[img_frame_idxs.index(fid)]['file_name']
                        # rgb_path = str(self.dataset_location / img_path)
                        # img_paths.append(rgb_path)

                        fgt = gt_here[frame_ids == fid]
                        if fid in frame_ids and (fgt[:, 2+3*kp_idx+2] > 0):
                            trajs.append(fgt[:, 2+3*kp_idx:4+3*kp_idx])
                            visibs.append(np.array([1]))
                        else:
                            trajs.append(np.array([[0, 0]]))
                            visibs.append(np.array([0]))
                    trajs = np.concatenate(trajs)
                    visibs = np.concatenate(visibs)

                    # it's actually kind of slow to load the rgbs
                    # so we will deal with pixel coords

                    # img_frame_idxs = [info['frame_index'] for info in image_info]
                    # img0_path = image_info[img_frame_idxs.index(np.min(frame_ids))]['file_name']
                    # rgb0_path = str(self.dataset_location / img_path)
                    # # print('img0_path', img0_path)
                    # rgb0 = cv2.imread(str(rgb0_path))[..., ::-1]
                    # print('rgb0', rgb0.shape)
                    # H,W,C = rgb0.shape

                    S_local = len(trajs)
                    for stride in strides:
                        for ii in range(0, S_local, clip_step*stride):
                            full_idx = ii + np.arange(self.S) * stride
                            full_idx = [ij for ij in full_idx if ij < S_local]

                            if len(frame_ids) < (self.S if fullseq else 8): continue

                            trajs_here = trajs[full_idx]
                            visibs_here = visibs[full_idx]
                            # print('visibs_here', visibs_here)

                            # posetrack does not really have visibility annotations;
                            # if it is annotated we call it visible;
                            # and some of these are super hard anyway
                            # so let's require a lot of "visibility"
                            # if np.sum(visibs_here[:2])<2 or np.sum(visibs_here)<6:
                            #     # sys.stdout.write('o')
                            #     continue

                            visibs_safe = visibs_here*0
                            for ij in range(1,len(full_idx)-1):
                                visibs_safe[ij] = visibs_here[ij-1] * visibs_here[ij] * visibs_here[ij+1]
                            if np.sum(visibs_safe)<4:
                                sys.stdout.write('o')
                                continue

                            for si in range(1,len(trajs_here)):
                                if visibs_here[si]==0:
                                    trajs_here[si] = trajs_here[si-1]

                            # if np.sum(visibs[:2])<2 or np.sum(visibs)<3:
                            #     continue

                            # # the data includes some cuts
                            # # we will find them by checking vel and accel
                            # # vels = (trajs_here[1:] - trajs_here[:-1])/max(H,W)
                            vels = trajs_here[1:] - trajs_here[:-1]
                            accels = vels[1:] - vels[:-1]
                            mean_vel = np.mean(np.linalg.norm(vels, axis=-1))
                            max_accel = np.max(np.linalg.norm(accels, axis=-1))

                            if max_accel > 250 or mean_vel < 2:
                                # print('mean_vel %.1f, max_accel %.1f' % (mean_vel, max_accel))
                                continue
                            
                            for zoom in zooms:
                                self.all_gt_fn.append(gt_fn)
                                self.all_track_idx.append(track_id)
                                self.all_kp_idx.append(kp_idx)
                                self.all_full_idx.append(full_idx)
                                self.all_zooms.append(zoom)
                            sys.stdout.write('.')
                sys.stdout.flush()
        print('found {:d} {} samples in {}'.format(len(self.all_gt_fn), ('train' if is_training else 'val'), self.dataset_location))
    
    def __len__(self):
        return len(self.all_gt_fn)
    
    def getitem_helper(self, index):

        gt_fn = self.all_gt_fn[index]
        track_id = self.all_track_idx[index]
        kp_idx = self.all_kp_idx[index]
        full_idx = self.all_full_idx[index]
        zoom = self.all_zooms[index]

        # print('gt_fn', gt_fn)
        # print('track_id', track_id)
        # print('kp_idx', kp_idx)
        # print('full_idx', full_idx)
        
        gt = np.loadtxt(gt_fn, delimiter=',').astype(int)
        image_info = json.load(open(gt_fn.parent.parent / 'image_info.json'))
        gt_here = gt[(gt[:, 1] == track_id) & (gt[:, 2+3*kp_idx+2] > 0)]
        
        frame_ids = gt_here[:, 0]

        img_paths = []
        xys = []
        visibs = []
        
        # in rare cases, not all frame indexs in image_info
        img_frame_idxs = [info['frame_index'] for info in image_info]
        for fid in range(np.min(frame_ids), np.max(frame_ids) + 1):
            img_path = image_info[img_frame_idxs.index(fid)]['file_name']
            rgb_path = str(self.dataset_location / img_path)
            img_paths.append(rgb_path)
            
            fgt = gt_here[frame_ids == fid]
            if fid in frame_ids and (fgt[:, 2+3*kp_idx+2] > 0):
                xys.append(fgt[:, 2+3*kp_idx:4+3*kp_idx])
                visibs.append(np.array([1]))
            else:
                xys.append(np.array([[0, 0]]))
                visibs.append(np.array([0]))
        
        xys = np.concatenate(xys)
        visibs = np.concatenate(visibs)
        
        xys = xys[full_idx]
        visibs = visibs[full_idx]
        img_paths = [img_paths[idx] for idx in full_idx]

        if np.sum(visibs) < 4: return None

        rgbs = [cv2.imread(str(path))[..., ::-1] for path in img_paths]

        valids = visibs.copy()

        # scene_manager = SceneManager()
        # scene_manager.add_detector(ContentDetector())
        # scene_manager.detect_scenes(video=video, duration=total_frames)
        # for si in range(len(rgbs)):
        #     # SceneDetector.process_frame(si, Image.fromarray(rgbs[si]))
        #     SceneDetector.process_frame(si, rgbs[si])
        
        rgbs = np.stack(rgbs) # S,H,W,C

        S,H,W,C = rgbs.shape

        # replace invalid xys with nearby ones
        xys = utils.misc.data_replace_with_nearest(xys, visibs)

        # # if an apparent cut coincides with a teleport,
        # # it is probably a cut, and we will discard.
        # vels = xys[1:] - xys[:-1]
        # accels = np.linalg.norm(vels[1:] - vels[:-1], axis=-1)
        # if np.max(accels) > 250:
        #     return None
        # print('accels', accels)
        # grays = np.mean(rgbs, axis=-1).reshape(S,-1) #> 64
        # diffs = np.mean(np.abs(grays[1:] - grays[:-1]) > 16, axis=1)
        # ddiffs_bad = diffs[1:] > 0.2
        # print('ddiffs_bad', ddiffs_bad)
        # both_bad = np.max(accels_bad * ddiffs_bad)
        # if both_bad:
        #     print('accels', np.round(accels))
        #     print('diffs', diffs[1:])
        #     return None

        # if np.max(accels_bad) > 0:
        #     print('accels', np.round(accels))
        #     print('diffs', diffs[1:])
        #     return None
        
        if zoom > 1:
            xys, visibs, valids, rgbs = utils.misc.data_zoom(zoom, xys, visibs, valids, rgbs)
            _, H, W, _ = rgbs.shape
        if np.sum(visibs) < 4: return None
        
        sample = {
            'xys': xys,
            'rgbs': rgbs,
            'visibs': visibs,
            # 'valids': valids,
        }
        return sample
