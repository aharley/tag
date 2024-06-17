import torch
import numpy as np
import os
import scipy.ndimage
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import random
import glob
import json
import imageio
import cv2
from datasets.dataset import PointDataset
import pickle
import utils.misc
from pathlib import Path
from moviepy.editor import VideoFileClip
from datasets.dataset_utils import make_split
import sys


class DriveTrackDataset(PointDataset):
    def __init__(
            self,
            dataset_location,
            S=32, fullseq=False, chunk=None,
            strides=[1,2,3,4],
            zooms=[2,3],
            crop_size=((384, 512)),
            use_augs=False,
            is_training=True,
    ):
        print("loading drivetrack dataset...")
        super().__init__(
            dataset_location=dataset_location,
            S=S, 
            strides=strides,
            fullseq=fullseq,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training,
        )

        clip_step = S // 2 if is_training else S
        if not is_training:
            strides = [1]
            zooms = [2]

        self.N_per = 32

        self.dataset_location = Path(dataset_location)
        self.S = S
        self.video_fns = sorted(list(Path('/orion/group/drivetrack').glob('*.npz')))

        print(f"found {len(self.video_fns)} unique videos in {dataset_location}")

        if chunk is not None:
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.video_fns = chunkify(self.video_fns,100)[chunk]
            print('filtered to %d video_fns' % len(self.video_fns))
            # print('self.video_fns', self.video_fns)
        
        # self.video_fns = make_split(self.video_fns, is_training, shuffle=True)

        self.all_info = []
        # for video_name in self.video_names[:]:
        for video_fn in self.video_fns:
            print('video_fn', video_fn)
            ds = np.load(video_fn, allow_pickle=True)
            video, tracks, visibles = ds['video'], ds['tracks'], ds['visibles']

            S_local = len(video)
            print('S_local', S_local)

            N, T, D = tracks.shape
            # N is around 10k-100k
            print('N', N)

            # new_inds = np.arange(N)
            # if N > 1000:
            #     # take points spaced apart
            #     keep = 1000
            #     inds = utils.misc.farthest_point_sample_py(tracks[:,0], keep, deterministic=True)
            #     new_inds = new_inds[inds]
            #     tracks = tracks[inds]
            #     visibles = visibles[inds]
            
            # print('tracks', tracks.shape)
            
            # if N > 1000: 
            #     keep = 1000
            #     dists = np.linalg.norm(tracks[1:] - tracks[:-1], axis=-1) # S-1,N
            #     # only count motion if it's valid
            #     mot_mean = np.mean(dists*valids[1:], axis=0) # N
            #     inds = np.argsort(-mot_mean)[:keep]
            #     tracks = tracks[:,inds]
            #     visibs = visibs[:,inds]
            #     valids = valids[:,inds]
            #     inbounds = inbounds[:,inds]
            # print('N3', tracks.shape[1])


            for stride in strides:
                for ii in range(0, S_local, clip_step*stride):
                    full_idx = ii + np.arange(self.S) * stride
                    full_idx = [ij for ij in full_idx if ij < S_local]
                    if len(full_idx) < (((S_local - 1) // stride + 1 if self.S == -1 else self.S) if fullseq else 24):
                        continue

                    N, T, D = tracks.shape

                    tids_here = np.arange(N)

                    visibs_here = visibles[:,full_idx].astype(np.float32)
                    trajs_here = tracks[:,full_idx].astype(np.float32)
                    safe_here = visibs_here[:,:-2] * visibs_here[:,1:-1] * visibs_here[:,2:]
                    safe_ok = np.sum(safe_here, axis=1)
                    # print('safe_ok', safe_ok.shape)
                    inds = np.nonzero(safe_ok >= 8)[0]
                    # print('safe_inds', safe_inds, len(safe_inds))

                    trajs_here = trajs_here[inds]
                    visibs_here = visibs_here[inds]
                    tids_here = tids_here[inds]
                    N = len(tids_here)
                    S = len(full_idx)
                    # print('trajs_here', trajs_here.shape)
                    # print('tids_here', tids_here.shape)

                    # # to speed up we'll first randomly sample 10k
                    # if N > 10000:
                    #     inds = np.random.permutation(N)[:10000]
                    #     trajs_here = trajs_here[inds]
                    #     visibs_here = visibs_here[inds]
                    #     tids_here = tids_here[inds]
                    #     N = len(tids_here)

                    if N > self.N_per*len(zooms):
                        mean_xy = np.sum(trajs_here * visibs_here.reshape(N,S,1), axis=1) / (1 + np.sum(visibs_here, axis=1).reshape(N,1)) # N,2

                        # vel_here = trajs_here[:,1:] - trajs_here[:,:-1]
                        # visibs_here_ = visibs_here[:,1:] - visibs_here[:,:-1]
                        # mean_vel = np.sum(vel_here * visibs_here_.reshape(N,S-1,1), axis=1) / (1 + np.sum(visibs_here_, axis=1).reshape(N,1)) # N,2
                        
                        # # print('mean_xy', mean_xy.shape)
                        # mean_xyv = np.concatenate([mean_xy, mean_vel], axis=1) # N,4

                        # take points spaced apart
                        keep = self.N_per*len(zooms)
                        inds = utils.misc.farthest_point_sample_py(mean_xy, keep, deterministic=True)
                        trajs_here = trajs_here[inds]
                        visibs_here = visibs_here[inds]
                        tids_here = tids_here[inds]
                        N = len(tids_here)

                    # if N > 500:
                    #     # we prefer points that travel
                    #     keep = 500
                    #     dists_here = np.linalg.norm(trajs_here[:,1:] - trajs_here[:,:-1], axis=-1) # N,S-1
                    #     # only count motion if it's visible
                    #     mot_mean = np.sum(dists_here*visibs_here[:,1:], axis=1)  / (1 + np.sum(visibs_here[:,1:], axis=1))
                    #     inds = np.argsort(-mot_mean)[:keep]
                        
                    #     trajs_here = trajs_here[inds]
                    #     visibs_here = visibs_here[inds]
                    #     tids_here = tids_here[inds]
                    #     N = len(tids_here)

                    # choose a different point for each zoom level
                    for ni in range(self.N_per):
                        for zoom in zooms:
                            tid = ni + self.N_per*(zoom-1)
                            if tid < N:
                                tid = tids_here[tid]
                                self.all_info.append([video_fn, tid, stride, full_idx, zoom])
                    sys.stdout.write('.')
                    sys.stdout.flush()

        print(f"found {len(self.all_info)} samples in {dataset_location}")

        

    def getitem_helper(self, index):
        video_fn, tid, stride, full_idx, zoom = self.all_info[index]

        ds = np.load(video_fn, allow_pickle=True)
        rgbs, tracks, visibs = ds['video'], ds['tracks'], ds['visibles']
        rgbs = rgbs[full_idx]
        visibs = visibs[tid][full_idx].astype(np.float32)
        
        xys = tracks[tid][full_idx]
        S, H, W, C = rgbs.shape
        print('rgbs', rgbs.shape)

        xys = utils.misc.data_replace_with_nearest(xys, visibs)
        
        safe = np.concatenate(([0], visibs[:-2] * visibs[1:-1] * visibs[2:], [0]))
        print('safe', safe)
        if np.sum(safe) < 2: return None

        valids = visibs[:]
        if zoom > 1:
            xys, visibs, valids, rgbs = utils.misc.data_zoom(zoom, xys, visibs, valids, rgbs)
            S,H,W,C = rgbs.shape
        visibs = 1.0 * (visibs>0.5)

        print('rgbs zoom', rgbs.shape)
        
        d = {
            "rgbs": rgbs.astype(np.uint8),  # S, H, W, C
            "xys": xys.astype(np.int64),  # S, 2
            "visibs": visibs.astype(np.float32),  # S
        }
        return d

    def __len__(self):
        return len(self.all_info)
