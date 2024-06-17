from numpy import random
from numpy.core.numeric import full
import torch
import numpy as np
import pickle
from datasets.dataset import PointDataset
from datasets.dataset_utils import make_split
import utils.misc

class RoboTapDataset(PointDataset):
    def __init__(
            self,
            dataset_location="/data/robotap",
            dset="TRAIN",
            use_augs=False,
            rand_frames=False,
            S=8, fullseq=False, chunk=None,
            strides=[1,2,3],
            crop_size=(368, 496),
            is_training=True,
    ):
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            strides=strides,
            crop_size=crop_size, 
            is_training=is_training,
        )
        self.train_pkls = ['robotap_split0.pkl', 'robotap_split1.pkl', 'robotap_split2.pkl']
        # self.train_pkls = ['robotap_split3.pkl', 'robotap_split4.pkl']
        self.val_pkls = ['robotap_split3.pkl', 'robotap_split4.pkl']

        clip_step = S//2
        if not is_training:
            strides = [2]

        self.use_augs = use_augs
        print("loading robotap dataset...")
        self.vid_pkls = self.train_pkls if is_training else self.val_pkls
        self.data = []

        for vid_pkl in self.vid_pkls[:1]:
            print(vid_pkl)
            input_path = "%s/%s" % (dataset_location, vid_pkl)
            print('input_path', input_path)
            with open(input_path, "rb") as f:
                data = pickle.load(f)
            keys = list(data.keys())
            print(len(keys))
            self.data += [data[key] for key in keys]

        print("found %d videos in %s" % (len(self.data), dataset_location))

        self.all_data_idx = []
        self.all_full_idx = []
        self.all_kp_idx = []

        
        for di in range(len(self.data)):
            dat = self.data[di]
            trajs = dat["points"]  # N,S,2 array
            visibs = 1 - dat["occluded"]  # N,S array

            trajs = trajs.transpose(1, 0, 2)  # S,N,2
            visibs = visibs.transpose(1, 0)  # S,N
            
            S_local, N = visibs.shape
            for stride in strides:
                for ii in range(0, S_local, clip_step*stride):
                    full_idx = ii + np.arange(self.S) * stride
                    full_idx = [ij for ij in full_idx if ij < S_local]
                    if len(full_idx) < (((S_local - 1) // stride + 1 if self.S == -1 else self.S) if fullseq else 8):
                        continue
                    
                    visibs_here = visibs[full_idx] # S,N
                    trajs_here = trajs[full_idx] # S,N,2
                    for ni in range(N):
                        traj_here = trajs_here[:,ni]
                        val_here = visibs_here[:,ni]
                        if np.sum(val_here) < 4: continue
                        traj = traj_here[val_here]
                        # print('traj', traj, traj.shape)
                        travel = np.linalg.norm(traj[1:]-traj[:-1], axis=-1).sum() # [0,1]
                        # print('travel', travel)
                        if travel > 0.01:
                            self.all_data_idx.append(di)
                            self.all_full_idx.append(full_idx)
                            self.all_kp_idx.append(ni)
                        # print('di', di) 
                        # print('full_idx', full_idx, 'visibs_here', visibs_here)
        print('found %d samples in %s' % (len(self.all_full_idx), dataset_location))

    def __len__(self):
        return len(self.all_full_idx)

    def getitem_helper(self, index):
        # print('index', index)
        data_idx = self.all_data_idx[index]
        full_idx = self.all_full_idx[index]
        kp_idx = self.all_kp_idx[index]
        # print('data_idx', data_idx)
        
        dat = self.data[data_idx]
        rgbs = dat["video"]  # list of H,W,C uint8 images
        trajs = dat["points"]  # N,S,2 array
        visibs = 1 - dat["occluded"]  # N,S array

        # note the annotations are only valid when not occluded
        
        trajs = trajs.transpose(1, 0, 2)  # S,N,2
        visibs = visibs.transpose(1, 0)  # S,N

        rgbs = [rgbs[idx] for idx in full_idx]
        trajs = trajs[full_idx]
        visibs = visibs[full_idx]

        # 1.0,1.0 should lie at the bottom-right corner pixel
        H, W, C = rgbs[0].shape
        trajs[:, :, 0] *= W - 1
        trajs[:, :, 1] *= H - 1
        trajs = trajs.round().astype(int)

        S = len(rgbs)

        # clamp to image bounds
        trajs = np.minimum(
            np.maximum(trajs, np.zeros((2,), dtype=int)), np.array([W, H]) - 1
        )  # S,2

        N = visibs.shape[1]

        xys = trajs[:, kp_idx]  # S,2
        visibs = visibs[:, kp_idx]  # S

        rgbs = np.stack(rgbs, 0)

        xys = utils.misc.data_replace_with_nearest(xys, visibs)

        sample = {
            "rgbs": rgbs,
            "xys": xys,
            "visibs": visibs,
        }
        return sample
