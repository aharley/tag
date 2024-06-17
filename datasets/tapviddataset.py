from numpy import random
from numpy.core.numeric import full
import torch
import numpy as np
import pickle
from datasets.dataset import PointDataset
from datasets.dataset_utils import make_split
import utils.misc

class TapVidDataset(PointDataset):
    def __init__(
            self,
            dataset_location="/data/tapvid_davis",
            use_augs=False,
            fullseq=False,  # just hack, this dataset not refactored yet
            chunk=None,
            S=8,
            strides=[1],
            zooms=[1],
            crop_size=(368, 496),
            is_training=True,
    ):
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            fullseq=fullseq,
            strides=strides,
            crop_size=crop_size, 
            is_training=is_training,
        )
        
        # if not is_training:
        #     strides = [1]
        #     zooms = [1]
        #     clip_step = S
        # else:
        #     clip_step = S // 2

        clip_step = S // 2
        
            
        self.use_augs = use_augs
        print("loading tapvid dataset...")
        # use the whole dataset as val
        if not is_training:
            input_path = "%s/tapvid_davis.pkl" % dataset_location
            with open(input_path, "rb") as f:
                data = pickle.load(f)
            keys = list(data.keys())
            self.data = [data[key] for key in keys]
        else:
            self.data = []
        print("found %d videos in %s" % (len(self.data), dataset_location))

        self.all_data_idx = []
        self.all_full_idx = []
        self.all_kp_idx = []
        self.all_zooms = []
        
        for di in range(len(self.data)):
            dat = self.data[di]
            trajs = dat["points"]  # N,S,2 array
            visibs = 1 - dat["occluded"]  # N,S array

            trajs = trajs.transpose(1, 0, 2)  # S,N,2
            visibs = visibs.transpose(1, 0)  # S,N

            S_local, N = visibs.shape
            for stride in strides:
                for ii in range(0, max(S_local - self.S * stride, 1), clip_step):
                    # print('trimming tapvid')
                    # if ii > 0: continue 
                    full_idx = ii + np.arange(self.S) * stride
                    full_idx = [ij for ij in full_idx if ij < S_local]
                    if len(full_idx) == self.S: # always fullseq
                        visibs_here = visibs[full_idx] # S,N
                        for ni in range(N):
                            if np.sum(visibs_here[:,ni])<4: continue
                            if visibs_here[0,ni]==0: continue
                            for zoom in zooms:
                                self.all_data_idx.append(di)
                                self.all_full_idx.append(full_idx)
                                self.all_kp_idx.append(ni)
                                self.all_zooms.append(zoom)
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
        zoom = self.all_zooms[index]
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

        if zoom > 1:
            valids = visibs[:]
            xys, visibs, valids, rgbs = utils.misc.data_zoom(zoom, xys, visibs, valids, rgbs)

        if np.sum(visibs) < 4:
            print('visibs', np.sum(visibs))
            return None

        sample = {
            "rgbs": rgbs,
            "xys": xys,
            "visibs": visibs,
        }
        return sample
