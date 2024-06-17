import os
import json
import numpy as np
from pycocotools.coco import COCO
from PIL import Image
import torch
from datasets.dataset import PointDataset
import csv
from datasets.dataset_utils import make_split
import glob
import pickle
import utils.misc


class DeepFly3dDataset(PointDataset):
    def __init__(self,
                 dataset_location='/orion/group/deepfly3d',
                 use_augs=False,
                 S=16, fullseq=False, chunk=None,
                 strides=[1,2],
                 zooms=[1,1.5],
                 crop_size=(368, 496),
                 is_training=True,
    ):
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size, 
            use_augs=use_augs,
            is_training=is_training
        )
        print('loading deepfly dataset...')

        clip_step = S//2
        if not is_training:
            clip_step = S
            strides = [1]
            zooms = [1]
            
        self.S = S

        self.use_augs = use_augs
        self.data = []

        seqs = sorted(os.listdir(dataset_location))
        seqs = [seq for seq in seqs if 'image' in seq]
        print('found', len(seqs))
        seqs = make_split(seqs, is_training, shuffle=True)
        # print('seqs', seqs)

        for seq in seqs:
            seq_path = os.path.join(dataset_location, seq)
            pkl_path = glob.glob(os.path.join(seq_path, 'df3d', 'df3d_result*.pkl'))[0]
            annots = pickle.load(open(pkl_path, 'rb'))
            cam_num, S_local = annots['points2d'].shape[:2]
            print('S_local', S_local)
            for stride in strides:
                for cam in range(cam_num):
                    for ii in range(0, S_local, clip_step*stride):
                        # we'll collect info on the zeroth timestep here
                        joints2d = annots['points2d'][cam, ii][:19] # 19,2
                        # print('joints2d', joints2d.shape)
                        valids_2d = np.logical_and(joints2d[:, 0] > 0, joints2d[:, 1] > 0)
                        confidence = annots['heatmap_confidence'][cam, ii]
                        # print('confidence', confidence.shape)
                        valids = confidence > 0.8
                        # print('valids', valids.shape)
                        # input()
                        valids = np.logical_and(valids[:, 0], valids_2d)
                        joint_idxs = np.where(valids)[0]
                        full_idx = ii + np.arange(self.S) * stride
                        if full_idx[-1] >= S_local: continue # this enforces fullseq
                        for j_idx in joint_idxs:
                            img_path = [os.path.join(seq_path, 'camera_{}_img_{}.jpg'.format(cam, str(jj).zfill(6))) for jj in full_idx]
                            joint_img = [annots['points2d'][cam, jj, j_idx] for jj in full_idx]
                            vis = [(annots['heatmap_confidence'][cam, jj, j_idx, 0] > 0.5) * (annots['points2d'][cam, jj, j_idx][0] > 0)
                                   * (annots['points2d'][cam, jj, j_idx][1] > 0) for jj in full_idx]

                            # measure travel to avoid boring samples
                            xys = np.stack(joint_img)
                            # eliminate jitter before measuring motion
                            xys = xys[1:]*0.5 + xys[:-1]*0.5
                            travel = np.sum(np.linalg.norm(xys[1:] - xys[:-1], axis=-1))
                            if travel < 0.2: continue
                            if np.sum(vis) < 4: continue

                            for zoom in zooms:
                                self.data.append({
                                    'img_path': img_path,
                                    'joint_img': joint_img,
                                    'vis': vis,
                                    'zoom': zoom,
                                })
                                
        print('loaded {} samples'.format(len(self.data)))

        if chunk is not None:
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.data = chunkify(self.data,100)[chunk]
            print('filtered to %d samples' % len(self.data))
            # print('self.data', self.data)

    def __len__(self):
        return len(self.data)

    def getitem_helper(self, index):
        # Collecting continuous frames for the clip
        clip_data = self.data[index]

        img_path = clip_data['img_path']
        img_list = [np.array(Image.open(im))[..., None].repeat(3, axis=2) for im in img_path]
        xys = clip_data['joint_img']
        visibs = clip_data['vis']
        zoom = clip_data['zoom']

        # print('joint_img', np.stack(joint_img).shape)
        # print('visibs', np.stack(visibs).shape)

        rgbs = np.stack(img_list, axis=0)
        xys = np.stack(xys, axis=0)
        
        xys[..., 0] = xys[..., 0] * 480
        xys[..., 1] = xys[..., 1] * 960
        xys = xys[..., ::-1]
        visibs = np.stack(visibs, axis=0)

        xys = xys.reshape(-1, 2)
        visibs = visibs.reshape(-1)

        S = xys.shape[0]

        xys = utils.misc.data_replace_with_nearest(xys, visibs)
        valids = np.ones_like(visibs)
        
        if zoom > 1:
            xys, visibs, valids, rgbs = utils.misc.data_zoom(zoom, xys, visibs, valids, rgbs)
        safe = np.concatenate(([0], visibs[:-2] * visibs[1:-1] * visibs[2:], [0]))
        if np.sum(safe) < 2: return None

        sample = {
            'rgbs': rgbs,
            'xys': xys,
            'visibs': visibs,
        }
        return sample

