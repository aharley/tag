import os
import json
import numpy as np
from pycocotools.coco import COCO
from PIL import Image
import torch
from datasets.dataset import PointDataset
import csv
from datasets.dataset_utils import make_split
import cv2
import utils.misc

# uids_exclude = [26,27,28,30,33,34,35,37,43,45,46,48,49,
# 3048,319,9714,9965,11136,10016,10478,11829,12160,11138,11891,11139,11840,
#               12169,12169,12170,12171,12172,]

uids_exclude = []

class AcinoDataset(PointDataset):
    def __init__(self,
                 dataset_location='/orion/group/Acino_full/exported_data/',
                 use_augs=False,
                 S=16,
                 fullseq=False,
                 chunk=None,
                 # strides=[1,2,3,4], 
                 # zooms=[1,2],
                 strides=[2], 
                 zooms=[2],
                 crop_size=(368, 496),
                 is_training=True):
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size, 
            use_augs=use_augs,
            is_training=is_training
        )
        print('loading acino dataset...')

        clip_step = S//2
        if not is_training:
            clip_step = S
            strides = [1]
            zooms = [2]

        self.S = S

        self.use_augs = use_augs
        self.data = []

        seqs = os.listdir(dataset_location)
        print("found {:d} videos in {}".format(len(seqs), dataset_location))
        seqs = sorted(seqs)
        
        # seqs = make_split(seqs, is_training, shuffle=True)
        # print('seqs', len(seqs))

        assert(chunk is None)
        # if chunk is not None:
        #     def chunkify(lst,n):
        #         return [lst[i::n] for i in range(n)]
        #     seqs = chunkify(seqs,100)[chunk]
        #     print('filtered to %d sequences' % len(seqs))
        #     print('seqs', seqs)

        max_K = 25
        for sq, seq in enumerate(seqs):
            seq_path = os.path.join(dataset_location, seq)
            annots = self.load_data(seq_path)
            S_local = len(annots)
            if S_local==0:
                continue

            print('S_local', S_local)

            # import ipdb; ipdb.set_trace()

            all_img_paths = [annot["image_name"] for annot in annots]
            labels = [annot["labels"] for annot in annots]

            labels = np.stack(labels) # S,K,3
            trajs = labels[:,:,:2]
            visibs = labels[:,:,2]
            for j_idx in range(max_K):
                
                # for j_idx in [1,

                uid = sq*max_K + j_idx

                if uid in uids_exclude:
                    continue

                # if uid < np.max(uids_exclude):
                #     continue
                
                inds = np.nonzero(visibs[:,j_idx])[0]
                if len(inds)<5:
                    continue
                    
                # print('inds', inds)
                img_paths = [all_img_paths[ind] for ind in inds]
                xys = trajs[inds,j_idx]
                vis = visibs[inds,j_idx]

                S_here = xys.shape[0]
                
                if S_here > self.S:
                    full_idx = np.linspace(0, S_here-1, min(self.S, S_here), dtype=np.int32)

                    img_paths = [img_paths[fi] for fi in full_idx]
                    xys = xys[full_idx]
                    vis = vis[full_idx]

                # print('vis', vis)
                self.data.append({
                    'img_paths': img_paths,
                    'xys': xys,
                    'vis': vis,
                    # 'zoom': zoom,
                    'uid': uid,
                })
            
            

            # ii = 0
            
            # # for stride in strides:
            # #     for ii in range(0, S_local, clip_step*stride):
            # frame_ii = annots[ii]
            # img_paths = frame_ii['image_name']
            # trajs = frame_ii['labels'][:,:2]
            # visibs = frame_ii['labels'][:,2]
            # # print('joint_labels', joint_labels.shape)
            # # valids = joint_labels[:, 2] > 0

            # print('trajs', trajs.shape)
            

            # K = joint_labels.shape[0]
            # # # print('K', K)


            # # full_idx = np.linspace(0, S_local-1, min(self.S, S_local), dtype=np.int32)

            # # # if np.sum(valids) < 2: continue
            # # # joint_idxs = np.where(valids)[0]
            # # # # print('joint_idxs', joint_idxs)
            # # # # input()
            # # # full_idx = ii + np.arange(self.S)*stride
            # # # full_idx = [ij for ij in full_idx if ij < S_local]

            # # if len(full_idx) < (((S_local - 1) // stride + 1 if self.S == -1 else self.S) if fullseq else 6):
            # #     continue

            # for j_idx in range(K):

            #     for zoom in zooms:

            #         # jj = 0
                    
            #         # img_path = [annots[jj]["image_name"] for jj in full_idx]
            #         # xys = [annots[jj]["labels"][j_idx, :2] for jj in full_idx]
            #         # vis = [annots[jj]["labels"][j_idx, 2] for jj in full_idx]

            #         # if vis[0] == 0:
            #         #     continue

            #         # # req multiple visible timesteps
            #         # if np.sum(vis) < 4:
            #         #     continue

            #         # # discard the craziest data
            #         # xys_ = xys[vis>0]
            #         # vels = np.linalg.norm(xys_[1:]-xys_[:-1], axis=-1)/1920
            #         # if np.max(vels) > 0.4:
            #         #     continue

            #         # print('xys', xys)
            #         # print('vis', vis)
            #         self.data.append({
            #             'img_path': img_path,
            #             'xys': xys,
            #             'vis': vis,
            #             'zoom': zoom,
            #             'uid': uid,
            #         })
        print('loaded {} samples'.format(len(self.data)))

    def load_data(self, seq_path):
        # print('loading annotations from %s' % seq_path)

        mach = False # these machinelabels are even less reliable than the human annot
        if mach:
            annots_path = os.path.join(seq_path, 'machinelabels.csv')
        else:
            annots_path = os.path.join(seq_path, 'CollectedData_UCT.csv')
            
        if not os.path.exists(annots_path):
            # print('could not find', annots_path)
            return []
        all_data = []
        anno_data = {}
        print('annots_path', annots_path)

        with open(annots_path, 'r') as file:
            csv_reader = csv.reader(file)

            # Read the headers
            scorers = next(csv_reader)
            bodyparts = next(csv_reader)
            coords = next(csv_reader)

            # Read the image data
            for row in csv_reader:
                # Create a dictionary for each image
                image_data = {
                    "image_name": os.path.join(seq_path, row[0].split('/')[-1]),
                    "labels": []
                }

                if mach:
                    for i in range(len(row[1:]) // 3):
                        x = row[3 * i + 1]
                        y = row[3 * i + 2]
                        c = row[3 * i + 3]
                        pt = np.zeros(3)
                        if x != '' and y != '':
                            pt[0] = float(x)
                            pt[1] = float(y)
                            pt[2] = float(c)
                            # print('xyc', x,y,c)
                            # input()
                        image_data["labels"].append(pt)
                else:
                    for i in range(len(row[1:]) // 2):
                        x = row[2 * i + 1]
                        y = row[2 * i + 2]
                        pt = np.zeros(3)
                        if x != '' and y != '':
                            pt[0] = float(x)
                            pt[1] = float(y)
                            pt[2] = 1
                        image_data["labels"].append(pt)

                        
                image_data["labels"] = np.stack(image_data["labels"])
                anno_data[image_data['image_name']] = image_data["labels"]
        img_list = os.listdir(seq_path)
        img_list = [img for img in img_list if img.endswith('.png')]
        img_list.sort()
        for img in img_list:
            img_path = os.path.join(seq_path, img)
            if img_path in anno_data.keys():
                all_data.append({
                    "image_name": img_path,
                    "labels": anno_data[img_path]
                })
            else:
                all_data.append({
                    "image_name": img_path,
                    "labels": np.zeros((25, 3))
                })
        return all_data

    def __len__(self):
        return len(self.data)

    def getitem_helper(self, index):
        # Collecting continuous frames for the clip
        clip_data = self.data[index]
        xys = clip_data['xys']
        visibs = clip_data['vis']
        image_paths = clip_data['img_paths']
        # zoom = clip_data['zoom']
        uid = clip_data['uid']
        zoom = 2
        # print('image_paths', image_paths)

        print('uid', uid)


        # rgb = cv2.imread(str(image_paths[0]))
        # H, W = rgb.shape[:2]
        # sc = 1.0
        # if H > 384:
        #     sc = 384 / H
        #     H_, W_ = int(H * sc), int(W * sc)
        img_list = []
        for path in image_paths:
            rgb = cv2.imread(str(path))[..., ::-1].copy()
            # if sc < 1.0:
            #     rgb = cv2.resize(rgb, (W_, H_), interpolation=cv2.INTER_AREA)
            img_list.append(rgb)


        # print('img_list', len(img_list)

        # print('xys', np.stack(xys).shape)
        # print('visibs', np.stack(visibs).shape)

        rgbs = np.stack(img_list, axis=0)
        # xys = np.stack(xys, axis=0) #* sc
        # visibs = np.stack(visibs, axis=0)
        
        print('rgbs', rgbs.shape)
        
        xys = xys.reshape(-1,2)
        # print('xys', xys)
        visibs = visibs.reshape(-1)
        # print('visibs', visibs)
        # visibs = visibs / np.max(visibs)

        S = xys.shape[0]
        
        xys = utils.misc.data_replace_with_nearest(xys, visibs)
        valids = np.ones_like(visibs)
        
        # if zoom > 1:
        #     xys, visibs, valids, rgbs = utils.misc.data_zoom(zoom, xys, visibs, valids, rgbs)

        if zoom > 1:
            xys, visibs, valids, rgbs = utils.misc.data_zoom(zoom, xys, visibs, valids, rgbs)
            _, H, W, _ = rgbs.shape

        print('rgbs after zoom', rgbs. shape)
        
        # # in this dataset, the "safe" strategy produces nearly 0 samples,
        # # so we use visibs directly        
        # if np.sum(visibs) < 3:
        #     print('visibs', visibs)
        #     return None

        # rgbs = rgbs[visibs>0]
        # xys = xys[visibs>0]
        # visibs = visibs[visibs>0]

        # print('rgbs', rgbs.shape)
        # print('uid', uid)

        sample = {
            'rgbs': rgbs,
            'xys': xys,
            'visibs': visibs,
            'uid': np.array([uid]),
        }
        return sample

