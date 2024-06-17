from numpy import random
import torch
import numpy as np
import os
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import random
from torch._C import dtype, set_flush_denormal
import utils.basic
import utils.improc
import glob
import cv2
from torchvision.transforms import ColorJitter, GaussianBlur
from datasets.dataset import PointDataset
from scipy import stats

class PointOdysseyDataset(PointDataset):
    def __init__(self,
                 dataset_location,
                 use_augs=False,
                 S=8, fullseq=False, chunk=None,
                 strides=[2,3,4],
                 zooms=[1,2,3],
                 crop_size=(368, 496),
                 is_training=True):
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training
        )
        print('loading pointodyssey dataset...')

        if self.is_training:
            dset = 'train'
            self.N_per = 32
        else:
            dset = 'val'
            self.N_per = 2
            strides = [1]
            
        self.S = S
        self.base_strides = strides
        self.num_strides = len(strides)

        self.use_augs = use_augs
        self.traj_paths = []
        self.subdirs = []
        self.sequences = []

        self.subdirs.append(os.path.join(dataset_location, dset))

        for subdir in self.subdirs:
            for seq in glob.glob(os.path.join(subdir, "*")):
                if os.path.isdir(seq):
                    seq_name = seq.split('/')[-1]
                    self.sequences.append(seq)
        self.sequences = sorted(self.sequences)
        print('found %d unique videos in %s (dset=%s)' % (len(self.sequences), dataset_location, dset))

        if chunk is not None:
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.sequences = chunkify(self.sequences,100)[chunk]
            print('filtered to %d sequences' % len(self.sequences))
            print('self.sequences', self.sequences)
        
        self.rgb_paths = []
        self.mask_paths = []
        self.full_idxs = []
        # self.annotation_paths = []
        self.tids = []
        self.sids = []
        self.seq_names = []
        self.strides = []
        self.zooms = []

        self.all_trajs = []
        self.all_visibs = []
        self.all_valids = []

        H,W = 540,960

        clip_step = S//2

        self.N_total = self.N_per*len(zooms)*len(strides)

        print('N_total', self.N_total)
        
        # self.anno_dict = {}
        # self.anno_list= []
        self.trajs_list= []
        self.visibs_list= []
        self.valids_list= []
        
        ## load trajectories
        print('loading trajectories...')
        for sid, seq in enumerate(self.sequences):
            rgb_path = os.path.join(seq, 'rgbs')
            annotations_path = os.path.join(seq, 'anno.npz')
            if os.path.isfile(annotations_path):
                annotations = np.load(annotations_path, allow_pickle=True)
                trajs = annotations['trajs_2d'].astype(np.float32) # S,N,2
                visibs = annotations['visibs'].astype(np.float32) # S,N
                valids = annotations['valids'].astype(np.float32) # S,N

                # self.all_trajs.append(trajs[full_idx])
                # self.all_visibs.append(visibs[full_idx])
                # self.all_valids.append(valids[full_idx])

                # seqname = seq.split('/')[-1]
                # print('seq', seq)
                # print('seqname', seqname)

                # self.anno_dict[seqname] = annotations

                # self.anno_list.append(annotations.copy())
                self.trajs_list.append(trajs)
                self.visibs_list.append(visibs)
                self.valids_list.append(valids)
                
                for si, stride in enumerate(strides):
                    S_local = len(os.listdir(rgb_path))
                    for ii in range(0,max(S_local-self.S*stride+1, 1), clip_step*stride):
                        full_idx = ii + np.arange(self.S)*stride
                        full_idx = [ij for ij in full_idx if ij < S_local]

                        if len(full_idx)==self.S: # always fullseq
                            for zi, zoom in enumerate(zooms):
                                for ni in range(self.N_per):
                                    ti = ni + zi*self.N_per + si*len(zooms)*self.N_per

                                    # if ii==0:
                                    #     print('si, zi, ti', si, zi, ti)
                                        
                                    self.rgb_paths.append([os.path.join(seq, 'rgbs', 'rgb_%05d.jpg' % idx) for idx in full_idx])
                                    self.mask_paths.append([os.path.join(seq, 'masks', 'mask_%05d.png' % idx) for idx in full_idx])
                                    # self.annotation_paths.append(os.path.join(seq, 'anno.npz'))
                                    self.full_idxs.append(full_idx)
                                    self.tids.append(ti)
                                    self.strides.append(stride)
                                    # self.seq_names.append(seqname)
                                    self.sids.append(sid)
                                    self.zooms.append(zoom)
                                    # print('added trajs')

        print('collected %d clips of length %d in %s (dset=%s)' % (
            len(self.rgb_paths), self.S, dataset_location, dset))


    def getitem_helper(self, index):

        rgb_paths = self.rgb_paths[index]
        mask_paths = self.mask_paths[index]
        # seq_name = self.seq_names[index]

        sid = self.sids[index]
        # annotations = self.anno_list[sid]
        # 
        # annotations = self.anno_dict[seq_name]
        # print('annotations', annotations)
        # print('annotations[trajs_2d', annotations['trajs_2d'].shape)
        
        full_idx = self.full_idxs[index]

        trajs = self.trajs_list[sid][full_idx].astype(np.float32) # S,N,2
        visibs = self.visibs_list[sid][full_idx].astype(np.float32) # S,N
        valids = self.valids_list[sid][full_idx].astype(np.float32) # S,N
        
        # annotations_path = self.annotation_paths[index]
        # print('annotations_path', annotations_path)
        # annotations = np.load(annotations_path, allow_pickle=True)
        # trajs = annotations['trajs_2d'][full_idx].astype(np.float32) # S,N,2
        # visibs = annotations['visibs'][full_idx].astype(np.float32) # S,N
        # valids = annotations['valids'][full_idx].astype(np.float32) # S,N
        # trajs = self.all_trajs[index]
        # visibs = self.all_visibs[index]
        # valids = self.all_valids[index]
        tid = self.tids[index]
        stride = self.strides[index]
        zoom = self.zooms[index]
        stride_ind = self.base_strides.index(stride)
        # print('full_idx', full_idx)
        # print('stride', stride, 'stride_ind', stride_ind)

        S,N,D = trajs.shape
        assert(D==2)
        assert(S==self.S)

        if N < self.N_per*self.num_strides:
            print('returning early: N=%d; need N=%d' % (N, self.N_per*self.num_strides))
            return None
        
        # get rid of infs and nans
        valids_xy = np.ones_like(trajs)
        inf_idx = np.where(np.isinf(trajs))
        trajs[inf_idx] = 0
        valids_xy[inf_idx] = 0
        nan_idx = np.where(np.isnan(trajs))
        trajs[nan_idx] = 0
        valids_xy[nan_idx] = 0
        inv_idx = np.where(np.sum(valids_xy, axis=2)<2) # S,N
        visibs[inv_idx] = 0
        valids[inv_idx] = 0

        # print('N notinf', trajs.shape[1])

        H,W = 540,960

        inbounds = valids.copy()

        # update visibility annotations
        for si in range(S):
            # avoid 1px edge
            oob_inds = np.logical_or(
                np.logical_or(trajs[si,:,0] < 1, trajs[si,:,0] > W-2),
                np.logical_or(trajs[si,:,1] < 1, trajs[si,:,1] > H-2))
            visibs[si,oob_inds] = 0
            inbounds[si,oob_inds] = 0

            # when a point moves far oob, don't supervise with it
            very_oob_inds = np.logical_or(
                np.logical_or(trajs[si,:,0] < -128, trajs[si,:,0] > W+128),
                np.logical_or(trajs[si,:,1] < -128, trajs[si,:,1] > H+128))
            valids[si,very_oob_inds] = 0

        # # ensure that the point is good in frame0
        # vis_and_val = valids * visibs
        # vis0 = vis_and_val[0] > 0
        # trajs = trajs[:,vis0]
        # visibs = visibs[:,vis0]
        # valids = valids[:,vis0]

        # # ensure that the point is good in frame1
        # vis_and_val = valids * visibs
        # vis1 = vis_and_val[1] > 0
        # trajs = trajs[:,vis1]
        # visibs = visibs[:,vis1]
        # valids = valids[:,vis1]

        # req 3 inbounds
        safe = inbounds[:-2] * inbounds[1:-1] * inbounds[2:]
        safe_ok = np.sum(safe, axis=0) >= 4
        trajs = trajs[:,safe_ok]
        visibs = visibs[:,safe_ok]
        valids = valids[:,safe_ok]
        inbounds = inbounds[:,safe_ok]
        print('N inb', trajs.shape[1])
        
        # req 3 safe
        safe = visibs[:-2] * visibs[1:-1] * visibs[2:]
        safe_ok = np.sum(safe, axis=0) >= 4
        trajs = trajs[:,safe_ok]
        visibs = visibs[:,safe_ok]
        valids = valids[:,safe_ok]
        inbounds = inbounds[:,safe_ok]
        print('N safe', valids.shape[1])

        # # ensure that the per-frame motion isn't too crazy
        # vels = trajs[1:] - trajs[:-1] # S-1,N,2
        # accels = vels[1:] - vels[:-1] # S-2,N,2
        # vnorms = np.linalg.norm(vels, axis=-1)
        # anorms = np.linalg.norm(accels, axis=-1)
        # vel_max = np.max(vnorms, axis=0) # N
        # accel_max = np.max(anorms, axis=0) # N
        # vel_mean = np.mean(vnorms, axis=0) # N
        # # print('vel_mean', vel_mean)
        # # print('vel_max', vel_max)
        # # print('accel_max', accel_max)
        # mot_ok = (vel_max < 256) & (accel_max < 64)
        # trajs = trajs[:,mot_ok]
        # visibs = visibs[:,mot_ok]
        # valids = valids[:,mot_ok]
        # inbounds = inbounds[:,mot_ok]
        # print('N1', valids.shape[1])
        
        N = trajs.shape[1]
        # print('N', N)

        if N < self.N_per*self.num_strides:
            print('N=%d' % (N))
            return None

        segs = []
        for mask_path in mask_paths:
            with Image.open(mask_path) as im:
                mask = np.array(im)
                if np.sum(mask==0) > 128:
                    # fill holes caused by fog/smoke
                    mask_filled = cv2.medianBlur(mask, 7)
                    mask[mask==0] = mask_filled[mask==0]
                segs.append(mask) # H,W
        segs = np.stack(segs, axis=0) # S,H,W
        # print('segs', segs.shape)

        # # compute edge map 
        # edges = []
        # kernel = np.ones((3,3), np.uint8)
        # dilate_iters = 2 #max(int(H/self.resize_size[0]),1)
        # for si in range(S):
        #     edge = cv2.Canny(segs[si], 1, 1)
        #     # # block apparent edges from fog/smoke
        #     # block all seg0
        #     # keep = 1 - (segs[si]==0).astype(np.uint8)
        #     # keep = 1 - cv2.dilate((segs[si]==0).astype(np.uint8), kernel, iterations=1) 
        #     # edge = edge * keep
        #     edge = cv2.dilate(edge, kernel, iterations=dilate_iters)
        #     edges.append(edge)
            
        # # discard trajs that begin exactly on segmentation boundaries
        # # since their labels are ambiguous
        # # xys = np.minimum(np.maximum(xys, np.zeros((2,), dtype=int)), np.array([W, H]) - 1) # S,2
        # x0, y0 = trajs[0,:,0].astype(np.int32), trajs[0,:,1].astype(np.int32)
        # x0 = np.minimum(np.maximum(x0, 0), W-1)
        # y0 = np.minimum(np.maximum(y0, 0), H-1)
        # on_edge = edges[0][y0,x0] > 0
        # trajs = trajs[:,~on_edge]
        # visibs = visibs[:,~on_edge]
        # valids = valids[:,~on_edge]
        # inbounds = inbounds[:,~on_edge]

        if N < tid:
            print('N=%d' % (N))
            return None
        
        # # let's take points spaced apart
        # if N > self.N_per*self.num_strides:
        #     inds = utils.misc.farthest_point_sample_py(trajs[0], self.N_per*self.num_strides, deterministic=True)
        #     trajs = trajs[:,inds]
        #     visibs = visibs[:,inds]
        #     valids = valids[:,inds]

        # # if we have a surplus of points, take ones spaced apart
        # if N > 2*self.N_per*self.num_strides:
        #     inds = utils.misc.farthest_point_sample_py(trajs[0], 2*self.N_per*self.num_strides, deterministic=True)
        #     trajs = trajs[:,inds]
        #     visibs = visibs[:,inds]
        #     valids = valids[:,inds]

        # # if we have a surplus of points, take ones spaced apart
        # if N > 2*self.N_per*self.num_strides:
        #     inds = utils.misc.farthest_point_sample_py(trajs[0], 2*self.N_per*self.num_strides, deterministic=True)
        #     trajs = trajs[:,inds]
        #     visibs = visibs[:,inds]
        #     valids = valids[:,inds]

        # print('N1', trajs.shape[1])
        if N > 10000: 
            # we prefer points that stay inbounds
            keep = 10000
            inbound_mean = np.mean(inbounds, axis=0) # N
            inds = np.argsort(-inbound_mean)[:keep]
            trajs = trajs[:,inds]
            visibs = visibs[:,inds]
            valids = valids[:,inds]
            inbounds = inbounds[:,inds]
        # print('N2', trajs.shape[1])

        if N > self.N_total*3:
            # we prefer points that travel
            keep = self.N_total*3
            dists = np.linalg.norm(trajs[1:] - trajs[:-1], axis=-1) # S-1,N
            # only count motion if it's valid
            mot_mean = np.mean(dists*valids[1:], axis=0) # N
            inds = np.argsort(-mot_mean)[:keep]
            trajs = trajs[:,inds]
            visibs = visibs[:,inds]
            valids = valids[:,inds]
            inbounds = inbounds[:,inds]
        
        if N > self.N_total:
            # take points spaced apart
            keep = self.N_total
            inds = utils.misc.farthest_point_sample_py(trajs.mean(axis=0), keep, deterministic=True)
            trajs = trajs[:,inds]
            visibs = visibs[:,inds]
            valids = valids[:,inds]
            inbounds = inbounds[:,inds]
        # print('N5', trajs.shape[1])
        # print('N3', trajs.shape[1])

        # if N > self.N_total:
        #     # we prefer points with a mix of vis and occ (when inbounds)
        #     keep = self.N_total
        #     prod = visibs*inbounds # S,N
        #     numer = np.sum(prod, axis=0)
        #     denom = 1+np.sum(valids, axis=0) # count all valids in denom
        #     mean = numer/denom
        #     dist = np.abs(mean-0.5)

        #     inds = np.argsort(dist)[:keep]
        #     trajs = trajs[:,inds]
        #     visibs = visibs[:,inds]
        #     valids = valids[:,inds]
        #     inbounds = inbounds[:,inds]
        #     # print('kept vis means', mean[inds])
        # print('N4', trajs.shape[1])
        
        # lock to N=1
        # pick a different index for each stride
        xys = trajs[:,tid] # S,2
        visibs = visibs[:,tid] # S
        valids = valids[:,tid] # S
        inbounds = inbounds[:,tid] # S

        # clamp to image bounds
        xys = np.minimum(np.maximum(xys, np.zeros((2,), dtype=int)), np.array([W, H]) - 1) # S,2

        # read images last
        rgbs = []
        for rgb_path in rgb_paths:
            with Image.open(rgb_path) as im:
                rgbs.append(np.array(im)[:, :, :3])
        rgbs = np.stack(rgbs, axis=0) # S,H,W,3
        # print('pod in rgbs', rgbs.shape)

        segA = segs.reshape(-1)
        obj_ids = np.unique(segA) # NSeg

        xys_ = xys.round().astype(np.int32)
        gotit = False

        vis_valid = visibs * valids * inbounds
        safe = np.concatenate([visibs[0:1]*0, visibs[:-2] * visibs[1:-1] * visibs[2:], visibs[0:1]*0])
        # print('sum(safe)', np.sum(safe))
        oids = []
        for si in range(S):
            if safe[si]:
                oid = segs[si,xys_[si,1],xys_[si,0]]
                oids.append(oid)
        oid = stats.mode(oids)[0]
        oids = np.array(oids)
        match = np.mean(oid==oids)
        print('oids', oids, 'picked', oid, 'match', match)
        if match < 0.7:
            print('bad match')
            return None
        
        if oid==0:
            # mask may be weird, so we will not use it
            masks = np.zeros_like(rgbs[:,:,:,0])
            # print('accidentally picked oid=0 (too risky in pod)')
            # return None
        else:
            masks = [(seg == oid).astype(np.float32) for seg in segs]
            masks = np.stack(masks, axis=0)

            # other_masks = [(seg != oid).astype(np.float32) for seg in segs]
            # other_masks = np.stack(other_masks, axis=0)
            
        masks = masks.astype(np.float32)

        kernel = np.ones((3,3), np.uint8)
        masks2 = masks*0
        for si in range(S):
            xy = xys[si].round().astype(np.int32)
            x, y = xy[0], xy[1]
            x = x.clip(0,W-1)
            y = y.clip(0,H-1)
            
            mask = masks[si].copy()
            mask[y,x] = 1
            mask_fat = cv2.dilate(mask.astype(np.uint8), kernel, iterations=8).astype(np.float32)

            # if mask[y,x] > 0:
            
            for alt_id in obj_ids:
                if alt_id!=oid and alt_id!=0:
                    alt_mask = (segs[si]==alt_id).astype(np.float32)
                    if np.sum(alt_mask*mask_fat)==0:
                        masks2[si] += alt_mask
            # if not oid==0:
            #     # add bkg
            #     alt_mask = (segs[si]==0).astype(np.float32)
            #     masks2[si] += alt_mask
                    
            # keep = 1 - cv2.dilate((segs[si]==0).astype(np.uint8), kernel, iterations=1) 
            # edge = edge * keep
            # edge = cv2.dilate(edge, kernel, iterations=dilate_iters)
        masks2 = 1-np.clip(masks2,0.5,1)
        masks2[masks>0.5] = 1

        # if np.random.rand() < 0.5:
        #     # reverse the video
        #     rgbs = np.flip(rgbs, axis=0)
        #     trajs = np.flip(trajs, axis=0)
        #     visibs = np.flip(visibs, axis=0)

        # sc = 0.5
        # W_, H_ = int(W*sc), int(H*sc)
        # rgbs = [cv2.resize(rgb, (W_, H_), interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
        # rgbs = np.stack(rgbs, axis=0)
        # trajs *= 0.5

        # print('pod resized rgbs', rgbs.shape)
        
        if zoom > 1:
            xys, visibs, valids, rgbs, masks, masks2 = utils.misc.data_zoom(zoom, xys, visibs, valids, rgbs, masks, masks2)
            _, H, W, _ = rgbs.shape
            safe = visibs[:-2] * visibs[1:-1] * visibs[2:]
            if np.sum(safe) < 2: return None

        masks0 = np.zeros_like(masks)
        masks1 = np.zeros_like(masks)
        # masks2 = np.zeros_like(masks)
        for si in range(len(masks)):
            if valids[si]:
                xy = xys[si].round().astype(np.int32)
                x, y = xy[0], xy[1]
                x = x.clip(0,W-1)
                y = y.clip(0,H-1)
                masks0[si,y,x] = 1

                if masks[si,y,x] > 0:
                    # use the smallest component 
                    num_labels, labels = cv2.connectedComponents(masks[si].astype(np.uint8))
                    best = labels[y,x]
                    bin_mask = labels==best
                    masks1[si] = bin_mask.astype(np.float32)
                    # masks2[si] = masks[si]
                else:
                    masks1[si] = masks[si]
                        # else:
                    #     masks1[si] = 0.5
                    #     masks2[si] = 0.5
            else:
                masks0[si] = 0.5
                masks1[si] = 0.5
                # masks2[si] = 0.5
                
        if oid > 0:
            full_masks = np.stack([masks0, masks1, masks2], axis=-1)

            # chans 0,1,2 valid when points are valid
            masks_valid = np.zeros((S,3), dtype=np.float32)
            masks_valid[:,0] = valids
            masks_valid[:,1] = valids
            masks_valid[:,2] = valids
        else:
            # only chan0 valid
            full_masks = np.stack([masks0, masks*0, masks*0], axis=-1)
            masks_valid = np.zeros((S,3), dtype=np.float32)
            masks_valid[:,0] = valids
            
        sample = {
            'rgbs': rgbs,
            'masks': full_masks,
            'masks_valid': masks_valid,
            'xys': xys,
            'visibs': visibs,
            'valids': valids,
        }
        return sample

    def __len__(self):
        return len(self.rgb_paths)
