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
from datasets.dataset import PointDataset, BBoxDataset
import albumentations as A
from datasets.dataset_utils import make_split
import utils.misc

class CrohdDataset(PointDataset):
    def __init__(
            self,
            dataset_location,
            S=32, fullseq=False, chunk=None,
            strides=[2],
            zooms=[1],
            crop_size=(384, 512),
            use_augs=False,
            is_training=True,
    ):
        print("loading crohd dataset...")
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training,
        )

        clip_step = S//2
        if not is_training:
            strides = [2]
            zooms = [2]
            clip_step = S
            

        dataset_dir = "%s/HT21" % dataset_location
        label_location = "%s/HT21Labels" % dataset_location
        subfolders = []

        dataset_dir = os.path.join(dataset_dir, "train")
        label_location = os.path.join(label_location, "train")
        subfolders = ["HT21-01", "HT21-02", "HT21-03", "HT21-04"]
        subfolders = make_split(subfolders, is_training, shuffle=True)

        print("loading data from {0}".format(dataset_dir))
        # read gt for subfolders
        self.dataset_dir = dataset_dir
        # self.seqlen = seqlen
        self.subfolders = subfolders
        self.folder_to_gt = (
            {}
        )  # key: folder name, value: dict with fields boxlist, valids, visibs
        # self.subfolder_lens = []
        # print('subfolders', len(subfolders))

        self.N_per = 16

        self.all_folders = []
        self.all_full_idx = []
        self.all_dicts = []
        self.all_tids = []
        self.all_zooms = []

        for fid, subfolder in enumerate(subfolders):
            print("loading labels for folder {0}/{1}".format(fid + 1, len(subfolders)))
            label_path = os.path.join(dataset_dir, subfolder, "gt/gt.txt")
            labels = np.loadtxt(label_path, delimiter=",")

            S_local = int(labels[-1, 0])
            n_heads = int(labels[:, 1].max())

            bboxes = np.zeros((S_local, n_heads, 4))
            valids = -1 * np.ones((S_local, n_heads))
            visibs = np.zeros((S_local, n_heads))

            # print('labels', labels.shape)

            for i in range(labels.shape[0]):
                (
                    frame_id,
                    head_id,
                    bb_left,
                    bb_top,
                    bb_width,
                    bb_height,
                    conf,
                    cid,
                    vis,
                ) = labels[i]
                frame_id = int(frame_id) - 1  # convert 1 indexed to 0 indexed
                head_id = int(head_id) - 1  # convert 1 indexed to 0 indexed

                valids[frame_id, head_id] = 1
                visibs[frame_id, head_id] = vis
                box_cur = np.array(
                    [bb_left, bb_top, bb_left + bb_width, bb_top + bb_height]
                )  # convert xywh to x1, y1, x2, y2
                bboxes[frame_id, head_id] = box_cur

            d = {
                "bboxes": np.copy(bboxes),
                "valids": np.copy(valids),
                "visibs": np.copy(visibs),
            }

            for stride in strides:
                for ii in range(0, S_local, clip_step*stride):
                    full_idx = ii + np.arange(self.S) * stride
                    full_idx = [ij for ij in full_idx if ij < S_local]
                    if len(full_idx) < (self.S if fullseq else 8): continue
                    
                    for tid in range(self.N_per):
                        for zoom in zooms:
                            self.all_folders.append(subfolder)
                            self.all_dicts.append(d)
                            self.all_tids.append(tid)
                            self.all_full_idx.append(full_idx)
                            self.all_zooms.append(zoom)

        print("found %d samples" % len(self.all_folders))


    def __len__(self):
        return len(self.all_folders)

    def getitem_helper(self, index):
        folder = self.all_folders[index]
        d = self.all_dicts[index]
        full_idx = self.all_full_idx[index]
        tid = self.all_tids[index]
        zoom = self.all_zooms[index]

        # # use different samples for different strides
        # tid = nid+(stride-1)*self.N_per

        S = self.S
        # bboxes = d["bboxes"][start_frame:end_frame] * 0.5  # we downsample by half, since the data is huge
        bboxes = d["bboxes"][full_idx]
        valids = d["valids"][full_idx]
        visibs = d["visibs"][full_idx]

        rgbs = []
        for ii in full_idx:
            rgb_path = os.path.join(
                self.dataset_dir, folder, "img1", str(ii + 1).zfill(6) + ".jpg"
            )
            rgb = Image.open(rgb_path)
            # rgb = cv2.imread(str(image_name))[..., ::-1].copy()
            # rgb = rgb.resize(
            #     (int(rgb.size[0] / 2), int(rgb.size[1] / 2)), Image.BILINEAR
            # )  # downsample by half
            
            rgbs.append(rgb)
        rgbs = np.stack(rgbs)  # S,C,H,W

        # print('rgbs', rgbs.shape)
        trajs = np.stack( [bboxes[:, :, [0, 2]].mean(2), bboxes[:, :, [1, 3]].mean(2)], axis=2)  # S,N,2
        # print('trajs', trajs.shape)
        # print('bboxes', bboxes.shape)

        S, N = trajs.shape[:2]
        H, W = 1080, 1920
        # update visibility annotations
        for si in range(S):
            # crohd annotations get noisy/wrong near edges
            oob_inds = np.logical_or(
                np.logical_or(trajs[si, :, 0] < 32, trajs[si, :, 0] > W - 32),
                np.logical_or(trajs[si, :, 1] < 32, trajs[si, :, 1] > H - 32),
            )
            visibs[si, oob_inds] = 0
            # exclude oob from eval
            valids[si, oob_inds] = 0

        vis_ok0 = visibs[0] > 0  # N
        vis_okE = visibs[-1] > 0  # N
        valid_ok0 = valids[0] > 0  # N
        valid_okE = valids[-1] > 0  # N
        mot_ok = (np.sum(np.linalg.norm(trajs[1:] - trajs[:-1], axis=-1), axis=0) > S)  # N
        # mot = np.sum(np.linalg.norm(trajs[1:] - trajs[:-1], axis=-1), axis=0)
        # print('mot', mot)

        # mot_ok = np.mean(np.linalg.norm(trajs[1:] - trajs[:-1]), axis=0) > 2.0 # N
        all_ok = (vis_ok0 & valid_ok0 & mot_ok & vis_okE & valid_okE)
        # print('np.sum(all_ok)', np.sum(all_ok))

        trajs = trajs[:, all_ok]
        bboxes = bboxes[:, all_ok]
        visibs = visibs[:, all_ok]
        valids = valids[:, all_ok]

        S, N = trajs.shape[:2]
        # print('trajs.shape', trajs.shape)
        

        # trajs = trajs[:,tid+(stride-1)*self.N_per] # S,2
        # visibs = visibs[:,tid+(stride-1)*self.N_per] # S
        # valids = valids[:,tid+(stride-1)*self.N_per] # S
        


        # # to improve the visuals, let's avoid shooting the traj oob
        # for ni in range(N):
        #     for si in range(1,S):
        #         if visibs[si,ni]==0:
        #             trajs[si,ni] = trajs[si-1,ni]

        if N <= tid:
            return None

        bboxes = bboxes[:, tid]
        visibs = visibs[:, tid]
        xys = trajs[:, tid]

        if zoom > 1:
            valids = visibs[:]
            xys, visibs, valids, rgbs = utils.misc.data_zoom(zoom, xys, visibs, valids, rgbs)
        
        sample = {
            "rgbs": rgbs,
            "xys": xys,
            "visibs": visibs,
        }
        return sample


class CrohdTestDataset(PointDataset):
    def __init__(
        self,
        dataset_location,
        S=32,
        crop_size=(384, 512),
    ):
        print("loading crohd dataset...")
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size,
        )

        dataset_dir = "%s/HT21" % dataset_location
        label_location = "%s/HT21Labels" % dataset_location
        subfolders = []

        dataset_dir = os.path.join(dataset_dir, "train")
        label_location = os.path.join(label_location, "train")
        subfolders = ["HT21-01", "HT21-02", "HT21-03", "HT21-04"] # TODO, add more
        subfolders = make_split(subfolders, False, shuffle=False)
        
        print("loading data from {0}".format(dataset_dir))
        # read gt for subfolders
        self.dataset_dir = dataset_dir
        # self.seqlen = seqlen
        self.subfolders = subfolders
        self.folder_to_gt = (
            {}
        )  # key: folder name, value: dict with fields boxlist, valids, visibs
        # self.subfolder_lens = []

        self.N_per = 16

        self.all_folders = []
        self.all_start_frames = []
        self.all_end_frames = []
        self.all_dicts = []
        self.all_tids = []
        self.all_strides = []

        for fid, subfolder in enumerate(subfolders):
            print("loading labels for folder {0}/{1}".format(fid + 1, len(subfolders)))
            label_path = os.path.join(dataset_dir, subfolder, "gt/gt.txt")
            labels = np.loadtxt(label_path, delimiter=",")

            n_frames = int(labels[-1, 0])
            n_heads = int(labels[:, 1].max())

            bboxes = np.zeros((n_frames, n_heads, 4))
            valids = -1 * np.ones((n_frames, n_heads))
            visibs = np.zeros((n_frames, n_heads))

            # print('labels', labels.shape)

            for i in range(labels.shape[0]):
                (
                    frame_id,
                    head_id,
                    bb_left,
                    bb_top,
                    bb_width,
                    bb_height,
                    conf,
                    cid,
                    vis,
                ) = labels[i]
                frame_id = int(frame_id) - 1  # convert 1 indexed to 0 indexed
                head_id = int(head_id) - 1  # convert 1 indexed to 0 indexed

                valids[frame_id, head_id] = 1
                visibs[frame_id, head_id] = vis
                box_cur = np.array(
                    [bb_left, bb_top, bb_left + bb_width, bb_top + bb_height]
                )  # convert xywh to x1, y1, x2, y2
                bboxes[frame_id, head_id] = box_cur

            d = {
                "bboxes": np.copy(bboxes),
                "valids": np.copy(valids),
                "visibs": np.copy(visibs),
            }

            stride = 1
            for start_frame in range(0, max(n_frames - self.S * stride, 1), 8):
                end_frame = min(start_frame + self.S * stride, n_frames)

                if end_frame - start_frame >= 8 * stride:
                    # print('%d:%d' % (start_frame, end_frame))

                    for tid in range(self.N_per):
                        self.all_folders.append(subfolder)
                        self.all_start_frames.append(start_frame)
                        self.all_end_frames.append(end_frame)
                        self.all_dicts.append(d)
                        self.all_tids.append(tid)
                        self.all_strides.append(stride)

        print("found %d samples" % len(self.all_folders))

    def __len__(self):
        return len(self.all_folders)

    def __getitem__(self, index):
        folder = self.all_folders[index]
        d = self.all_dicts[index]
        start_frame = self.all_start_frames[index]
        end_frame = self.all_end_frames[index]
        tid = self.all_tids[index]
        stride = self.all_strides[index]

        S = self.S
        bboxes = d["bboxes"][start_frame:end_frame] * 0.5  # downsample by half
        valids = d["valids"][start_frame:end_frame]
        visibs = d["visibs"][start_frame:end_frame]

        bboxes = bboxes[::stride]
        valids = valids[::stride]
        visibs = visibs[::stride]

        rgbs = []
        for ii in range(start_frame, min(start_frame + S * stride, end_frame), stride):
            rgb_path = os.path.join(
                self.dataset_dir, folder, "img1", str(ii + 1).zfill(6) + ".jpg"
            )
            rgb = Image.open(rgb_path)
            # rgb = cv2.imread(str(image_name))[..., ::-1].copy()
            rgb = rgb.resize(
                (int(rgb.size[0] / 2), int(rgb.size[1] / 2)), Image.BILINEAR
            )  # downsample by half
            rgbs.append(rgb)
        rgbs = np.stack(rgbs)  # S,C,H,W

        # print('rgbs', rgbs.shape)
        trajs = np.stack(
            [bboxes[:, :, [0, 2]].mean(2), bboxes[:, :, [1, 3]].mean(2)], axis=2
        )  # S,N,2
        # print('trajs', trajs.shape)
        # print('bboxes', bboxes.shape)

        S, N = trajs.shape[:2]
        H, W = 1080, 1920
        # update visibility annotations
        for si in range(S):
            # crohd annotations get noisy/wrong near edges
            oob_inds = np.logical_or(
                np.logical_or(trajs[si, :, 0] < 32, trajs[si, :, 0] > W - 32),
                np.logical_or(trajs[si, :, 1] < 32, trajs[si, :, 1] > H - 32),
            )
            visibs[si, oob_inds] = 0
            # exclude oob from eval
            valids[si, oob_inds] = 0

        vis_ok0 = visibs[0] > 0  # N
        vis_okE = visibs[-1] > 0  # N
        valid_ok0 = valids[0] > 0  # N
        valid_okE = valids[-1] > 0  # N
        mot_ok = (
            np.sum(np.linalg.norm(trajs[1:] - trajs[:-1], axis=-1), axis=0) > 50
        )  # N
        all_ok = (
            vis_ok0 & valid_ok0 & mot_ok & vis_okE & valid_okE
        )

        trajs = trajs[:, all_ok]
        bboxes = bboxes[:, all_ok]
        visibs = visibs[:, all_ok]
        valids = valids[:, all_ok]

        S, N = trajs.shape[:2]
        if N <= tid:
            return None

        bboxes = bboxes[:, tid]
        visibs = visibs[:, tid]
        trajs = trajs[:, tid]
        sample = {
            "rgbs": rgbs,
            "trajs": trajs,
            "visibs": visibs.astype(np.float32),
        }
        
        
        rgbs, pts = sample['rgbs'], sample['trajs']  # len S
        bboxes[:, 2:] = np.maximum(bboxes[:, 2:], bboxes[:, :2] + 1)  # make sure bbox is valid
        
        crop_size = self.crop_size
        
        rgbs_cr = []
        for i in range(len(rgbs)):
            crop_resize = A.Compose([
                    A.Resize(*crop_size)],
                keypoint_params=A.KeypointParams(format='xy', label_fields=[], remove_invisible=False))

            data = crop_resize(image=rgbs[i], keypoints=[pts[i]])
            pts[i] = data['keypoints'][0]
            rgbs_cr.append(data['image'])
        
        sample['rgbs'] = np.stack(rgbs_cr)
        return sample
