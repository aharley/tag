# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

cam_idxs = ['400002', '400004', '400006', '400008', \
            '400009', '400010', '400012', '400013', \
            '400015', '400016', '400017', '400018', \
            '400019', '400023', '400026', '400027', \
            '400028', '400029', '400030', '400031', \
            '400035', '400037', '400039', '400041', \
            '400042', '400048', '400049', '400051', \
            '400053', '400059', '400060', '400063', \
            '400064']
ignore_list = [('Capture0', '0002_good_luck'), ('Capture0', '0001_neutral_rigid'), ('Capture0', '0000_neutral_relaxed')]

# get integer version of camera index
cam_idxs = [int(cam_idx) for cam_idx in cam_idxs]

import numpy as np
import torch
import torch.utils.data
import cv2
from glob import glob
import os.path as osp
from datasets.interhand_config import cfg
from datasets.interhand_utils.preprocessing import (
    load_img,
    load_skeleton,
    get_bbox,
    process_bbox,
    transform_input_to_output_space,
    trans_point2d,
)
from datasets.interhand_utils.transforms import world2cam, cam2pixel, pixel2cam
from datasets.interhand_utils.vis import vis_keypoints, vis_3d_keypoints
from PIL import Image, ImageDraw
import random
import json
import math
import pickle
from pycocotools.coco import COCO
import scipy.io as sio
from collections import defaultdict
from datasets.dataset import PointDataset
import torchvision.transforms as transforms
import os

CLIP_STEP = 8


def to_image(joint_coord):
    joint_coord[:, 0] = (
        joint_coord[:, 0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
    )
    joint_coord[:, 1] = (
        joint_coord[:, 1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
    )
    return joint_coord


# class InterHandDataset(torch.utils.data.Dataset):
class InterHandDataset(PointDataset):
    def __init__(
        self,
        dataset_location="/orion/group/InterHand",
        S=6,
        rand_frames=False,
        crop_size=None,
        use_augs=False,
        is_training=True,
        chunk=None,
        transform=transforms.ToTensor(),
    ):
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            rand_frames=rand_frames,
            crop_size=crop_size, 
            use_augs=use_augs,
            is_training=is_training,
        )
        self.S = S
        # length_pickle_path = "/orion/group/InterHand/sorted_interhand_frame_length.pkl"
        length_pickle_path = "/orion/group/InterHand/sorted_interhand_frame_length_30fps.pkl"
        with open(
            length_pickle_path, "rb"
        ) as f:
            self.sorted_interhand_frame_length = pickle.load(f)
        self.geqS = [
            (x[0], x[1]) for x in self.sorted_interhand_frame_length if x[2] >= self.S
        ]

        self.mode = "train" if is_training else "val"  # train, test, val
        self.img_path = f"{dataset_location}/images"
        self.annot_path = f"{dataset_location}/annotations"
        if self.mode == "val":
            self.rootnet_output_path = "../data/InterHand2.6M/rootnet_output/rootnet_interhand2.6m_output_val.json"
        else:
            self.rootnet_output_path = "../data/InterHand2.6M/rootnet_output/rootnet_interhand2.6m_output_test.json"
        self.transform = transform
        self.joint_num = 21  # single hand
        self.root_joint_idx = {"right": 20, "left": 41}
        self.joint_type = {
            "right": np.arange(0, self.joint_num),
            "left": np.arange(self.joint_num, self.joint_num * 2),
        }
        self.skeleton = load_skeleton(
            osp.join(self.annot_path, "skeleton.txt"), self.joint_num * 2
        )

        self.datalist = []
        self.datalist_sh = []
        self.datalist_ih = []
        self.sequence_names = []

        self.num_sh = 0
        self.num_ih = 0
        self.video_end = defaultdict(list)
        self.all_processed_data = defaultdict(list)
        # load annotation
        print("Loading annotations from " + osp.join(self.annot_path, self.mode))
        db = COCO(
            osp.join(
                self.annot_path, self.mode, "InterHand2.6M_" + self.mode + "_data.json"
            )
        )
        print('loaded data json')
        with open(
            osp.join(
                self.annot_path,
                self.mode,
                "InterHand2.6M_" + self.mode + "_camera.json",
            )
        ) as f:
            cameras = json.load(f)
        print('loaded camera json')
        with open(
            osp.join(
                self.annot_path,
                self.mode,
                "InterHand2.6M_" + self.mode + "_joint_3d.json",
            )
        ) as f:
            joints = json.load(f)
        print('loaded joints json')

        if (self.mode == "val" or self.mode == "test") and cfg.trans_test == "rootnet":
            print("Get bbox and root depth from " + self.rootnet_output_path)
            rootnet_result = {}
            with open(self.rootnet_output_path) as f:
                annot = json.load(f)
            for i in range(len(annot)):
                rootnet_result[str(annot[i]["annot_id"])] = annot[i]
        else:
            print("Get bbox and root depth from groundtruth annotation")
        print('loaded box annot')

        # print(len(db.anns.keys()))

        keys = list(db.anns.keys())
        # # print('keys', keys, len(keys))
        # print('len(keys)', len(keys))
        
        # if chunk is not None:
        #     def chunkify(lst,n):
        #         return [lst[i::n] for i in range(n)]
        #     keys = chunkify(keys,100)[chunk]
        #     print('filtered to %d sequences' % len(keys))
        #     print('keys', keys)
        
        # return
        count = 0
        # for i, aid in enumerate(db.anns.keys()):
        for i, aid in enumerate(keys):
            ann = db.anns[aid]
            image_id = ann["image_id"]
            img = db.loadImgs(image_id)[0]
            capture_id = img["capture"]
            seq_name = img["seq_name"]
            cam = img["camera"]
            frame_idx = img["frame_idx"]
            # ************************************************************************
            # Filter the data
            # Set a max capture ID
            if not self.filter_data(capture_id, cam, seq_name):
                continue
            #             if cam not in [400008, 400030, 400053, 400059]:
            # 400030 is the front view camera
            # name_key = (capture_id, seq_name)
            name_key = (cam, capture_id, seq_name)
            # ************************************************************************

            frame_idx = img["frame_idx"]
            img_path = osp.join(self.img_path, self.mode, img["file_name"])

            campos, camrot = np.array(
                cameras[str(capture_id)]["campos"][str(cam)], dtype=np.float32
            ), np.array(cameras[str(capture_id)]["camrot"][str(cam)], dtype=np.float32)
            focal, princpt = np.array(
                cameras[str(capture_id)]["focal"][str(cam)], dtype=np.float32
            ), np.array(cameras[str(capture_id)]["princpt"][str(cam)], dtype=np.float32)
            joint_world = np.array(
                joints[str(capture_id)][str(frame_idx)]["world_coord"], dtype=np.float32
            )
            joint_cam = world2cam(
                joint_world.transpose(1, 0), camrot, campos.reshape(3, 1)
            ).transpose(1, 0)
            joint_img = cam2pixel(joint_cam, focal, princpt)[:, :2]

            joint_valid = np.array(ann["joint_valid"], dtype=np.float32).reshape(
                self.joint_num * 2
            )
            # if root is not valid -> root-relative 3D pose is also not valid. Therefore, mark all joints as invalid
            joint_valid[self.joint_type["right"]] *= joint_valid[
                self.root_joint_idx["right"]
            ]
            joint_valid[self.joint_type["left"]] *= joint_valid[
                self.root_joint_idx["left"]
            ]
            hand_type = ann["hand_type"]
            hand_type_valid = np.array((ann["hand_type_valid"]), dtype=np.float32)

            img_width, img_height = img["width"], img["height"]
            bbox = np.array(ann["bbox"], dtype=np.float32)  # x,y,w,h
            bbox = process_bbox(bbox, (img_height, img_width))
            abs_depth = {
                "right": joint_cam[self.root_joint_idx["right"], 2],
                "left": joint_cam[self.root_joint_idx["left"], 2],
            }

            cam_param = {"focal": focal, "princpt": princpt}
            joint = {
                "cam_coord": joint_cam,
                "img_coord": joint_img,
                "valid": joint_valid,
            }
            data = {
                "img_path": img_path,
                "seq_name": seq_name,
                "cam_param": cam_param,
                "bbox": bbox,
                "joint": joint,
                "hand_type": hand_type,
                "hand_type_valid": hand_type_valid,
                "abs_depth": abs_depth,
                "file_name": img["file_name"],
                "capture": capture_id,
                "cam": cam,
                "frame": frame_idx,
            }
            # ************************************************************************

            self.all_processed_data[name_key].append(data)

        self.total_number = sum([len(x) for x in self.all_processed_data.values()])
        print("Total number of images:", self.total_number)
        # print('Number of annotations in single hand sequences: ' + str(self.num_sh))
        # print('Number of annotations in interacting hand sequences: ' + str(self.num_ih))

        self.all_samples = []

        self.video_names = sorted(list(self.all_processed_data.keys()))

        # valid_num = {}
        # read valid_num from pickle
        if os.path.isfile("/orion/group/interhand_valid_num_30fps.pkl"):
            with open("/orion/group/interhand_valid_num_30fps.pkl", "rb") as f:
                valid_num = pickle.load(f)

        for i in range(len(self.video_names)):
            if i % 50 == 0:
                print("Processing video", i, "of", len(self.video_names), "videos")
            video = self.video_names[i]
            all_vis_scores = self.get_all_vis_scores(video)
            if all_vis_scores == None:
                continue

            for start in range(
                0, len(self.all_processed_data[video]) - self.S + 1, CLIP_STEP
            ):
                # print(video)
                # samples = self.get_samples_for_video_at_start_frame(
                #     video, start, all_vis_scores
                # )
                # valid_num[(video, start)] = len(samples)
                len_samples = valid_num[(video, start)]
                # remember to change it back!
                # len_samples = 2
                # Hack: to avoid using too much memory, just store the indices and
                # then reload the frames later
                # self.all_samples.extend(
                #     [(video, start, i) for i in range(len(samples))]
                # )
                self.all_samples.extend(
                    [(video, start, i) for i in range(len_samples)]
                )

        # save valid_num to pickle
        # with open("valid_num_30fps.pkl", "wb") as f:
        #     pickle.dump(valid_num, f)


    def get_all_vis_scores(self, video):
        # print(video)
        cam, capture_id, seq_name = video
        # cam_idx = "400030"
        cam_idx = str(cam)
        vis_root = "/orion/u/xs15/InterHand2.6M/tool/MANO_render"
        # save_path = osp.join(
        #     vis_root, "visibility", "train", str(capture_id), seq_name, cam_idx
        # )
        # save_path = osp.join(
        #     vis_root, "visibility_30fps", "train", str(capture_id), seq_name, cam_idx
        # )
        save_path = osp.join(
            vis_root, "visibility_30fps_thresh5e-3", "train", str(capture_id), seq_name, cam_idx
        )
        if os.path.isfile(f"{save_path}/vis.pkl"):
            with open(f"{save_path}/vis.pkl", "rb") as f:
                all_vis_scores = pickle.load(f)
            return all_vis_scores
        else:
            return None

    def get_samples_for_video_at_start_frame(self, video, start, all_vis_scores):
        stop = start + self.S

        rgbs = []
        trajs = []
        visibs = []
        for j in range(start, stop):
            image, keypoints, valids = self.retrieve_data(video, j)
            # vis = np.array(all_vis_scores[j])
            # remember to change it back!
            vis = 1
            image = image.permute(1, 2, 0).numpy()
            image = image * 255
            image = image.astype(np.uint8)
            rgbs.append(image)
            trajs.append(torch.tensor(keypoints))
            visibs.append(torch.tensor(valids.astype(bool) & vis).to(torch.float32))

        rgbs = np.array(rgbs)
        trajs = torch.stack(trajs, 0)
        visibs = torch.stack(visibs, 0)
        samples = self.prep_samples(rgbs, trajs, visibs)

        return samples

    def filter_data(self, capture, cam, seqname):
        res = True
        # if capture > 4:
        if capture != 0:
            res &= False
        # if int(cam) not in [400030]:
        if int(cam) not in cam_idxs:
            res &= False
        if (f"Capture{capture}", seqname) not in self.geqS:
            res &= False
        if (f"Capture{capture}", seqname) in ignore_list:
            res &= False

        return res

    def handtype_str2array(self, hand_type):
        if hand_type == "right":
            return np.array([1, 0], dtype=np.float32)
        elif hand_type == "left":
            return np.array([0, 1], dtype=np.float32)
        elif hand_type == "interacting":
            return np.array([1, 1], dtype=np.float32)
        else:
            assert 0, print("Not supported hand type: " + hand_type)

    def __len__(self):
        return len(self.all_samples)

    def prep_samples(
        self,
        rgbs,
        trajs,
        visibs,
    ):
        seq_vis_init = visibs[:2].sum(0) == 2
        seq_valid = seq_vis_init

        if seq_valid.sum() == 0:
            print("no valid")
            return []

        trajs = trajs[:, seq_valid > 0]
        visibs = visibs[:, seq_valid > 0]

        N = trajs.shape[1]

        samples = []
        for i in range(N):
            d = {
                "rgbs": rgbs.astype(np.uint8),  # S, C, H, W
                "trajs": trajs[:, i].numpy().astype(np.int64),  # S, 2
                "visibs": visibs[:, i].numpy().astype(np.float32),  # S
            }
            samples.append(d)

        return samples

    def getitem_helper(self, index):
        video, start, i = self.all_samples[index]
        all_vis_scores = self.get_all_vis_scores(video)
        assert all_vis_scores is not None

        samples = self.get_samples_for_video_at_start_frame(
            video, start, all_vis_scores
        )

        return samples[i]

    def retrieve_data(self, key, idx):
        data = self.all_processed_data[key][idx]
        img_path, bbox, joint, hand_type, hand_type_valid = (
            data["img_path"],
            data["bbox"],
            data["joint"],
            data["hand_type"],
            data["hand_type_valid"],
        )
        joint_cam = joint["cam_coord"].copy()
        joint_img = joint["img_coord"].copy()
        joint_valid = joint["valid"].copy()
        hand_type = self.handtype_str2array(hand_type)
        joint_coord = np.concatenate((joint_img, joint_cam[:, 2, None]), 1)
        # ************************************************************************
        # image load
        img = load_img(img_path)
        rel_root_depth = np.array(
            [
                joint_coord[self.root_joint_idx["left"], 2]
                - joint_coord[self.root_joint_idx["right"], 2]
            ],
            dtype=np.float32,
        ).reshape(1)
        root_valid = (
            np.array(
                [
                    joint_valid[self.root_joint_idx["right"]]
                    * joint_valid[self.root_joint_idx["left"]]
                ],
                dtype=np.float32,
            ).reshape(1)
            if hand_type[0] * hand_type[1] == 1
            else np.zeros((1), dtype=np.float32)
        )

        # transform to output heatmap space
        (
            joint_coord,
            joint_valid,
            rel_root_depth,
            root_valid,
        ) = transform_input_to_output_space(
            joint_coord,
            joint_valid,
            rel_root_depth,
            root_valid,
            self.root_joint_idx,
            self.joint_type,
        )
        img = self.transform(img.astype(np.float32)) / 255.0

        # Get the image coordinates of keypoints
        keypoints = joint_coord[:, :2]
        keypoints = to_image(keypoints)
        # ************************************************************************

        return img, keypoints, joint_valid
