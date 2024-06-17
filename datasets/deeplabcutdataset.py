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
import sys
import csv

from datasets.dataset_utils import make_split


class DeepLabCutDataset(PointDataset):
    def __init__(
        self,
        dataset_location="/orion/group/deeplabcut",
        use_augs=False,
        S=8,
        strides=[1, 2],
        crop_size=(368, 496),
        is_training=True,
    ):
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size, 
            use_augs=use_augs,
            is_training=is_training,
        )
        print("loading deeplabcut datasets...")

        self.S = S

        self.use_augs = use_augs
        self.traj_paths = []
        self.subdirs = [
            "fish-dlc-2021-05-07",
            "marmoset-dlc-2021-05-07",
            "pups-dlc-2021-03-24",
            "trimice-dlc-2021-06-22",
        ]
        self.sequences = []

        self.videos = []
        self.gt = {}

        for subdir in self.subdirs:
            subset_location = dataset_location + "/" + subdir
            for video in glob.glob(subset_location + "/labeled-data/*"):
                video_name = video.split("/")[-1]
                # print(video_name)
                self.videos.append(video_name)
                self.gt[video_name] = {}
                self.gt[video_name]["img_paths"] = []
                # self.gt[video_name]["pt_coords"] = []
                csv_file = open(video + "/CollectedData_dlc.csv", "r")
                csv_reader = csv.reader(csv_file, delimiter=",")
                row_count = 0
                for row in csv_reader:
                    # print(row)
                    if row_count == 1:
                        individual_names = row[1:]
                        self.gt[video_name]["individuals"] = list(set(individual_names))
                        for individual in individual_names:
                            self.gt[video_name][individual] = []
                    if row_count >= 4:
                        self.gt[video_name]["img_paths"].append(row[0])
                        for i in range(1, len(row)):
                            coord = row[i]
                            individual = individual_names[i - 1]
                            if coord == "":
                                coord = "-1000"
                            self.gt[video_name][individual].append(float(coord))
                    row_count += 1
                for individual in self.gt[video_name]["individuals"]:
                    self.gt[video_name][individual] = np.asarray(
                        self.gt[video_name][individual]
                    )  # S*N_individual*2
                    self.gt[video_name][individual] = np.reshape(
                        self.gt[video_name][individual],
                        (len(self.gt[video_name]["img_paths"]), -1, 2),
                    )
                self.gt[video_name]["subset"] = subdir
                csv_file.close()
            print("found %d unique videos in %s" % (len(self.videos), subset_location))
        # print(self.gt)
        # exit(0)

        # self.sequences = sorted(self.sequences)
        print("found %d unique videos in %s" % (len(self.videos), dataset_location))
        self.videos = make_split(self.videos, is_training, shuffle=True, train_ratio=0.8)

        self.rgb_paths = []
        self.full_idxs = []
        self.video_names = []
        self.tids = []
        self.individual_names = []

        ## load trajectories
        print("loading trajectories...")
        for video in self.videos:
            video_name = video.split("/")[-1]
            for individual in self.gt[video]["individuals"]:
                S_local, N_local, _ = self.gt[video_name][individual].shape
                # print(video_name, S_local, N_local)
                for stride in strides:
                    for ii in range(0, max(S_local - self.S * stride + 1, 1), 8):
                        for ni in range(N_local):

                            full_idxs_here = []
                            
                            tries = 0
                            # try shuffling the video to see if we can get more valid clips
                            while tries < 20:
                                full_idx = ii + np.arange(self.S) * stride
                                full_idx = [ij for ij in full_idx if ij < S_local]

                                random.seed(2048)
                                if tries > 0:
                                    random.shuffle(full_idx)

                                traj = self.gt[video_name][individual][full_idx, ni].astype(np.float32)
                                full_traj = self.gt[video_name][individual][full_idx].astype(np.float32)
                                visib = (traj[:, 0] > 0) * (traj[:, 1] > 0)

                                if len(full_idx) > 3 and visib[0] and visib[1]:
                                    # print('full_idx', full_idx)

                                    travel = 0
                                    for si in range(1, len(full_idx)):
                                        dist = np.linalg.norm(full_traj[si-1] - full_traj[si])
                                        # print('dist', dist)
                                        if dist > 220:  # cut the seq at the first discontinuity
                                            full_idx = full_idx[:si]
                                            break
                                        travel += dist
                                    if travel < 40:  # discard seqs with no travel
                                        # sys.stdout.write('m')
                                        pass
                                    elif len(full_idx) < 4:
                                        # discard seqs that are too short
                                        # sys.stdout.write('l')
                                        pass
                                    elif full_idx in full_idxs_here:
                                        pass;
                                    else:
                                        self.rgb_paths.append(
                                            [
                                                os.path.join(
                                                    dataset_location,
                                                    self.gt[video]["subset"],
                                                    self.gt[video]["img_paths"][idx],
                                                )
                                                for idx in full_idx
                                            ]
                                        )
                                        self.video_names.append(video)
                                        self.full_idxs.append(full_idx)
                                        self.tids.append(ni)
                                        self.individual_names.append(individual)
                                        full_idxs_here.append(full_idx)
                                        sys.stdout.write(".")
                                    sys.stdout.flush()
                                tries += 1
        print("done")

        print("collected %d clips in %s" % (len(self.rgb_paths), dataset_location))

    def getitem_helper(self, index):
        rgb_paths = self.rgb_paths[index]
        full_idx = self.full_idxs[index]
        video_name = self.video_names[index]
        tid = self.tids[index]
        individual = self.individual_names[index]
        # print('video_name', video_name)

        trajs = self.gt[video_name][individual][full_idx, tid, :].astype(np.float32)
        visibs = (trajs[:, 0] > 0) * (trajs[:, 1] > 0)
        S_video, D = trajs.shape
        assert D == 2

        rgbs = []
        for rgb_path in rgb_paths:
            with Image.open(rgb_path) as im:
                rgbs.append(np.array(im)[:, :, :3])
        rgbs = np.stack(rgbs, axis=0)  # S,H,W,3

        H, W, C = rgbs[0].shape
        assert C == 3

        sample = {
            "rgbs": rgbs,
            "trajs": trajs,
            "visibs": visibs,
        }
        return sample

    def __len__(self):
        return len(self.rgb_paths)
