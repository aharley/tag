import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from datasets.dataset import PointDataset
from datasets.dataset_utils import make_split
from datasets.deepposekit_patch import DataGenerator


class DeepPoseKitDataset(PointDataset):
    def __init__(
        self,
        dataset_location="/orion/group/deepposekit-data/datasets/",
        S=32,
        strides=[1, 2],
        clip_step=8,
        crop_size=(384, 512),
        use_augs=False,
        is_training=True,
    ):
        print("loading DeepPoseKit point dataset...")
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training,
        )
        self.dataset_location = dataset_location

        self.root = Path(dataset_location)
        datasets = [dataset for dataset in os.listdir(self.root) if dataset != "human"]

        # if is_training:
        #     datasets = [dataset for dataset in datasets if dataset != 'fly']
        # print('datasets', datasets)
        # datasets = make_split(datasets, is_training, shuffle=True, train_ratio=(2/3))
        # print('split datasets', datasets)
        if not is_training:
            datasets = []

        # datasets = [dataset for dataset in os.listdir(self.root)]
        # print('datasets', datasets)
        # datasets = make_split(datasets, is_training, shuffle=True, train_ratio=(2/3))
        # print('split datasets', datasets)
        
        # Skip "human"?
        self.generators = []
        for dataset in datasets:
            self.generators.append(DataGenerator(os.path.join(self.root, dataset, "annotation_data_release.h5")))

        self.all_generator_idx = []
        self.all_frame_idx = []
        self.all_kp_idx = []

        for i, generator in enumerate(self.generators):
            S_local = len(generator)
            for stride in strides:
                for ii in range(0, max(S_local - self.S * stride, 1), clip_step):
                    keypoints = generator[ii][1]
                    coordinates = [(b, n) for b in range(keypoints.shape[0]) for n in range(keypoints.shape[1])]
                    full_idx = ii + np.arange(self.S) * stride
                    full_idx = [ij for ij in full_idx if ij < S_local]
                    for coordinate in coordinates: # For each keypoint
                        self.all_generator_idx.append(i)
                        self.all_frame_idx.append(full_idx)
                        self.all_kp_idx.append(coordinate)

        print(f"Created a dataset of length {len(self.all_frame_idx)}")

    def __len__(self):
        return len(self.all_frame_idx)

    def getitem_helper(self, index):
        generator_idx = self.all_generator_idx[index]
        frame_idx = self.all_frame_idx[index]
        kp_idx = self.all_kp_idx[index]
        xys = []
        visibs = []

        generator = self.generators[generator_idx]

        rgbs = []
        xys = []
        for frame in frame_idx:
            img, keypoints = generator[frame]
            img = np.squeeze(img, axis=0)
            rgbs.append(img)
            xys.append(keypoints[kp_idx])
            visibs.append([1])

        rgbs = np.stack(rgbs)  # S, H, W, C
        if rgbs.shape[-1] == 1:
            rgbs = np.repeat(rgbs, 3, axis=-1)
        xys = np.stack(xys)
        visibs = np.concatenate(visibs)
        print(xys.shape, rgbs.shape, visibs.shape)

        sample = {
            "rgbs": rgbs,
            "visibs": visibs.astype(np.float32),  # S
            "xys": xys,
        }

        return sample
