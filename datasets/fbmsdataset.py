import time
from numpy import random
from numpy.core.numeric import full
import torch
import numpy as np
import os
import scipy.ndimage
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import random
from torch._C import dtype, set_flush_denormal
import utils.basic
import utils.improc
import glob
import json
import imageio
import cv2
import re
import sys
from torchvision.transforms import ColorJitter, GaussianBlur
from datasets.dataset import MaskDataset
from icecream import ic

# reference https://github.com/antonilo/unsupervised_detection/blob/master/data/fbms_data_utils.py to read FBMS dataset

# np.random.seed(125)
# torch.multiprocessing.set_sharing_strategy('file_system')


class FBMSDataset(MaskDataset):
    def __init__(
        self,
        dataset_location="../FBMS",
        S=32,
        crop_size=(384, 512),
        strides=[1, 2],
        clip_step=8,
        use_augs=False,
        is_training=True,
    ):
        print("loading FBMS dataset...")
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training,
        )
        self.dataset_location = os.path.join(
            dataset_location, "Trainingset" if is_training else "Testset"
        )
        self.video_names = os.listdir(os.path.join(self.dataset_location))
        print(
            "found {:d} videos in {}".format(
                len(self.video_names), self.dataset_location
            )
        )

        self.all_video_names = []
        self.all_full_idx = []

        for video_name in self.video_names:
            # these sequences contain group of multiple objects that can cause extreme bbox change (as one object leaves the frame, the overall bbox will drastically change)
            if video_name in ["people05", "horse06", "cats07", "marple10", "people04"]:
                continue
            video_dir = os.path.join(self.dataset_location, video_name)
            type_weird = (
                len(glob.glob(os.path.join(video_dir, "GroundTruth", "*.ppm"))) > 0
            )
            frames = [
                fname
                for fname in sorted(os.listdir(video_dir))
                if os.path.exists(
                    os.path.join(
                        video_dir,
                        "GroundTruth",
                        fname.replace(
                            ".jpg", "{}".format("_gt.ppm" if type_weird else ".pgm")
                        ),
                    )
                )
            ]
            S_local = len(frames)

            for stride in strides:
                for ii in range(0, max(S_local - self.S * stride, 1), clip_step):
                    full_idx = ii + np.arange(self.S) * stride
                    full_idx = [ij for ij in full_idx if ij < S_local]

                    if len(full_idx) >= 8:
                        self.all_video_names.append(video_name)
                        self.all_full_idx.append(full_idx)
                        sys.stdout.write(".")
                        sys.stdout.flush()
        print("done")

    def __len__(self):
        return len(self.all_video_names)

    def getitem_helper(self, index):
        video_name = self.all_video_names[index]
        video_dir = os.path.join(self.dataset_location, video_name)

        full_idx = self.all_full_idx[index]
        type_weird = len(glob.glob(os.path.join(video_dir, "GroundTruth", "*.ppm"))) > 0
        frames = [
            fname
            for fname in sorted(os.listdir(video_dir))
            if os.path.exists(
                os.path.join(
                    video_dir,
                    "GroundTruth",
                    fname.replace(
                        ".jpg", "{}".format("_gt.ppm" if type_weird else ".pgm")
                    ),
                )
            )
        ]
        frames = [frames[idx] for idx in full_idx]

        rgbs = [cv2.imread(os.path.join(video_dir, fn))[..., ::-1] for fn in frames]
        masks = []
        for fn in frames:
            mask = (
                cv2.imread(
                    os.path.join(
                        video_dir,
                        "GroundTruth",
                        fn.replace(
                            ".jpg", "{}".format("_gt.ppm" if type_weird else ".pgm")
                        ),
                    ),
                    cv2.IMREAD_GRAYSCALE,
                )
                / 255.0
            )
            if (
                type_weird
            ):  # https://github.com/antonilo/unsupervised_detection/blob/master/data/fbms_data_utils.py#L116C17-L123C36
                mask[mask > 0.99] = 0.0
            if "marple7" == video_name:
                mask = mask > 0.05
            elif "marple2" == video_name:
                mask = mask > 0.4
            else:
                mask = mask > 0.1
            masks.append(mask.astype(np.float32))

        if np.mean(masks[0]) < 0.001:
            print("mask0_mean", np.mean(masks[0]))
            return None

        sample = {
            "rgbs": np.stack(rgbs),
            "masks": np.stack(masks),
        }

        return sample
