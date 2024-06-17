import csv
import json
import os
from PIL import Image
import sys

from numpy import random
import numpy as np
import pandas as pd
import random

from datasets.dataset import PointDataset
from pathlib import Path
from datasets.dataset_utils import make_split
import scipy.io as sio
import cv2
from scipy.interpolate import griddata

NUM_CAMERAS = 6


def interpolate_nan_points(points):
    # Convert to numpy array if not already
    points = np.array(points)

    # Function to find nearest valid point index
    def find_nearest_valid_index(index):
        valid_indices = np.array([i for i, p in enumerate(points) if not np.isnan(p).any()])
        nearest_index = valid_indices[np.abs(valid_indices - index).argmin()]
        return nearest_index

    # Interpolate NaN points
    for i, point in enumerate(points):
        if np.isnan(point).any():  # Check if the point has NaN
            nearest_index = find_nearest_valid_index(i)
            points[i] = points[nearest_index]

    return points


class Rat7MDataset(PointDataset):
    def __init__(
        self,
        dataset_location="/orion/group/Rat7M/",
        use_augs=False,
        S=8,
        strides=[1],
        clip_step=1024, # the videos are super long, so let's help uniqueness via large clip_step
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
        print("loading Rat7M dataset...")

         # validation gt not provided
        self.meta_fns = list(Path(dataset_location).glob('mocap*.mat'))
        print(
            "found {:d} videos in {}".format(
                len(self.meta_fns), dataset_location
            )
        )
        self.meta_fns = make_split(self.meta_fns, is_training, shuffle=True)

        self.all_seq_ids = []
        self.all_ids = []
        self.all_strides = []
        self.all_full_idx = []
        self.all_xys = []
        self.all_cam_ids = []
        
        for fn in self.meta_fns[:]:
            print('fn', fn)
            seq_id = '-'.join(fn.stem.split('-')[1:])
            if seq_id in ['s2-d1', 's4-d1']:
                continue
            meta = sio.loadmat(fn)
            
            absolute_position_data = np.stack([meta['mocap'][0][0][i] for i in range(20)], 1)
            
            cam_mapping = {
                1: 1,
                2: 2,
                3: 5,
                4: 3,
                5: 4,
                6: 6
            }
            
            if 's5' in seq_id:
                cam_mapping[3] = 3
                cam_mapping[4] = 4
                cam_mapping[5] = 5

            for cam_id in range(1, 1 + NUM_CAMERAS):
                cam_info = meta['cameras'][0][0][cam_mapping[cam_id] - 1][0][0]
                data_frames = cam_info[0].reshape(-1).tolist()
                intrinsic_matrix = cam_info[1].T
                rotation_matrix = cam_info[2].T
                rotation_vector, _ = cv2.Rodrigues(rotation_matrix)
                translation_vector = cam_info[3].reshape(-1, 1)
                distortion_coefficients = np.concatenate([cam_info[5][0], cam_info[4][0]])
            
                
                track_ids = list(range(20))
                S_local = meta['mocap'][0][0][0].shape[0]
                print('S_local', S_local)
                for i, track_id in enumerate(track_ids):
                    for stride in strides:
                        for ii in range(0, max(S_local - self.S * stride, 1), clip_step):
                        # for ii in range(0, 1):  # DEBUG
                            full_idx = ii + np.arange(self.S) * stride
                            full_idx = np.array([ij for ij in full_idx if ij < S_local])
                            full_idx = [ij for ij in full_idx if ij in data_frames]
                            pos = np.array([absolute_position_data[data_frames.index(ij), track_id, :] for ij in full_idx])
                            # since the videos are so long, let's require full S
                            if len(full_idx) == S:
                                image_points, _ = cv2.projectPoints(
                                    pos,
                                    rotation_vector,
                                    translation_vector,
                                    intrinsic_matrix,
                                    distortion_coefficients
                                )
                                image_points = image_points.reshape(-1, 2)  # S x 2
                                if np.any(np.isnan(image_points[0])) or np.any(np.isnan(image_points[1])):
                                    continue

                                vel = image_points[1:] - image_points[:-1]
                                accel = vel[1:] - vel[:-1]

                                # print('vel', np.mean(np.linalg.norm(vel, axis=-1)))
                                # print('accel', np.max(np.linalg.norm(accel, axis=-1)))

                                # many boring seqs
                                if np.mean(vel) < 4:
                                    continue

                                # some teleports
                                if np.max(accel) > 40:
                                    sys.stdout.write('a')
                                    continue
                                

                                # travel = np.sum(np.linalg.norm(image_points[1:] - image_points[:-1], axis=-1))
                                # H, W = rgbs[0,0].shape
                                # print('travel', travel)

                                # if travel < 100:
                                #     # sys.stdout.write('t')
                                #     continue
                                
                                # if travel/max(H,W) < 0.1:
                                #     print('travel', travel/max(H,W))
                                #     return None
                                
                                self.all_seq_ids.append(seq_id)
                                self.all_ids.append(track_id)
                                self.all_strides.append(stride)
                                self.all_full_idx.append(full_idx)
                                self.all_xys.append(image_points)
                                self.all_cam_ids.append(cam_id)
                                sys.stdout.write('.')
                                sys.stdout.flush()
        print(
            "found {:d} samples in {}".format(
                len(self.all_xys), self.dataset_location
            )
        )

    def getitem_helper(self, index):
        seq_id = self.all_seq_ids[index]
        track_id = self.all_ids[index]
        full_idx = self.all_full_idx[index]
        image_points = self.all_xys[index]
        cam_id = self.all_cam_ids[index]
        
        idx_start = full_idx[0]
        idx_end = full_idx[-1]
        video_idx = sorted(list(set([idx_start // 3500, idx_end // 3500])))  # assume we don't have seqs longer than 3500 frames
        rgbs = []
        
        for vid in video_idx:
            video_fn = os.path.join(self.dataset_location, '{}-camera{}-{}.mp4'.format(seq_id, cam_id, vid * 3500))
            cap = cv2.VideoCapture(str(video_fn))
            
            frame_indices = [idx % 3500 for idx in full_idx if idx >= vid * 3500 and idx < (vid + 1) * 3500]
            for frame_index in frame_indices:
                # Set the current frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                
                # Read the frame
                ret, frame = cap.read()
                
                # If the frame is read correctly, add it to the list
                if ret:
                    rgbs.append(frame[:,:,::-1])
                else:
                    print(f"Frame at index {frame_index} could not be read")
            cap.release()
        
        rgbs = np.stack(rgbs, 0)
        visibs = np.all(~np.isnan(image_points), 1)
        points = image_points.copy()
        
        points = interpolate_nan_points(points)
        return {
            'rgbs': rgbs,
            'trajs': points,
            'visibs': visibs,
        }


    def __len__(self):
        return len(self.all_xys)
