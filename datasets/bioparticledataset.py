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
import pickle
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

from datasets.dataset import PointDataset
from datasets.dataset_utils import make_split
import utils.misc

# root = "/orion/group/particle_challenge/"
# targets = ["MICROTUBULE", "VESICLE", "RECEPTOR"]
# S = 32
# STEP = 2

def parse_xml_file(file_path, S):
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Initialize a list to hold all particle tracks
    all_tracks = []

    # Iterate over each particle in the XML
    for particle in root.findall('.//particle'):
        track = []

        # Iterate over each detection in the particle
        for detection in particle.findall('detection'):
            # Extract the attributes (t, x, y, z)
            t = detection.get('t')
            x = detection.get('x')
            y = detection.get('y')
            z = detection.get('z')
            # if z > 0:
            #     print('z', z)

            # if float(z) > 1:
            #     continue

            # Add the detection data to the track
            # track.append({'t': t, 'x': x, 'y': y, 'z': z})
            # track.append([t, x, y])
            # convert to float
            track.append([int(t), float(x), float(y), float(z)])

        if S>0 and len(track) < S:
            continue
        # Add the completed track to the list of all tracks
        all_tracks.append(track)

    return all_tracks

class BioparticleDataset(PointDataset):
    def __init__(
            self,
            dataset_location,
            S=8, fullseq=False, chunk=None,
            zooms=[2,3,4,5],
            crop_size=None,
            use_augs=False,
            is_training=True,
    ):
        print("loading bioparticle dataset...")
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            zooms=zooms,
            crop_size=crop_size, 
            use_augs=use_augs,
            is_training=is_training,
        )


        clip_step = S//2
        
        if not is_training:
            clip_step = S
            zooms = [1]

        self.dataset_location = dataset_location
        self.S = S
        self.zooms = zooms
        self.targets = ["MICROTUBULE", "VESICLE", "RECEPTOR", "VIRUS"]

        self.all_video_names = []
        self.all_video_info = {}
        self.all_sample_idcs = []

        for target in self.targets:
            self.all_video_names += glob.glob(self.dataset_location + target + "/*")

        print('found %d videos' % len(self.all_video_names))

        self.all_video_names = make_split(self.all_video_names, is_training, shuffle=True)
        print('split to %d videos' % len(self.all_video_names))

        for video_name in self.all_video_names:
            # when SNR is 3 or less, the task is nearly impossible to my eye
            # http://www.bioimageanalysis.org/track/#data
            if 'snr 1' in video_name:
                continue
            if 'snr 2' in video_name:
                continue
            if 'snr 3' in video_name:
                continue
            if 'VIRUS' in video_name: # looking at the GT it looks impossible/wrong
                continue
            # print('video_name', video_name)
            video_info = {}
            all_image_names = glob.glob(video_name + "/*.tif")
            all_image_names = sorted(all_image_names)
            target_info = glob.glob(video_name + "/*.xml")
            assert len(target_info) == 1
            target_info = target_info[0]
            # Read the .xml
            if fullseq:
                gt_track = parse_xml_file(target_info, self.S)
            else:
                gt_track = parse_xml_file(target_info, 0)
            video_info["gt_track"] = gt_track
            video_info["image_names"] = all_image_names
            self.all_video_info[video_name] = video_info
            for track_id, track in enumerate(gt_track):
                # print('len(track)', len(track))
                # print('track', track)
                # the videos are 30-50 frames long
                for i in range(0, len(track)-S//2, clip_step):
                    for zoom in self.zooms:
                        self.all_sample_idcs.append([video_name, track_id, i, zoom])

        if chunk is not None:
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.all_sample_idcs = chunkify(self.all_sample_idcs,100)[chunk]
            print('filtered to %d all_sample_idcs' % len(self.all_sample_idcs))
            print('self.all_sample_idcs', self.all_sample_idcs)
        
        print(f"found {len(self.all_sample_idcs)} samples in {dataset_location} (training={is_training})")

    def getitem_helper(self, index):
        video_name, track_id, start_ind, zoom = self.all_sample_idcs[index]
        
        video_info = self.all_video_info[video_name]

        gt_track = video_info["gt_track"][track_id]

        S_max = len(gt_track)

        gt_track = gt_track[start_ind:min(start_ind+self.S, S_max)]
        S = len(gt_track)
        
        # # print('video_info', video_info['image_names'][0])
        # ara = np.arange(start_ind, min(start_ind + self.S, S_here))
        
        start_time = gt_track[0][0]
        all_images = video_info["image_names"]
        images = []

        # S = len(ara)
        
        for i in range(S):
            with Image.open(all_images[start_time + i]) as img:
                img_array = np.array(img)
                torgb = cv2.cvtColor(img_array,cv2.COLOR_GRAY2RGB)
            images.append(torgb)

        images = [np.expand_dims(img, 0) for img in images]
        rgbs = np.concatenate(images, axis=0)  # S, H, W, C
        gt_track = np.array(gt_track)
        z = gt_track[:,-1]
        xys = gt_track[:, 1:3]
        # print('z', z)

        rgbs = rgbs.astype(np.float32)
        rgbs = rgbs - np.min(rgbs)
        rgbs = rgbs / np.max(rgbs)
        rgbs = rgbs * 255.0

        # print('rgbs', rgbs.shape)

        visibs = np.ones((S,))  # S
        valids = np.ones((S,))  # S
        
        if zoom > 1:
            xys, visibs, valids, rgbs = utils.misc.data_zoom(zoom, xys, visibs, valids, rgbs)

        visibs = 1.0 * (visibs>0.5)
        safe = visibs*0
        for si in range(1,S-1):
            safe[si] = visibs[si-1]*visibs[si]*visibs[si+1]
            
        if np.sum(safe) < 2:
            print('safe', safe)
            return None
            
        d = {
            "rgbs": rgbs.astype(np.uint8),  # S, H, W, C
            "xys": xys.astype(np.int64),  # S, 2
            "visibs": visibs.astype(np.float32),  # S
            "valids": valids.astype(np.float32),  # S
        }
        return d

    def __len__(self):
        return len(self.all_sample_idcs)
