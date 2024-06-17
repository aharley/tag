from datasets.waymo_utils.coco import COCODetection
from datasets.waymo_utils.vision import ToBGR
from datasets.dataset import BBoxDataset
import utils.misc

import numpy as np
import torch
from typing import Optional
from pathlib import Path
import os
import cv2
import numpy as np
import sys

WIDTH = 1920
HEIGHT = 1280
NUM_CAMERAS = 5
# waymo has 5 cameras in total
# however, in some of the sequences, not all five cameras exist
# this creates difficulty in tracking
# only use front cameras for now
POST_NUM_CAMERAS = 1

def retrieve_only_box_trajectory(subdataset, object_id):
    bboxes = []
    # prev_box = None
    for i in range(len(subdataset)):
        sample = subdataset[i]
        cur_object_ids = sample["object_id"]
        # get the index of the object_id
        if object_id not in cur_object_ids:
            # print(np.array(sample['input']).shape)
            # bboxes.append(prev_box)
            pass
        else:
            idx = list(cur_object_ids).index(object_id)
            # print(np.array(sample['input']).shape)
            # print(sample['bbox'][idx])
            bbox = sample['bbox'][idx][:4]
            # prev_box = bbox
            bboxes.append(bbox)
    return bboxes


def retrieve_trajectory(subdataset, object_id):
    images = []
    bboxes = []
    vis = []
    labels = []
    prev_box = None
    prev_label = None
    for i in range(len(subdataset)):
        sample = subdataset[i]
        cur_object_ids = sample["object_id"]
        # get the index of the object_id
        if object_id not in cur_object_ids:
            vis.append(False)
            # print(np.array(sample['input']).shape)
            images.append(np.array(sample['input']))
            bboxes.append(prev_box)
            labels.append(prev_label)
        else:
            idx = list(cur_object_ids).index(object_id)
            vis.append(True)
            # print(np.array(sample['input']).shape)
            images.append(np.array(sample['input']))
            # print(sample['bbox'][idx])
            bbox = sample['bbox'][idx][:4]
            prev_box = bbox
            bboxes.append(bbox)

            label = sample['bbox'][idx][4]
            prev_label = label
            labels.append(label)

    return images, bboxes, vis, labels


class ScenarioDataset(COCODetection):
    def __init__(self, images_root, annotations, ingore_empty=False, bgr=False):
        super(ScenarioDataset, self).__init__(images_root, annotations)
        if ingore_empty:
            self.ids = [i for i in self.ids if self._get_target(i)]
        self.to_bgr = ToBGR() if bgr else None

    def getitem(self, index):
        sample = super().getitem(index)

        if 'bbox' in sample:
            # filter bbox
            bbox = sample['bbox']
            valid = (bbox[:, 2] > bbox[:, 0] + 0.01) & (bbox[:, 3] > bbox[:, 1] + 0.01)
            sample['bbox'] = bbox[valid]
            sample['object_id'] = sample['object_id'][valid]
            sample['tracking_difficulty_level'] = sample['tracking_difficulty_level'][valid]

        if self.to_bgr:
            sample = self.to_bgr(sample)

        return sample

    def get_anno(self, index):
        sample = super().getitem(index, no_img=True)

        if 'bbox' in sample:
            # filter bbox
            bbox = sample['bbox']
            valid = (bbox[:, 2] > bbox[:, 0] + 0.01) & (bbox[:, 3] > bbox[:, 1] + 0.01)
            sample['bbox'] = bbox[valid]
            sample['object_id'] = sample['object_id'][valid]
            sample['tracking_difficulty_level'] = sample['tracking_difficulty_level'][valid]

        # if self.to_bgr:
        #     sample = self.to_bgr(sample)

        return sample
    
# ****************************************************************************************** #
def check_camera(image_id):
    if "FRONT_RIGHT" in image_id:
        return False
    elif "FRONT_LEFT" in image_id:
        return False
    elif "SIDE_LEFT" in image_id:
        return False
    elif "SIDE_RIGHT" in image_id:
        return False
    elif "FRONT" in image_id:
        return True
    else:
        print("Unknown camera")
        return False

class WaymoDataset(BBoxDataset):
    def __init__(self, dataset_location='../Hoot',
                 S=32,
                 fullseq=False, chunk=None,
                 crop_size=(384,512),
                 zooms=[1,2],
                 use_augs=False,
                 is_training=True):
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training)

        # this dset does not use a clip_step or strides
        
        self.dset = 'train' if is_training else 'test'
        # self.data_root = "/orion/group/waymo_alltrack_coco/training/"# /segment-9747453753779078631_940_000_960_000_with_camera_labels.tfrecord
        if is_training:
            self.data_root = f"{self.dataset_location}/training/"
            self.dataset_location = self.data_root
        # list directories in data_root
        self.scenario_names = sorted([d for d in os.listdir(self.data_root) if os.path.isdir(os.path.join(self.data_root, d))])
        # only use the first 10 for quick check
        self.scenario_names = self.scenario_names[:10]

        self.data_include_empty = False
        self.data_bgr = True

        self.scenario_datasets = {}
        for scenario_name in self.scenario_names:
            self.scenario_datasets[scenario_name] = ScenarioDataset(f"{self.data_root}/{scenario_name}", f"{self.data_root}/{scenario_name}/annotations.json", ingore_empty=not self.data_include_empty, bgr=self.data_bgr)

        self.interval_length = self.S * NUM_CAMERAS
        # idx to sample key mapping
        self.all_info = []
        # for scenario_name, scenario_dataset in self.scenario_datasets.items():
        #     print(scenario_name, len(scenario_dataset))
        #     scenario_max_stop = len(scenario_dataset) - self.interval_length
        #     scenario_dataset = scenario_dataset[:scenario_max_stop]
            
        #     for sample_start in range(len(scenario_dataset)):
        #         # we skip the last two cameras every five cameras
        #         if sample_start % NUM_CAMERAS > (NUM_CAMERAS - POST_NUM_CAMERAS):
        #             continue
        #         sample = scenario_dataset.get_anno(sample_start)
        #         num_queries = len(sample['object_id'])
        #         # Could add tracking difficultly check here.
        #         # Skipped for now
        #         query_indices = list(range(num_queries))
        #         query_keys = [(scenario_name, sample_start, query_idx) for query_idx in query_indices]
        #         self.all_info.extend(query_keys)
        
        # It appears that not all scenarios have the same number of cameras
        # We only keep the front cameras for now
        self.front_camera_idxs = {}
        for scenario_name, scenario_dataset in self.scenario_datasets.items():
            self.front_camera_idxs[scenario_name] = []
            print(scenario_name, len(scenario_dataset))

            scenario_id2keys = []
            for sample_start in range(len(scenario_dataset)):
                sample = scenario_dataset.get_anno(sample_start)
                image_id = sample['image_id']
                if not check_camera(image_id):
                    continue
                self.front_camera_idxs[scenario_name].append(sample_start)
                sample_start_idx = len(self.front_camera_idxs[scenario_name]) - 1
                num_queries = len(sample['object_id'])
                # Could add tracking difficultly check here.
                # Skipped for now
                query_indices = list(range(num_queries))
                for zoom in zooms:
                    query_keys = [(scenario_name, sample_start_idx, query_idx, zoom) for query_idx in query_indices]
                    scenario_id2keys.extend(query_keys)

            max_stop = len(self.front_camera_idxs[scenario_name]) - self.S
            # remove the scenario_id2key that exceeds max_stop
            scenario_id2keys = [x for x in scenario_id2keys if x[1] < max_stop]
            self.all_info.extend(scenario_id2keys)

        self.scale = np.array([WIDTH, HEIGHT, WIDTH, HEIGHT], dtype=np.float32)
        print('found %d samples in %s (dset=%s)' % (len(self.all_info), dataset_location, self.dset))

        if chunk is not None:
            assert(len(self.all_info) > 100)
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.all_info = chunkify(self.all_info,100)[chunk]
            print('filtered to %d' % len(self.all_info))
        
        # self.final_all_info = []
        # self.final_all_zooms = []
        # for index in range(len(self.all_info)):
        #     scenario_name, sample_start_idx, query_idx = self.all_info[index]
        #     sample_starts = self.front_camera_idxs[scenario_name][sample_start_idx:sample_start_idx+self.S]
        #     subdataset = self.scenario_datasets[scenario_name][sample_starts]
        #     query_object_id = subdataset[0]['object_id'][query_idx]
        #     # get object_id for the query and slice to get sub dataset
        #     # scenario_name, sample_start, query_idx = "segment-12681651284932598380_3585_280_3605_280_with_camera_labels.tfrecord", 245, 1
        #     # query_object_id = self.scenario_datasets[scenario_name][sample_start]['object_id'][query_idx]
        #     # print(self.interval_length)
        #     # subdataset = self.scenario_datasets[scenario_name][sample_start:sample_start+self.interval_length][::NUM_CAMERAS]
        #     # print(len(subdataset))
        #     # print(scenario_name, sample_start, query_idx)
        #     bboxes = retrieve_only_box_trajectory(subdataset, query_object_id)
        #     bboxes = np.array(bboxes)
        #     bboxes = bboxes * self.scale
        #     print(bboxes.shape)

        #     mots = []
        #     for i in range(1, len(bboxes)):
        #         xy_prev = (bboxes[i-1, :2] + bboxes[i-1, 2:]) / 2
        #         xy = (bboxes[i, :2] + bboxes[i, 2:]) / 2
        #         dist = np.linalg.norm(xy - xy_prev)
        #         mots.append(dist)
        #     if len(mots) > 4:
        #         mots = np.stack(mots)
        #         mean_mot = np.mean(mots)
        #         print("mean_mot", mean_mot)
        #         if mean_mot > 2:
        #             for zoom in zooms:
        #                 self.final_all_info.append(self.all_info[index])
        #                 self.final_all_zooms.append(zoom)
        # self.all_info = self.final_all_info
        # self.all_zooms = self.final_all_zooms
        # print('filtered to %d moving samples in %s (dset=%s)' % (len(self.all_info), dataset_location, self.dset))
        
    def getitem_helper(self, index):
        scenario_name, sample_start_idx, query_idx, zoom = self.all_info[index]
        
        sample_starts = self.front_camera_idxs[scenario_name][sample_start_idx:sample_start_idx+self.S]
        subdataset = self.scenario_datasets[scenario_name][sample_starts]
        query_object_id = subdataset[0]['object_id'][query_idx]
        # get object_id for the query and slice to get sub dataset
        # scenario_name, sample_start, query_idx = "segment-12681651284932598380_3585_280_3605_280_with_camera_labels.tfrecord", 245, 1
        # query_object_id = self.scenario_datasets[scenario_name][sample_start]['object_id'][query_idx]
        # print(self.interval_length)
        # subdataset = self.scenario_datasets[scenario_name][sample_start:sample_start+self.interval_length][::NUM_CAMERAS]
        # print(len(subdataset))
        # print(scenario_name, sample_start, query_idx)
        rgbs, bboxes, visibs, labels = retrieve_trajectory(subdataset, query_object_id)
        # print('labels', labels)
        rgbs = np.array(rgbs)
        bboxes = np.array(bboxes)
        visibs = np.array(visibs)
        
        # (32, 720, 1280, 3)
        # (32, 4)
        # (32,)
        # (32, 1280, 1920, 3) (32, 4) (32,)
        # scale bboxes back to original size
        bboxes = bboxes * self.scale
        # print(rgbs.shape, bboxes.shape, vis.shape)
        # exit()

        mots = []
        for i in range(1, len(bboxes)):
            xy_prev = (bboxes[i-1, :2] + bboxes[i-1, 2:]) / 2
            xy = (bboxes[i, :2] + bboxes[i, 2:]) / 2
            dist = np.linalg.norm(xy - xy_prev)
            mots.append(dist)
        mots = np.stack(mots)
        
        mean_mot = np.mean(mots)
        print('zoom', zoom, 'mean_mot', mean_mot)

        if mean_mot < 2.0:
            print('no motion detected')
            return None

        rgbs, bboxes = utils.misc.data_pad_if_necessary_b(rgbs, bboxes, visibs)
        S,H,W,C = rgbs.shape
        if zoom > 1:
            whs = bboxes[:,2:4] - bboxes[:,0:2]
            whs = whs[visibs > 0.5]
            if np.mean(whs[:,0])*zoom >= W and np.mean(whs[:,1])*zoom >= H:
                return None
            bboxes, visibs, rgbs = utils.misc.data_zoom_bbox(zoom, bboxes, visibs, rgbs)
        safe = np.concatenate(([0], visibs[:-2] * visibs[1:-1] * visibs[2:], [0]))
        if np.sum(safe) < 2: return None

        sample = {
            "rgbs": rgbs.astype(np.uint8),
            "bboxes": bboxes.astype(np.float64),
            "visibs": visibs.astype(np.float32),
        }
        return sample
    
    def __len__(self):
        return len(self.all_info)


