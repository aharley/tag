import torch
import multiprocessing
from typing import Optional
from pathlib import Path
import json
import pickle
import os
import cv2
import numpy as np
import sys
from datasets.dataset import TrackingDataset, MaskDataset, BBoxDataset
from pycocotools.coco import COCO
from collections import defaultdict
import utils.misc

class SportsMot(BBoxDataset):
    def __init__(
            self,
            dataset_location,
            S=32, fullseq=False, chunk=None,
            strides=[2,4],
            zooms=[1,2],
            crop_size=(384, 512),
            use_augs=False,
            is_training=True,
    ):
        print("loading sportsmot dataset...")
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training,
        )

        clip_step = S//2 if is_training else S

        # /orion/group/sportsmot/sportsmot_publish/dataset/annotations
        self.root = Path(dataset_location)
        split = "train" if is_training else "val"
        db = COCO(
            f"/orion/group/sportsmot/sportsmot_publish/dataset/annotations/{split}.json"
        )
        self.all_processed_data = defaultdict(list)
        for i, aid in enumerate(db.anns):
            # {'id': 1,
            # 'category_id': 1,
            # 'image_id': 1,
            # 'track_id': 0,
            # 'bbox': [83.0, 421.0, 88.0, 130.0],
            # 'conf': 1.0,
            # 'iscrowd': 0,
            # 'area': 11440.0}
            ann = db.anns[aid]
            image_id = ann["image_id"]
            # {'file_name': 'v_-6Os86HzwCs_c001/img1/000001.jpg',
            # 'id': 1,
            # 'frame_id': 1,
            # 'prev_image_id': -1,
            # 'next_image_id': 2,
            # 'video_id': 1,
            # 'height': 720,
            # 'width': 1280}
            img = db.loadImgs(image_id)[0]
            video_id = img["video_id"]
            frame_id = img["frame_id"]
            track_id = ann["track_id"]
            conf = ann["conf"]
            video_id = f"{split}_{video_id}"
            name_key = (video_id, track_id)
            img_path = f"/orion/group/sportsmot/sportsmot_publish/dataset/{split}/{img['file_name']}"
            # data = {'img_path': img_path, 'frame_id': frame_id, 'bbox': ann['bbox']}
            data = {
                "img_path": img_path,
                "frame_id": frame_id,
                "bbox": ann["bbox"],
                "conf": conf,
            }
            self.all_processed_data[name_key].append(data)
        # Make sure sorted
        for key in self.all_processed_data.keys():
            self.all_processed_data[key] = sorted(
                self.all_processed_data[key], key=lambda x: x["frame_id"]
            )
        self.total_num = sum(
            [len(v) // self.S for v in self.all_processed_data.values()]
        )
        print(f"total number of available tracks of length {self.S}: ", self.total_num)

        self.all_info = []
        for video in sorted(list(self.all_processed_data.keys())):
            data = self.all_processed_data[video]
            S_local = len(data)
            print('S_local', S_local)
            for stride in strides:
                for start in range(0, S_local - self.S*stride + 1, clip_step*stride): # always fullseq
                    for zoom in zooms:
                        self.all_info.append([video, start, stride, zoom])
        print("found {:d} samples in {}".format(len(self.all_info), self.dataset_location))
        if chunk is not None:
            assert(len(self.all_info) > 100)
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.all_info = chunkify(self.all_info,100)[chunk]
            print('filtered to %d' % len(self.all_info))
                    
                    
    def __len__(self):
        return len(self.all_info)

    def getitem_helper(self, index):
        video, start, stride, zoom = self.all_info[index]
        data = self.all_processed_data[video]
        stop = start + self.S*stride
        rgbs = []
        bboxes = []
        visibs = []
        for i in range(start, stop, stride):
            d = data[i]
            img = cv2.imread(d["img_path"])[:, :, ::-1].copy()
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgbs.append(img)
            dim = img.shape
            # print(dim)
            bboxes.append(d["bbox"])
            visibs.append((d["conf"] > 0.75))

        rgbs = np.stack(rgbs)  # S, C, H, W
        bboxes = np.stack(bboxes)  # S, 4
        visibs = np.stack(visibs).reshape(self.S)  # S
        bboxes[..., 2:] += bboxes[..., :2]
        # max the bbox to be within the image
        bboxes[..., 0] = np.clip(bboxes[..., 0], 0, dim[1] - 1)
        bboxes[..., 1] = np.clip(bboxes[..., 1], 0, dim[0] - 1)
        bboxes[..., 2] = np.clip(bboxes[..., 2], 0, dim[1] - 1)
        bboxes[..., 3] = np.clip(bboxes[..., 3], 0, dim[0] - 1)
        # print(bboxes)
        # print(rgbs.shape)
        # print(bboxes.shape)
        # print(visibs.shape)

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
