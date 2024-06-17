import json
import os
from pathlib import Path
import cv2
import numpy as np
from datasets.dataset import BBoxDataset
import utils.misc

MIN_SEQUENCE_LENGTH = 10

class APTv2BBoxDataset(BBoxDataset):
    def __init__(
            self,
            dataset_location="/orion/group/APTv2/videos/APTv2/",
            S=32, fullseq=False, chunk=None,
            crop_size=(384, 512),
            zooms=[1.2,1.5,2],
            use_augs=False,
            is_training=True,
    ):
        # note every most videos in aptv2 are 15 frames long,
        # so we do not bother with strides, and we don't support fullseq
        assert(fullseq==False)
        
        print("loading APTv2 bbox dataset...")
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training,
        )
        self.dataset_location = dataset_location

        self.root = Path(dataset_location)

        annotations_path = os.path.join(
            self.root,
            "annotations",
            f"{'train' if is_training else 'val'}_annotations.json",
        )
        with open(annotations_path) as f:
            self.gts = json.load(f)

        images_dicts = self.gts["images"]
        video_to_img_path = {}
        for entry in images_dicts:
            filename = entry["file_name"]
            video_id = entry["video_id"]
            id_ = entry["id"]
            key = (video_id, id_)

            if key in video_to_img_path:
                assert False
            else:
                video_to_img_path[key] = filename

        annotation_dicts = self.gts["annotations"]
        ids_to_annot = {}
        for entry in annotation_dicts:
            for zoom in zooms:
                # Note: I think that "id" is the id of the annotation, not of the image. So, we must use "image_id" here
                video_id_and_track = (entry["video_id"], entry["track_id"], zoom)
                if video_id_and_track in ids_to_annot:
                    ids_to_annot[video_id_and_track].append((entry["image_id"], entry["bbox"])) 
                else:
                    ids_to_annot[video_id_and_track] = [(entry["image_id"], entry["bbox"])]
        ids_to_annot_filtered_and_sorted = {}
        for key in ids_to_annot:
            # Comment this out to do filtering by sequence length
            # if len(ids_to_annot[key]) >= MIN_SEQUENCE_LENGTH:
            ids_to_annot_filtered_and_sorted[key] = sorted(ids_to_annot[key])

        self.video_to_img_path = video_to_img_path
        self.ids_to_annot = ids_to_annot_filtered_and_sorted
        self.sequences = list(self.ids_to_annot.keys())

        print('found %d sequences in %s' % (len(self.sequences), dataset_location))
        if chunk is not None:
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.sequences = chunkify(self.sequences,100)[chunk]
            print('filtered to %d sequences' % len(self.sequences))
            # print('self.sequences', self.sequences)
        
    def __len__(self):
        return len(self.sequences)

    def getitem_helper(self, index):
        video_id, track_id, zoom = self.sequences[index]

        img_paths = []
        bboxes = []
        visibs = []
        for id_, bbox in self.ids_to_annot[(video_id, track_id, zoom)]:
            img_paths.append(self.video_to_img_path[(video_id, id_)])
            bboxes.append(bbox)
            visibs.append(np.array([1]))

        bboxes = np.stack(bboxes)
        visibs = np.concatenate(visibs)

        rgbs = [cv2.imread(str(os.path.join(self.dataset_location, "data", path)))[..., ::-1].copy() for path in img_paths]

        rgbs = np.stack(rgbs)
        bboxes[..., 2:] += bboxes[..., :2]

        S,H,W,C = rgbs.shape
        if S > self.S:
            rgbs = rgbs[:self.S]
            bboxes = bboxes[:self.S]
            visibs = visibs[:self.S]

        if zoom > 1:
            whs = bboxes[:,2:4] - bboxes[:,0:2]
            whs = whs[visibs > 0.5]
            if np.mean(whs[:,0])*zoom >= W and np.mean(whs[:,1])*zoom >= H:
                return None
            bboxes, visibs, rgbs = utils.misc.data_zoom_bbox(zoom, bboxes, visibs, rgbs)
        S,H,W,C = rgbs.shape
        safe = np.concatenate(([0], visibs[:-2] * visibs[1:-1] * visibs[2:], [0]))
        if np.sum(safe) < 2: return None
        
        sample = {
            "rgbs": rgbs,
            "visibs": visibs,
            "bboxes": bboxes,
        }
        return sample


if __name__ == "__main__":
    dataset = APTv2BBoxDataset()
