import json
import os
from pathlib import Path
import cv2
import numpy as np
from datasets.dataset import PointDataset
import utils.misc

MIN_SEQUENCE_LENGTH = 10

class APTv2PointDataset(PointDataset):
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
        
        print("loading APTv2 point dataset...")
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training,
        )
        self.dataset_location = dataset_location

        self.root = Path(dataset_location)

        # note the clips in aptv2point are short, so there is no clip_step or stride
        
        annotations_path = os.path.join(
            self.root,
            "annotations",
            f"{'train' if is_training else 'val'}_annotations.json",
        )
        with open(annotations_path) as f:
            self.gts = json.load(f)

        images_dicts = self.gts["images"]

        # print('images_dicts', len(images_dicts))
        
        video_to_img_path = {}
        for entry in images_dicts:
            filename = entry["file_name"]
            video_id = entry["video_id"]
            id_ = entry["id"]
            key = (video_id, id_)

            # print('entry, key', entry, key)

            if key in video_to_img_path:
                assert False
            else:
                video_to_img_path[key] = filename

        # if is_training:
        #     flips = [False, True]
        # else:
        #     flips = [False]
            
        # if is_training:
        #     zooms = [1, 2]
        # else:
        #     zooms = [1]
            
        annotation_dicts = self.gts["annotations"]
        # annotation_dicts = sorted(annotation_dicts)

        # print('annotation_dicts', len(annotation_dicts))
        # if chunk is not None:
        #     def chunkify(lst,n):
        #         return [lst[i::n] for i in range(n)]
        #     annotation_dicts = chunkify(annotation_dicts,100)[chunk]
        #     print('filtered to %d annots' % len(annotation_dicts))
        #     # print('annotation_dicts', annotation_dicts)
        
        ids_to_annot = {}
        for entry in annotation_dicts:
            # Note: I think that "id" is the id of the annotation, not of the image. So, we must use "image_id" here
            keypoints = entry["keypoints"]
            for i in range(len(keypoints) // 3):
                for zoom in zooms: # inflate the dataset with zooms
                    # keypoint4 ("tail root") seems very noisy, so we discard it
                    # ... same for keypoint 11 and 14, which are left and right hip
                    if i!=4 and i!=11 and i!=14:
                        video_id_and_track = (entry["video_id"], entry["track_id"], i, zoom)
                        to_append = (entry["image_id"], keypoints[i * 3], keypoints[i * 3 + 1], keypoints[i * 3 + 2])
                        if video_id_and_track in ids_to_annot:
                            ids_to_annot[video_id_and_track].append(to_append) 
                        else:
                            ids_to_annot[video_id_and_track] = [to_append]
        ids_to_annot_filtered_and_sorted = {}
        for key in ids_to_annot:
            # Comment this out to do filtering by sequence length
            # if len(ids_to_annot[key]) >= MIN_SEQUENCE_LENGTH:
            # Some sequences have certain keypoints invisible for the entire sequence

            # num_visibs should be 2 when it's visible on 1 timestep
            # so we ask for >4, to get >2 timesteps
            if sum(num_visibs for _, _, _, num_visibs in ids_to_annot[key]) > 4:
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
        video_id, track_id, keypoint_id, zoom = self.sequences[index]

        # print('video_id', video_id)
        # print('track_id', track_id)
        # print('keypoint_id', keypoint_id)
        
        img_paths = []
        xys = []
        visibs = []
        for id_, x, y, num_visibs in self.ids_to_annot[(video_id, track_id, keypoint_id, zoom)]:
            img_paths.append(self.video_to_img_path[(video_id, id_)])
            xys.append(np.array([x, y]))
            visibs.append([int(num_visibs == 2)])

        xys = np.stack(xys).astype(np.float32)
        visibs = np.concatenate(visibs).astype(np.float32)
        # print('S', len(xys))

        # xys_ = xys[visibs>0]
        # vels = xys_[1:] - xys_[:-1]
        # accels = vels[1:] - vels[:-1]
        # max_vel = np.max(np.linalg.norm(vels))
        # max_accel = np.max(np.linalg.norm(accels))
        # print('max_accel', max_accel)
        
        rgbs = [cv2.imread(str(os.path.join(self.dataset_location, "data", path)))[..., ::-1].copy() for path in img_paths]
        rgbs = np.stack(rgbs) # S,H,W,C

        S = len(rgbs)

        if S > self.S:
            rgbs = rgbs[:self.S]
            xys = xys[:self.S]
            visibs = visibs[:self.S]

        # if S <= 2:
        #     print('S', S)
        #     return None

        xys = utils.misc.data_replace_with_nearest(xys, visibs)
        valids = np.ones_like(visibs)
        
        if zoom > 1:
            xys, visibs, valids, rgbs = utils.misc.data_zoom(zoom, xys, visibs, valids, rgbs)
        S,H,W,C = rgbs.shape

        visibs = 1.0 * (visibs>0.5)
        safe = visibs*0
        for si in range(1,S-1):
            safe[si] = visibs[si-1]*visibs[si]*visibs[si+1]

        if np.sum(safe) < 2:
            print('safe', safe)
            return None

        # in this dset, we will use visibs=safe,
        # since the vis is sometimes wrong during self-occlusions,
        # and it is risky to start tracks there
        safe[0] = visibs[0]
        safe[-1] = visibs[-1]
        visibs = safe

        sample = {
            "rgbs": rgbs,
            "xys": xys,
            "visibs": visibs,
        }
        return sample


if __name__ == "__main__":
    dataset = APTv2PointDataset()
