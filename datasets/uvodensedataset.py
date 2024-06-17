import numpy as np
import os
import pycocotools.mask as cocomask
import json
import cv2
from datasets.dataset import MaskDataset, mask2bbox
import utils.misc

import sys
# np.random.seed(125)
# torch.multiprocessing.set_sharing_strategy('file_system')

class UVODenseDataset(MaskDataset):
    def __init__(self,
                 dataset_location='../uvo',
                 S=32, fullseq=False, 
                 crop_size=(384,512), 
                 # zooms=[1,2,3],
                 zooms=[1,1.5,2,3],
                 use_augs=False,
                 is_training=True,
                 chunk=None,
    ):

        print('loading UVO dataset...')
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size, 
            use_augs=use_augs,
            is_training=is_training
        )
        self.dataset_location = dataset_location
        self.split = 'train' if is_training else 'val'
        self.annotations = json.load(open(os.path.join(self.dataset_location, 'VideoDenseSet', 'UVO_video_train_dense_with_label.json' if is_training else 'UVO_video_val_dense_with_label.json')))

        vid_dict = {}
        for video in self.annotations["videos"]:
            vid_dict[video['id']] = video
        print('found {:d} videos in {}'.format(len(vid_dict), self.dataset_location))

        self.all_info = []

        for track in self.annotations["annotations"]:
            # print(video_name)
            # sys.stdout.write(video_name)
            video_frame_paths = vid_dict[track['video_id']]
            
            valid_idxs = []
            for i in range(len(track['segmentations'])):
                if track['segmentations'][i] is not None:
                    valid_idxs.append(i)
                    
            S_local = len(valid_idxs)
            print('S_local', S_local)
            if S_local < 8:
                continue

            # enforce a kind of fullseq, by just converting each video into the seq of max length
            full_idx = np.linspace(0, S_local-1, min(self.S, S_local), dtype=np.int32)
            full_idx = [valid_idxs[fi] for fi in full_idx]

            for zoom in zooms:
                self.all_info.append([video_frame_paths, track['id'], track['segmentations'], full_idx, zoom])
                sys.stdout.write('.')
                sys.stdout.flush()
        
        print('found {:d} samples in {}'.format(len(self.all_info), self.dataset_location))
        if chunk is not None:
            assert(len(self.all_info) > 100)
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.all_info = chunkify(self.all_info,100)[chunk]
            print('filtered to %d' % len(self.all_info))
        
        
    def __len__(self):
        return len(self.all_info)
        
    def getitem_helper(self, index):
        video, tid, segs, full_idx, zoom = self.all_info[index]

        # print('video', video)
        # print('full_idx', full_idx)
        # print('tid', tid)
        # print('segs', segs)
        
        h, w = video["height"], video["width"]
        base_dir = os.path.join(self.dataset_location, 'uvo_videos_dense_frames')
        image_paths = [os.path.join(base_dir, img_path) for img_path in video['file_names']]
        
        num_frames = len(image_paths)
        
        image_paths = [image_paths[pi] for pi in full_idx]
        segs = [segs[pi] for pi in full_idx]
        
        rgb = cv2.imread(str(image_paths[0]))
        H, W = rgb.shape[:2]
        rgbs = []
        for path in image_paths:
            rgb = cv2.imread(str(path))[..., ::-1].copy()
            rgbs.append(rgb)

        masks = [cocomask.decode(seg) if seg is not None else np.zeros((h, w), dtype=bool) for seg in segs]

        rgbs = np.stack(rgbs)
        masks = np.stack(masks).astype(np.float32)

        S,H,W,C = rgbs.shape
        # print('H, W', H, W)

        if np.sum(masks) == 0:
            print('np.sum(masks)', np.sum(masks))
            return None

        mask_areas = (masks > 0).reshape(S,-1).sum(axis=1)
        mask_areas_norm = mask_areas / np.max(mask_areas)
        visibs = mask_areas_norm
        rgbs, masks = utils.misc.data_pad_if_necessary(rgbs, masks)
        S,H,W,C = rgbs.shape
        if zoom > 1:
            xys, _, valids, fills = utils.misc.data_get_traj_from_masks(masks)
            xys = utils.misc.data_replace_with_nearest(xys, valids)
            bboxes = np.stack([mask2bbox(mask) for mask in masks])
            whs = bboxes[:,2:4] - bboxes[:,0:2]
            whs = whs[visibs > 0.5]
            if np.mean(whs[:,0])*zoom >= W and np.mean(whs[:,1])*zoom >= H:
                return None
            xys, visibs, valids, rgbs, masks = utils.misc.data_zoom(zoom, xys, visibs, valids, rgbs, masks=masks)
        visibs = 1.0 * (visibs>0.5)
        safe = np.concatenate(([0], visibs[:-2] * visibs[1:-1] * visibs[2:], [0]))
        if np.sum(safe) < 2: return None

        sample = {
            "rgbs": rgbs,
            "masks": masks,
        }
        return sample

if __name__ == '__main__':
    uvo = UVODenseDataset('/projects/katefgroup/datasets/UVO')
    sample = uvo[1]
