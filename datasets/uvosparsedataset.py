
import numpy as np
import os
import pycocotools.mask as cocomask
import json
import cv2
from datasets.dataset import MaskDataset

# np.random.seed(125)
# torch.multiprocessing.set_sharing_strategy('file_system')

class UVOSparseDataset(MaskDataset):
    def __init__(self,
                 dataset_location='../uvo',
                 S=32,
                 crop_size=(384,512), 
                 strides=[1],
                 fullseq=False, chunk=None,
                 use_augs=False,
                 is_training=True):

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
        self.annotations = json.load(open(os.path.join(self.dataset_location, 'VideoSparseSet', 'UVO_sparse_train_video.json' if is_training else 'UVO_sparse_val_video.json')))

        self.mask_fill_thr = 0.1
        self.mask_fill_thr0 = 0.2
        strides = [1] # Annotations are too sparse, hardcode stride to be 1

        vid_dict = {}
        for video in self.annotations["videos"]:
            vid_dict[video['id']] = video
        print('found {:d} videos in {}'.format(len(vid_dict), self.dataset_location))

        self.all_vids = []
        self.all_tids = []
        self.all_segs = []
        self.all_strides = []
        self.all_starts = []

        for track in self.annotations["annotations"]:
            # print(video_name)
            # sys.stdout.write(video_name)
            video_frame_paths = vid_dict[track['video_id']]
            
            for stride in strides:
                valid_idxs = []
                for i in range(len(track['segmentations'])):
                    if track['segmentations'][i] is not None:
                        valid_idxs.append(i)
                if len(valid_idxs) > 0:

                    # print('len(valid_idxs)', len(valid_idxs))

                    sidx = valid_idxs[0]
                    eidx = valid_idxs[-1]
                    for ii in range(sidx, max(eidx - self.S * stride + 1, sidx + 1), min(self.S * stride // 2,8)):
                        start_idx = ii
                        self.all_vids.append(video_frame_paths)
                        self.all_tids.append(track['id'])
                        self.all_segs.append(track['segmentations'])
                        self.all_strides.append(stride)
                        self.all_starts.append(start_idx)
        
        print('found {:d} samples in {}'.format(len(self.all_vids), self.dataset_location))
        
    def __len__(self):
        return len(self.all_vids)
        
    def getitem_helper(self, index):
        video = self.all_vids[index]
        segs = self.all_segs[index]
        stride = self.all_strides[index]
        
        h, w = video["height"], video["width"]
        base_dir = os.path.join(self.dataset_location, 'uvo_videos_sparse_frames')
        image_paths = [os.path.join(base_dir, img_path) for img_path in video['file_names']]
        
        num_frames = len(image_paths)
        perm = np.arange(num_frames)

        S = self.S * stride

        start_ind = self.all_starts[index]
        perm = perm[start_ind:start_ind + S:stride]
        
        image_paths = [image_paths[pi] for pi in perm]
        rgb = cv2.imread(str(image_paths[0]))
        H, W = rgb.shape[:2]
        #print('H, W', H, W)
        sc = 1.0
        if H > 384:
            sc = 384/H
            H_, W_ = int(H*sc), int(W*sc)
        rgbs = []
        for path in image_paths:
            rgb = cv2.imread(str(path))[..., ::-1].copy()
            if sc < 1.0:
                rgb = cv2.resize(rgb, (W_, H_), interpolation=cv2.INTER_AREA)
            rgbs.append(rgb)

        segs = [segs[pi] for pi in perm]
        
        masks = [cocomask.decode(seg) if seg is not None else np.zeros((h, w), dtype=bool) for seg in segs]

        masks = [cv2.resize(mask.astype(np.uint8), (W_, H_), interpolation=cv2.INTER_NEAREST) for mask in masks]

        rgbs = np.stack(rgbs)
        masks = np.stack(masks).astype(np.float32)

        vis_g = np.ones((S))
        for si, mask in enumerate(masks):
            if np.sum(mask) > 8:
                ys, xs = np.where(mask)
                x0, x1 = np.min(xs), np.max(xs)
                y0, y1 = np.min(ys), np.max(ys)
                crop = mask[y0:y1,x0:x1]
                fill = np.mean(crop)
                if fill < self.mask_fill_thr or (si==0 and fill < self.mask_fill_thr0):
                    print('uvo: fill %.2f on frame %d' % (fill, si))
                    return None
            else:
                vis_g[si] = 0

        if np.sum(vis_g) < np.sqrt(S):
            print('ovis: vis %d/%d' % (np.sum(vis_g), S))
            return None

        sample = {
            'rgbs': rgbs,
            'masks': masks,
        }
        
        return sample

if __name__ == '__main__':
    uvo = UVOSparseDataset('/projects/katefgroup/datasets/UVO')
    sample = uvo[1]
