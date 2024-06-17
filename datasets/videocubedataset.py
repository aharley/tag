from __future__ import absolute_import, print_function
import glob
import os
from pathlib import Path
import cv2
import numpy as np
import six
from datasets.dataset import BBoxDataset
# from datasets.dataset_utils import make_split
import utils.misc


class VideoCubeDataset(BBoxDataset):
    def __init__(
            self,
            dataset_location="../VideoTube",
            S=32, fullseq=False, chunk=None,
            strides=[1,2],
            zooms=[1,2],
            crop_size=(384, 512),
            use_augs=False,
            is_training=True,
    ):
        """
        Args: (specific to videotube)
            strides (tuple[int])
        """
        print("Loading VideoCube dataset.")
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training,
        )

        self.root = Path(dataset_location)
        self.seq_names = sorted(Path(self.root / "MGIT-Train").glob("*/"))
        self.seq_names = [seq_name.name for seq_name in self.seq_names]
        # self.seq_names = make_split(self.seq_names, is_training, shuffle=True)
        print(f"Found {len(self.seq_names)} videos in {self.root}")

        self.all_info = []

        clip_step = S // 2
        num_skipped = 0
        for s_i, seq_name in enumerate(self.seq_names):
            img_files = sorted(Path(self.root / "MGIT-Train" / seq_name / f"frame_{seq_name}").glob("*.jpg"))
            S_local = len(img_files)
            # print('S_local', S_local)
            
            if S_local < S: continue
            # some folders are empty

            print('seq_name', seq_name, 'S_local', S_local)
            
            absence = np.loadtxt(self.root / "VideoCube-Info/attribute/absent" / f"{seq_name}.txt", delimiter=",")
            occlusion = np.loadtxt(self.root / "VideoCube-Info/attribute/occlusion" / f"{seq_name}.txt", delimiter=",")
            vis = 1 - (absence * occlusion)

            for stride in strides:
                for ii in range(0, max(S_local - self.S * stride, 1), clip_step*stride):
                    full_idx = ii + np.arange(self.S) * stride
                    full_idx = [ij for ij in full_idx if ij < S_local]

                    if len(full_idx) < (((S_local - 1) // stride + 1 if self.S == -1 else self.S) if fullseq else 8): continue

                    vis_here = vis[full_idx]

                    if np.sum(vis_here) < 3: continue

                    for zoom in zooms:
                        self.all_info.append((s_i, full_idx, zoom))
                    
        print("found {:d} samples in {}".format(len(self.all_info), self.dataset_location))
        if chunk is not None:
            assert(len(self.all_info) > 100)
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.all_info = chunkify(self.all_info,100)[chunk]
            print('filtered to %d' % len(self.all_info))
        

    def getitem_helper(self, index):
        seq_index, full_idx, zoom = self.all_info[index]
        seq_name = self.seq_names[seq_index]
        img_files = sorted(Path(self.root / "MGIT-Train" / seq_name / f"frame_{seq_name}").glob("*.jpg"))
        bboxes = np.loadtxt(self.root / "VideoCube-Info/attribute/groundtruth" / f"{seq_name}.txt",delimiter=",")
        absence = np.loadtxt(self.root / "VideoCube-Info/attribute/absent" / f"{seq_name}.txt", delimiter=",")
        occlusion = np.loadtxt(self.root / "VideoCube-Info/attribute/occlusion" / f"{seq_name}.txt", delimiter=",")
        visibs = 1 - (absence * occlusion)

        img_paths = [img_files[fi] for fi in full_idx]
        bboxes = bboxes[full_idx]
        visibs = visibs[full_idx]
        
        rgbs = []
        for path in img_paths:
            rgb = cv2.imread(str(path))[..., ::-1].copy()
            rgbs.append(rgb)

        rgbs = np.stack(rgbs)  # Shape (S, C, H, W)
        bboxes = np.stack(bboxes)  # Shape (S, 4)
        # From xywh to xyxy
        bboxes[..., 2:] += bboxes[..., :2]

        rgbs, bboxes = utils.misc.data_pad_if_necessary_b(rgbs, bboxes, visibs)
        S,H,W,C = rgbs.shape
        if zoom > 1:
            whs = bboxes[:,2:4] - bboxes[:,0:2]
            whs = whs[visibs > 0.5]
            if np.mean(whs[:,0])*zoom >= W or np.mean(whs[:,1])*zoom >= H: return None
            bboxes, visibs, rgbs = utils.misc.data_zoom_bbox(zoom, bboxes, visibs, rgbs)
        safe = np.concatenate(([0], visibs[:-2] * visibs[1:-1] * visibs[2:], [0]))
        if np.sum(safe) < 2: return None

        sample = {
            "rgbs": rgbs,
            "bboxes": bboxes,
            "visibs": visibs,
        }
        return sample

    def __len__(self):
        return len(self.all_info)
