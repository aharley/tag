import os
from pathlib import Path
import cv2
import numpy as np
from datasets.dataset import BBoxDataset
from datasets.dataset_utils import make_split
import sys
import utils.misc


class TlpDataset(BBoxDataset):
    def __init__(
            self,
            dataset_location="../TLP_V2",
            S=32, fullseq=False, chunk=None,
            strides=[2,4],  # the videos seem like the frames repeat 2x
            zooms=[1,2],
            crop_size=(384, 512),
            use_augs=False,
            is_training=True,
    ):
        print("loading tlp dataset...")
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training,
        )

        clip_step = S//2
        if not is_training:
            strides = [2]
            clip_step = S

        self.root = Path(dataset_location)

        skip_sequences = [
            "CarChase4",
        ]  # Sequences that are missing the ground-truth file

        sequences = sorted([os.path.join(self.root, sequence_name) for sequence_name in os.listdir(self.root) if sequence_name not in skip_sequences])

        self.gt_fns = [
            os.path.join(sequence_path, "groundtruth_rect.txt")
            for sequence_path in sequences
        ]
        sequences_and_gt_fns = make_split(
            zip(sequences, self.gt_fns), is_training, shuffle=True
        )
        sequences = [elem[0] for elem in sequences_and_gt_fns]
        self.gt_fns = [elem[1] for elem in sequences_and_gt_fns]
        print("found {:d} videos in {}".format(len(self.gt_fns), self.dataset_location))

        self.all_info = []
        for sequence_id, gtf in enumerate(self.gt_fns):
            gt = np.loadtxt(gtf, delimiter=",").astype(int)
            frame_ids = gt[:, 0]

            img_paths = []
            bboxes = []
            visibs = []
            for fid in range(np.min(frame_ids), np.max(frame_ids) + 1):
                img_paths.append(
                    os.path.join(os.path.dirname(gtf), "img", f"{fid:05d}.jpg")
                )
                if fid in frame_ids:
                    bboxes.append(gt[frame_ids == fid, 1:5])
                    # Final index is `isLost`, where 1 means target object is not at
                    # all visible
                    visibs.append(1 - gt[frame_ids == fid, -1])
                else:
                    bboxes.append(np.array([[0, 0, 0, 0]]))
                    visibs.append(np.array([0]))
            bboxes = np.concatenate(bboxes)
            visibs = np.concatenate(visibs)
            

            bboxes[..., 2:] += bboxes[..., :2]
            xys = bboxes[:,0:2]*0.5 + bboxes[:,2:4]*0.5
            
            S_local = len(img_paths)
            for stride in strides:
                for ii in range(0, max(S_local - self.S * stride, 1), clip_step*stride):
                    full_idx = ii + np.arange(self.S) * stride
                    full_idx = [ij for ij in full_idx if ij < S_local]
                    contains_missing_frame = False
                    for ij in full_idx:
                        if img_paths[ij] is None:
                            contains_missing_frame = True
                            break
                    if contains_missing_frame: continue
                    if len(full_idx) < (((S_local - 1) // stride + 1 if self.S == -1 else self.S) if fullseq else 8):
                        continue
                    xys_here = xys[full_idx]
                    visibs_here = visibs[full_idx]
                    if np.sum(visibs_here) < self.S: continue # some with occlusions are weird
                    travel = np.sum(np.linalg.norm(xys_here[1:]-xys_here[:-1], axis=-1))
                    if travel < self.S*2: continue
                    for zoom in zooms:
                        self.all_info.append([gtf, full_idx, zoom])
            sys.stdout.write('%d ' % sequence_id)
            sys.stdout.flush()

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
        gt_fn, full_idx, zoom = self.all_info[index]

        gt = np.loadtxt(gt_fn, delimiter=",").astype(int)
        frame_ids = gt[:, 0]

        img_paths = []
        bboxes = []
        visibs = []
        for fid in range(np.min(frame_ids), np.max(frame_ids) + 1):
            img_paths.append(
                os.path.join(os.path.dirname(gt_fn), "img", f"{fid:05d}.jpg")
            )
            if fid in frame_ids:
                bboxes.append(gt[frame_ids == fid, 1:5])
                # Final index is `isLost`, where 1 means target object is not at
                # all visible
                visibs.append(1 - gt[frame_ids == fid, -1])
            else:
                bboxes.append(np.array([[0, 0, 0, 0]]))
                visibs.append(np.array([0]))
        bboxes = np.concatenate(bboxes)
        visibs = np.concatenate(visibs)

        img_paths = [img_paths[fi] for fi in full_idx]
        bboxes = bboxes[full_idx]
        visibs = visibs[full_idx]

        rgbs = [cv2.imread(str(path))[..., ::-1].copy() for path in img_paths]

        rgbs = np.stack(rgbs)  # S, C, H, W
        # from xywh to xyxy
        bboxes[..., 2:] += bboxes[..., :2]

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
            "rgbs": rgbs,
            "visibs": visibs,
            "bboxes": bboxes,
        }
        return sample


if __name__ == "__main__":
    dataset = TlpDataset()
    for i in range(10):
        print(dataset[i])
