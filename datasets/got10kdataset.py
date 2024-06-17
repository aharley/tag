from __future__ import absolute_import, print_function

import glob
import os
from pathlib import Path

import cv2
import numpy as np
import six
import sys

from datasets.dataset import BBoxDataset
import utils.misc


class Got10kDataset(BBoxDataset):
    # Maps "cover" label from GOT to minimum in that cover bucket.
    COVER_TO_VIS = np.array([0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0 - 1e-5, 1.0])

    def __init__(
            self,
            dataset_location="../got10k",
            S=32, fullseq=False, chunk=None,
            crop_size=(384,512), 
            use_augs=False,
            is_training=True,
            strides=[1,2],
            zooms=[1,2],
    ):

        print("Loading GOT-10k dataset.")
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training,
        )

        clip_step = S//2
        if not is_training:
            clip_step = S

        self.root = Path(dataset_location)
        self.got = _GOT10k(
            dataset_location,
            subset="train" if is_training else "val",
            return_meta=True,
            check_integrity=False,
        )

        print(f"Found {len(self.got.seq_dirs)} videos in {self.root}")

        # Get a list of all possible sequences of length S
        # List of (sequence_index, start, stride) tuples.
        self.all_info = []

        # num_skipped = 0
        for s_i, seq_dir in enumerate(self.got.seq_dirs):
            # Shape (S_local, 4)
            img_files, anno, meta = self.got[s_i]
            absence, cover = meta["absence"].astype(int), meta["cover"].astype(int)
            S_local = anno.shape[0]
            # print('S_local', S_local)

            if S_local < S: continue

            for stride in strides:
                clip_length = self.S * stride
                last_start = S_local - clip_length + 1

                for clip_start in range(0, last_start, clip_step*stride):
                    # vis0 = self.COVER_TO_VIS[cover[clip_start]]
                    # vis1 = self.COVER_TO_VIS[cover[clip_start+1]]
                    # if absence[clip_start] == 0 and absence[clip_start+1] == 0 and vis0 >= min_vis_start and vis1 >= min_vis_start:
                    for zoom in zooms:
                        self.all_info.append([s_i, clip_start, stride, zoom])
                    # else:
                    #     num_skipped += 1
            sys.stdout.write('.')
            sys.stdout.flush()

        print(f"Found {len(self.all_info)} clips from {len(self.got.seq_dirs)} videos.")
        # print(f"Skipped {num_skipped} clips due to visibility constraints")

        if chunk is not None:
            assert(len(self.all_info) > 100)
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.all_info = chunkify(self.all_info,100)[chunk]
            print('filtered to %d' % len(self.all_info))
        

    def getitem_helper(self, index):
        seq_index, clip_start, stride, zoom = self.all_info[index]
        img_files, anno, meta = self.got[seq_index]
        clip_end = clip_start + self.S * stride
        img_paths = img_files[clip_start:clip_end:stride]
        bboxes = anno[clip_start:clip_end:stride]
        cover = meta["cover"][clip_start:clip_end:stride].astype(int)

        rgbs = []

        rgb = cv2.imread(img_paths[0])
        H, W = rgb.shape[:2]
        # print('H, W', H, W)
        sc = 1.0
        if H > 512:
            sc = 512/H
            H_, W_ = int(H*sc), int(W*sc)
        
        for path in img_paths:
            rgb = cv2.imread(path)[..., ::-1].copy()
            if sc < 1.0:
                rgb = cv2.resize(rgb, (W_, H_), interpolation=cv2.INTER_AREA)
            rgbs.append(rgb)

        rgbs = np.stack(rgbs)  # Shape (S, C, H, W)
        bboxes = np.stack(bboxes)  # Shape (S, 4)
        bboxes = bboxes * sc

        visibs = self.COVER_TO_VIS[cover]
        # From xywh to xyxy
        bboxes[..., 2:] += bboxes[..., :2]
        visibs = (visibs > 0.5).astype(np.float32)

        # print('rgbs', rgbs.shape)

        rgbs, bboxes = utils.misc.data_pad_if_necessary_b(rgbs, bboxes, visibs)
        S,H,W,C = rgbs.shape
        if zoom > 1:
            whs = bboxes[:,2:4] - bboxes[:,0:2]
            whs = whs[visibs > 0.5]
            if np.mean(whs[:,0])*zoom >= W and np.mean(whs[:,1])*zoom >= H:
                return None
            bboxes, visibs, rgbs = utils.misc.data_zoom_bbox(zoom, bboxes, visibs, rgbs)
        safe = np.concatenate(([0], visibs[:-2] * visibs[1:-1] * visibs[2:], [0]))
        if np.sum(safe) < 2:
            return None
        
        sample = {
            "rgbs": rgbs,
            "bboxes": bboxes,
            "visibs": visibs,
        }
        return sample

    def __len__(self):
        return len(self.all_info)


class _GOT10k(object):
    r"""`GOT-10K <http://got-10k.aitestunion.com//>`_ Dataset.

    From <https://github.com/got-10k/toolkit/blob/956e7286fdf209cbb125adac9a46376bd8297ffb/got10k/datasets/got10k.py>

    Publication:
        ``GOT-10k: A Large High-Diversity Benchmark for Generic Object
        Tracking in the Wild``, L. Huang, X. Zhao and K. Huang, ArXiv 2018.

    Args:
        root_dir (string): Root directory of dataset where ``train``,
            ``val`` and ``test`` folders exist.
        subset (string, optional): Specify ``train``, ``val`` or ``test``
            subset of GOT-10k.
        return_meta (string, optional): If True, returns ``meta``
            of each sequence in ``__getitem__`` function, otherwise
            only returns ``img_files`` and ``anno``.
        list_file (string, optional): If provided, only read sequences
            specified by the file instead of all sequences in the subset.
    """

    def __init__(
        self,
        root_dir,
        subset="test",
        return_meta=False,
        list_file=None,
        check_integrity=True,
    ):
        super(_GOT10k, self).__init__()
        assert subset in ["train", "val", "test"], "Unknown subset."
        self.root_dir = root_dir
        self.subset = subset
        self.return_meta = False if subset == "test" else return_meta

        if list_file is None:
            list_file = os.path.join(root_dir, subset, "list.txt")
        if check_integrity:
            self._check_integrity(root_dir, subset, list_file)

        with open(list_file, "r") as f:
            self.seq_names = f.read().strip().split("\n")
        self.seq_dirs = [os.path.join(root_dir, subset, s) for s in self.seq_names]
        self.anno_files = [os.path.join(d, "groundtruth.txt") for d in self.seq_dirs]

    def __getitem__(self, index):
        r"""
        Args:
            index (integer or string): Index or name of a sequence.

        Returns:
            tuple: (img_files, anno) if ``return_meta`` is False, otherwise
                (img_files, anno, meta), where ``img_files`` is a list of
                file names, ``anno`` is a N x 4 (rectangles) numpy array, while
                ``meta`` is a dict contains meta information about the sequence.
        """
        if isinstance(index, six.string_types):
            if not index in self.seq_names:
                raise Exception("Sequence {} not found.".format(index))
            index = self.seq_names.index(index)

        img_files = sorted(glob.glob(os.path.join(self.seq_dirs[index], "*.jpg")))
        anno = np.loadtxt(self.anno_files[index], delimiter=",")

        if self.subset == "test" and anno.ndim == 1:
            assert len(anno) == 4
            anno = anno[np.newaxis, :]
        else:
            assert len(img_files) == len(anno)

        if self.return_meta:
            meta = self._fetch_meta(self.seq_dirs[index])
            return img_files, anno, meta
        else:
            return img_files, anno

    def __len__(self):
        return len(self.seq_names)

    def _check_integrity(self, root_dir, subset, list_file=None):
        assert subset in ["train", "val", "test"]
        if list_file is None:
            list_file = os.path.join(root_dir, subset, "list.txt")

        if os.path.isfile(list_file):
            with open(list_file, "r") as f:
                seq_names = f.read().strip().split("\n")

            # check each sequence folder
            for seq_name in seq_names:
                seq_dir = os.path.join(root_dir, subset, seq_name)
                if not os.path.isdir(seq_dir):
                    print("Warning: sequence %s not exists." % seq_name)
        else:
            # dataset not exists
            raise Exception("Dataset not found or corrupted.")

    def _fetch_meta(self, seq_dir):
        # meta information
        meta_file = os.path.join(seq_dir, "meta_info.ini")
        with open(meta_file) as f:
            meta = f.read().strip().split("\n")[1:]
        meta = [line.split(": ") for line in meta]
        meta = {line[0]: line[1] for line in meta}

        # attributes
        attributes = ["cover", "absence", "cut_by_image"]
        for att in attributes:
            meta[att] = np.loadtxt(os.path.join(seq_dir, att + ".label"))

        return meta
