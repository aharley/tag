import time
from numpy import random
from numpy.core.numeric import full
import torch
import numpy as np
import os
import scipy.ndimage
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import random
from torch._C import dtype, set_flush_denormal
import utils.basic
import utils.improc
import glob
import json
import imageio
import cv2
import re
import sys
from torchvision.transforms import ColorJitter, GaussianBlur
from datasets.dataset import MaskDataset, mask2bbox
from icecream import ic

from datasets.dataset_utils import make_split


__author__ = "jyq"
# Interface for accessing the OVIS dataset.

# The following API functions are defined:
#  OVIS       - OVIS api class that loads OVIS annotation file and prepare data structures.
#  decodeMask - Decode binary mask M encoded via run-length encoding.
#  encodeMask - Encode binary mask M using run-length encoding.
#  getAnnIds  - Get ann ids that satisfy given filter conditions.
#  getCatIds  - Get cat ids that satisfy given filter conditions.
#  getImgIds  - Get img ids that satisfy given filter conditions.
#  loadAnns   - Load anns with the specified ids.
#  loadCats   - Load cats with the specified ids.
#  loadImgs   - Load imgs with the specified ids.
#  annToMask  - Convert segmentation in an annotation to binary mask.
#  loadRes    - Load algorithm results and create API for accessing them.

# Microsoft COCO Toolbox.      version 2.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
# Licensed under the Simplified BSD License [see bsd.txt]


import json
import time
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
import copy
import itertools
import pycocotools.mask as maskUtils
import os
from collections import defaultdict
import sys

PYTHON_VERSION = sys.version_info[0]


def _isArrayLike(obj):
    return hasattr(obj, "__iter__") and hasattr(obj, "__len__")


class OVIS:
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset, self.anns, self.cats, self.vids = dict(), dict(), dict(), dict()
        self.vidToAnns, self.catToVids = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print("loading annotations into memory...")
            tic = time.time()
            dataset = json.load(open(annotation_file, "r"))
            assert (
                type(dataset) == dict
            ), "annotation file format {} not supported".format(type(dataset))
            print("Done (t={:0.2f}s)".format(time.time() - tic))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        print("creating index...")
        anns, cats, vids = {}, {}, {}
        vidToAnns, catToVids = defaultdict(list), defaultdict(list)
        if "annotations" in self.dataset:
            if self.dataset["annotations"] is None:
                del self.dataset["annotations"]
            else:
                for ann in self.dataset["annotations"]:
                    vidToAnns[ann["video_id"]].append(ann)
                    anns[ann["id"]] = ann

        if "videos" in self.dataset:
            for vid in self.dataset["videos"]:
                vids[vid["id"]] = vid

        if "categories" in self.dataset:
            for cat in self.dataset["categories"]:
                cats[cat["id"]] = cat

        if "annotations" in self.dataset and "categories" in self.dataset:
            for ann in self.dataset["annotations"]:
                catToVids[ann["category_id"]].append(ann["video_id"])

        print("index created!")

        # create class members
        self.anns = anns
        self.vidToAnns = vidToAnns
        self.catToVids = catToVids
        self.vids = vids
        self.cats = cats

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.dataset["info"].items():
            print("{}: {}".format(key, value))

    def getAnnIds(self, vidIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param vidIds  (int array)     : get anns for given vids
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        vidIds = vidIds if _isArrayLike(vidIds) else [vidIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(vidIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset["annotations"]
        else:
            if not len(vidIds) == 0:
                lists = [
                    self.vidToAnns[vidId] for vidId in vidIds if vidId in self.vidToAnns
                ]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset["annotations"]
            anns = (
                anns
                if len(catIds) == 0
                else [ann for ann in anns if ann["category_id"] in catIds]
            )
            anns = (
                anns
                if len(areaRng) == 0
                else [
                    ann
                    for ann in anns
                    if ann["avg_area"] > areaRng[0] and ann["avg_area"] < areaRng[1]
                ]
            )
        if not iscrowd == None:
            ids = [ann["id"] for ann in anns if ann["iscrowd"] == iscrowd]
        else:
            ids = [ann["id"] for ann in anns]
        return ids

    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        supNms = supNms if _isArrayLike(supNms) else [supNms]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset["categories"]
        else:
            cats = self.dataset["categories"]
            cats = (
                cats
                if len(catNms) == 0
                else [cat for cat in cats if cat["name"] in catNms]
            )
            cats = (
                cats
                if len(supNms) == 0
                else [cat for cat in cats if cat["supercategory"] in supNms]
            )
            cats = (
                cats
                if len(catIds) == 0
                else [cat for cat in cats if cat["id"] in catIds]
            )
        ids = [cat["id"] for cat in cats]
        return ids

    def getVidIds(self, vidIds=[], catIds=[]):
        """
        Get vid ids that satisfy given filter conditions.
        :param vidIds (int array) : get vids for given ids
        :param catIds (int array) : get vids with all given cats
        :return: ids (int array)  : integer array of vid ids
        """
        vidIds = vidIds if _isArrayLike(vidIds) else [vidIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(vidIds) == len(catIds) == 0:
            ids = self.vids.keys()
        else:
            ids = set(vidIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToVids[catId])
                else:
                    ids &= set(self.catToVids[catId])
        return list(ids)

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if _isArrayLike(ids):
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def loadVids(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying vid
        :return: vids (object array) : loaded vid objects
        """
        if _isArrayLike(ids):
            return [self.vids[id] for id in ids]
        elif type(ids) == int:
            return [self.vids[ids]]

    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = OVIS()
        res.dataset["videos"] = [img for img in self.dataset["videos"]]

        print("Loading and preparing results...")
        tic = time.time()
        if type(resFile) == str:
            anns = json.load(open(resFile))
        elif type(resFile) == np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = resFile
        assert type(anns) == list, "results in not an array of objects"
        annsVidIds = [ann["video_id"] for ann in anns]
        assert set(annsVidIds) == (
            set(annsVidIds) & set(self.getVidIds())
        ), "Results do not correspond to current coco set"
        if "segmentations" in anns[0]:
            res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
            for id, ann in enumerate(anns):
                ann["areas"] = []
                if not "bboxes" in ann:
                    ann["bboxes"] = []
                for seg in ann["segmentations"]:
                    # now only support compressed RLE format as segmentation results
                    if seg:
                        ann["areas"].append(maskUtils.area(seg))
                        if len(ann["bboxes"]) < len(ann["areas"]):
                            ann["bboxes"].append(maskUtils.toBbox(seg))
                    else:
                        ann["areas"].append(None)
                        if len(ann["bboxes"]) < len(ann["areas"]):
                            ann["bboxes"].append(None)
                ann["id"] = id + 1
                l = [a for a in ann["areas"] if a]
                if len(l) == 0:
                    ann["avg_area"] = 0
                else:
                    ann["avg_area"] = np.array(l).mean()
                ann["iscrowd"] = 0
        print("DONE (t={:0.2f}s)".format(time.time() - tic))

        res.dataset["annotations"] = anns
        res.createIndex()
        return res

    def annToRLE(self, ann, frameId):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        t = self.vids[ann["video_id"]]
        h, w = t["height"], t["width"]
        segm = ann["segmentations"][frameId]
        if type(segm) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, h, w)
            rle = maskUtils.merge(rles)
        elif type(segm["counts"]) == list:
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = segm
        return rle

    def annToMask(self, ann, frameId):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, frameId)
        m = maskUtils.decode(rle)
        return m


# REF: https://github.com/qjy981010/CMaskTrack-RCNN/blob/main/mmdet/datasets/ovis.py
class OVISDataset(MaskDataset):
    def __init__(
            self,
            dataset_location="../OVIS",
            S=32, fullseq=False, chunk=None,
            rand_frames=False,
            crop_size=(384, 512),
            use_augs=False,
            strides=[1,2,3],
            zooms=[1,1.5,2],
            is_training=True,
    ):
        print("loading OVIS dataset...")
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            rand_frames=rand_frames,
            crop_size=crop_size, 
            is_training=is_training,
        )

        clip_step = S//2

        if not is_training:
            strides = [1]
            zooms = [1]
            clip_step = S
        
        # Validation set has no annotations
        self.dataset_location = os.path.join(dataset_location, "train")
        self.ovis = OVIS(
            os.path.join(dataset_location, "annotations_{}.json".format("train"))
        )

        self.cat_ids = self.ovis.getCatIds()
        self.cat2label = {cat_id: i + 1 for i, cat_id in enumerate(self.cat_ids)}

        self.vid_ids = self.ovis.getVidIds()

        assert(is_training)
        # self.vid_ids = make_split(self.vid_ids, is_training, shuffle=True)

        print('found %d unique videos in %s' % (len(self.vid_ids), dataset_location))

        if chunk is not None:
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.vid_ids = chunkify(self.vid_ids,100)[chunk]
            print('filtered to %d video_names' % len(self.vid_ids))
            print('self.vid_ids', self.vid_ids)

        
        self.all_infos = []
        self.all_full_idxs = []
        self.all_obj_ids = []
        self.all_anno_infos = []
        self.all_zooms = []

        for vi in self.vid_ids:
            info = self.ovis.loadVids([vi])[0]
            info["filenames"] = info["file_names"]
            frames = info["filenames"]
            vid_id = info["id"]
            ann_ids = self.ovis.getAnnIds(vidIds=[vid_id])
            ann_info = self.ovis.loadAnns(ann_ids)

            S_local = len(frames)
            print('S_local', S_local)

            for stride in strides:
                for ii in range(0, S_local, clip_step*stride):
                    full_idx = ii + np.arange(self.S) * stride
                    full_idx = [ij for ij in full_idx if ij < S_local]
                    if len(full_idx) < (((S_local - 1) // stride + 1 if self.S == -1 else self.S) if fullseq else 8):
                        continue
                    
                    obj_ids = self._parse_ann_info(ann_info, ii)["obj_ids"]
                    
                    for obj_id in obj_ids:
                        # don't load mask here, just check if it exists
                        
                        skip = False
                        for frame_id in full_idx:
                            ok = False
                            for _, ann in enumerate(ann_info):
                                bbox = ann["bboxes"][frame_id]
                                area = ann["areas"][frame_id]
                                if bbox is None:
                                    continue
                                x1, y1, w, h = bbox
                                if area <= 0 or w < 1 or h < 1:
                                    continue
                                bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
                                if not ann["iscrowd"]:
                                    if ann['id'] == obj_id:
                                        ok = True
                                        break
                            if not ok:
                                skip = True
                                break
                        if skip:
                            continue

                        for zoom in zooms:
                            self.all_infos.append(info)
                            self.all_full_idxs.append(full_idx)
                            self.all_obj_ids.append(obj_id)
                            self.all_anno_infos.append(ann_info)
                            self.all_zooms.append(zoom)

                sys.stdout.write(".")
                sys.stdout.flush()

        print("found %d samples in %s" % (len(self.all_infos), self.dataset_location))

    def get_ann_info(self, vid_id, frame_id):
        ann_ids = self.ovis.getAnnIds(vidIds=[vid_id])
        ann_info = self.ovis.loadAnns(ann_ids)
        return self._parse_ann_info(ann_info, frame_id)

    def _parse_ann_info(self, ann_info, frame_id, with_mask=True):
        gt_bboxes = []
        gt_labels = []
        gt_ids = []
        gt_bboxes_ignore = []
        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.
        if with_mask:
            gt_masks = []
            gt_mask_polys = []
            gt_poly_lens = []
        for i, ann in enumerate(ann_info):
            # each ann is a list of masks
            # ann:
            # bbox: list of bboxes
            # segmentation: list of segmentation
            # category_id
            # area: list of area
            bbox = ann["bboxes"][frame_id]
            area = ann["areas"][frame_id]
            segm = ann["segmentations"][frame_id]
            if bbox is None:
                continue
            x1, y1, w, h = bbox
            if area <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann["iscrowd"]:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_ids.append(ann["id"])
                gt_labels.append(self.cat2label[ann["category_id"]])
            if with_mask:
                gt_masks.append(self.ovis.annToMask(ann, frame_id))
                mask_polys = [
                    p for p in segm if len(p) >= 6
                ]  # valid polygons have >= 3 points (6 coordinates)
                poly_lens = [len(p) for p in mask_polys]
                gt_mask_polys.append(mask_polys)
                gt_poly_lens.extend(poly_lens)
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            obj_ids=gt_ids,
            bboxes_ignore=gt_bboxes_ignore,
        )

        if with_mask:
            ann["masks"] = gt_masks
            # poly format is not used in the current implementation
            ann["mask_polys"] = gt_mask_polys
            ann["poly_lens"] = gt_poly_lens
        return ann

    def get_masks_given_id(self, vid_id, obj_id, full_idx):
        segs = []
        for i in full_idx:
            ann = self.get_ann_info(vid_id, i)
            if obj_id in ann["obj_ids"]:
                mask = ann["masks"][ann["obj_ids"].index(obj_id)]
                segs.append(mask * 255)
            else:
                return None

        valid_segs = np.array([v for v in np.unique(segs.reshape(-1)) if v > 0])
        for label in valid_segs:
            masks = [(seg == label).astype(np.float32) for seg in segs]
            if np.sum(masks) > len(masks)*4:
                all_masks.append(masks)

        return all_masks

    def getitem_helper(self, index):
        vid_info = self.all_infos[index]
        full_idx = self.all_full_idxs[index]
        anno_info = self.all_anno_infos[index]
        zoom = self.all_zooms[index]

        obj_id = self.all_obj_ids[index]
        frames = vid_info["filenames"]

        frames = [frames[idx] for idx in full_idx]

        image_paths = [os.path.join(self.dataset_location, fn) for fn in frames]
        rgb = cv2.imread(str(image_paths[0]))
        H, W = rgb.shape[:2]
        # print("H, W", H, W)
        rgbs = []
        for path in image_paths:
            rgb = cv2.imread(str(path))[..., ::-1].copy()
            rgbs.append(rgb)
        
        masks = []
        for i in full_idx:
            ann = self._parse_ann_info(anno_info, i)
            if obj_id in ann["obj_ids"]:
                mask = ann["masks"][ann["obj_ids"].index(obj_id)]
                masks.append(mask)
            else:
                return None
            
        rgbs = np.stack(rgbs)
        masks = np.stack(masks)
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

    def __len__(self):
        return len(self.all_infos)
