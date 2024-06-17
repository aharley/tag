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
import pycocotools.mask as mask_util
import logging
logger = logging.getLogger(__name__)

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/youtubevos/cocoapi

# Interface for accessing the YouTubeVIS dataset.

# The following API functions are defined:
#  YTVOS       - YTVOS api class that loads YouTubeVIS annotation file and prepare data structures.
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
from pycocotools import mask as maskUtils
import os
from collections import defaultdict
import sys
PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    from urllib import urlretrieve
elif PYTHON_VERSION == 3:
    from urllib.request import urlretrieve


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class LVVIS:
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset,self.anns,self.cats,self.vids = dict(),dict(),dict(),dict()
        self.vidToAnns, self.catToVids = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, vids = {}, {}, {}
        vidToAnns,catToVids = defaultdict(list),defaultdict(list)
        if 'annotations' in self.dataset and self.dataset['annotations'] != None:
            for ann in self.dataset['annotations']:
                vidToAnns[ann['video_id']].append(ann)
                anns[ann['id']] = ann

        if 'videos' in self.dataset:
            for vid in self.dataset['videos']:
                vids[vid['id']] = vid

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and self.dataset['annotations'] != None and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToVids[ann['category_id']].append(ann['video_id'])

        print('index created!')

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
        for key, value in self.dataset['info'].items():
            print('{}: {}'.format(key, value))

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
            anns = self.dataset['annotations']
        else:
            if not len(vidIds) == 0:
                lists = [self.vidToAnns[vidId] for vidId in vidIds if vidId in self.vidToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['avg_area'] > areaRng[0] and ann['avg_area'] < areaRng[1]]
        if not iscrowd == None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
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
            cats = self.dataset['categories']
        else:
            cats = self.dataset['categories']
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name']          in catNms]
            cats = cats if len(supNms) == 0 else [cat for cat in cats if cat['supercategory'] in supNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id']            in catIds]
        ids = [cat['id'] for cat in cats]
        return ids

    def getVidIds(self, vidIds=[], catIds=[]):
        '''
        Get vid ids that satisfy given filter conditions.
        :param vidIds (int array) : get vids for given ids
        :param catIds (int array) : get vids with all given cats
        :return: ids (int array)  : integer array of vid ids
        '''
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
        res = LVVIS()
        res.dataset['videos'] = [img for img in self.dataset['videos']]

        print('Loading and preparing results...')
        tic = time.time()
        if type(resFile) == str:
            anns = json.load(open(resFile))
        elif type(resFile) == np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = resFile
        assert type(anns) == list, 'results in not an array of objects'
        annsVidIds = [ann['video_id'] for ann in anns]
        assert set(annsVidIds) == (set(annsVidIds) & set(self.getVidIds())), \
               'Results do not correspond to current coco set'
        if 'segmentations' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                ann['areas'] = []
                if not 'bboxes' in ann:
                    ann['bboxes'] = []
                for seg in ann['segmentations']:
                    # now only support compressed RLE format as segmentation results
                    if seg:
                        ann['areas'].append(maskUtils.area(seg))
                        if len(ann['bboxes']) < len(ann['areas']):
                            ann['bboxes'].append(maskUtils.toBbox(seg))
                    else:
                        ann['areas'].append(None)
                        if len(ann['bboxes']) < len(ann['areas']):
                            ann['bboxes'].append(None)
                ann['id'] = id+1
                l = [a for a in ann['areas'] if a]
                if len(l)==0:
                  ann['avg_area'] = 0
                else:
                  ann['avg_area'] = np.array(l).mean() 
                ann['iscrowd'] = 0
        print('DONE (t={:0.2f}s)'.format(time.time()- tic))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res

    def annToRLE(self, ann, frameId):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        t = self.vids[ann['video_id']]
        h, w = t['height'], t['width']
        segm = ann['segmentations'][frameId]
        if type(segm) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, h, w)
            rle = maskUtils.merge(rles)
        elif type(segm['counts']) == list:
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
    
    
def load_lvvis_json(json_file, image_root, extra_annotation_keys=None):
    ytvis_api = LVVIS(json_file)
    # sort indices for reproducible results
    vid_ids = sorted(ytvis_api.vids.keys())
    # vids is a list of dicts, each looks something like:
    # {'license': 1,
    #  'flickr_url': ' ',
    #  'file_names': ['ff25f55852/00000.jpg', 'ff25f55852/00005.jpg', ..., 'ff25f55852/00175.jpg'],
    #  'height': 720,
    #  'width': 1280,
    #  'length': 36,
    #  'date_captured': '2019-04-11 00:55:41.903902',
    #  'id': 2232}
    vids = ytvis_api.loadVids(vid_ids)

    anns = [ytvis_api.vidToAnns[vid_id] for vid_id in vid_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(ytvis_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    vids_anns = list(zip(vids, anns))
    logger.info("Loaded {} videos in YTVIS format from {}".format(len(vids_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "category_id", "id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for (vid_dict, anno_dict_list) in vids_anns:
        record = {}
        #record["file_names"] = [os.path.join(image_root, vid_dict["file_names"][i]) for i in range(vid_dict["length"])]
        record["file_names"] = [os.path.join(image_root, '/'.join(vid_dict["file_names"][i].split('\\')[-2:])) for i in range(vid_dict["length"])]
        record["height"] = vid_dict["height"]
        record["width"] = vid_dict["width"]
        record["length"] = vid_dict["length"]
        video_id = record["video_id"] = vid_dict["id"]

        video_objs = []
        for frame_idx in range(record["length"]):
            frame_objs = []
            for anno in anno_dict_list:
                assert anno["video_id"] == video_id

                obj = {key: anno[key] for key in ann_keys if key in anno}

                _segm = anno.get("segmentations", None)

                if not ( _segm and _segm[frame_idx]):
                    continue

                segm = _segm[frame_idx]


                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                elif segm:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm
                frame_objs.append(obj)
            video_objs.append(frame_objs)
        record["annotations"] = video_objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process. "
            "A valid polygon should be a list[float] with even length >= 6."
        )
    return dataset_dicts


class LVVISDataset(MaskDataset):
    def __init__(self,
                 dataset_location='../LVVIS',
                 S=32, fullseq=False, chunk=None,
                 crop_size=(384,512), 
                 strides=[1,2,3],
                 zooms=[1,1.25,1.5,2], # lots of zooms, to ensure we get >100 clips
                 use_augs=False,
                 is_training=True,
    ):

        print('loading LV-VIS dataset...')
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size, 
            use_augs=use_augs,
            is_training=is_training
        )

        clip_step = S//2 if is_training else S
        
        self.dataset_location = os.path.join(dataset_location, 'train' if is_training else 'val')
        
        self.data = load_lvvis_json(os.path.join(self.dataset_location, '../{}_instances.json'.format('train' if is_training else 'val')), os.path.join(self.dataset_location, 'JPEGImages'))
        print('found {:d} videos in {}'.format(len(self.data), self.dataset_location))

        self.all_info = []
        for vid, data in enumerate(self.data):
            # obj_ids = [anno['id'] for anno in data['annotations'][0]]
            # print('obj_ids', obj_ids)
            # import ipdb; ipdb.set_trace()
            all_frame_obj_ids = [[a['id'] for a in anno] for anno in data['annotations']]
            obj_ids = np.unique(sum(all_frame_obj_ids,[]))
            for obj_id in obj_ids:
                frame_ids = [i for i, frame_obj_ids in enumerate(all_frame_obj_ids) if obj_id in frame_obj_ids]
                S_local = max(frame_ids) + 1
                for stride in strides:
                    for ii in range(0, S_local, clip_step*stride):
                        full_idx = ii + np.arange(self.S)*stride
                        full_idx = [ij for ij in full_idx if ij < S_local]
                        if len(full_idx) < (((S_local - 1) // stride + 1 if self.S == -1 else self.S) if fullseq else 8):
                            continue
                        for zoom in zooms:
                            self.all_info.append([vid, obj_id, full_idx, zoom])
                        sys.stdout.write('.')
                        sys.stdout.flush()
        print('\nloaded {} clips'.format(len(self.all_info)))

        if chunk is not None:
            assert(len(self.all_info) > 100)
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.all_info = chunkify(self.all_info,100)[chunk]
            print('filtered to %d clips' % len(self.all_info))
            # print('self.all_info', self.all_info)
        
        
    def __len__(self):
        return len(self.all_info)
        
    def getitem_helper(self, index):
        vid, tid, full_idx, zoom = self.all_info[index]
        video = self.data[vid]
        
        print('vid', vid, 'tid', tid, 'full_idx', full_idx, 'zoom', zoom)

        frames = video['file_names']
        frames = [frames[idx] for idx in full_idx]

        rgb = cv2.imread(frames[0])[..., ::-1].copy()
        H, W = rgb.shape[:2]
        
        rgbs = []
        segs = []
        for fn in frames:
            rgb = cv2.imread(fn)[..., ::-1].copy()
            H_, W_ = rgb.shape[:2]
            if not (H_==H and W_==W):
                print('one image is a weird size')
                return None
            rgbs.append(rgb)
        
        masks = []
        visibs = []
        for idx in full_idx:
            found = False
            for anno in video['annotations'][idx]:
                if anno['id'] == tid:
                    mask = mask_util.decode(anno['segmentation'])
                    masks.append(mask)
                    found = True
                    break
            if not found:
                if idx==full_idx[0]:
                    print('first mask missing')
                    return None
                masks.append(np.zeros_like(masks[-1]))
            visibs.append(found)

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
    data = load_lvvis_json('/orion/group/LVVIS/train_instances.json', '/orion/group/LVVIS/train')
    import pdb; pdb.set_trace()
