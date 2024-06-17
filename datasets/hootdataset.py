import torch
import multiprocessing
from typing import Optional
from pathlib import Path
import json
import pickle
import os
# from hoot.anno import load_video_from_file, Video
import cv2
import numpy as np
import sys
from datasets.dataset import BBoxDataset
import utils.misc

COLORS = {"tgt_poly": (0, 255, 0), 
          "tgt_aa_bb": (0, 153, 0),
          "tgt_rot_bb": (102, 255, 178),
          "occ" : (192, 192, 192),
          "s": (0, 0, 204), 
          "sp": (0, 128, 255),
          "st": (255 ,102, 178),
          "t": (255 ,178, 102)
         }


import json
from dataclasses import dataclass, field
from dacite import from_dict
from typing import List, Union
from types import SimpleNamespace
from typing import List, Union, Optional, Tuple
from pycocotools import mask
from pathlib import Path

## Occlusion tags class, provides a mapping from attr to str
class OcclusionTags(SimpleNamespace):
    solid = "solid"
    sparse = "sparse"
    semi_transparent = "semi_transparent"
    transparent = "transparent"
    absent = "absent"
    full_occlusion = "full_occlusion"
    similar_occluder = "similar_occluder"
    cut_by_frame = "cut_by_frame"
    partial_obj_occlusion = "partial_obj_occlusion"

## Motion tags class, provides a mapping from attr to str
class MotionTags(SimpleNamespace):
    blur = "blur"
    moving_occluder = "moving_occluder"
    parallax = "parallax"
    dynamic = "dynamic"
    camera_motion = "camera_motion"

## Target tags class, provides a mapping from attr to str
class TargetTags(SimpleNamespace):
    deformable = "deformable"
    self_propelled = "self_propelled"
    animate = "animate"

## Mask class that holds a COCO RLE encoded mask
## Provides a property to return the decoded binary mask
@dataclass
class Mask:
    size: List[int]
    counts: str

    ## Function that converts mask counts to bytes and decodes it
    ## Returns a binary 2D array
    @property
    def mask(self) -> np.ndarray:
        counts_bytes = bytes.fromhex(self.counts)
        mask_mat = mask.decode({"size": self.size, "counts":counts_bytes})
        return mask_mat

## Occlusion Masks class that holds all occlusion masks for a frame
## If a certain type of occluder does not exists for the frame, stores empty list
## Provides get_masks fn to return masks for a given list of occ. types
OptionalMask = Union[List, Mask]
@dataclass
class OcclusionMasks:
    all: OptionalMask=field(default_factory=list)
    s: OptionalMask=field(default_factory=list)
    sp: OptionalMask=field(default_factory=list)
    st: OptionalMask=field(default_factory=list)
    t: OptionalMask=field(default_factory=list)

    ## Function to iterate through occlusion masks for the frame
    ## Iterates a given occlusion types list or all types
    def get_masks(self, occ_types: Optional[List[str]]=None):
        if occ_types is None:
            occ_types = ["all", "s", "sp", "st", "t"]
        for occ_type in occ_types:
            if occ_type == 'all':
                yield (occ_type, self.all)
            elif occ_type == 's':
                yield (occ_type, self.s)
            elif occ_type == 'sp':
                yield (occ_type, self.sp)
            elif occ_type == 'st':
                yield (occ_type, self.st)
            elif occ_type == 't':
                yield (occ_type, self.t)
            else:
                assert False, 'unrecognized occ_type'


## Frame Attributes class which holds frame-level occlusion attributes 
@dataclass
class FrameAttributes:
    absent: bool
    full_occlusion: bool
    similar_occluder: bool
    cut_by_frame: bool
    partial_obj_occlusion: bool

## Definition of rotated and axis-aligned bounding box object types
RotatedBoundingBox = List[List[float]]
AxisAlignedBoundingBox = List[List[float]]
## Frame class that holds frame id, path and other annotations
@dataclass
class Frame:
    frame_id: int
    frame_path: str ## "path/to/hoot/class/video/padded_frame_id.png"
    rot_bb: RotatedBoundingBox
    aa_bb: AxisAlignedBoundingBox
    occ_masks: OcclusionMasks
    attributes: FrameAttributes

    ## Function to compute an x,y,w,h style box from aa_bb (polygon points)
    @property
    def xywh(self) -> List[float]:
        min_x = min([pt[0] for pt in self.aa_bb])
        min_y = min([pt[1] for pt in self.aa_bb])
        max_x = max([pt[0] for pt in self.aa_bb])
        max_y = max([pt[1] for pt in self.aa_bb])
        w = max_x - min_x
        h = max_y - min_y
        return [min_x+w/2.0, min_y+h/2.0, w, h] # adam made the xy part centered
    @property
    def xyxy(self) -> List[float]:
        min_x = min([pt[0] for pt in self.aa_bb])
        min_y = min([pt[1] for pt in self.aa_bb])
        max_x = max([pt[0] for pt in self.aa_bb])
        max_y = max([pt[1] for pt in self.aa_bb])
        return [min_x, min_y, max_x, max_y] # adam made the xy part centered

## Video class that hold video key, path and a list of frame objects, as well as other video data
@dataclass
class Video:
    video_key: str
    video_path: str
    frames: List[Frame]
    frame_occlusion_level: float
    median_target_occlusion_level: float
    mean_target_occlusion_level: float
    #from metadata.info
    height: int
    width: int
    motion_tags: List[str]
    target_tags: List[str]
    
    ## Makes sure frames are sorted by id - in case json read/write messed it up
    def __post_init__(self):
        self.frames.sort(key=lambda f: f.frame_id)

    ## Computes video-level occlusion tags from frame tags
    ## e.g. if any frame is video has solid occluder, it gets added to video tags
    @property
    def occlusion_tags(self) -> List[str]:
        video_tags = set()
        for frame in self.frames:
            if type(frame.occ_masks.s) != list:
                video_tags.add(OcclusionTags.solid)
            if type(frame.occ_masks.sp) != list:
                video_tags.add(OcclusionTags.sparse)
            if type(frame.occ_masks.st) != list:
                video_tags.add(OcclusionTags.semi_transparent)
            if type(frame.occ_masks.t) != list:
                video_tags.add(OcclusionTags.transparent)

            if frame.attributes.absent:
                video_tags.add(OcclusionTags.absent)
            if frame.attributes.full_occlusion:
                video_tags.add(OcclusionTags.full_occlusion)
            if frame.attributes.similar_occluder:
                video_tags.add(OcclusionTags.similar_occluder)
            if frame.attributes.cut_by_frame:
                video_tags.add(OcclusionTags.cut_by_frame)
            if frame.attributes.partial_obj_occlusion:
                video_tags.add(OcclusionTags.partial_obj_occlusion)
        return list(video_tags)

## Loads video annotations for HOOT
def load_video_from_file(videopath: Path, in_test=None, annopath=None, metapath=None) -> Video:
    ## If not given specifically, load anno.json/meta.info from default path
    if annopath is None:
        annopath = videopath.joinpath('anno.json')
        assert annopath.exists()
    if metapath is None:
        metapath = videopath.joinpath('meta.info')
        assert metapath.exists()

    # Edit annotation dict to add path and test video info
    with open(annopath, 'r') as f:
        anno_data = json.load(f)
    anno_data['video_path'] = str(videopath)
    anno_data['in_test'] = in_test
    for f in anno_data['frames']:
        frame_id = int(f['frame_id'])
        f['frame_path'] = str(videopath.joinpath(f'{frame_id:06}.png'))

    # Load metadata.info json
    with open(metapath, 'r') as f:
        meta_data = json.load(f)
    motion_tags, target_tags = load_tags_from_metadata(meta_data)
    anno_data['height'] = int(meta_data['height'])
    anno_data['width'] = int(meta_data['width'])
    anno_data['motion_tags'] = motion_tags
    anno_data['target_tags'] = target_tags

    # Load rest o the annotations from the anno.json file
    video = from_dict(data_class=Video, data=anno_data)
    return video

## Loads video-level tags like motion and target tags from the meta.info
def load_tags_from_metadata(meta_data) -> Tuple[List[str], List[str]]:
    motion_tags = []
    target_tags = []

    if meta_data['video_tags'][MotionTags.blur]:
        motion_tags.append(MotionTags.blur)
    if meta_data['video_tags'][MotionTags.moving_occluder]:
        motion_tags.append(MotionTags.moving_occluder)
    if meta_data['video_tags'][MotionTags.parallax]:
        motion_tags.append(MotionTags.parallax)
    if meta_data['video_tags'][MotionTags.dynamic]:
        motion_tags.append(MotionTags.dynamic)
    if meta_data['video_tags'][MotionTags.camera_motion]:
        motion_tags.append(MotionTags.camera_motion)
    
    if meta_data['video_tags'][TargetTags.animate]:
        target_tags.append(TargetTags.animate)
    if meta_data['video_tags'][TargetTags.deformable]:
        target_tags.append(TargetTags.deformable)
    if meta_data['video_tags'][TargetTags.self_propelled]:
        target_tags.append(TargetTags.self_propelled)

    return (motion_tags, target_tags)


import matplotlib.pyplot as plt
class HootDataset(BBoxDataset):
    def __init__(self, dataset_location='../Hoot',
                 S=32, fullseq=False, chunk=None,
                 strides=[1,2,3,4,6,8], # very hq dataset and high fps
                 zooms=[1,1.25,1.5,2],
                 crop_size=(384,512),
                 use_augs=False,
                 is_training=True):
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training)
        self.dset = 'train' if is_training else 'test'

        clip_step = S//2
        if not is_training:
            clip_step = S
            strides = [2]
        
        dataset_path = '%s/%s.txt' % (dataset_location, self.dset)
        with open(dataset_path) as f:
            content = f.readlines()
        content = sorted(content)
        video_list = [line.strip() for line in content]
        print('found %d videos in %s (dset=%s)' % (len(video_list), dataset_location, self.dset))

        if chunk is not None:
            assert(len(video_list) > 100)
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            video_list = chunkify(video_list,100)[chunk]
            print('filtered to %d' % len(video_list))

        self.vis_thr = 0.9
        self.all_info = []
        
        for vid in video_list:
            obj_name, vid_num = vid.split('-')
            video_path = '%s/%s/%s' % (self.dataset_location, obj_name, vid_num)
            video_path = Path(video_path)
            video_anno_json = '%s/%s/%s/anno.json' % (self.dataset_location, obj_name, vid_num)
            video_meta_json = '%s/%s/%s/meta.info' % (self.dataset_location, obj_name, vid_num)
            video_data = load_video_from_file(video_path, None, video_anno_json, video_meta_json)
            all_frames = video_data.frames
            S_local = len(all_frames)

            vis_g = np.ones((S_local,), dtype=np.float32)
            for si, frame in enumerate(all_frames):
                if frame.attributes.absent:
                    vis_g[si] = 0
                else:
                    x0, y0, x1, y1 = np.maximum(np.array(frame.xyxy).astype(int), 0)
                    if frame.occ_masks.all:       
                        occ_mask_mat = frame.occ_masks.all.mask
                        local_occ = occ_mask_mat[y0:y1, x0:x1]
                        occ_frac = np.mean(local_occ)
                        vis_g[si] = 1. - occ_frac
            
            for stride in strides:
                for ii in range(0, max(S_local-self.S*stride,1), clip_step*stride):
                    full_idx = ii + np.arange(self.S)*stride
                    full_idx = [ij for ij in full_idx if ij < S_local]
                    if len(full_idx) == self.S: # fullseq
                        vis_local = vis_g[full_idx]
                        if np.sum(vis_local > self.vis_thr) > 3:
                            for zoom in zooms:
                                self.all_info.append([vid, full_idx, zoom])
            sys.stdout.write('.')
            sys.stdout.flush()

        print('found %d samples in %s (dset=%s)' % (len(self.all_info), dataset_location, self.dset))
        
    def getitem_helper(self, index):
        video_path, full_idx, zoom = self.all_info[index]

        obj_name, vid_num = video_path.split('-')

        video_path = '%s/%s/%s' % (self.dataset_location, obj_name, vid_num)
        video_path = Path(video_path)
        video_anno_json = '%s/%s/%s/anno.json' % (self.dataset_location, obj_name, vid_num)
        video_meta_json = '%s/%s/%s/meta.info' % (self.dataset_location, obj_name, vid_num)
        video_data = load_video_from_file(video_path, None, video_anno_json, video_meta_json)
        all_frames = video_data.frames
        all_frames = [all_frames[idx] for idx in full_idx]
        S = len(all_frames)

        rgbs = []
        masks = []
        visibs = []
        bboxes = []
        for si, frame in enumerate(all_frames):
            rgb = cv2.imread(str(frame.frame_path))[..., ::-1]
            mask = np.zeros((*rgb.shape[:-1],), dtype=np.float32)
            xyxy = np.zeros((4,), dtype=int)
            rgbs.append(rgb)
            visib = 0.

            if not frame.attributes.absent:
                visib = 1.
                x0, y0, x1, y1 = xyxy = np.maximum(np.array(frame.xyxy).astype(int), 0)
                mask[y0:y1, x0:x1] = 0.5
                if frame.occ_masks.all:
                    occ_mask_mat = frame.occ_masks.all.mask
                    local_occ = occ_mask_mat[y0:y1, x0:x1]
                    occ_frac = np.mean(local_occ)
                    visib = 1. - occ_frac
                    mask[y0:y1, x0:x1][local_occ > 0] = 0
                    mask = np.maximum(mask, 0.)
            masks.append(mask)
            visibs.append(visib)
            bboxes.append(xyxy)
        
        rgbs = np.stack(rgbs)
        masks = np.stack(masks)
        visibs = np.stack(visibs)
        bboxes = np.stack(bboxes)

        # hoot is special for providing valid boxes even during occlusions
        valids = (visibs > 0.01).astype(np.float32)
        visibs = (visibs > self.vis_thr).astype(np.float32)

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
            'rgbs': rgbs,
            'visibs': visibs,
            'valids': valids,
            'bboxes': bboxes,
        }
        return sample
    
    def __len__(self):
        return len(self.all_info)
