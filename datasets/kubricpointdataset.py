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
from datasets.dataset import PointDataset
from icecream import ic
import matplotlib
from pathlib import Path
import matplotlib.pyplot as plt

from datasets.dataset_utils import make_split


import functools
import itertools
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
# from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from matplotlib.animation import FuncAnimation
import matplotlib
from scipy import stats


TOTAL_TRACKS = 200
MAX_SAMPLED_FRAC = .1
MAX_SEG_ID = 25
INPUT_SIZE = (None, 256, 256)
STRIDE = 4  # Make sure this divides all axes of INPUT_SIZE

def from_quaternion(quaternion):
    """Convert a quaternion to a rotation matrix.
    Note:
    In the following, A1 to An are optional batch dimensions.
    Args:
    quaternion: A tensor of shape `[A1, ..., An, 4]`, where the last dimension
        represents a normalized quaternion.
    name: A name for this op that defaults to
        "rotation_matrix_3d_from_quaternion".
    Returns:
    A tensor of shape `[A1, ..., An, 3, 3]`, where the last two dimensions
    represent a 3d rotation matrix.
    Raises:
    ValueError: If the shape of `quaternion` is not supported.
    """

    x, y, z, w = tf.unstack(quaternion, axis=-1)
    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z
    matrix = tf.stack((1.0 - (tyy + tzz), txy - twz, txz + twy,
                        txy + twz, 1.0 - (txx + tzz), tyz - twx,
                        txz - twy, tyz + twx, 1.0 - (txx + tyy)),
                        axis=-1)  # pyformat: disable
    output_shape = tf.concat((tf.shape(input=quaternion)[:-1], (3, 3)), axis=-1)
    return tf.reshape(matrix, shape=output_shape)

def project_point(cam, point3d, num_frames):
    """Compute the image space coordinates [0, 1] for a set of points.

    Args:
        cam: The camera parameters, as returned by kubric.  'matrix_world' and
            'intrinsics' have a leading axis num_frames.
        point3d: Points in 3D world coordinates.  it has shape [num_frames,
            num_points, 3].
        num_frames: The number of frames in the video.

    Returns:
        Image coordinates in 2D.  The last coordinate is an indicator of whether
            the point is behind the camera.
    """

    homo_transform = tf.linalg.inv(cam['matrix_world'])
    homo_intrinsics = tf.zeros((num_frames, 3, 1), dtype=tf.float32)
    homo_intrinsics = tf.concat([cam['intrinsics'], homo_intrinsics], axis=2)

    point4d = tf.concat([point3d, tf.ones_like(point3d[:, :, 0:1])], axis=2)
    projected = tf.matmul(point4d, tf.transpose(homo_transform, (0, 2, 1)))
    projected = tf.matmul(projected, tf.transpose(homo_intrinsics, (0, 2, 1)))
    image_coords = projected / projected[:, :, 2:3]
    image_coords = tf.concat(
            [image_coords[:, :, :2],
             tf.sign(projected[:, :, 2:])], axis=2)
    return image_coords


def unproject(coord, cam, depth):
    """Unproject points.

    Args:
        coord: Points in 2D coordinates.  it has shape [num_points, 2].  Coord is in
            integer (y,x) because of the way meshgrid happens.
        cam: The camera parameters, as returned by kubric.  'matrix_world' and
            'intrinsics' have a leading axis num_frames.
        depth: Depth map for the scene.

    Returns:
        Image coordinates in 3D.
    """
    shp = tf.convert_to_tensor(tf.shape(depth))
    idx = coord[:, 0] * shp[1] + coord[:, 1]
    coord = tf.cast(coord[..., ::-1], tf.float32)
    shp = tf.cast(shp[1::-1], tf.float32)[tf.newaxis, ...]
    projected_pt = coord / shp

    projected_pt = tf.concat(
            [
                projected_pt,
                tf.ones_like(projected_pt[:, -1:]),
            ],
            axis=-1,
    )

    camera_plane = projected_pt @ tf.linalg.inv(tf.transpose(cam['intrinsics']))
    camera_ball = camera_plane / tf.sqrt(
            tf.reduce_sum(
                    tf.square(camera_plane),
                    axis=1,
                    keepdims=True,
            ),)
    new_depth_shape = tf.math.reduce_prod(tf.convert_to_tensor(tf.shape(depth)))
    camera_ball *= tf.gather(tf.reshape(depth, [new_depth_shape]), idx)[:, tf.newaxis]

    camera_ball = tf.concat(
            [
                camera_ball,
                tf.ones_like(camera_plane[:, 2:]),
            ],
            axis=1,
    )
    points_3d = camera_ball @ tf.transpose(cam['matrix_world'])
    return points_3d[:, :3] / points_3d[:, 3:]


def reproject(coords, camera, camera_pos, num_frames, bbox=None):
    """Reconstruct points in 3D and reproject them to pixels.

    Args:
        coords: Points in 3D.  It has shape [num_points, 3].  If bbox is specified,
            these are assumed to be in local box coordinates (as specified by kubric),
            and bbox will be used to put them into world coordinates; otherwise they
            are assumed to be in world coordinates.
        camera: the camera intrinsic parameters, as returned by kubric.
            'matrix_world' and 'intrinsics' have a leading axis num_frames.
        camera_pos: the camera positions.  It has shape [num_frames, 3]
        num_frames: the number of frames in the video.
        bbox: The kubric bounding box for the object.  Its first axis is num_frames.

    Returns:
        Image coordinates in 2D and their respective depths.  For the points,
        the last coordinate is an indicator of whether the point is behind the
        camera.  They are of shape [num_points, num_frames, 3] and
        [num_points, num_frames] respectively.
    """
    # First, reconstruct points in the local object coordinate system.
    if bbox is not None:
        coord_box = list(itertools.product([-.5, .5], [-.5, .5], [-.5, .5]))
        coord_box = np.array([np.array(x) for x in coord_box])
        coord_box = np.concatenate(
                [coord_box, np.ones_like(coord_box[:, 0:1])], axis=1)
        coord_box = tf.tile(coord_box[tf.newaxis, ...], [num_frames, 1, 1])
        bbox_homo = tf.concat([bbox, tf.ones_like(bbox[:, :, 0:1])], axis=2)

        local_to_world = tf.linalg.lstsq(tf.cast(coord_box, tf.float32), bbox_homo)
        world_coords = tf.matmul(
                tf.cast(
                        tf.concat([coords, tf.ones_like(coords[:, 0:1])], axis=1),
                        tf.float32)[tf.newaxis, :, :], local_to_world)
        world_coords = world_coords[:, :, 0:3] / world_coords[:, :, 3:]
    else:
        world_coords = tf.tile(coords[tf.newaxis, :, :], [num_frames, 1, 1])

    # Compute depths by taking the distance between the points and the camera
    # center.
    depths = tf.sqrt(
            tf.reduce_sum(
                    tf.square(world_coords - camera_pos[:, np.newaxis, :]),
                    axis=2,
            ),)

    # Project each point back to the image using the camera.
    projections = project_point(camera, world_coords, num_frames)

    return tf.transpose(projections, (1, 0, 2)), tf.transpose(depths)


def estimate_scene_depth_for_point(data, x, y, num_frames):
    """Estimate depth at a (floating point) x,y position.

    We prefer overestimating depth at the point, so we take the max over the 4
    neightoring pixels.

    Args:
        data: depth map. First axis is num_frames.
        x: x coordinate. First axis is num_frames.
        y: y coordinate. First axis is num_frames.
        num_frames: number of frames.

    Returns:
        Depth for each point.
    """
    x0 = tf.cast(tf.floor(x), tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), tf.int32)
    y1 = y0 + 1

    shp = tf.shape(data)
    assert len(data.shape) == 3
    x0 = tf.clip_by_value(x0, 0, shp[2] - 1)
    x1 = tf.clip_by_value(x1, 0, shp[2] - 1)
    y0 = tf.clip_by_value(y0, 0, shp[1] - 1)
    y1 = tf.clip_by_value(y1, 0, shp[1] - 1)

    new_data_shape = tf.math.reduce_prod(tf.convert_to_tensor(tf.shape(data)))
    data = tf.reshape(data, [new_data_shape])
    rng = tf.range(num_frames)[:, tf.newaxis]
    i1 = tf.gather(data, rng * shp[1] * shp[2] + y0 * shp[2] + x0)
    i2 = tf.gather(data, rng * shp[1] * shp[2] + y1 * shp[2] + x0)
    i3 = tf.gather(data, rng * shp[1] * shp[2] + y0 * shp[2] + x1)
    i4 = tf.gather(data, rng * shp[1] * shp[2] + y1 * shp[2] + x1)

    return tf.maximum(tf.maximum(tf.maximum(i1, i2), i3), i4)


def get_camera_matrices(
        cam_focal_length,
        cam_positions,
        cam_quaternions,
        cam_sensor_width,
        num_frames=None,
):
    """Tf function that converts camera positions into projection matrices."""
    intrinsics = []
    matrix_world = []
    assert cam_quaternions.shape[0] == num_frames
    for frame_idx in range(cam_quaternions.shape[0]):
        focal_length = tf.cast(cam_focal_length, tf.float32)
        sensor_width = tf.cast(cam_sensor_width, tf.float32)
        f_x = focal_length / sensor_width
        f_y = focal_length / sensor_width * INPUT_SIZE[1] / INPUT_SIZE[2]
        p_x = 0.5
        p_y = 0.5
        intrinsics.append(
                tf.stack([
                        tf.stack([f_x, 0., -p_x]),
                        tf.stack([0., -f_y, -p_y]),
                        tf.stack([0., 0., -1.]),
                ]))

        position = cam_positions[frame_idx]
        quat = cam_quaternions[frame_idx]
        rotation_matrix = from_quaternion(
                tf.concat([quat[1:], quat[0:1]], axis=0))
        transformation = tf.concat(
                [rotation_matrix, position[:, tf.newaxis]],
                axis=1,
        )
        transformation = tf.concat(
                [transformation,
                 tf.constant([0.0, 0.0, 0.0, 1.0])[tf.newaxis, :]],
                axis=0,
        )
        matrix_world.append(transformation)

    return tf.cast(tf.stack(intrinsics),
                                 tf.float32), tf.cast(tf.stack(matrix_world), tf.float32)


def single_object_reproject(
        bbox_3d=None,
        pt=None,
        camera=None,
        cam_positions=None,
        num_frames=None,
        depth_map=None,
        window=None,
):
    """Reproject points for a single object.

    Args:
        bbox_3d: The object bounding box from Kubric.  If none, assume it's
            background.
        pt: The set of points in 3D, with shape [num_points, 3]
        camera: Camera intrinsic parameters
        cam_positions: Camera positions, with shape [num_frames, 3]
        num_frames: Number of frames
        depth_map: Depth map video for the camera
        window: the window inside which we're sampling points

    Returns:
        Position for each point, of shape [num_points, num_frames, 2], in pixel
        coordinates, an occlusion flag for each point, of shape
        [num_points, num_frames], and a valid flag for each point, of shape [num_points, num_frames].
        These are respect to the image frame, not the window.

    """
    # Finally, reproject
    reproj, depth_proj = reproject(
            pt,
            camera,
            cam_positions,
            num_frames,
            bbox=bbox_3d,
    )

    occluded = tf.less(reproj[:, :, 2], 0)
    valid = tf.greater(reproj[:, :, 2], 0)#tf.math.equal(tf.math.reduce_sum(reproj[:, :, 2], axis=1), num_frames)
    reproj = reproj[:, :, 0:2] * np.array(INPUT_SIZE[2:0:-1])[np.newaxis, np.newaxis, :]
    occluded = tf.logical_or(occluded, tf.less(tf.transpose(estimate_scene_depth_for_point(depth_map[:, :, :, 0],
                                                                                           tf.transpose(reproj[:, :, 0]),
                                                                                           tf.transpose(reproj[:, :, 1]),
                                                                                           num_frames)), depth_proj * .99))
    obj_occ = occluded
    obj_reproj = reproj

    obj_occ = tf.logical_or(obj_occ, tf.less(obj_reproj[:, :, 1], window[0]))
    obj_occ = tf.logical_or(obj_occ, tf.less(obj_reproj[:, :, 0], window[1]))
    obj_occ = tf.logical_or(obj_occ, tf.greater(obj_reproj[:, :, 1], window[2]))
    obj_occ = tf.logical_or(obj_occ, tf.greater(obj_reproj[:, :, 0], window[3]))
    return obj_reproj, obj_occ, valid


def get_num_to_sample(counts):
    """Computes the number of points to sample for each object.

    Args:
        counts: The number of points available per object.  An int array of length
            n, where n is the number of objects.

    Returns:
        The number of points to sample for each object.  An int array of length n.
    """
    seg_order = tf.argsort(counts)
    sorted_counts = tf.gather(counts, seg_order)
    initializer = (0, TOTAL_TRACKS, 0)

    def scan_fn(prev_output, count_seg):
        index = prev_output[0]
        remaining_needed = prev_output[1]
        desired_frac = 1 / (tf.shape(seg_order)[0] - index)
        want_to_sample = (
                tf.cast(remaining_needed, tf.float32) *
                tf.cast(desired_frac, tf.float32))
        want_to_sample = tf.cast(tf.round(want_to_sample), tf.int32)
        max_to_sample = (
                tf.cast(count_seg, tf.float32) * tf.cast(MAX_SAMPLED_FRAC, tf.float32))
        max_to_sample = tf.cast(tf.round(max_to_sample), tf.int32)
        num_to_sample = tf.minimum(want_to_sample, max_to_sample)

        remaining_needed = remaining_needed - num_to_sample
        return (index + 1, remaining_needed, num_to_sample)

    # outputs 0 and 1 are just bookkeeping; output 2 is the actual number of
    # points to sample per object.
    res = tf.scan(scan_fn, sorted_counts, initializer)[2]
    invert = tf.argsort(seg_order)
    num_to_sample = tf.gather(res, invert)
    num_to_sample = tf.concat(
            [
                num_to_sample,
                tf.zeros([MAX_SEG_ID - tf.shape(num_to_sample)[0]], dtype=tf.int32),
            ],
            axis=0,
    )
    return num_to_sample


#  pylint: disable=cell-var-from-loop


def track_points(
        object_coordinates,
        depth,
        depth_range,
        segmentations,
        bboxes_3d,
        cam_focal_length,
        cam_positions,
        cam_quaternions,
        cam_sensor_width,
        window,
        num_frames=None,
):
    """Track points in 2D using Kubric data.

    Args:
        object_coordinates: Video of coordinates for each pixel in the object's
            local coordinate frame.  Shape [num_frames, height, width, 3]
        depth: uint16 depth video from Kubric.  Shape [num_frames, height, width]
        depth_range: Values needed to normalize Kubric's int16 depth values into
            metric depth.
        segmentations: Integer object id for each pixel.  Shape
            [num_frames, height, width]
        bboxes_3d: The set of all object bounding boxes from Kubric
        cam_focal_length: Camera focal length
        cam_positions: Camera positions, with shape [num_frames, 3]
        cam_quaternions: Camera orientations, with shape [num_frames, 4]
        cam_sensor_width: Camera sensor width parameter
        window: the window inside which we're sampling points.  Integer valued
            in the format [x_min, y_min, x_max, y_max], where min is inclusive and
            max is exclusive.
        num_frames: number of frames in the video

    Returns:
        A set of queries, randomly sampled from the video (with a bias toward
            objects), of shape [num_points, 3].  Each point is [t, y, x], where
            t is time.  All points are in pixel/frame coordinates.
        The trajectory for each query point, of shape [num_points, num_frames, 3].
            Each point is [x, y].  Points are in pixel coordinates
        Occlusion flag for each point, of shape [num_points, num_frames].  This is
            a boolean, where True means the point is occluded.
        Valid flag for each point, of shape [num_points, num_frames].  This is
            a boolean, where False means the point went behind the camera at some point.

    """
    chosen_points = []
    all_reproj = []
    all_occ = []
    all_valid = []

    # Denote if points belong to the same obj
    # When we use this in training it'll be used to penalize self attention on points outside
    # of the same obj
    same_obj_flag = tf.zeros([TOTAL_TRACKS, TOTAL_TRACKS], dtype=tf.bool)
    # Denote which objects the points belong to, starting at bg=-1
    pt_obj_mapping = []

    # Convert to metric depth

    depth_range_f32 = tf.cast(depth_range, tf.float32)
    depth_min = depth_range_f32[0]
    depth_max = depth_range_f32[1]
    depth_f32 = tf.cast(depth, tf.float32)
    depth_map = depth_min + depth_f32 * (depth_max-depth_min) / 65535

    # We first sample query points within the given window.  That means first
    # extracting the window from the segmentation tensor, because we want to have
    # a bias toward moving objects.
    # Note: for speed we sample points on a grid.  The grid start position is
    # randomized within the window.
    start_vec = [
            tf.random.uniform([], minval=0, maxval=STRIDE, dtype=tf.int32)
            for _ in range(len(INPUT_SIZE))
    ]
    start_vec[1] += window[0]
    start_vec[2] += window[1]
    end_vec = [num_frames, window[2], window[3]]

    def extract_box(x):
        x = x[start_vec[0]::STRIDE, start_vec[1]:window[2]:STRIDE,
                    start_vec[2]:window[3]:STRIDE]
        return x

    segmentations_box = extract_box(segmentations)
    object_coordinates_box = extract_box(object_coordinates)

    # Next, get the number of points to sample from each object.  First count
    # how many points are available for each object.

    new_seg_box_shape = tf.math.reduce_prod(tf.convert_to_tensor(tf.shape(segmentations_box)))
    cnt = tf.math.bincount(tf.cast(tf.reshape(segmentations_box, [new_seg_box_shape]), tf.int32))
    num_to_sample = get_num_to_sample(cnt)
    num_to_sample.set_shape([MAX_SEG_ID])
    intrinsics, matrix_world = get_camera_matrices(
            cam_focal_length,
            cam_positions,
            cam_quaternions,
            cam_sensor_width,
            num_frames=num_frames,
    )

    def get_camera(fr=None):
        if fr is None:
            return {'intrinsics': intrinsics, 'matrix_world': matrix_world}
        return {'intrinsics': intrinsics[fr], 'matrix_world': matrix_world[fr]}

    # Construct pixel coordinates for each pixel within the window.
    window = tf.cast(window, tf.float32)
    z, y, x = tf.meshgrid(
            *[tf.range(st, ed, STRIDE) for st, ed in zip(start_vec, end_vec)],
            indexing='ij')
    pix_size = tf.math.reduce_prod(tf.convert_to_tensor(tf.shape(z)))
    pix_coords = tf.reshape(tf.stack([z, y, x], axis=-1), [pix_size, 3])

    pt_cnt = 0

    for i in range(MAX_SEG_ID):
        # sample points on object i in the first frame.  obj_id is the position
        # within the object_coordinates array, which is one lower than the value
        # in the segmentation mask (0 in the segmentation mask is the background
        # object, which has no bounding box).
        obj_id = i - 1
        mask = tf.equal(tf.reshape(segmentations_box, [new_seg_box_shape]), i)
        new_obj_box_shape = tf.math.reduce_prod(tf.convert_to_tensor(tf.shape(object_coordinates_box))) / 3
        pt = tf.boolean_mask(tf.reshape(object_coordinates_box, [new_obj_box_shape, 3]), mask)
        idx = tf.cond(
                tf.shape(pt)[0] > 0,
                lambda: tf.multinomial(  # pylint: disable=g-long-lambda
                        tf.zeros(tf.shape(pt)[0:1])[tf.newaxis, :],
                        tf.gather(num_to_sample, i))[0],
                lambda: tf.zeros([0], dtype=tf.int64))
        pt_coords = tf.gather(tf.boolean_mask(pix_coords, mask), idx)

        if obj_id == -1:
            # For the background object, no bounding box is available.  However,
            # this doesn't move, so we use the depth map to backproject these points
            # into 3D and use those positions throughout the video.
            pt_3d = []
            pt_coords_reorder = []
            for fr in range(num_frames):
                # We need to loop over frames because we need to use the correct depth
                # map for each frame.
                pt_coords_chunk = tf.boolean_mask(pt_coords, tf.equal(pt_coords[:, 0], fr))
                pt_coords_reorder.append(pt_coords_chunk)

                pt_3d.append(
                        unproject(pt_coords_chunk[:, 1:], get_camera(fr), depth_map[fr]))
            pt = tf.concat(pt_3d, axis=0)
            chosen_points.append(tf.concat(pt_coords_reorder, axis=0))
            bbox = None
            # HACK: We don't assume any behavior for bg points, this is equivalent to
            # setting same_obj flags to one for all points
            #same_obj_flag[pt_cnt:pt_cnt+tf.gather(num_to_sample, i), :] = True
        else:
            # For any other object, we just use the point coordinates supplied by
            # kubric.
            pt = tf.gather(pt, idx)
            pt = pt / np.iinfo(np.uint16).max - .5
            chosen_points.append(pt_coords)
            # if obj_id>num_objects, then we won't have a box.  We also won't have
            # points, so just use a dummy to prevent tf from crashing.
            bbox = tf.cond(obj_id >= tf.shape(bboxes_3d)[0], lambda: bboxes_3d[0, :],
                                         lambda: bboxes_3d[obj_id, :])
            # Set same_obj flag
            #same_obj_flag[pt_cnt:pt_cnt+tf.gather(num_to_sample, i), pt_cnt:pt_cnt+tf.gather(num_to_sample, i)] = True

        # add mapping idx info
        pt_obj_mapping.append(tf.fill([tf.gather(num_to_sample, i)], obj_id))
        # increment point count
        pt_cnt += tf.gather(num_to_sample, i)

        # Finally, compute the reprojections for this particular object.
        obj_reproj, obj_occ, obj_valid = tf.cond(
                tf.shape(pt)[0] > 0,
                functools.partial(
                        single_object_reproject,
                        bbox_3d=bbox,
                        pt=pt,
                        camera=get_camera(),
                        cam_positions=cam_positions,
                        num_frames=num_frames,
                        depth_map=depth_map,
                        window=window,
                ),
                lambda:  # pylint: disable=g-long-lambda
                (tf.zeros([0, num_frames, 2], dtype=tf.float32),
                 tf.zeros([0, num_frames], dtype=tf.bool),
                 tf.zeros([0, num_frames], dtype=tf.bool)))
        all_reproj.append(obj_reproj)
        all_occ.append(obj_occ)
        all_valid.append(obj_valid)

    # Points are currently in pixel coordinates of the original video.  We now
    # convert them to coordinates within the window frame, and rescale to
    # pixel coordinates.  Note that this produces the pixel coordinates after
    # the window gets cropped and rescaled to the full image size.
    wd = tf.concat(
            [np.array([0.0]), window[0:2],
             np.array([num_frames]), window[2:4]],
            axis=0)
    wd = wd[tf.newaxis, tf.newaxis, :]
    coord_multiplier = [num_frames, INPUT_SIZE[1], INPUT_SIZE[2]]
    all_reproj = tf.concat(all_reproj, axis=0)
    # We need to extract x,y, but the format of the window is [t1,y1,x1,t2,y2,x2]
    window_size = wd[:, :, 5:3:-1] - wd[:, :, 2:0:-1]
    window_top_left = wd[:, :, 2:0:-1]
    all_reproj = (all_reproj - window_top_left) / window_size
    all_reproj = all_reproj * coord_multiplier[2:0:-1]
    all_occ = tf.concat(all_occ, axis=0)
    all_valid = tf.concat(all_valid, axis=0)

    # chosen_points is [num_points, (z,y,x)]
    chosen_points = tf.concat(chosen_points, axis=0)

    chosen_points = tf.cast(chosen_points, tf.float32)

    # renormalize so the box corners are at [-1,1]
    chosen_points = (chosen_points - wd[:, 0, :3]) / (wd[:, 0, 3:] - wd[:, 0, :3])
    chosen_points = chosen_points * coord_multiplier
    # Note: all_reproj is in (x,y) format, but chosen_points is in (z,y,x) format

    pt_obj_mapping = tf.concat(pt_obj_mapping, axis=0)

    return tf.cast(chosen_points, tf.float32), tf.cast(all_reproj, tf.float32), all_occ, all_valid, pt_obj_mapping


def _get_distorted_bounding_box(
        jpeg_shape,
        bbox,
        min_object_covered,
        aspect_ratio_range,
        area_range,
        max_attempts,
):
    """Sample a crop window to be used for cropping."""
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
            jpeg_shape,
            bounding_boxes=bbox,
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True)

    # Crop the image to the specified bounding box.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack(
            [offset_y, offset_x, offset_y + target_height, offset_x + target_width])
    return crop_window


def add_tracks(data, train_size=(256,256), vflip=False, random_crop=False):
    """Track points in 2D using Kubric data.

    Args:
        data: kubric data, including RGB/depth/object coordinate/segmentation
            videos and camera parameters.
        train_size: cropped output will be at this resolution
        vflip: whether to vertically flip images and tracks (to test generalization)
        random_crop: whether to randomly crop videos

    Returns:
        A dict with the following keys:
        query_points:
            A set of queries, randomly sampled from the video (with a bias toward
            objects), of shape [num_points, 3].  Each point is [t, y, x], where
            t is time.  Points are in pixel/frame coordinates.
            [num_frames, height, width].
        target_points:
            The trajectory for each query point, of shape [num_points, num_frames, 3].
            Each point is [x, y].  Points are in pixel/frame coordinates.
        occlusion:
            Occlusion flag for each point, of shape [num_points, num_frames].  This is
            a boolean, where True means the point is occluded.
        video:
            The cropped video, normalized into the range [-1, 1]

    """
    shp = data['video'].shape.as_list()
    num_frames = shp[0]

    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    min_area = 0.8
    max_area = 1.0
    min_aspect_ratio = 0.5
    max_aspect_ratio = 2.0
    if random_crop:
        crop_window = _get_distorted_bounding_box(
                jpeg_shape=shp[1:4],
                bbox=bbox,
                min_object_covered=min_area,
                aspect_ratio_range=(min_aspect_ratio, max_aspect_ratio),
                area_range=(min_area, max_area),
                max_attempts=10)
    else:
        crop_window = tf.constant([0, 0, INPUT_SIZE[1], INPUT_SIZE[2]],
                                                            dtype=tf.int32,
                                                            shape=[4])

    query_points, target_points, occluded, valid, pt_obj_mapping = track_points(
            data['object_coordinates'], data['depth'],
            data['metadata']['depth_range'], data['segmentations'],
            data['instances']['bboxes_3d'], data['camera']['focal_length'],
            data['camera']['positions'], data['camera']['quaternions'],
            data['camera']['sensor_width'], crop_window, num_frames)
    video = data['video']
    seg = data['segmentations'] # Note that obj_id = 0 is background, [S, H, W, 1]

    inb = target_points[:, :, 0] > 0
    inb = tf.math.logical_and(inb, target_points[:, :, 0] < shp[2] - 1)
    inb = tf.math.logical_and(inb, target_points[:, :, 1] > 0)
    inb = tf.math.logical_and(inb, target_points[:, :, 1] < shp[1] - 1)
    valid = tf.math.logical_and(valid, inb)

    shp = video.shape.as_list()
    num_frames = shp[0]
    query_points.set_shape([TOTAL_TRACKS, 3])
    target_points.set_shape([TOTAL_TRACKS, num_frames, 2])
    occluded.set_shape([TOTAL_TRACKS, num_frames])
    valid.set_shape([TOTAL_TRACKS, num_frames])

    # Crop the video to the sampled window, in a way which matches the coordinate
    # frame produced the track_points functions.
    crop_window = crop_window / (
            np.array(shp[1:3] + shp[1:3]).astype(np.float32) - 1)
    crop_window = tf.tile(crop_window[tf.newaxis, :], [num_frames, 1])
    # video = tf.image.crop_and_resize(
    #         video,
    #         tf.cast(crop_window, tf.float32),
    #         tf.range(num_frames),
    #         train_size,
    # )
    # seg = tf.image.crop_and_resize(
    #         seg,
    #         tf.cast(crop_window, tf.float32),
    #         tf.range(num_frames),
    #         train_size,
    # ) # [S, H, W, 1]
    seg = seg[:, :, :, 0] # [S, H, W]
    if vflip:
        video = video[:, ::-1, :, :]
        seg = seg[:, ::-1, :]
        target_points = target_points * np.array([1, -1])
        query_points = query_points * np.array([1, -1, 1])
    res = {
            'query_points': query_points,
            'target_points': target_points,
            'occluded': occluded,
            'valid': valid,
            'pt_obj_mapping': pt_obj_mapping,
            'video': video,# / (255. / 2.) - 1.,
            'masks': seg,
            'num_objs': data['metadata']['num_instances'],
    }
    return res


def create_point_tracking_dataset(
        train_size=(256, 256),
        shuffle_buffer_size=256,
        split='train',
        batch_dims=tuple(),
        repeat=True,
        vflip=False,
        random_crop=True,
        **kwargs):
    """Construct a dataset for point tracking using Kubric: go/kubric.

    Args:
        train_size: Tuple of 2 ints. Cropped output will be at this resolution
        shuffle_buffer_size: Int. Size of the shuffle buffer
        split: Which split to construct from Kubric.  Can be 'train' or
            'validation'.
        batch_dims: Sequence of ints. Add multiple examples into a batch of this
            shape.
        repeat: Bool. whether to repeat the dataset.
        vflip: Bool. whether to vertically flip the dataset to test generalization.
        random_crop: Bool. whether to randomly crop videos
        **kwargs: additional args to pass to tfds.load.

    Returns:
        The dataset generator.
    """
    input_context = tf.distribute.InputContext(
        input_pipeline_id=1,  # Worker id
        num_input_pipelines=1,  # Total number of workers
    )
    read_config = tfds.ReadConfig(
        input_context=input_context,
    )

    ds = tfds.load(
            'movi_e/256x256',
            data_dir=dataset_location,
            split=split,
            shuffle_files=shuffle_buffer_size is not None,
            read_config=read_config,
            decoders=tfds.decode.PartialDecoding({
                'video': True,
                'object_coordinates': True,
                'depth': True,
                'segmentations': True,
                'camera': {'positions', 'quaternions', 'sensor_width', 'focal_length'},
                'metadata': {'depth_range', 'num_instances'},
                'instances': {'bboxes_3d'},
            }),
            **kwargs)

    # if repeat:
    # ds = ds.repeat()
    ds = ds.map(functools.partial(add_tracks), num_parallel_calls=4)
    # if shuffle_buffer_size is not None:
    #     ds = ds.shuffle(shuffle_buffer_size)

    # for bs in batch_dims[::-1]:
    #     ds = ds.batch(bs)
    #     ds = ds.prefetch(buffer_size=2)

    return ds

class KubricPointDataset(PointDataset):
    def __init__(self,
                 dataset_location='../kubric',
                 S=32, fullseq=False, chunk=None,
                 crop_size=(384,512),
                 strides=[1],
                 clip_step=8,
                 use_augs=False,
                 is_training=True):
        print('loading kubpt dataset...')
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size, 
            use_augs=use_augs,
            is_training=is_training
        )

        if is_training:
            split = 'train'
        else:
            split = 'validation'

        input_context = tf.distribute.InputContext(
            input_pipeline_id=1,  # Worker id
            num_input_pipelines=4,  # Total number of workers
        )
        read_config = tfds.ReadConfig(
            input_context=input_context,
        )

        ds = tfds.load(
                'movi_e/256x256',
                data_dir=dataset_location,
                split=split,
                shuffle_files=False,
                read_config=read_config,
                decoders=tfds.decode.PartialDecoding({
                    'video': True,
                    'object_coordinates': True,
                    'depth': True,
                    'segmentations': True,
                    'camera': {'positions', 'quaternions', 'sensor_width', 'focal_length'},
                    'metadata': {'depth_range', 'num_instances'},
                    'instances': {'bboxes_3d'},
                }),
        )        
        # self.ds = tfds.load(
        #     'movi_e/256x256',
        #     data_dir=dataset_location,
        #     split=split,
        #     shuffle_files=False, 
        #     read_config=read_config,
        #     decoders=tfds.decode.PartialDecoding({
        #         'video': True,
        #         'object_coordinates': True,
        #         'depth': True,
        #         'segmentations': True,
        #         'camera': {'positions', 'quaternions', 'sensor_width', 'focal_length'},
        #         'metadata': {'depth_range', 'num_instances'},
        #         'instances': {'bboxes_3d'},
        #     })
        # )
        
        # ds = ds.map(functools.partial(add_tracks))#, num_parallel_calls=4)

        # ds = ds.add_tracks)

        # self.iterator = ds.make_one_shot_iterator()
        
        # ds = tfds.as_numpy(ds)
        
        # ds = ds.map(
        #         functools.partial(
        #                 add_tracks,
        #                 train_size=train_size,
        #                 vflip=vflip,
        #                 random_crop=random_crop),
        #         num_parallel_calls=4)
        

        self.ds = ds
        # self.ds = tfds.as_numpy(ds)

        # self.ds = tfds.as_numpy(self.ds)#create_point_tracking_dataset(shuffle_buffer_size=16, split='train', batch_dims=[hyp.batch_size], random_crop=True))

        # self.ds = self.ds.as_numpy_iterator()

        self.iterator = iter(self.ds)

        # length = self.ds.reduce(0, lambda x,_: x+1).numpy() # resulted in 2437
        self.length = 2437
        # and i want some number of trajs per sample, which will hopefully be diverse
        # print('length', length)
        # self.N_per = 100
        self.total_len = self.length * TOTAL_TRACKS

        # maybe we can say: each chunk will take a different traj across the N

        # assert(chunk is not None)
        assert(chunk is None)
        # self.chunk = chunk

        # actually let's just export all in one shot, since it seems that this requires a lot of memory anyway

        self.my_count = 0

        # self.iterloader = iter(self.ds)

        self.curr_data = self.iterator.get_next()
        self.curr_data = add_tracks(self.curr_data)

    def __len__(self):
        return self.total_len
        
    def getitem_helper(self, index):

        traj_id = index % TOTAL_TRACKS
        count_id = index//TOTAL_TRACKS
        
        while self.my_count < count_id:
            print('catching up dataloder: ind %d, want %d' % (self.my_count, index//TOTAL_TRACKS))
            self.curr_data = self.iterator.get_next()
            self.my_count += 1
            self.curr_data = add_tracks(self.curr_data)
        
        # data = self.ds.take(1)
        # data = self.ds.next()
        # data = self.iterator.get_next()

        # iterator = dataset.make_one_shot_iterator()
        # print('data', data.keys())
        
        # print('data', data)
        # import ipdb; ipdb.set_trace()
        
        rgbs = self.curr_data["video"].numpy()
        trajs = self.curr_data["target_points"].numpy().transpose(1,0,2) # S,N,2
        valids = self.curr_data["valid"].numpy().transpose(1,0) # S,N
        visibs = 1.0-self.curr_data["occluded"].numpy().transpose(1,0) # S,N
        segs = self.curr_data["masks"].numpy()


        # utils.basic.print_stats_py('seg', seg)

        # print('labels', labels)

        # # # we prefer points with a mix of vis and occ (when inbounds)
        # # keep = 800#4*self.N_per*self.num_strides
        # # keep = 800
        # numer = np.sum(visibs, axis=0)
        # denom = 1+np.sum(valids, axis=0) # count all valids in denom
        # mean = numer/denom
        # dist = np.abs(mean-0.5)

        # we prefer points that travel
        # # keep = 600#3*self.N_per*self.num_strides
        # dists = np.linalg.norm(trajs[:,1:] - trajs[:,:-1], axis=-1) # S-1,N
        # mot_mean = np.mean(dists*valids[:,1:], axis=0) # N
        # inds = np.argsort(-mot_mean)
        # trajs = trajs[inds]
        # visibs = visibs[inds]
        # valids = valids[inds]

        N = trajs.shape[1]
        # print('N', N)

        if False: # we disabled this bc doersch already wrote it to help get diversity 

            if N > 300:
                # take points spaced apart
                keep = 300
                inds = utils.misc.farthest_point_sample_py(trajs[0], keep, deterministic=True)
                trajs = trajs[:,inds]
                visibs = visibs[:,inds]
                valids = valids[:,inds]

            # keep = 600#3*self.N_per*self.num_strides
            dists = np.linalg.norm(trajs[1:] - trajs[:-1], axis=-1) # S-1,N
            # only count motion if it's valid
            mot_mean = np.mean(dists*valids[1:], axis=0) # N
            inds = np.argsort(-mot_mean)#[:keep]
            trajs = trajs[:,inds]
            visibs = visibs[:,inds]
            valids = valids[:,inds]
            # inbounds = inbounds[:,inds]
            # print('sorted mot_mean[:10]', mot_mean[inds[:10]])
        
        # inbounds = inbounds[inds]
        # print('kept vis means', mean[inds])
        # print('N2', trajs.shape[1])

        # print('video', video.shape)
        # print('trajs', trajs.shape)
        # print('valids', valids.shape)
        # print('labels', labels.shape)
        S_local, H, W, C = rgbs.shape
        xys = trajs[:,traj_id]
        # move to [0,1] then move to 0,W-1
        # xys = xys + 1
        # xys[:,0] *= W-1
        # xys[:,1] *= H-1
        visibs = visibs[:,traj_id]
        valids = valids[:,traj_id]
        # print('xys', xys)
        # print('visibs', visibs)

        segA = segs.reshape(-1)
        obj_ids = np.unique(segA) # NSeg
        xys_ = xys.round().astype(np.int32)
        gotit = False
        masks = None

        vis_valid = visibs * valids
        safe = visibs*0
        for si in range(1,S_local-1):
            safe[si] = vis_valid[si-1]*vis_valid[si]*vis_valid[si+1]
        if np.sum(safe) == 0:
            print('safe', safe)
            return None
        oids = []
        for si in range(S_local):
            if safe[si]:
                oid = segs[si,xys_[si,1],xys_[si,0]]
                oids.append(oid)
        oid = stats.mode(oids)[0]
        # print('oids', oids, 'picked', oid)
        masks = [(seg == oid).astype(np.float32) for seg in segs]
        masks = np.stack(masks, axis=0)

        print('rgbs', rgbs.shape)
        
        if masks is None:
            return None
            
        # # for this data,
        # # maybe it will be better to inflate via some random over-sampling subseq,
        # # rather than zigzag
        # if S_local < self.S:
        #     ara = np.arange(S_local, self.S)
        #     ara = (np.arange(self.S)/(self.S-1)*(S_local-1)).round().astype(np.int32)
        #     print('ara', ara)
        #     # print('rgbs', rgbs.shape)
        #     rgbs = rgbs[ara]
        #     masks = masks[ara]
        #     xys = xys[ara]
        #     visibs = visibs[ara]
        #     valids = valids[ara]
        if S_local > self.S:
            rgbs = rgbs[:self.S]
            masks = masks[:self.S]
            xys = xys[:self.S]
            visibs = visibs[:self.S]
            valids = valids[:self.S]
            S_local = self.S

        zoom = 1.2
        xys, visibs, valids, rgbs, masks = utils.misc.data_zoom(zoom, xys, visibs, valids, rgbs, masks)
        _, H, W, _ = rgbs.shape
        print('rgbs zoom', rgbs.shape)
            
        masks0 = np.zeros_like(masks)
        for i in range(len(masks)):
            if valids[i]:
                xy = xys[i].round().astype(np.int32)
                x, y = xy[0], xy[1]
                x = x.clip(0,W-1)
                y = y.clip(0,H-1)
                masks0[i,y,x] = 1
            else:
                masks0[i] = 0.5
        full_masks = np.stack([masks0, masks, masks], axis=-1)

        # chans 0,1 valid when points are valid
        masks_valid = np.zeros((S_local,3), dtype=np.float32)
        masks_valid[:,0] = valids
        masks_valid[:,1] = valids
        
        sample = {
            'rgbs': rgbs,
            'masks': full_masks,
            'masks_valid': masks_valid,
            'xys': xys,
            'visibs': visibs,
            'valids': valids,
        }
        return sample
        
        
        return sample

