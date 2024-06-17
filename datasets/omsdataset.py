import os
import numpy as np
from scipy.io import loadmat
from PIL import Image
import torch
from datasets.dataset import PointDataset
import glob

from datasets.dataset_utils import make_split


class OMSDataset(PointDataset):
    def __init__(
            self,
            dataset_location="/orion/group/OMS_Dataset",
            use_augs=False,
            S=8, fullseq=False, chunk=None,
            strides=[1, 2, 3, 4],
            crop_size=(512, 512),
            is_training=True,
    ):
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training,
        )
        print("loading OMS dataset...")

        clip_step = S//2
        if not is_training:
            strides = [2]
            clip_step = S

        self.S = S

        # Paths for images and annotation data
        self.image_dir = os.path.join(dataset_location, "Images")

        batches = sorted([f for f in os.listdir(dataset_location) if f.startswith("Batch")])
        batches = make_split(batches, is_training, shuffle=True)

        self.data = []

        for btch in batches:
            with open(os.path.join(dataset_location, "{}/intrinsic.txt".format(btch))) as f:
                lines = f.readlines()
                cameras = {}
                for i in range(0, len(lines), 5):
                    cam_line = lines[i]
                    K_lines = lines[i + 1 : i + 4]
                    ds = lines[i + 4].rstrip("\n")
                    d = ds.split(" ")
                    d1 = float(d[0])
                    d2 = float(d[1])
                    cam = cam_line.strip().split(" ")[1]
                    K = np.reshape(
                        np.array(
                            [
                                float(f)
                                for K_line in K_lines
                                for f in K_line.strip().split(" ")
                            ]
                        ),
                        [3, 3],
                    )
                    cameras[cam] = {"K": K, "d1": d1, "d2": d2}

            # Extrinsics
            with open(
                os.path.join(dataset_location, "{}/camera.txt".format(btch))
            ) as f:
                lines = f.readlines()
                for i in range(3, len(lines), 5):
                    cam_line = lines[i]
                    C_line = lines[i + 1]
                    R_lines = lines[i + 2 : i + 5]
                    cam = cam_line.strip().split(" ")[1]
                    C = np.array([float(f) for f in C_line.strip().split(" ")])
                    R = np.reshape(
                        np.array(
                            [
                                float(f)
                                for R_line in R_lines
                                for f in R_line.strip().split(" ")
                            ]
                        ),
                        [3, 3],
                    )
                    P = cameras[cam]["K"] @ (
                        R
                        @ (
                            np.concatenate(
                                (np.identity(3), -np.reshape(C, [3, 1])), axis=1
                            )
                        )
                    )
                    cameras[cam]["R"] = R
                    cameras[cam]["C"] = C
                    cameras[cam]["P"] = P

            annotations = loadmat(
                os.path.join(dataset_location, "{}/coords_3D.mat".format(btch))
            )
            parameters = loadmat(
                os.path.join(dataset_location, "{}/crop_para.mat".format(btch))
            )
            pt = parameters["crop"].transpose()[0]
            u = np.unique(pt, axis=0)
            num_frames = len(u)
            print(
                "{} has {} frames from {} cameras".format(
                    btch, num_frames, len(cameras.keys())
                )
            )

            for i in range(len(cameras.keys())):
                for stride in strides:
                    for ii in range(0, max(num_frames - self.S * stride + 1, 1), clip_step*stride):
                        q = np.where(pt == u[ii])
                        if len(q[0]) < i + 1:
                            break
                        cmr = parameters["crop"][q[0][i]][1]
                        cam_info = cameras[str(cmr)]
                        h = parameters["crop"][q[0][i]][5]
                        w = parameters["crop"][q[0][i]][4]

                        params = (
                            parameters["crop"][q[0][i]][2],
                            parameters["crop"][q[0][i]][3],
                        )
                        z_list_ii = []
                        for jt in range(13):
                            coords = annotations["coords"][ii * 13 + jt, 1:4]
                            if coords is not None:
                                proj, z = self.get_projection(cam_info, coords)
                                x, y = proj[0], proj[1]
                                y = y - params[1]
                                x = x - params[0]

                                visibles = (
                                    1 if (x >= 0 and x < w and y >= 0 and y < h) else 0
                                )
                                z = z if visibles else 10000
                                z_list_ii.append(z)
                                # proj = self.distort_point(x, y, cam_info)
                        # choose the closest 3 joint idx
                        z_arg = np.argsort(z_list_ii)
                        z_min_idx_list = z_arg[:4]

                        for z_min_idx in z_min_idx_list:
                            full_idx = ii + np.arange(self.S) * stride
                            if full_idx[-1] >= num_frames:
                                continue
                            proj_list = []
                            img_list = []
                            vis_list = []
                            for jj in full_idx:
                                q = np.where(pt == u[jj])
                                cam_ = [
                                    parameters["crop"][q[0][i_]][1]
                                    for i_ in range(len(q[0]))
                                ]
                                if cmr not in cam_:
                                    break
                                i_ = cam_.index(cmr)
                                frame = parameters["crop"][q[0][i_]][0]
                                h = parameters["crop"][q[0][i_]][5]
                                w = parameters["crop"][q[0][i_]][4]

                                params = (
                                    parameters["crop"][q[0][i_]][2],
                                    parameters["crop"][q[0][i_]][3],
                                )
                                img_name = (
                                    "batch"
                                    + str(btch[5:])
                                    + "_"
                                    + str(frame).zfill(9)
                                    + "_"
                                    + str(cmr)
                                    + ".jpg"
                                )
                                coords = annotations["coords"][jj * 13 + z_min_idx, 1:4]
                                if coords is not None:
                                    proj, z = self.get_projection(cam_info, coords)
                                    x, y = proj[0], proj[1]
                                    y = y - params[1]
                                    x = x - params[0]
                                    pt1 = np.array([x, y])

                                    visibles = (
                                        1
                                        if (x >= 0 and x < w and y >= 0 and y < h)
                                        else 0
                                    )
                                    vis_list.append(visibles)
                                    proj_list.append(pt1)
                                else:
                                    proj_list.append(np.zeros(2))
                                    vis_list.append(0)
                                img_list.append(os.path.join(self.image_dir, img_name))
                            if len(img_list) < self.S:
                                continue
                            self.data.append(
                                {
                                    "img_list": img_list,
                                    "proj_list": proj_list,
                                    "vis_list": vis_list,
                                }
                            )
        print("Done loading {} samples".format(len(self.data)))

    @staticmethod
    def get_projection(cam_info, coords_3d):
        P = cam_info["P"]
        u = P @ np.append(coords_3d, [1])
        z = u[2]
        u = u[0:2] / u[2]
        proj = OMSDataset.distort_point(cam_info, u[0], u[1])
        return proj, z

    @staticmethod
    def distort_point(cam_info, u_x, u_y):
        K = cam_info["K"]
        d1 = cam_info["d1"]
        d2 = cam_info["d2"]

        invK = np.linalg.inv(K)
        z = np.array([u_x, u_y, 1])
        nx = invK.dot(z)

        x_dn = nx[0] * (
            1
            + d1 * (nx[0] * nx[0] + nx[1] * nx[1])
            + d2 * (nx[0] * nx[0] + nx[1] * nx[1]) * (nx[0] * nx[0] + nx[1] * nx[1])
        )
        y_dn = nx[1] * (
            1
            + d1 * (nx[0] * nx[0] + nx[1] * nx[1])
            + d2 * (nx[0] * nx[0] + nx[1] * nx[1]) * (nx[0] * nx[0] + nx[1] * nx[1])
        )

        z2 = np.array([x_dn, y_dn, 1])
        x_d = K.dot(z2)

        return np.array([x_d[0], x_d[1]])

    def __len__(self):
        return len(self.data)

    def getitem_helper(self, index):
        # Collecting continuous frames for the clip
        clip_data = self.data[index]

        img_path = clip_data["img_list"]
        img_list = [np.array(Image.open(im))[:, :, :3] for im in img_path]
        max_h = max([im.shape[0] for im in img_list])
        max_w = max([im.shape[1] for im in img_list])

        # padding
        img_list = [
            np.pad(
                im,
                ((0, max_h - im.shape[0]), (0, max_w - im.shape[1]), (0, 0)),
                "constant",
            )
            for im in img_list
        ]

        xys = clip_data["proj_list"]
        visibs = clip_data["vis_list"]

        # print('joint_img', np.stack(joint_img).shape)
        # print('visibs', np.stack(visibs).shape)

        rgbs = np.stack(img_list, axis=0)
        xys = np.stack(xys, axis=0)
        visibs = np.stack(visibs, axis=0)

        xys = xys.reshape(-1, 2)
        visibs = visibs.reshape(-1)

        sample = {
            "rgbs": rgbs,
            "xys": xys,
            "visibs": visibs,
        }
        return sample
