import os
import json
import numpy as np
from pycocotools.coco import COCO
from PIL import Image
import torch
from datasets.dataset import PointDataset
import utils.misc

class UBodyDataset(PointDataset):
    def __init__(self,
                 dataset_location='/orion/group/UBody',
                 use_augs=False,
                 S=16, fullseq=False, chunk=None,
                 N=32,
                 strides=[2],
                 zooms=[1,2],
                 crop_size=(368, 496),
                 is_training=True):
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size, 
            use_augs=use_augs,
            is_training=is_training
        )
        print('loading ubody dataset...')

        self.S = S
        self.N = N
        # self.N_per = 4

        clip_step = S//2
        if not is_training:
            clip_step = S

        self.use_augs = use_augs
        self.data = []

        self.test_list = np.load(os.path.join(dataset_location, 'splits', 'intra_scene_test_list.npy')).tolist()
        self.scene_path = os.path.join(dataset_location, 'images')
        annot_path = os.path.join(dataset_location, 'annotations')

        scenes = sorted(os.listdir(self.scene_path))

        print('found %d scenes' % (len(scenes)))
        
        for scene in scenes[:]:
            print('working on scene', scene)
            datadict = self.load_data(os.path.join(annot_path, scene, 'keypoint_annotation.json'), os.path.join(self.scene_path, scene))
            if len(datadict.keys()) == 0: continue

            print('found %d videos' % len(datadict.keys()))
            for video in datadict.keys():
                S_local = len(datadict[video])
                print('S_local', S_local)
                for stride in strides:
                    for ii in range(0, max(S_local-self.S*stride+1, 1), clip_step*stride):
                        frame_ii = datadict[video][ii]
                        joint_valid_ii = frame_ii['joint_valid']
                        if np.sum(joint_valid_ii) == 0: continue
                        joint_img_ii = frame_ii['joint_img']
                        valid_ii = np.where(joint_valid_ii)[0]
                        # valid_ii_N_idx = np.random.choice(valid_ii, self.N_per, replace=False)
                        # N_per_idx = np.linspace(0, len(valid_ii)-1, self.N_per, dtype=int)
                        # valid_ii_N_idx = valid_ii[N_per_idx]
                        full_idx = ii + np.arange(self.S)*stride
                        if full_idx[-1] < len(datadict[video]): # fullseq
                            for ni in valid_ii:
                                full_data = [datadict[video][fi] for fi in full_idx]

                                traj = [d['joint_img'][ni] for d in full_data]
                                traj = np.stack(traj)
                                # print('traj', traj, traj.shape)
                                max_travel = np.max(np.linalg.norm(traj[1:] - traj[:-1], axis=-1))
                                total_travel = np.sum(np.linalg.norm(traj[1:] - traj[:-1], axis=-1))
                                # print('travel', travel)
                                if total_travel > self.S*2 and max_travel < 32:
                                    for zoom in zooms:
                                        self.data.append({
                                            'img_path': [d['img_path'] for d in full_data],
                                            'joint_img': [d['joint_img'][ni] for d in full_data],
                                            'joint_valid': [d['joint_valid'][ni] for d in full_data],
                                            'zoom': zoom,
                                        })
        print('ubody dataset loaded, total {} samples'.format(len(self.data)))
        if chunk is not None:
            assert(len(self.data) > 100)
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.data = chunkify(self.data,100)[chunk]
            print('filtered to %d' % len(self.data))

    def load_data(self, annot_path, image_path):
        print('loading annotations from %s' % annot_path)
        db = COCO(annot_path)

        datadict_all = {}
        i = 0
        for aid in db.anns.keys():
            i = i + 1
            ann = db.anns[aid]
            person_id = ann['person_id']
            if person_id != 0: continue

            img = db.loadImgs(ann['image_id'])[0]
            if img['file_name'].startswith('/'):
                file_name = img['file_name'][1:]  # [1:] means delete '/'
            else:
                file_name = img['file_name']
            video_name = file_name.split('/')[-2]
            if 'Trim' in video_name:
                video_name = video_name.split('_Trim')[0]
            if video_name in self.test_list and self.is_training: continue
            elif video_name not in self.test_list and not self.is_training: continue
            seq_name = file_name.split('/')[-2]
            img_path = os.path.join(image_path, file_name)
            if not os.path.exists(img_path):
                print('image not exists: {}'.format(img_path))
                continue

            # exclude the samples that are crowd or have few visible keypoints
            if ann['iscrowd'] or (ann['num_keypoints'] == 0): continue


            # joint coordinates
            joint_img = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3) # only use 17 joints
            # foot_img = np.array(ann['foot_kpts'], dtype=np.float32).reshape(-1, 3)
            # lhand_img = np.array(ann['lefthand_kpts'], dtype=np.float32).reshape(-1, 3)
            # rhand_img = np.array(ann['righthand_kpts'], dtype=np.float32).reshape(-1, 3)
            # face_img = np.array(ann['face_kpts'], dtype=np.float32).reshape(-1, 3)
            # joint_img = np.concatenate((joint_img, foot_img, lhand_img, rhand_img, face_img), axis=0)

            joint_valid = (joint_img[:, 2].copy().reshape(-1, 1) > 0).astype(np.float32)
            joint_img = joint_img[:, :2]

            data_dict = {'img_path': img_path, 'img_shape': (img['height'], img['width']),
                         'joint_img': joint_img, 'joint_valid': joint_valid}

            person_key = str(person_id) + '_' + seq_name
            if not person_key in datadict_all.keys():
                datadict_all[person_key] = []
            datadict_all[person_key].append(data_dict)

        return datadict_all


    def __len__(self):
        return len(self.data)

    def getitem_helper(self, index):
        # Collecting continuous frames for the clip
        clip_data = self.data[index]

        img_path = clip_data['img_path']
        img_list = [np.array(Image.open(im))[:, :, :3] for im in img_path]
        joint_img = clip_data['joint_img']
        visibs = clip_data['joint_valid']
        zoom = clip_data['zoom']

        # print('joint_img', np.stack(joint_img).shape)
        # print('visibs', np.stack(visibs).shape)

        rgbs = np.stack(img_list, axis=0)
        xys = np.stack(joint_img, axis=0)
        visibs = np.stack(visibs, axis=0)

        xys = xys.reshape(-1,2)
        visibs = visibs.reshape(-1)

        xys = utils.misc.data_replace_with_nearest(xys, visibs)
        valids = np.ones_like(visibs)
        if zoom > 1:
            xys, visibs, valids, rgbs = utils.misc.data_zoom(zoom, xys, visibs, valids, rgbs)
        visibs = 1.0 * (visibs>0.5)
        safe = np.concatenate(([0], visibs[:-2] * visibs[1:-1] * visibs[2:], [0]))
        if np.sum(safe) < 2: return None

        sample = {
            'rgbs': rgbs,
            'xys': xys,
            'visibs': visibs,
        }
        return sample

