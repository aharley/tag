import numpy as np
from torch.utils.data import Dataset, DataLoader
import utils.misc
import utils.geom
import torch
from datasets.acinodataset import AcinoDataset
from datasets.animaltrackdataset import AnimalTrackDataset
from datasets.antdataset import AntDataset
from datasets.aptv2bboxdataset import APTv2BBoxDataset
from datasets.aptv2pointdataset import APTv2PointDataset
from datasets.autotrajdataset import AutotrajDataset
from datasets.avistdataset import AVisTDataset
from datasets.badjadataset import BadjaDataset
from datasets.bdd100kdataset import BDD100KDataset
from datasets.bedlamsegdataset import BEDLAMSEGDataset
from datasets.bedlamjointsdataset import BEDLAMJOINTSDataset
from datasets.bee23dataset import BEE23Dataset
from datasets.biodronedataset import BioDroneDataset
from datasets.bioparticledataset import BioparticleDataset
from datasets.birdsaidataset import BirdSAIDataset
from datasets.bl30kdataset import Bl30kDataset
from datasets.burstdataset import BurstDataset
from datasets.caltechfish import CaltechFishDataset
from datasets.cattleboxdataset import CattleBoxDataset
from datasets.cattlepointdataset import CattlePointDataset
from datasets.cocodataset import COCODataset
from datasets.crohddataset import CrohdDataset
from datasets.ctmcdataset import CTMCMaskDataset
from datasets.dancetrackdataset import DanceTrackDataset
from datasets.davisdataset import DAVISDataset
from datasets.deepfly3d import DeepFly3dDataset
from datasets.deeplabcutdataset import DeepLabCutDataset
# from datasets.deepposekit_dataset import DeepPoseKitDataset
from datasets.egotracksdataset import EgoTracksDataset
from datasets.fbmsdataset import FBMSDataset
from datasets.flyingthingsdataset_raw import FlyingThingsDataset
from datasets.gmotdataset import GMOTDataset
from datasets.got10kdataset import Got10kDataset
from datasets.hobdataset import HOBDataset
from datasets.hootdataset import HootDataset
from datasets.horsedataset import HorseDataset
from datasets.interhanddataset import InterHandDataset
from datasets.kittidataset import KittiDataset
from datasets.kubricmaskdataset import KubricRandomDataset, KubricContainersDataset
from datasets.kubricpointdataset import KubricPointDataset
from datasets.lasotdataset import LaSOTDataset
from datasets.latotdataset import LaTOTDataset
from datasets.lepertdataset import LepertDataset
from datasets.lvvisdataset import LVVISDataset
from datasets.mocadataset import MoCADataset
from datasets.mosedataset import MOSEDataset
from datasets.mot17dataset import MOT17Dataset
from datasets.motsdataset import MOTSDataset
from datasets.newzealanddataset import NewZealandDataset
from datasets.nuscenesdataset import NuScenesDataset
from datasets.omsdataset import OMSDataset
from datasets.otbdataset import OTBDataset
from datasets.ovisdataset import OVISDataset
from datasets.pacedataset import PacePointDataset
from datasets.personpathdataset import PersonPathDataset
from datasets.podmaskdataset import PodMaskDataset
from datasets.pointodysseydataset import PointOdysseyDataset
from datasets.posetrackdataset import PoseTrackDataset
from datasets.rat7mdataset import Rat7MDataset
from datasets.robotapdataset import RoboTapDataset
from datasets.sailvosdataset import SAILVOSDataset
from datasets.sportsmotdataset import SportsMot
from datasets.surgicalhandsdataset import SurgicalHandsDataset
from datasets.tapviddataset import TapVidDataset
from datasets.templecolordataset import TempleColorDataset
from datasets.teyeddataset import TeyedPointDataset, TeyedSegDataset
from datasets.thirdantiuavdataset import ThirdAntiUAVDataset
from datasets.tknetdataset import TknetDataset
from datasets.tlpdataset import TlpDataset
from datasets.totbdataset import TOTBDataset
from datasets.uavdataset import UAVDataset
from datasets.ubodydataset import UBodyDataset
from datasets.uvodensedataset import UVODenseDataset
from datasets.uvosparsedataset import UVOSparseDataset
from datasets.videocubedataset import VideoCubeDataset
from datasets.visdronedataset import VisDroneDataset
from datasets.visordataset import VISORDataset
from datasets.vostdataset import VOSTDataset
from datasets.vsb100dataset import VSB100Dataset
from datasets.waymodataset import WaymoDataset
from datasets.ycbineoatdataset import YCBInEOATDataset
from datasets.youtubevos import YoutubeVOSDataset
from datasets.zef20dataset import Zef20BBoxDataset, Zef20PointDataset
from datasets.drivetrackdataset import DriveTrackDataset

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# rejected  dnames (for now): 
# "antleg", < no images
# "badja", < vis unreliable
# "deeplabcut", < bad
# "deepposekit", < bad
# "fbms", bad
# "mot17", superseded by mots
# "oms", # weird/buggy; nice looking monkeys but the data seems useless
# "rat7m", # bad
# "uvosparse", # superseded by uvodense
# "newzealand", # bad?
# "thirdantiuav", # only holds "exist" label; better labels seem available if we dig through the json

dnames = [
    "interhand", # broken
    "acino", # mostly wrong annotations 
    "animaltrack", # exporting on oriong4
    "ant", # exporting on oriong5
    "aptv2bbox", # short; exporting S16 on oriong8; exporting S24 on oriong8
    "aptv2point", # short; exporting S16 on oriong8; exporting S24 on oriong8
    "autotraj", # exporting on oriong7
    "avist", # exporting on oriong6
    "bdd", # exporting on oriong5
    "bedlamseg", # exporting on oriong6, but want to wait for sync
    "bedlamjoints", # (re-)exporting on oriong2
    "bee23", # exporting on oriong7
    "biodrone", # exporting on oriong5
    "bioparticle", # exporting on oriong5
    "birdsai", # exporting on oriong8
    "blk", # exporting on oriong7
    "burst",  # exporting on oriong6
    "caltechfish", # exporting on oriong8
    "cattlebox", # exporting on oriong7
    "cattlepoint", # exporting on oriong7
    "coco", # exporting on oriong4
    "crohd", # test
    "ctmc", # test
    "dancetrack", # exporting on oriong2
    "davis", # test
    "deepfly3d", # exporting on oriong2
    "drivetrack", 
    "egotracks", # exporting on oriong3
    "flt", # short; exporting S=16 on oriong9; exporting S24 on oriong9; 
    "gmot", # exporting on oriong5
    "got10k", # exporting on oriong1
    "hob", # exporting on oriong2
    "hoot", # exporting on oriong8
    "horse", # exporting on oriong4
    "kitti", # short; exporting S24 on oriong3
    "kubpt", # short (S=24); exporting S=16 on oriong2; exporting S24 on oriong2
    "kubrand", # short (S=36); exporting S=16 on oriong4; exporting S24 on oriong6
    "kubcont", # short (S=36); exporting S=16 on oriong4; exporting S24 on oriong6
    "lasot", # exporting on oriong2
    "latot", # exporting on oriong5
    "lepert", # 
    "lvvis", # exporting on oriong1
    "moca", # test set!
    "mose", # exporting on oriong4
    "mots", # exporting on oriong7
    "nuscenes", # short; exporting on oriong5
    "otb", # exporting on oriong4
    "ovis", # exporting on oriong4
    "pacepoint", # exporting on oriong9 
    "personpath", # exporting oriong5
    "podmask", # exporting on oriong6
    "podpt", # exporting on oriong6
    "posetrack", # short; exporting S=16 on oriong6; exporting S24 on oriong6
    "robotap", # test
    "sailvos", # exporting on oriong2
    "sportsmot", # exporting on oriong8
    "surgicalhands", # test
    "tapvid", # test
    "templecolor", # test
    "teyedmask",  # exporting on oriong5
    "teyedpt", # exporting on oriong5
    "tknet", #  short (after filtering to stride30 to avoid lin terp boxes); exporting on oriong7
    "tlp", # exporting on oriong8
    "totb", # test
    "uav", # test
    "ubody", # exporting on oriong9
    "uvodense", # exporting on oriong8
    "videocube", # exporting on oriong5
    "visdrone", # exporting on oriong5
    "visor", # exporting on oriong4
    "vost", # test
    "vsb100", # short; exporting S24 on oriong5
    "waymo", # exporting on oriong4
    "ycb", # test
    "ytvos", # short; exporting on oriong7
    "zefbox", # test
    "zefpt", # test
]


class AnyDataset:
    def __init__(
            self,
            dname,
            is_training=True,
            B=1,
            S=4,
            fullseq=False,
            shuffle=True,
            crop_size=(256, 448),
            chunk=None, 
            use_augs=True,
            cache_len=0,
            cache_freq=0,
            num_workers=8,
    ):
        print("creating %s dataset" % dname)

        self.cache_len = cache_len
        self.cache_freq = cache_freq

        assert dname in dnames, f"{dname} is an invalid dataset name"

        import socket
        host = socket.gethostname()

        if dname == "acino":
            if host=='oriong3.stanford.edu':
                dataset_location = "/scr/aharley/Acino_full/exported_data"
            else:
                dataset_location = "/orion/group/Acino_full/exported_data"
            dataset = AcinoDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "animaltrack":
            if host=='oriong4.stanford.edu':
                dataset_location = "/scr/aharley/AnimalTrack"
            else:
                dataset_location = "/orion/group/AnimalTrack"
            dataset = AnimalTrackDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "ant":
            if host=='oriong5.stanford.edu':
                dataset_location = "/scr/aharley/ant_det_and_track"
            else:
                dataset_location = "/orion/u/aharley/datasets/ant_det_and_track"
            dataset = AntDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "aptv2bbox":
            dataset_location = "/orion/group/APTv2/videos/APTv2/"
            dataset = APTv2BBoxDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training, 
            )
        elif dname == "aptv2point":
            if host=='oriong8.stanford.edu':
                dataset_location = "/scr/aharley/APTv2/"
            else:
                dataset_location = "/orion/group/APTv2/videos/APTv2/"
            dataset = APTv2PointDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training, 
            )
        elif dname == "avist":
            if host=='oriong6.stanford.edu':
                dataset_location = "/scr/aharley/avist/"
            else:
                dataset_location = "/orion/group/avist"
            dataset = AVisTDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "bdd":
            if host=='oriong5.stanford.edu':
                dataset_location = "/scr/aharley/BDD100K"
            else:
                dataset_location = "/orion/group/BDD100K"
            dataset = BDD100KDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == 'bedlamseg':
            dataset_location = "/orion/group/BEDLAM/"
            dataset = BEDLAMSEGDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == 'bedlamjoints':
            if host=='oriong2.stanford.edu':
                dataset_location = "/scr/aharley/BEDLAM/"
            else:
                dataset_location = "/orion/group/BEDLAM/"
            dataset = BEDLAMJOINTSDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
                chunk=chunk,
            )
        elif dname == "bee23":
            if host=='oriong7.stanford.edu':
                dataset_location = "/scr/aharley/BEE23"
            else:
                dataset_location = "/orion/u/yangyou/datasets/BEE23"
            dataset = BEE23Dataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "biodrone":
            dataset_location = "/orion/group/biodrone"
            dataset = BioDroneDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "bioparticle":
            if host=='oriong5.stanford.edu' and is_training:
                dataset_location = "/scr/aharley/particle_challenge/"
            else:
                dataset_location = "/orion/group/particle_challenge/"
            dataset = BioparticleDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
                chunk=chunk,
            )
        elif dname == 'birdsai':
            dataset_location = "/orion/group/birdsai"
            dataset = BirdSAIDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "blk":
            if host=='oriong7.stanford.edu' and is_training:
                dataset_location = "/scr/aharley/bl30k/BL30K"
            else:
                dataset_location = "/orion/u/aharley/datasets/bl30k/BL30K"
            dataset = Bl30kDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "burst":
            dataset_location = "/orion/group/burst"
            dataset = BurstDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "caltechfish":
            dataset_location = "/orion/group/CaltechFish"
            dataset = CaltechFishDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "cattlebox":
            if host=='oriong7.stanford.edu':
                dataset_location = "/scr/aharley/CattleEyeView"
            else:
                dataset_location = "/orion/group/CattleEyeView"
            dataset = CattleBoxDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "cattlepoint":
            if host=='oriong7.stanford.edu':
                dataset_location = "/scr/aharley/CattleEyeView"
            else:
                dataset_location = "/orion/group/CattleEyeView"
            dataset = CattlePointDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "coco":
            if host=='oriong2.stanford.edu':
                dataset_location = "/scr/aharley/coco"
            else:
                dataset_location = "/orion/group/coco"
            dataset = COCODataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
                chunk=chunk,
            )
        elif dname == "crohd":
            dataset_location = "/orion/group/head_tracking"
            dataset = CrohdDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "ctmc":
            dataset_location = "/orion/group/CTMC"
            dataset = CTMCMaskDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "dancetrack":
            if host=='oriong2.stanford.edu' and is_training:
                dataset_location = "/scr/aharley/DanceTrack"
            else:
                dataset_location = "/orion/group/DanceTrack"
                
            dataset = DanceTrackDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "davis":
            dataset_location = "/orion/group/DAVIS"
            dataset = DAVISDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "deepfly3d":
            if host=='oriong2.stanford.edu' and is_training:
                dataset_location = "/scr/aharley/deepfly3d"
            else:
                dataset_location = "/orion/group/deepfly3d"
            dataset = DeepFly3dDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == 'drivetrack':
            dataset_location = "/orion/group/drivetrack"
            dataset = DriveTrackDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == 'egotracks':
            dataset_location = "/orion/group/egotracks/v2"
            dataset = EgoTracksDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "flt":
            if host=='oriong9.stanford.edu' or host=='oriong3.stanford.edu':
                dataset_location = '/scr/aharley/flyingthings'
            else:
                dataset_location = '/orion/u/aharley/datasets/flyingthings'
            dataset = FlyingThingsDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "gmot":
            if host=='oriong5.stanford.edu':
                dataset_location = "/scr/aharley/GMOT"
            else:
                dataset_location = "/orion/group/GMOT"
            dataset = GMOTDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "got10k":
            dataset_location = "/orion/group/got10k"
            dataset = Got10kDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "hob":
            dataset_location = "/orion/group/hob"
            dataset = HOBDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "hoot":
            dataset_location = "/orion/u/aharley/datasets/hoot"
            dataset = HootDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                use_augs=use_augs,
                crop_size=crop_size,
                is_training=is_training,
            )
        elif dname == "horse":
            if host=='oriong3.stanford.edu':
                dataset_location = '/scr/aharley/horse10/horse10/'
            else:
                dataset_location = "/orion/group/horse10/horse10/"
            dataset = HorseDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "kitti":
            dataset_location = "/orion/u/yangyou/datasets/kitti"
            dataset = KittiDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "kubrand":
            dataset_location = "/orion/group/kubric_random"
            dataset = KubricRandomDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
                chunk=chunk,
            )
        elif dname == "kubcont":
            dataset_location = "/orion/group/kubric_containers"
            dataset = KubricContainersDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "kubpt":
            if host=='oriong2.stanford.edu':
                dataset_location = "/scr/aharley/kubric_movi_e"
            else:
                dataset_location = "/orion/group/kubric_movi_e"
            dataset = KubricPointDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "lasot":
            dataset_location = "/orion/u/yangyou/datasets/lasot"
            dataset = LaSOTDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "latot":
            dataset_location = "/orion/group/LaTOT"
            dataset = LaTOTDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "lepert":
            dataset_location = "/orion/group/lepert"
            dataset = LepertDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "mose":
            if host=='oriong4.stanford.edu':
                dataset_location = '/scr/aharley/MOSE'
            else:
                dataset_location = "/orion/u/yangyou/datasets/MOSE"
            dataset = MOSEDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "mot17":
            dataset_location = "/orion/group/MOT17"
            dataset = MOT17Dataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "mots":
            if host=='oriong7.stanford.edu':
                dataset_location = "/scr/aharley/MOTS"
            else:
                dataset_location = "/orion/group/MOTS"
            dataset = MOTSDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
                chunk=chunk,
            )
        elif dname == "nuscenes":
            dataset_location = "/orion/group/nuscenes"
            dataset = NuScenesDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "newzealand":
            dataset_location = "/orion/group/newzealand"
            dataset = NewZealandDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "oms":
            dataset_location = "/orion/group/OMS_Dataset"
            dataset = OMSDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "otb":
            dataset_location = "/orion/u/yangyou/datasets/OTB"
            dataset = OTBDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "ovis":
            if host=='oriong4.stanford.edu':
                dataset_location = '/scr/aharley/OVIS'
            else:
                dataset_location = "/orion/u/yangyou/datasets/OVIS"
            dataset = OVISDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "pacepoint":
            if host=='oriong9.stanford.edu' and is_training:
                dataset_location = "/scr/aharley/colspa_tracking"
            else:
                dataset_location = "/orion/group/bop/colspa_tracking"
            print('host', host)
            print('dataset_location', dataset_location)
            
            dataset = PacePointDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq,
                crop_size=crop_size,
                use_augs=use_augs,
                chunk=chunk,
                is_training=is_training,
            )
        elif dname == "personpath":
            dataset_location = "/orion/group/personpath/tracking-dataset/dataset/personpath22"
            dataset = PersonPathDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "uav":
            dataset_location = "/orion/group/UAV123"
            dataset = UAVDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "ycb":
            dataset_location = "/orion/group/YCBInEOAT"
            dataset = YCBInEOATDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "totb":
            dataset_location = "/orion/group/TOTB"
            dataset = TOTBDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=None,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "videocube":
            dataset_location = "/orion/group/VideoCube"
            dataset = VideoCubeDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "deeplabcut":
            dataset_location = "/orion/group/deeplabcut"
            dataset = DeepLabCutDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "vost":
            if host=='oriong7.stanford.edu':
                dataset_location = "/scr/aharley/VOST"
            else:
                dataset_location = "/orion/group/VOST"
            dataset = VOSTDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
                chunk=chunk,
            )
        elif dname == "lvvis":
            if host=='oriong7.stanford.edu' and is_training:
                dataset_location = "/scr/aharley/LVVIS/"
            else:
                dataset_location = "/orion/group/LVVIS"
            dataset = LVVISDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "moca":
            dataset_location = "/orion/group/MoCA"
            dataset = MoCADataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "fbms":
            dataset_location = "/orion/group/FBMS"
            dataset = FBMSDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "vsb100":
            dataset_location = "/orion/group/VSB100"
            dataset = VSB100Dataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "uvodense":
            dataset_location = "/orion/group/UVO"
            dataset = UVODenseDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
                chunk=chunk,
            )
        elif dname == "uvosparse":
            dataset_location = "/orion/group/UVO"
            dataset = UVOSparseDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "teyedpt":
            dataset_location = "/orion/group/teyed"
            dataset = TeyedPointDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "teyedmask":
            dataset_location = "/orion/group/teyed"
            dataset = TeyedSegDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "ubody":
            dataset_location = "/orion/group/UBody"
            dataset = UBodyDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "tlp":
            dataset_location = "/orion/group/TLP_V2"
            dataset = TlpDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "badja":
            dataset_location = "/orion/u/aharley/badja2"
            dataset = BadjaDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "visdrone":
            dataset_location = "/orion/group/visdrone"
            dataset = VisDroneDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "templecolor":
            dataset_location = "/orion/group/TempleColor128"
            dataset = TempleColorDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "tknet":
            dataset_location = "/orion/group/tknet"
            dataset = TknetDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "tapvid":
            dataset_location = "/orion/group/tapvid_davis"
            dataset = TapVidDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, 
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "ytvos":
            if host=='oriong7.stanford.edu':
                dataset_location = "/scr/aharley/ytvos"
            else:
                dataset_location = "/orion/group/ytvos"
            dataset = YoutubeVOSDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "zefpt":
            dataset_location = "/orion/u/yangyou/datasets/3DZeF20"
            dataset = Zef20PointDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "zefbox":
            dataset_location = "/orion/u/yangyou/datasets/3DZeF20"
            dataset = Zef20BBoxDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "podpt":
            if host=='oriong9.stanford.edu' or host=='oriong7.stanford.edu' or host=='oriong6.stanford.edu' and is_training:
                dataset_location = "/scr/aharley/point_odyssey_v1.2"
            else:
                dataset_location = "/orion/group/point_odyssey_v1.2"
            print('host', host)
            print('dataset_location', dataset_location)
            dataset = PointOdysseyDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "podmask":
            if host=='oriong9.stanford.edu' or host=='oriong7.stanford.edu' and is_training:
                dataset_location = "/scr/aharley/point_odyssey_v1.2"
            else:
                dataset_location = "/orion/group/point_odyssey_v1.2"
            print('host', host)
            print('dataset_location', dataset_location)
            dataset = PodMaskDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
                chunk=chunk,
            )
        elif dname == "posetrack":
            dataset_location = "/orion/group/posetrack/PoseTrack21"
            dataset = PoseTrackDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "sailvos":
            if host=='oriong2.stanford.edu':
                dataset_location = "/scr/aharley/SAIL-VOS"
            else:
                dataset_location = "/orion/group/SAIL-VOS"
            dataset = SAILVOSDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "sportsmot":
            dataset_location = (
                "/orion/group/sportsmot/sportsmot_publish/dataset/annotations"
            )
            dataset = SportsMot(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "surgicalhands":
            dataset_location = "/orion/group/surgicalhands"
            dataset = SurgicalHandsDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=None,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == 'robotap':
            dataset_location = "/orion/group/robotap"
            dataset = RoboTapDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=None,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
            
        elif dname == "thirdantiuav":
            dataset_location = "/orion/group/3rd_anti_uav_raw"
            dataset = ThirdAntiUAVDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "visor":
            if host=='oriong4.stanford.edu':
                dataset_location = "/scr/aharley/VISOR/VISOR_2022_Dense"
            else:
                dataset_location = "/orion/group/VISOR/VISOR_2022_Dense"
            dataset = VISORDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == 'waymo':
            dataset_location = "/orion/group/waymo_alltrack_coco/"
            dataset = WaymoDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "autotraj":
            dataset_location = "/orion/group/autotraj_mp4"
            dataset = AutotrajDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "interhand":
            # dataset_location = "/orion/group/InterHand"
            dataset_location = "/orion/group/InterHand_30fps"
            dataset = InterHandDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "deepposekit":
            dataset_location = "/orion/group/deepposekit-data/datasets"
            dataset = DeepPoseKitDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "rat7m":
            dataset_location = "/orion/group/Rat7M"
            dataset = Rat7MDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )
        elif dname == "visorsparse":
            dataset_location = "/orion/group/VISOR/VISOR_2022_Sparse"
            dataset = VISORDataset(
                dataset_location=dataset_location,
                S=S, fullseq=fullseq, chunk=chunk,
                crop_size=crop_size,
                use_augs=use_augs,
                is_training=is_training,
            )


        dataloader = DataLoader(
            dataset,
            batch_size=B,
            shuffle=shuffle,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            drop_last=True,
        )
        iterloader = iter(dataloader)

        self.dname = dname
        self.dataset = dataset
        self.dataloader = dataloader
        self.iterloader = iterloader

        if cache_len:
            print("we will cache %d into RAM" % cache_len)
            self.sample_pool = utils.misc.SimplePool(cache_len, version="np")

        print("finished creating %s; len %d" % (dname, len(self.dataset)))

    def get_sample(self, global_step):
        read_new = True  # read something from the dataloder

        if self.cache_len:
            read_new = False

            if len(self.sample_pool) < self.cache_len:
                read_new = True

            if self.cache_freq > 0 and global_step % self.cache_freq == 0:
                read_new = True

        if read_new:
            try:
                sample = next(self.iterloader)
            except StopIteration:
                self.iterloader = iter(self.dataloader)
                sample = next(self.iterloader)

            if self.cache_len:
                self.sample_pool.update([sample])
                print(
                    "cached a new sample into %s pool (len %d)"
                    % (self.dname, len(self.sample_pool))
                )

        if self.cache_len:
            # load from cache
            sample = self.sample_pool.sample()

        return sample



    
