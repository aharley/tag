import time
import numpy as np
import timeit
import saverloader
from nets.tag_base import Tag
import utils.improc
import utils.geom
import random
from utils.basic import print_, print_stats
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from fire import Fire
import sys
import torchvision
import glob
import imageio.v2 as imageio

from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

random.seed(125)
np.random.seed(125)
torch.manual_seed(125)

from torch.utils.data import Dataset, DataLoader

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from prettytable import PrettyTable

from PIL import Image

import torch.autograd.profiler as profiler

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        if param > 100000:
            table.add_row([name, param])
        total_params+=param
    print(table)
    print('total params: %.2f M' % (total_params/1000000.0))
    return total_params

def run_subseq(model, rgbs, prompts, xys_e, vis_e, itr, ara, sw=None):
    B,S,C,H,W = rgbs.shape
    
    
    scope = 'itr_%d' % itr
    
    rgbs_local = []
    prompts_local = []
    offsets_local = xys_e*0
    scales_local = torch.ones_like(xys_e)

    cH, cW = model.module.H, model.module.W
    device = rgbs.device

    assert(B==1)
    for b in range(B):
        prompt_b = torch.sum((prompts[b].reshape(S,-1)>0).float(), dim=1) > 0 # S
        anchor_ind = torch.nonzero(prompt_b)[0].item()
        print('anchor_ind', anchor_ind)

        # to simplify this one, we will ensure a fixed cropsize
        crop_W = cW
        crop_H = cH

        xys = xys_e[b].clone()

        xys[anchor_ind] = xys_e[b,anchor_ind] # on anchor, set to gt

        # clamp and smooth out, to mimic a camera following this traj
        xys[:,0] = torch.clamp(xys[:,0], crop_W//2, W-crop_W//2)
        xys[:,1] = torch.clamp(xys[:,1], crop_H//2, H-crop_H//2)

        for _ in range(3):
            xys_new = xys.clone()
            for ii in range(S):
                if ii==0:
                    xys_new[ii] = xys[ii]*0.5 + xys[ii+1]*0.5
                elif ii==S-1:
                    xys_new[ii] = xys[ii]*0.5 + xys[ii-1]*0.5
                else:
                    xys_new[ii] = xys[ii-1]*0.25 + xys[ii]*0.5 + xys[ii+1]*0.25
            xys = xys_new.clone()
            xys = (xys + xys_new.clone())/2.0

        xys[:,0] = torch.clamp(xys[:,0], crop_W//2, W-crop_W//2)
        xys[:,1] = torch.clamp(xys[:,1], crop_H//2, H-crop_H//2)

        xys = xys.round().long()

        rgbs_crop = []
        prompts_crop = []
        for ii in range(S):
            xy_mid = xys[ii]
            xmid, ymid = xy_mid[0], xy_mid[1]
            xmid = torch.clamp(xmid, crop_W//2, W-crop_W//2)
            ymid = torch.clamp(ymid, crop_H//2, H-crop_H//2)
            x0, x1 = torch.clamp(xmid-crop_W//2, 0), torch.clamp(xmid+crop_W//2, 0, W)
            y0, y1 = torch.clamp(ymid-crop_H//2, 0), torch.clamp(ymid+crop_H//2, 0, H)
            assert(x1-x0==crop_W)
            assert(y1-y0==crop_H)

            offset = torch.stack([x0, y0], dim=0).reshape(2)
            rgbs_crop.append(rgbs[b,ii,:,y0:y1,x0:x1])
            prompts_crop.append(prompts[b,ii,:,y0:y1,x0:x1])
            offsets_local[b,ii] = offset

        rgbs_crop = torch.stack(rgbs_crop, dim=0)
        prompts_crop = torch.stack(prompts_crop, dim=0)

        rgbs_local.append(rgbs_crop)
        prompts_local.append(prompts_crop)

    mC = 3
    rgbs_local = torch.stack(rgbs_local, dim=0).reshape(B,S,3,cH,cW)
    prompts_local = torch.stack(prompts_local, dim=0).reshape(B,S,1,cH,cW)

    prompt_stride = 1
    if prompt_stride > 1:
        prompts_local_ = prompts_local.reshape(B*S,1,cH,cW)
        prompts_local_ = F.max_pool2d(prompts_local_, kernel_size=prompt_stride, stride=prompt_stride)
        prompts_local = prompts_local_.reshape(B,S,1,cH//prompt_stride,cW//prompt_stride)

    xys_e_local, ltrbs_e_local, vis_e_local, all_heats_e_local = model(rgbs_local, prompts_local)
    
    xy_heats_e_local = all_heats_e_local[:,:,0:1]
    obj_heats_e_local = all_heats_e_local[:,:,1:2]
    amo_heats_e_local = all_heats_e_local[:,:,2:3]

    vis_e_local = torch.sigmoid(vis_e_local)

    xy_anchor_bak = xys_e[:,anchor_ind]
    xys_e = xys_e_local / scales_local + offsets_local

    xys_e[:,anchor_ind] = xy_anchor_bak

    vis_e_local[:,anchor_ind] = 1
    
    if sw is not None and sw.save_this:

        sw.summ_oneds('%s/xy_heats_e_local' % scope, xy_heats_e_local.unbind(1))
        sw.summ_oneds('%s/obj_heats_e_local' % scope, obj_heats_e_local.unbind(1))
        sw.summ_oneds('%s/amo_heats_e_local' % scope, amo_heats_e_local.unbind(1))
        prep_rgbs = utils.basic.normalize(rgbs_local[0:1])-0.5
        sw.summ_rgbs('%s/rgbs_local' % scope, prep_rgbs.unbind(1), frame_ids=list(range(S)))
        sw.summ_traj2ds_on_rgb('%s/trajs_e_on_g' % scope,
                               xys_e_local[0:1].unsqueeze(2), prep_rgbs[:,anchor_ind], cmap='spring',
                               frame_str='%d' % ara[anchor_ind], linewidth=2)
        mask_stride = 2
        if mask_stride > 1:
            xy_heats_ = F.interpolate(utils.basic.normalize(xy_heats_e_local)[0], scale_factor=mask_stride, mode='bilinear').unsqueeze(0)
        else:
            xy_heats_ = xy_heats_e_local

        rgbs_vis = sw.summ_traj2ds_on_rgbs('', xys_e_local[0:1].unsqueeze(2), prep_rgbs, cmap='spring', visibs=vis_e_local[0:1].unsqueeze(2), linewidth=1, only_return=True)
        xys_vis = sw.summ_oneds('', xy_heats_.unbind(1), frame_ids=vis_e_local.reshape(S), only_return=True)
        xys_vis = sw.summ_traj2ds_on_rgbs('', xys_e_local[0:1].unsqueeze(2), utils.improc.preprocess_color(xys_vis), cmap='spring', visibs=vis_e_local[0:1].unsqueeze(2), linewidth=1, only_return=True)
        xys_vis = torch.cat([rgbs_vis, xys_vis], dim=-2)
        sw.summ_rgbs('%s/rgbs_and_xys' % scope, xys_vis.unbind(1))

        prep_rgbs = utils.basic.normalize(rgbs[0:1])-0.5
        sw.summ_traj2ds_on_rgb('%s/full_vis' % scope, xys_e[0:1].unsqueeze(2), prep_rgbs[:,0], cmap='spring', linewidth=2)
        
        prep_rgbs = utils.basic.normalize(rgbs_local[0:1])-0.5
        segs_ge = utils.basic.normalize(xy_heats_e_local)[0:1]
        if mask_stride > 1:
            segs_ge = F.interpolate(segs_ge[0], scale_factor=mask_stride, mode='nearest').unsqueeze(0)

        slist = [0, S//4, S//2, 3*S//4, S-1]
        nearest = np.argmin(np.abs(np.array(slist) - anchor_ind))
        slist[nearest] = anchor_ind
        s0, s1, s2, s3, s4 = slist

        seg_vis0 = sw.summ_oned('', segs_ge[0:1,s0], norm=False, only_return=True, frame_id=vis_e_local[0,s0])
        seg_vis1 = sw.summ_oned('', segs_ge[0:1,s1], norm=False, only_return=True, frame_id=vis_e_local[0,s1])
        seg_vis2 = sw.summ_oned('', segs_ge[0:1,s2], norm=False, only_return=True, frame_id=vis_e_local[0,s2])
        seg_vis3 = sw.summ_oned('', segs_ge[0:1,s3], norm=False, only_return=True, frame_id=vis_e_local[0,s3])
        seg_visE = sw.summ_oned('', segs_ge[0:1,s4], norm=False, only_return=True, frame_id=vis_e_local[0,s4])

        strs = []
        for si in slist:
            st = ""
            if si==anchor_ind:
                st = "(A)"
            strs.append(st)

        rgb_vis0 = sw.summ_rgb('', prep_rgbs[0:1,s0], only_return=True, frame_id=slist[0], frame_str=strs[0])
        rgb_vis1 = sw.summ_rgb('', prep_rgbs[0:1,s1], only_return=True, frame_id=slist[1], frame_str=strs[1])
        rgb_vis2 = sw.summ_rgb('', prep_rgbs[0:1,s2], only_return=True, frame_id=slist[2], frame_str=strs[2])
        rgb_vis3 = sw.summ_rgb('', prep_rgbs[0:1,s3], only_return=True, frame_id=slist[3], frame_str=strs[3])
        rgb_visE = sw.summ_rgb('', prep_rgbs[0:1,s4], only_return=True, frame_id=slist[4], frame_str=strs[4])

        seg0_vis = torch.cat([rgb_vis0, seg_vis0], dim=-1)
        seg1_vis = torch.cat([rgb_vis1, seg_vis1], dim=-1)
        seg2_vis = torch.cat([rgb_vis2, seg_vis2], dim=-1)
        seg3_vis = torch.cat([rgb_vis3, seg_vis3], dim=-1)
        segE_vis = torch.cat([rgb_visE, seg_visE], dim=-1)

        sw.summ_rgb('%s/segs_0123E' % scope, torch.cat([seg0_vis, seg1_vis, seg2_vis, seg3_vis, segE_vis], dim=-2))#, frame_id=total_loss[itr][0].item())

        masks0_ = F.interpolate(all_heats_e_local[0,:,0:1], scale_factor=mask_stride, mode='bilinear').unsqueeze(0)
        masks1_ = F.interpolate(all_heats_e_local[0,:,1:2], scale_factor=mask_stride, mode='bilinear').unsqueeze(0)
        masks2_ = F.interpolate(all_heats_e_local[0,:,2:3], scale_factor=mask_stride, mode='bilinear').unsqueeze(0)
        rgbs_vis = sw.summ_traj2ds_on_rgbs('', xys_e_local[0:1].unsqueeze(2), prep_rgbs[0:1], visibs=vis_e_local[0:1].unsqueeze(2), cmap='spring', linewidth=2, only_return=True)
        seg0_vis = sw.summ_oneds('', utils.basic.normalize(masks0_).unbind(1), norm=False, frame_ids=vis_e_local[0], only_return=True)
        seg1_vis = sw.summ_oneds('', torch.sigmoid(masks1_).unbind(1), norm=False, only_return=True) 
        seg2_vis = sw.summ_oneds('', torch.sigmoid(masks2_).unbind(1), norm=False, only_return=True)
        segs_vis = torch.cat([rgbs_vis, seg0_vis, seg1_vis, seg2_vis], dim=-2)
        sw.summ_rgbs('%s/rgbs_and_segs' % scope, segs_vis.unbind(1))

    return xys_e, vis_e_local


def run_model(model, rgbs, S, device, rank=0, sw=None):
    device = 'cuda:%d' % rank

    rgbs = rgbs.to(device).float() # B,T,C,H,W

    rgbs_flip = torch.flip(rgbs, [1])
    rgbs = torch.cat([rgbs, rgbs_flip], axis=1)

    rgbs_flip = torch.flip(rgbs, [1])
    rgbs = torch.cat([rgbs, rgbs_flip], axis=1)

    print('rgbs', rgbs.shape)

    B,T,C,H,W = rgbs.shape
    
    if S > T:
        S = T

    xys_g = torch.ones((B,T,2), dtype=torch.float32, device=device)
    xys_g[:,:,0] = 128
    xys_g[:,:,1] = 126

    print_stats('rgbs', rgbs)
    
    mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).reshape(1,1,3,1,1)
    std = torch.as_tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).reshape(1,1,3,1,1)
    rgbs = rgbs / 255.0
    rgbs = (rgbs - mean)/std

    anchor_ind = 0

    prompt = torch.zeros_like(rgbs[:,0,0:1]) # B,T,1,H,W

    x, y = 128, 128 # prompt index
    prompt[:,:,y,x] = 1

    prompts_g = torch.zeros_like(rgbs[:,:,0:1])
    prompts_g[0:1,anchor_ind] = prompt

    # anchor with zero-vel
    xys_e = xys_g.clone()
    
    if sw is not None and sw.save_this:
        prep_rgbs = utils.basic.normalize(rgbs[0:1])-0.5
        sw.summ_rgbs('input/rgbs', prep_rgbs.unbind(1))

        # prep_rgbs = utils.basic.normalize(rgbs_local[0:1])-0.5
        sw.summ_traj2ds_on_rgb('input/xys_g', xys_g[0:1].unsqueeze(2), prep_rgbs[:,anchor_ind], cmap='winter', linewidth=2)
        sw.summ_oned('input/prompt', prompt, norm=False)
        
    
    cH, cW = model.module.H, model.module.W
    
    # anchor with zero-vel on anchor
    xys_e = 0*xys_g
    for b in range(B):
        prompt_b = torch.sum((prompts_g[b].reshape(T,-1)>0).float(), dim=1) > 0 # T
        anchor_ind = torch.nonzero(prompt_b)[0].item()
        # print('anchor_ind', anchor_ind)
        xys_e[b,:] = xys_g[b,anchor_ind:anchor_ind+1]

    # add a random alternate trajectory, to help push us off center
    alt_xys = np.random.randint(-cH//8, cH//8, (S,2))
    for _ in range(3): # smooth out
        for ii in range(1,S-1):
            alt_xys[ii] = (alt_xys[ii-1] + alt_xys[ii] + alt_xys[ii+1])/3.0
    xys_e += torch.from_numpy(alt_xys.reshape(1,S,2)).to(device).float()

    anchors_available = []
    anchors_available.append(anchor_ind)
    
    anchor_ind_bak = anchor_ind
    xy_bak = xys_e[:,anchor_ind]
    
    xys_e_soft = xys_e.clone()

    vis_e_soft = torch.zeros_like(xys_e[:,:,0])
    vis_e_soft[:,anchor_ind] = 1

    visits_e = torch.zeros_like(xys_e[:,:,0])

    full_vis = []
    
    itr_count = 0
    vis_thr = 0.9
    sol_thr = 1.0
    max_itr = 3


    for itr in range(max_itr): # number of passes over the video
        print('='*10)
        
        ara = np.arange(S)
        vis_subseq = vis_e_soft[0,ara]
        local_anchor = torch.argmax(vis_subseq).item()
        anchor_ind = ara[local_anchor]
        
        if vis_subseq[local_anchor] < vis_thr:
            print('anchor (%d) no good; replacing it with something better' % anchor_ind)

            anchors_available = torch.where(vis_e_soft[0] > vis_thr)[0].cpu().numpy()
            K = len(anchors_available)
            print('anchors_available', anchors_available)
            
            ara_ = ara.reshape(S,1)
            anc_ = anchors_available.reshape(1,K)
            dist = np.abs(ara_-anc_) # S,K
            dist1 = np.min(dist, axis=1) # for each ind, this is the dist to the nearest anchor
            ind1 = np.argmin(dist1) # this is the ind we want to replace with a nearby anchor

            ind2 = np.argmin(dist[ind1]) # this is the anchor ind
            anchor_ind = anchors_available[ind2]
            print('replacing %d with %d' % (ara[ind1], anchor_ind))
            
            ara[ind1] = anchor_ind
            
            ara = np.sort(ara)
            print('new ara (anchor %d)' % anchor_ind, ara)
            
        if torch.min(vis_subseq) > sol_thr:
            print('all timesteps solved; skipping this')
            continue

        print('anchor', anchor_ind)

        xy = xys_e_soft[0,anchor_ind].cpu().round().long().numpy() # 2
        x, y = xy[0], xy[1]
        prompts = torch.zeros((1,T,1,H,W), dtype=torch.float32, device=device)
        prompts[:,anchor_ind,:,y,x] = 1
        prompts[0] = utils.improc.dilate2d(prompts[0], times=3).unsqueeze(0)


        print('set prompt on frame %d' % anchor_ind)
        print('ara', ara, len(ara))
        rgbs_subseq = rgbs[:,ara]
        prompts_subseq = prompts[:,ara]
        xys_e_subseq = xys_e_soft[:,ara]
        vis_e_subseq = vis_e_soft[:,ara]
        
        xys_e_subseq, vis_e_subseq = run_subseq(model, rgbs_subseq, prompts_subseq, xys_e_subseq, vis_e_subseq, itr, ara, sw=sw)

        vis_e_subseq = torch.clamp(vis_e_subseq, 0, vis_e_soft[0,anchor_ind])
        visits_subseq = visits_e[:,ara]
        take_new = 0.9**visits_subseq
        
        visits_e[:,ara] += 1
        xys_e_soft[:,ara] = take_new.unsqueeze(2)*xys_e_subseq + (1-take_new.unsqueeze(2))*xys_e_soft[:,ara]
        # use some momentum on vis, since we expect the xy to lock AFTER vis becomes high
        vis_e_soft[:,ara] = 0.1*vis_e_soft[:,ara] + 0.9*vis_e_subseq

        np_valids = vis_e_soft[0].cpu().numpy()
        np_xys = xys_e_soft[0].cpu().numpy()
        invalid_idx = np.where(np_valids==0)[0]
        valid_idx = np.where(np_valids>0)[0]
        for idx in invalid_idx:
            nearest = valid_idx[np.argmin(np.abs(valid_idx - idx))]
            np_xys[idx] = np_xys[nearest]
            xys_e_soft[:,idx] = xys_e_soft[:,nearest]

        xys_e_soft[:,anchor_ind_bak] = xy_bak
        vis_e_soft[:,anchor_ind_bak] = 1
        anchors_available = torch.where(vis_e_soft[0] > vis_thr)[0].cpu().numpy()

        if sw is not None and sw.save_this:
            scope = 'itr_%d' % itr
            full_vis.append(sw.summ_traj2ds_on_rgb('%s/fullseq_trajs_e_on_g' % scope, xys_e_soft[0:1].unsqueeze(2), prep_rgbs[:,0], cmap='spring', linewidth=2, only_return=True))
        itr_count += 1
    
    if sw is not None and sw.save_this:
        sw.summ_rgbs('full/rgbs', prep_rgbs.unbind(1))
        sw.summ_traj2ds_on_rgb('full/trajs_e', xys_e[0:1].unsqueeze(2), prep_rgbs[:,anchor_ind], cmap='spring', linewidth=2)
        sw.summ_rgbs('full/fullseq_traj_anim', full_vis)
        sw.summ_traj2ds_on_rgbs('full/trajs_on_rgbs', xys_e_soft[0:1].unsqueeze(2), prep_rgbs, visibs=vis_e_soft[0:1].unsqueeze(2), cmap='spring', linewidth=2, frame_ids=vis_e_soft[0])
        
    return None
    

def main(
        dname=None,
        exp_name='debug',
        data_version='cjS64', 
        target_shape=(224,224),
        B=1, # batchsize
        S=128, # data seqlen
        S_model=64, 
        K=8, # inference iters
        max_iters=100,
        log_freq=1,
        num_workers=8,
        log_dir='./logs_test_on_cups',
        init_dir='',
        ignore_load=None,
        device_ids=[0],
        quick=False,
        is_training=False,
):
    device = 'cuda:%d' % device_ids[0]

    exp_name = 'tb00' # copy from base repo

    init_dir = '32h_64_1e-6_cjS64_24_cjS24_ec84_3857'
    
    dsets = [dname]
    if is_training:
        dstring = '%s_trainA' % data_version
    else:
        dstring = '%s_valA' % data_version
        
    import socket
    host = socket.gethostname()

    aws = 'ip-10-237-6-12'
    ckpt_dir = './checkpoints'
    data_dir = './tag_export'
        
    ## autogen a descriptive name
    B_ = B
    model_name = "%d" % (B_)
    model_name += "_%d" % S
    model_name += "_%d" % K
    model_name += "_%s" % dname
    model_name += "_%s" % exp_name
    import datetime
    model_date = datetime.datetime.now().strftime('%H%M%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)

    if init_dir:
        init_dir = '%s/%s' % (ckpt_dir, init_dir)
    save_dir = '%s/%s' % (ckpt_dir, model_name)


    if is_training:
        writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)
    else:
        writer_t = SummaryWriter(log_dir + '/' + model_name + '/v', max_queue=10, flush_secs=60)

    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)


    sS, cH, cW = 64,target_shape[0],target_shape[1]
    model = Tag(sS,cH,cW,tstride=8,sstride=32).to(device)

    model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    count_parameters(model)

    _ = saverloader.load(init_dir, model.module, ignore_load=ignore_load)
    global_step = 0
    print('loaded model')
    
    parameters = list(model.parameters())
    model.eval()

    filenames = glob.glob('./demo_images/*.jpg')
    filenames = sorted(filenames)
    S_local = 64
    ara = np.linspace(0, len(filenames)-1, S_local, dtype=np.int32)
    filenames = [filenames[ar] for ar in ara]
    print('filenames', filenames)
    max_iters = np.ceil(len(filenames)/S) # run each unique subsequence
    
    while global_step < max_iters:
        global_step += 1

        iter_start_time = time.time()
        read_start_time = time.time()

        sw_t = utils.improc.Summ_writer(
            writer=writer_t,
            global_step=global_step,
            log_freq=log_freq,
            fps=10,
            scalar_freq=1,
            just_gif=True)

        try:
            rgbs = []
            for s in range(min(S, len(filenames))):
                fn = filenames[(global_step-1)*S+s]
                if s==0:
                    print('start frame', fn)
                im = imageio.imread(fn)
                im = im.astype(np.uint8)
                rgbs.append(torch.from_numpy(im).permute(2,0,1))
            rgbs = torch.stack(rgbs, dim=0).unsqueeze(0) # 1, S, C, H, W
            print('read rgbs', rgbs.shape)

            read_time = time.time()-read_start_time
            iter_start_time = time.time()

            with torch.no_grad():
                trajs_e = run_model(model, rgbs, S_model, device, sw=sw_t)

            iter_time = time.time()-iter_start_time
            print('%s; step %06d/%d; rtime %.2f; itime %.2f' % (
                model_name, global_step, max_iters, read_time, iter_time))
        except FileNotFoundError as e:
            print('error', e)

        print('done')
    print('done')
        
    writer_t.close()
            

if __name__ == '__main__':
    Fire(main)
