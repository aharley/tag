import torch
import numpy as np
import math
import torch.nn.functional as F
import utils.basic
# from datasets.dataset import mask2bbox

# prevent circular imports
def mask2bbox(mask):
    if mask.ndim == 3:
        mask = mask[..., 0]
    ys, xs = np.where(mask > 0.4)
    if ys.size == 0 or xs.size==0:
        return np.array((0, 0, 0, 0), dtype=int)
    lt = np.array([np.min(xs), np.min(ys)])
    rb = np.array([np.max(xs), np.max(ys)]) + 1
    return np.concatenate([lt, rb])

# def get_stark_2d_embedding(H, W, C=64, device='cuda:0', temperature=10000, normalize=True):
#     scale = 2*math.pi
#     mask = torch.ones((1,H,W), dtype=torch.float32, device=device)
#     y_embed = mask.cumsum(1, dtype=torch.float32)  # cumulative sum along axis 1 (h axis) --> (b, h, w)
#     x_embed = mask.cumsum(2, dtype=torch.float32)  # cumulative sum along axis 2 (w axis) --> (b, h, w)
#     if normalize:
#         eps = 1e-6
#         y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale  # 2pi * (y / sigma(y))
#         x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale  # 2pi * (x / sigma(x))

#     dim_t = torch.arange(C, dtype=torch.float32, device=device)  # (0,1,2,...,d/2)
#     dim_t = temperature ** (2 * (dim_t // 2) / C)

#     pos_x = x_embed[:, :, :, None] / dim_t # (b,h,w,d/2)
#     pos_y = y_embed[:, :, :, None] / dim_t # (b,h,w,d/2)
#     pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3) # (b,h,w,d/2)
#     pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3) # (b,h,w,d/2)
#     pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # (b,h,w,d)
#     return pos

# def get_1d_embedding(x, C, cat_coords=False):
#     B, N, D = x.shape
#     assert(D==1)

#     div_term = (torch.arange(0, C, 2, device=x.device, dtype=torch.float32) * (10000.0 / C)).reshape(1, 1, int(C/2))
    
#     pe_x = torch.zeros(B, N, C, device=x.device, dtype=torch.float32)
    
#     pe_x[:, :, 0::2] = torch.sin(x * div_term)
#     pe_x[:, :, 1::2] = torch.cos(x * div_term)
    
#     if cat_coords:
#         pe_x = torch.cat([pe, x], dim=2) # B,N,C*2+2
#     return pe_x

# def posemb_sincos_2d(h, w, dim, temperature=10000, dtype=torch.float32, device='cuda:0'):

#     y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
#     assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
#     omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
#     omega = 1. / (temperature ** omega)

#     y = y.flatten()[:, None] * omega[None, :]
#     x = x.flatten()[:, None] * omega[None, :] 
#     pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1) # B,C,H,W
#     return pe.type(dtype)

def get_2d_embedding(xy, C, cat_coords=False):
    B, N, D = xy.shape
    assert(D==2)

    x = xy[:,:,0:1]
    y = xy[:,:,1:2]
    div_term = (torch.arange(0, C, 2, device=xy.device, dtype=torch.float32) * (10000.0 / C)).reshape(1, 1, int(C/2))
    
    pe_x = torch.zeros(B, N, C, device=xy.device, dtype=torch.float32)
    pe_y = torch.zeros(B, N, C, device=xy.device, dtype=torch.float32)
    
    pe_x[:, :, 0::2] = torch.sin(x * div_term)
    pe_x[:, :, 1::2] = torch.cos(x * div_term)
    
    pe_y[:, :, 0::2] = torch.sin(y * div_term)
    pe_y[:, :, 1::2] = torch.cos(y * div_term)
    
    pe = torch.cat([pe_x, pe_y], dim=2) # B,N,C*2
    if cat_coords:
        pe = torch.cat([pe, xy], dim=2) # B,N,C*2+2
    return pe

def get_3d_embedding(xyz, C, cat_coords=False):
    B, N, D = xyz.shape
    assert(D==3)

    x = xyz[:,:,0:1]
    y = xyz[:,:,1:2]
    z = xyz[:,:,2:3]
    div_term = (torch.arange(0, C, 2, device=xyz.device, dtype=torch.float32) * (10000.0 / C)).reshape(1, 1, int(C/2))
    
    pe_x = torch.zeros(B, N, C, device=xyz.device, dtype=torch.float32)
    pe_y = torch.zeros(B, N, C, device=xyz.device, dtype=torch.float32)
    pe_z = torch.zeros(B, N, C, device=xyz.device, dtype=torch.float32)
    
    pe_x[:, :, 0::2] = torch.sin(x * div_term)
    pe_x[:, :, 1::2] = torch.cos(x * div_term)
    
    pe_y[:, :, 0::2] = torch.sin(y * div_term)
    pe_y[:, :, 1::2] = torch.cos(y * div_term)
    
    pe_z[:, :, 0::2] = torch.sin(z * div_term)
    pe_z[:, :, 1::2] = torch.cos(z * div_term)
    
    pe = torch.cat([pe_x, pe_y, pe_z], dim=2) # B, N, C*3
    if cat_coords:
        pe = torch.cat([pe, xyz], dim=2) # B, N, C*3+3
    return pe
    
class SimplePool():
    def __init__(self, pool_size, version='pt'):
        self.pool_size = pool_size
        self.version = version
        self.items = []
        
        if not (version=='pt' or version=='np'):
            print('version = %s; please choose pt or np')
            assert(False) # please choose pt or np
            
    def __len__(self):
        return len(self.items)
    
    def mean(self, min_size=1):
        if min_size=='half':
            pool_size_thresh = self.pool_size/2
        else:
            pool_size_thresh = min_size
            
        if self.version=='np':
            if len(self.items) >= pool_size_thresh:
                return np.sum(self.items)/float(len(self.items))
            else:
                return np.nan
        if self.version=='pt':
            if len(self.items) >= pool_size_thresh:
                return torch.sum(self.items)/float(len(self.items))
            else:
                return torch.from_numpy(np.nan)
    
    def sample(self, with_replacement=True):
        idx = np.random.randint(len(self.items))
        if with_replacement:
            return self.items[idx]
        else:
            return self.items.pop(idx)
    
    def fetch(self, num=None):
        if self.version=='pt':
            item_array = torch.stack(self.items)
        elif self.version=='np':
            item_array = np.stack(self.items)
        if num is not None:
            # there better be some items
            assert(len(self.items) >= num)
                
            # if there are not that many elements just return however many there are
            if len(self.items) < num:
                return item_array
            else:
                idxs = np.random.randint(len(self.items), size=num)
                return item_array[idxs]
        else:
            return item_array
            
    def is_full(self):
        full = len(self.items)==self.pool_size
        return full
    
    def empty(self):
        self.items = []
            
    def update(self, items):
        for item in items:
            if len(self.items) < self.pool_size:
                # the pool is not full, so let's add this in
                self.items.append(item)
            else:
                # the pool is full
                # pop from the front
                self.items.pop(0)
                # add to the back
                self.items.append(item)
        return self.items


class SimpleHeap():
    def __init__(self, pool_size, version='pt'):
        self.pool_size = pool_size
        self.version = version
        self.items = []
        self.vals = []
        
        if not (version=='pt' or version=='np'):
            print('version = %s; please choose pt or np')
            assert(False) # please choose pt or np
            
    def __len__(self):
        return len(self.items)
    
    def sample(self, random=True, with_replacement=True, semirandom=False):
        vals_arr = np.stack(self.vals)
        if random:
            ind = np.random.randint(len(self.items))
        else:
            if semirandom and len(vals_arr)>1:
                # choose from the harder half
                inds = np.argsort(vals_arr) # ascending
                inds = inds[len(vals_arr)//2:]
                ind = np.random.choice(inds)
            else:
                # find the most valuable element
                ind = np.argmax(vals_arr)

            
        if with_replacement:
            return self.items[ind]
        else:
            item = self.items.pop(ind)
            val = self.vals.pop(ind)
            return item
    
    def fetch(self, num=None):
        if self.version=='pt':
            item_array = torch.stack(self.items)
        elif self.version=='np':
            item_array = np.stack(self.items)
        if num is not None:
            # there better be some items
            assert(len(self.items) >= num)
                
            # if there are not that many elements just return however many there are
            if len(self.items) < num:
                return item_array
            else:
                idxs = np.random.randint(len(self.items), size=num)
                return item_array[idxs]
        else:
            return item_array
            
    def is_full(self):
        full = len(self.items)==self.pool_size
        return full
    
    def empty(self):
        self.items = []
            
    def update(self, vals, items):
        for val,item in zip(vals, items):
            if len(self.items) < self.pool_size:
                # the pool is not full, so let's add this in
                self.items.append(item)
                self.vals.append(val)
            else:
                # the pool is full
                # find our least-valuable element
                # and see if we should replace it
                vals_arr = np.stack(self.vals)
                ind = np.argmin(vals_arr)
                if vals_arr[ind] < val:
                    # pop the min
                    self.items.pop(ind)
                    self.vals.pop(ind)

                    # add to the back
                    self.items.append(item)
                    self.vals.append(val)
        return self.items
    

def farthest_point_sample(xyz, npoint, include_ends=False, deterministic=False):
    """
    Input:
        xyz: pointcloud data, [B, N, C], where C is probably 3
        npoint: number of samples
    Return:
        inds: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    xyz = xyz.float()
    inds = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones((B, N), dtype=torch.float32, device=device) * 1e10
    if deterministic:
        farthest = torch.randint(0, 1, (B,), dtype=torch.long, device=device)
    else:
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        if include_ends:
            if i==0:
                farthest = 0
            elif i==1:
                farthest = N-1
        inds[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

        if npoint > N:
            # if we need more samples, make them random
            distance += torch.randn_like(distance)
    return inds

def farthest_point_sample_py(xyz, npoint, deterministic=False):
    N,C = xyz.shape
    inds = np.zeros(npoint, dtype=np.int32)
    distance = np.ones(N) * 1e10
    if deterministic:
        farthest = 0
    else:
        farthest = np.random.randint(0, N, dtype=np.int32)
    for i in range(npoint):
        inds[i] = farthest
        centroid = xyz[farthest, :].reshape(1,C)
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
        if npoint > N:
            # if we need more samples, make them random
            distance += np.random.randn(*distance.shape)
    return inds
    
def balanced_ce_loss(pred, gt, pos_weight=0.5, valid=None, dim=None, return_both=False, use_halfmask=False, H=64, W=64):
    # pred and gt are the same shape
    for (a,b) in zip(pred.size(), gt.size()):
        if not a==b:
            print('mismatch: pred, gt', pred.shape, gt.shape)
        assert(a==b) # some shape mismatch!
    if valid is not None:
        for (a,b) in zip(pred.size(), valid.size()):
            assert(a==b) # some shape mismatch!
    else:
        valid = torch.ones_like(gt)

    pos = (gt > 0.95).float()
    if use_halfmask:
        pos_wide = (gt >= 0.5).float()
        halfmask = (gt == 0.5).float()
    else:
        pos_wide = pos

    neg = (gt < 0.05).float()

    label = pos_wide*2.0 - 1.0
    a = -label * pred
    b = F.relu(a)
    loss = b + torch.log(torch.exp(-b)+torch.exp(a-b))
    
    pos_loss = utils.basic.reduce_masked_mean(loss, pos*valid, dim=dim)
    neg_loss = utils.basic.reduce_masked_mean(loss, neg*valid, dim=dim)

    if use_halfmask:
        # here we will find the pixels which are already leaning positive,
        # and encourage them to be more positive
        B = loss.shape[0]
        loss_ = loss.reshape(B,-1)
        mask_ = halfmask.reshape(B,-1) * valid.reshape(B,-1)

        # to avoid the issue where spikes become spikier,
        # we will only apply this loss on batch els where we predicted zero positives
        pred_sig_ = torch.sigmoid(pred).reshape(B,-1)
        no_pred_ = torch.max(pred_sig_.round(), axis=1)[0] < 1 # B
        # and only on batch els where we have negatives available
        have_neg_ = torch.sum(neg, dim=1)>0 # B

        loss_ = loss_[no_pred_ & have_neg_] # N,H*W
        mask_ = mask_[no_pred_ & have_neg_] # N,H*W
        N = loss_.shape[0]

        if N > 0:
            # we want: 
            # in the neg pixels, 
            # set them to the max loss of the pos pixels,
            # so that they do not contribute to the min
            loss__ = loss_.reshape(-1)
            mask__ = mask_.reshape(-1)
            if torch.sum(mask__)>0:
                # print('loss_', loss_.shape, 'mask_', mask_.shape, 'loss__', loss__.shape, 'mask__', mask__.shape)
                mloss__ = loss__.detach()
                mloss__[mask__==0] = torch.max(loss__[mask__==1])
                mloss_ = mloss__.reshape(N,H*W)

                # now, in each batch el, take a tiny region around the argmin, so we can boost this region
                minloss_mask_ = torch.zeros_like(mloss_).scatter(1,mloss_.argmin(1,True),value=1)
                minloss_mask_ = utils.improc.dilate2d(minloss_mask_.view(N,1,H,W), times=3).reshape(N,H*W)

                loss__ = loss_.reshape(-1)
                minloss_mask__ = minloss_mask_.reshape(-1)
                half_loss = loss__[minloss_mask__>0].mean()

                # print('N', N, 'half_loss', half_loss)
                pos_loss = pos_loss + half_loss

        # if False:
        #     min_pos = 8

        #     # only apply the loss when we have some negatives available,
        #     # otherwise it's a whole "ignore" frame, which may mean
        #     # we are unsure if the target is even there
        #     if torch.sum(mask__==0) > 0: # negatives available
        #         # only apply the loss when the halfmask is larger area than
        #         # min_pos (the number of pixels we want to boost),
        #         # so that indexing will work 
        #         if torch.all(torch.sum(mask_==1, dim=1) >= min_pos): # topk indexing will work
        #             # in the pixels we will not use,
        #             # set them to the max of the pixels we may use,
        #             # so that they do not contribute to the min
        #             loss__[mask__==0] = torch.max(loss__[mask__==1])
        #             loss_ = loss__.reshape(B,-1)

        #             half_loss = torch.mean(torch.topk(loss_, min_pos, dim=1, largest=False)[0], dim=1) # B

        #             have_neg = (torch.sum(neg, dim=1)>0).float() # B
        #             pos_loss = pos_loss + half_loss*have_neg

                


            
        
        # half_loss = []
        # for b in range(B):
        #     loss_b = loss_[b]
        #     mask_b = mask_[b]
        #     if torch.sum(mask_b):
        #         inds = torch.nonzero(mask_b).reshape(-1)
        #         half_loss.append(torch.min(loss_b[inds]))
        # if len(half_loss):
        #     # # half_loss_ = half_loss.reshape(-1)
        #     # half_loss = torch.min(half_loss, dim=1)[0] # B
        #     # half_loss = torch.mean(torch.topk(half_loss, 4, dim=1, largest=False)[0], dim=1) # B
        #     pos_loss = pos_loss + torch.stack(half_loss).mean()

    if return_both:
        return pos_loss, neg_loss
    balanced_loss = pos_weight*pos_loss + (1-pos_weight)*neg_loss

    return balanced_loss

def dice_loss(pred, gt):
    # gt has ignores at 0.5
    # pred and gt are the same shape
    for (a,b) in zip(pred.size(), gt.size()):
        assert(a==b) # some shape mismatch!
    
    prob = pred.sigmoid()

    # flatten everything except batch
    prob = prob.flatten(1)
    gt = gt.flatten(1)
    
    pos = (gt > 0.95).float()
    neg = (gt < 0.05).float()
    valid = (pos+neg).float().clamp(0,1)
    
    numerator = 2 * (prob * pos * valid).sum(1)
    denominator = (prob*valid).sum(1) + (pos*valid).sum(1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss

def sigmoid_focal_loss(pred, gt, alpha=0.25, gamma=2):#, use_halfmask=False):
    # gt has ignores at 0.5
    # pred and gt are the same shape
    for (a,b) in zip(pred.size(), gt.size()):
        assert(a==b) # some shape mismatch!

    # flatten everything except batch
    pred = pred.flatten(1)
    gt = gt.flatten(1)
        
    pos = (gt > 0.95).float()
    neg = (gt < 0.05).float()
    # if use_halfmask:
    #     pos_wide = (gt >= 0.5).float()
    #     halfmask = (gt == 0.5).float()
    # else:
    #     pos_wide = pos
    valid = (pos+neg).float().clamp(0,1)
    
    prob = pred.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(pred, pos, reduction="none")
    p_t = prob * pos + (1 - prob) * (1 - pos)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * pos + (1 - alpha) * (1 - pos)
        loss = alpha_t * loss

    loss = (loss*valid).sum(1) / (1 + valid.sum(1))
    return loss 
    
    if use_halfmask:
        # here we will find the pixels which are already leaning positive,
        # and encourage them to be more positive
        B = loss.shape[0]
        loss_ = loss.reshape(B,-1)
        mask_ = halfmask.reshape(B,-1) * valid.reshape(B,-1)

        # to avoid the issue where spikes become spikier,
        # we will only apply this loss on batch els where we predicted zero positives
        pred_sig_ = torch.sigmoid(pred).reshape(B,-1)
        no_pred_ = torch.max(pred_sig_.round(), axis=1)[0] < 1 # B
        # and only on batch els where we have negatives available
        have_neg_ = torch.sum(neg, dim=1)>0 # B

        loss_ = loss_[no_pred_ & have_neg_] # N,H*W
        mask_ = mask_[no_pred_ & have_neg_] # N,H*W
        N = loss_.shape[0]

        if N > 0:
            # we want: 
            # in the neg pixels, 
            # set them to the max loss of the pos pixels,
            # so that they do not contribute to the min
            loss__ = loss_.reshape(-1)
            mask__ = mask_.reshape(-1)
            if torch.sum(mask__)>0:
                # print('loss_', loss_.shape, 'mask_', mask_.shape, 'loss__', loss__.shape, 'mask__', mask__.shape)
                mloss__ = loss__.detach()
                mloss__[mask__==0] = torch.max(loss__[mask__==1])
                mloss_ = mloss__.reshape(N,H*W)

                # now, in each batch el, take a tiny region around the argmin, so we can boost this region
                minloss_mask_ = torch.zeros_like(mloss_).scatter(1,mloss_.argmin(1,True),value=1)
                minloss_mask_ = utils.improc.dilate2d(minloss_mask_.view(N,1,H,W), times=3).reshape(N,H*W)

                loss__ = loss_.reshape(-1)
                minloss_mask__ = minloss_mask_.reshape(-1)
                half_loss = loss__[minloss_mask__>0].mean()

                # print('N', N, 'half_loss', half_loss)
                # pos_loss = pos_loss + half_loss
                loss = pos_loss + half_loss
        
    return loss

# def dice_loss(inputs, targets, normalizer=1):
#     inputs = inputs.sigmoid()
#     inputs = inputs.flatten(1)
#     numerator = 2 * (inputs * targets).sum(1)
#     denominator = inputs.sum(-1) + targets.sum(-1)
#     loss = 1 - (numerator + 1) / (denominator + 1)
#     return loss.sum() / normalizer

# def sigmoid_focal_loss(inputs, targets, normalizer=1, alpha=0.25, gamma=2):
#     prob = inputs.sigmoid()
#     ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
#     p_t = prob * targets + (1 - prob) * (1 - targets)
#     loss = ce_loss * ((1 - p_t) ** gamma)

#     if alpha >= 0:
#         alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
#         loss = alpha_t * loss

#     return loss.mean(1).sum() / normalizer
                                        
def data_replace_with_nearest(xys, valids):
    # replace invalid xys with nearby ones
    invalid_idx = np.where(valids==0)[0]
    valid_idx = np.where(valids==1)[0]
    for idx in invalid_idx:
        nearest = valid_idx[np.argmin(np.abs(valid_idx - idx))]
        xys[idx] = xys[nearest]
    return xys

def data_get_traj_from_masks(masks):
    if masks.ndim==4:
        masks = masks[...,0]
    S, H, W = masks.shape
    masks = (masks > 0.1).astype(np.float32)
    fills = np.zeros((S))
    xy_means = np.zeros((S,2))
    xy_rands = np.zeros((S,2))
    valids = np.zeros((S))
    for si, mask in enumerate(masks):
        if np.sum(mask) > 0:
            ys, xs = np.where(mask)
            inds = np.random.permutation(len(xs))
            xs, ys = xs[inds], ys[inds]
            x0, x1 = np.min(xs), np.max(xs)+1
            y0, y1 = np.min(ys), np.max(ys)+1
            # if (x1-x0)>0 and (y1-y0)>0:
            xy_means[si] = np.array([xs.mean(), ys.mean()])
            xy_rands[si] = np.array([xs[0], ys[0]])
            valids[si] = 1
            crop = mask[y0:y1, x0:x1]
            fill = np.mean(crop)
            fills[si] = fill
        # print('fills', fills)
    return xy_means, xy_rands, valids, fills

def data_zoom(zoom, xys, visibs, valids, rgbs, masks=None, masks2=None, masks3=None, masks4=None):
    S, H, W, C = rgbs.shape

    xys = xys.reshape(S,1,2)
    visibs = visibs.reshape(S,1)
    valids = valids.reshape(S,1)

    _, H, W, C = rgbs.shape
    assert(C==3)
    crop_W = int(W//zoom)
    crop_H = int(H//zoom)

    if np.random.rand() < 0.25: # follow-crop
        # start with xy traj
        smooth_xys = xys.copy()
        # make it inbounds
        smooth_xys = np.clip(smooth_xys, [crop_W // 2, crop_H // 2], [W - crop_W // 2, H - crop_H // 2])
        # smooth it out, to remove info about the traj, and simulate camera motion
        for _ in range(S*3):
            for ii in range(1,S-1):
                smooth_xys[ii] = (smooth_xys[ii-1] + smooth_xys[ii] + smooth_xys[ii+1])/3.0
    else: # static (no-hint) crop
        # zero-vel on random available coordinate
        vis_valid = visibs*valids
        anchor_inds = np.nonzero(vis_valid.reshape(-1)>0.5)[0]
        ind = anchor_inds[np.random.randint(len(anchor_inds))]
        smooth_xys = xys[ind:ind+1].repeat(S,axis=0)
        # xmid = np.random.randint(crop_W//2, W-crop_W//2)
        # ymid = np.random.randint(crop_H//2, H-crop_H//2)
        # smooth_xys = np.stack([xmid, ymid], axis=-1).reshape(1,1,2).repeat(S, axis=0) # S,1,2
        smooth_xys = np.clip(smooth_xys, [crop_W // 2, crop_H // 2], [W - crop_W // 2, H - crop_H // 2])
    # print('xys', xys)
    # print('smooth_xys', smooth_xys)
                
    if np.random.rand() < 0.5:
        # add a random alternate trajectory, to help push us off center
        alt_xys = np.random.randint(-crop_H//8, crop_H//8, (S,1,2))
        for _ in range(3):
            for ii in range(1,S-1):
                alt_xys[ii] = (alt_xys[ii-1] + alt_xys[ii] + alt_xys[ii+1])/3.0
        smooth_xys = smooth_xys + alt_xys
        
    smooth_xys = np.clip(smooth_xys, [crop_W // 2, crop_H // 2], [W - crop_W // 2, H - crop_H // 2])

    rgbs_crop = []
    if masks is not None:
        masks_crop = []
    if masks2 is not None:
        masks2_crop = []
    if masks3 is not None:
        masks3_crop = []
    if masks4 is not None:
        masks4_crop = []

    offsets = []
    for si in range(S):
        xy_mid = smooth_xys[si,0].round().astype(np.int32)

        xmid, ymid = xy_mid[0], xy_mid[1]

        x0, x1 = np.clip(xmid-crop_W//2, 0, W), np.clip(xmid+crop_W//2, 0, W)
        y0, y1 = np.clip(ymid-crop_H//2, 0, H), np.clip(ymid+crop_H//2, 0, H)
        offset = np.array([x0, y0]).reshape(1,2)

        rgbs_crop.append(rgbs[si,y0:y1,x0:x1])
        if masks is not None:
            masks_crop.append(masks[si,y0:y1,x0:x1])
        if masks2 is not None:
            masks2_crop.append(masks2[si,y0:y1,x0:x1])
        if masks3 is not None:
            masks3_crop.append(masks3[si,y0:y1,x0:x1])
        if masks4 is not None:
            masks4_crop.append(masks4[si,y0:y1,x0:x1])
        xys[si] -= offset

        offsets.append(offset)

    rgbs = np.stack(rgbs_crop, axis=0)
    if masks is not None:
        masks = np.stack(masks_crop, axis=0)
    if masks2 is not None:
        masks2 = np.stack(masks2_crop, axis=0)
    if masks3 is not None:
        masks3 = np.stack(masks3_crop, axis=0)
    if masks4 is not None:
        masks4 = np.stack(masks4_crop, axis=0)

    # update visibility annotations
    for si in range(S):
        # avoid 1px edge
        oob_inds = np.logical_or(
            np.logical_or(xys[si,:,0] < 1, xys[si,:,0] > W-2),
            np.logical_or(xys[si,:,1] < 1, xys[si,:,1] > H-2))
        visibs[si,oob_inds] = 0

        # when a point moves far oob, don't supervise with it
        very_oob_inds = np.logical_or(
            np.logical_or(xys[si,:,0] < -128, xys[si,:,0] > W+128),
            np.logical_or(xys[si,:,1] < -128, xys[si,:,1] > H+128))
        valids[si,very_oob_inds] = 0

    xys = xys.squeeze(1)
    visibs = visibs.squeeze(1)
    valids = valids.squeeze(1)

    # clamp to image bounds
    xys = np.minimum(np.maximum(xys, np.zeros((2,), dtype=int)), np.array([W, H]) - 1) # S,2

    if masks4 is not None:
        return xys, visibs, valids, rgbs, masks, masks2, masks3, masks4
    if masks3 is not None:
        return xys, visibs, valids, rgbs, masks, masks2, masks3
    if masks2 is not None:
        return xys, visibs, valids, rgbs, masks, masks2
    if masks is not None:
        return xys, visibs, valids, rgbs, masks
    else:
        return xys, visibs, valids, rgbs


def data_zoom_bbox(zoom, bboxes, visibs, rgbs):#, valids=None):
    S, H, W, C = rgbs.shape

    _, H, W, C = rgbs.shape
    assert(C==3)
    crop_W = int(W//zoom)
    crop_H = int(H//zoom)

    xys = bboxes[:,0:2]*0.5 + bboxes[:,2:4]*0.5
    
    # start with xy traj
    smooth_xys = xys[:]
    # make it inbounds
    smooth_xys = np.clip(smooth_xys, [crop_W // 2, crop_H // 2], [W - crop_W // 2, H - crop_H // 2])
    # smooth it out, to remove info about the traj, and simulate camera motion
    for _ in range(S*3):
        for ii in range(1,S-1):
            smooth_xys[ii] = (smooth_xys[ii-1] + smooth_xys[ii] + smooth_xys[ii+1])/3.0

    if np.random.rand() < 0.5:
        # add a random alternate trajectory, to help push us off center
        alt_xys = np.random.randint(-crop_H//8, crop_H//8, (S,2))
        for _ in range(3):
            for ii in range(1,S-1):
                alt_xys[ii] = (alt_xys[ii-1] + alt_xys[ii] + alt_xys[ii+1])/3.0
        smooth_xys = smooth_xys + alt_xys
        
    smooth_xys = np.clip(smooth_xys, [crop_W // 2, crop_H // 2], [W - crop_W // 2, H - crop_H // 2])
    
    rgbs_crop = []

    offsets = []
    for si in range(S):
        xy_mid = smooth_xys[si].round().astype(np.int32)
        xmid, ymid = xy_mid[0], xy_mid[1]

        x0, x1 = np.clip(xmid-crop_W//2, 0, W), np.clip(xmid+crop_W//2, 0, W)
        y0, y1 = np.clip(ymid-crop_H//2, 0, H), np.clip(ymid+crop_H//2, 0, H)
        offset = np.array([x0, y0]).reshape(2)

        rgbs_crop.append(rgbs[si,y0:y1,x0:x1])
        xys[si] -= offset
        bboxes[si,0:2] -= offset
        bboxes[si,2:4] -= offset
        
        offsets.append(offset)

    rgbs = np.stack(rgbs_crop, axis=0)

    # update visibility annotations
    for si in range(S):
        # avoid 1px edge
        oob_inds = np.logical_or(
            np.logical_or(xys[si,0] < 1, xys[si,0] > W-2),
            np.logical_or(xys[si,1] < 1, xys[si,1] > H-2))
        visibs[si,oob_inds] = 0

    # clamp to image bounds
    xys0 = np.minimum(np.maximum(bboxes[:,0:2], np.zeros((2,), dtype=int)), np.array([W, H]) - 1) # S,2
    xys1 = np.minimum(np.maximum(bboxes[:,2:4], np.zeros((2,), dtype=int)), np.array([W, H]) - 1) # S,2
    bboxes = np.concatenate([xys0, xys1], axis=1)
    return bboxes, visibs, rgbs 
        

def data_pad_if_necessary(rgbs, masks, masks2=None):
    S,H,W,C = rgbs.shape
    
    mask_areas = (masks > 0).reshape(S,-1).sum(axis=1)
    mask_areas_norm = mask_areas / np.max(mask_areas)
    visibs = mask_areas_norm
    
    bboxes = np.stack([mask2bbox(mask) for mask in masks])
    whs = bboxes[:,2:4] - bboxes[:,0:2]
    whs = whs[visibs > 0.5]
    # print('mean wh', np.mean(whs[:,0]), np.mean(whs[:,1]))
    if np.mean(whs[:,0]) >= W/2:
        # print('padding w')
        pad = ((0,0),(0,0),(W//4,W//4),(0,0))
        rgbs = np.pad(rgbs, pad, mode="constant")
        masks = np.pad(masks, pad[:3], mode="constant")
        if masks2 is not None:
            masks2 = np.pad(masks2, pad[:3], mode="constant")
    # print('rgbs', rgbs.shape)
    # print('masks', masks.shape)
    if np.mean(whs[:,1]) >= H/2:
        # print('padding h')
        pad = ((0,0),(H//4,H//4),(0,0),(0,0))
        rgbs = np.pad(rgbs, pad, mode="constant")
        masks = np.pad(masks, pad[:3], mode="constant")
        if masks2 is not None:
            masks2 = np.pad(masks2, pad[:3], mode="constant", constant_values=0.5)

    if masks2 is not None:
        return rgbs, masks, masks2
    return rgbs, masks

def data_pad_if_necessary_b(rgbs, bboxes, visibs):
    S,H,W,C = rgbs.shape
    whs = bboxes[:,2:4] - bboxes[:,0:2]
    whs = whs[visibs > 0.5]
    if np.mean(whs[:,0]) >= W/2:
        pad = ((0,0),(0,0),(W//4,W//4),(0,0))
        rgbs = np.pad(rgbs, pad, mode="constant")
        bboxes[:,0] += W//4
        bboxes[:,2] += W//4
    if np.mean(whs[:,1]) >= H/2:
        pad = ((0,0),(H//4,H//4),(0,0),(0,0))
        rgbs = np.pad(rgbs, pad, mode="constant")
        bboxes[:,1] += H//4
        bboxes[:,3] += H//4
    return rgbs, bboxes
