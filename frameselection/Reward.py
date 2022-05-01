import sys
import torch


# --------------The Rewards (inter-keyframe dissimilarity and keyframe representativeness)-------------------------
def intrinsicreward(seq, actions, ignore_far_sim=True, temp_dist_thre=3, use_gpu=False):
    """
    diverse reward and representative reward
    input:
        seq: the sequence feature shape (1, seq_len, dim)
        actions: the binary action sequence shape (1, seq_len, 1)
        ignore_far_sim (bool): whether temporal distance similarity is considered（defeat: True）
        temp_dist_thre (int): threshold is used for temporal distance similarity（默认为20）
        use_gpu (bool): whether to use gpus
    """
    _seq = seq.detach()          # separate and not update
    _actions = actions.detach()  # separate and not update
    pick_idxs = _actions.squeeze().nonzero().squeeze()                 # index of non-zero elements
    num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1    # the number of selected frames
    # no frames are selected, the reward returns 0
    if num_picks == 0:
        reward = torch.tensor(0.)
        if use_gpu: reward = reward.cuda()
        return reward
    _seq = _seq.squeeze()   # squeeze
    n = _seq.size(0)        # length of sequence
    # --------------------------------diverse reward------------------------------
    if num_picks == 1:
        # only one keyframe
        reward_div = torch.tensor(0.)
        if use_gpu: reward_div = reward_div.cuda()
    else:
        normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)     # normalize
        dissim_mat = 1. - torch.matmul(normed_seq, normed_seq.t())  # dissimilarity matrix
        dissim_submat = dissim_mat[pick_idxs,:][:,pick_idxs]        # the dissimilarity matrix of selected keyframe
        if ignore_far_sim:
            # considering temporal distance
            pick_mat = pick_idxs.expand(num_picks, num_picks)
            temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
            dissim_submat[temp_dist_mat > temp_dist_thre] = 1.
        reward_div = dissim_submat.sum() / (num_picks * (num_picks - 1.))   # the average of dissimilarity
    # ------------------------------representative reward---------------------------------
    if num_picks==1:
        reward_rep = torch.tensor(0.)  # only one keyframe
    else:
        dist_mat = torch.pow(_seq, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist_mat = dist_mat + dist_mat.t()
        dist_mat.addmm_(1, -2, _seq, _seq.t())
        dist_mat = dist_mat[:,pick_idxs]                                       # distance matrix
        dist_mat = dist_mat.min(1, keepdim=True)[0]                            # kmedoid
        reward_rep = torch.exp(-dist_mat.mean())                               # the minimum distance exponent square
    # add two rewards
    reward = reward_div + reward_rep
    # return the corresponding rewards
    return reward


# attacking reward
def attackreward(res,iter_num,vid,adv_vid):
    # get mean perturbations
    def pertubation(clean, adv):
        loss = torch.nn.L1Loss()
        average_pertubation = loss(clean, adv)
        return average_pertubation
    R = 0.0
    if (res):
        if (iter_num>15000):
            P = pertubation(vid,adv_vid)       # average perturbations per pixel
            R = 0.999*torch.exp(-(P/0.05))     # the smaller the perturbations, the greater the reward
            P = P.cpu()
        else:
            P = pertubation(vid,adv_vid)       # average perturbations per pixel
            R = torch.exp(-(P/0.05))           # the smaller the perturbations, the greater the reward
    else:
        R = -1
    # return the attacking reward
    return R
