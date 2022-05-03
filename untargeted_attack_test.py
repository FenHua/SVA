import os
import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
from Utils.utils import *
from torch.distributions import Bernoulli  # bernoulli
from attack.untargetedAttack import attack
from frameselection.Agent import frames_select, feature_extractor
from model_wrapper.vid_model_top_k import C3D_K_Model, LRCN_K_Model  # acquire top k results

gpus = [0]                # gpu setting
image_models = ['resnet50']
os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join([str(gpu) for gpu in gpus])    # visible gpu setting
model_name = 'c3d'
dataset_name = 'hmdb51'
resume = 'agent_checkpoints/{}_{}.pth'.format(model_name, dataset_name)      # checkpoint path
print('load {} dataset'.format(dataset_name))
model = generate_model(model_name, dataset_name)        # get the training dataset
print('Initialize model')
agent = frames_select()              # agent
checkpoint = torch.load(resume)
agent.load_state_dict(checkpoint)
GetFeatures = feature_extractor()    # feature extractor
agent.eval()
try:
    model.cuda()
    agent = agent.cuda()
except:
    pass
if model_name == 'c3d':
    vid_model = C3D_K_Model(model)
else:
    vid_model = LRCN_K_Model(model)

ids_labels = ["brush_hair", "cartwheel", "catch", "chew", "clap", "climb", "climb_stairs",
 "dive", "draw_sword", "dribble", "drink", "eat", "fall_floor", "fencing", "flic_flac",
 "golf", "handstand", "hit", "hug", "jump", "kick", "kick_ball", "kiss", "laugh", "pick",
 "pour", "pullup", "punch", "push", "pushup", "ride_bike", "ride_horse", "run", "shake_hands",
 "shoot_ball", "shoot_bow", "shoot_gun", "sit", "situp", "smile", "smoke", "somersault", "stand",
 "swing_baseball", "sword", "sword_exercise", "talk", "throw", "turn", "walk", "wave"]

# attack
i = 1   # attack the i-th video, there are total 5 video segments in directory TT
vid = torch.from_numpy(np.load('TT/{}.npy'.format(i)))
vid = vid.cuda()
vid_label = vid_model(vid[None,:])[1][0,0]   # the clean video label
print('The origin label is: {}'.format(ids_labels[vid_label]))

features = GetFeatures(vid)
probs = agent(features[None, :])
actions = Bernoulli(probs)
SS = torch.zeros(5, 16)                      # 记录采样的动作
SSlen = []
for t in range(5):
    Tactions = actions.sample()
    SS[t, :] = Tactions.squeeze().cpu().data
    SSlen.append(len(SS[t, :].nonzero()))
midx = SSlen.index(min(SSlen))
masklist = (SS[midx, :]).nonzero()
del SS
del SSlen
res, iter_num, adv_vid = attack(vid_model, vid, vid_label, masklist, image_models, gpus)
if res:
    print('Attack Successes!')
    label_P = vid_model(adv_vid[None,:])[1][0,0]
    print('The adversarial label is: {}'.format(ids_labels[label_P]))
else:
    print('--------------------Attack Fails-------------------------')