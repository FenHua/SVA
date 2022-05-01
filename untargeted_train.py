import os
import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
from Utils.utils import *
from torch.optim import lr_scheduler
from torch.distributions import Bernoulli
from attack.untargetedAttack import attack
from frameselection.Reward import intrinsicreward, attackreward
from frameselection.Agent import frames_select, feature_extractor
from model_wrapper.vid_model_top_k import C3D_K_Model, LRCN_K_Model  # acquire top k results

config =argparse.ArgumentParser()
config.add_argument('--model_name',type=str,default='c3d',
                    help='The action recognition')
config.add_argument('--dataset_name',type=str,default='hmdb51',
                    help='The dataset: hmdb51/ucf101')
config.add_argument('--gpus',nargs='+',type=int,required=True,
                    help='The gpus to use')
config.add_argument('--train_num',type=int,default=20,
                    help='The number of testing')
args = config.parse_args()

gpus = args.gpus
os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join([str(gpu) for gpu in gpus])  # visible gpu setting
model_name = args.model_name         # recognition model
dataset_name = args.dataset_name     # dataset
NUM = args.train_num                 # the number of training samples
image_models = ['resnet50']
max_epoch = 20                       # 20

# ---------------------------start-----------------------------------------------
print('load {} dataset'.format(dataset_name))
test_data = generate_dataset(model_name, dataset_name)  # dataset setting
print('load {} model'.format(model_name))
model = generate_model(model_name, dataset_name)        # threat model setting
print("Initialize model")
agent = frames_select()  # agent
GetFeatures = feature_extractor()  # feature extractor
optimizer = torch.optim.Adam(agent.parameters(), lr=1e-05, weight_decay=1e-05)  # optimizer
scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)             # learning rate setting
try:
    agent = agent.cuda()
    model.cuda()
except:
    pass
if model_name == 'c3d':
    vid_model = C3D_K_Model(model)
else:
    vid_model = LRCN_K_Model(model)

idx = 5
# get attack ids
attacked_ids = get_samples(model_name, dataset_name)
train_ids = attacked_ids[:NUM]              # training dataset
baselines = {key: 0. for key in train_ids}  # The reference point for each video
train_root = 'untargeted_exp/trains/{}_{}'.format(model_name, dataset_name)  # Record training results
reward_path = os.path.join(train_root, 'rewards.txt')  # recording rewards
f_reward = open(reward_path, 'a+')
for epoch in range(max_epoch):
    epoch_root = os.path.join(train_root, 'Epoch-{}'.format(epoch))
    os.mkdir(epoch_root)
    np.random.shuffle(train_ids)  # shuffle video id
    epoch_rewards = 0.0
    for idx in train_ids:
        out_path = os.path.join(epoch_root, 'vid-{}'.format(idx))
        os.mkdir(out_path)
        x0, label = test_data[idx]  # data
        x0 = image_to_vector(model_name, x0)
        vid = x0.cuda()
        features = GetFeatures(vid)
        probs = agent(features[None, :])
        actions = Bernoulli(probs)
        cost = 0.0
        epis_rewards = []
        SS = torch.zeros(5, 16)  # Record the action of sampling
        SSlog = []               # Record the probability of sampling
        SSlen = []
        for t in range(5):
            Sactions = actions.sample()
            log_probs = actions.log_prob(Sactions)
            SS[t, :] = Sactions.squeeze().cpu().data
            SSlen.append(len(SS[t, :].nonzero()))
            SSlog.append(log_probs.mean().cpu().data)
            reward = intrinsicreward(features, Sactions)  # Video intrinsic Rewards
            expected_reward = log_probs.mean() * (reward - baselines[idx])
            cost -= expected_reward
            epis_rewards.append(reward.item())
        # Calculate the reward value of the attack result
        midx = SSlen.index(min(SSlen))
        Faction = (SS[midx,:]).nonzero()
        ASSlog = np.sum(SSlog)
        print('Attacking.....')
        res, iter_num, adv_vid = attack(vid_model, vid, label[1], Faction, image_models, gpus)
        reward2 = attackreward(res, iter_num, vid, adv_vid)
        cost -= 5 * ASSlog * reward2  # X5(hyperparameter 5)
        epoch_rewards += np.mean(epis_rewards)+3*reward2
        del SS
        del SSlog
        del SSlen
        optimizer.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 5.0)
        optimizer.step()
        baselines[idx] = 0.9 * baselines[idx] + 0.1 * (np.mean(epis_rewards) + 3 * reward2)
        if res:
            # success
            AP = pertubation(vid, adv_vid)
            metric_path = os.path.join(out_path, 'metric.txt')  # save metric
            adv_path = os.path.join(out_path, 'adv_vid.npy')
            np.save(adv_path, adv_vid.cpu().numpy())
            f = open(metric_path, 'w')
            f.write(str(iter_num))
            f.write('\n')
            f.write(str(AP.cpu()))
            f.write('\n')
            f.write(str(Faction.cpu().data))
            f.close()
            print('untargeted attack succeed using {} quries'.format(iter_num))
            print('The average pertubation of video is: {}'.format(AP.cpu()))
        else:
            # fail
            print('--------------------Attack Fails-----------------------')

    epoch_rewards = epoch_rewards/NUM
    f_reward.write(str(epoch_rewards))
    f_reward.write('\n')
    '''
    if ((epoch + 1) % 5 == 0):
        model_state_dict = agent.state_dict()
        torch.save(model_state_dict, 'agent_checkpoints/{}_{}.pth'.format(model_name, dataset_name))    
    '''
