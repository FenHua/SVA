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

config =argparse.ArgumentParser()
config.add_argument('--model_name',type=str,default='c3d',
                    help='The action recognition')
config.add_argument('--dataset_name',type=str,default='hmdb51',
                    help='The dataset: hmdb51/ucf101')
config.add_argument('--gpus',nargs='+',type=int,required=True,
                    help='The gpus to use')
config.add_argument('--test_num',type=int,default=50,
                    help='The number of testing')
args = config.parse_args()
gpus = args.gpus                # gpu setting
image_models = ['resnet50']
os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join([str(gpu) for gpu in gpus])    # visible gpu setting
model_name = args.model_name
dataset_name = args.dataset_name
test_num = args.test_num
resume = 'agent_checkpoints/{}_{}.pth'.format(model_name, dataset_name)      # checkpoint path
print('load {} dataset'.format(dataset_name))
test_data = generate_dataset(model_name, dataset_name)  # dataset setting
print('load {} model'.format(model_name))
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
# get attack ids
attacked_ids = get_samples(model_name, dataset_name)
def GetPairs(test_data,idx):
    x0, label = test_data[attacked_ids[idx]]
    x0 = image_to_vector(model_name, x0)
    return x0.cuda(),label[1]

result_root = 'untargeted_exp/results/{}_{}'.format(model_name,dataset_name)
av_metric = os.path.join(result_root, 'Avmetric.txt')
success_num = 0          # Number of successful attacks
total_P_num = 0.0        # perturbation sum for all test samples
total_query_num = 0.0    # total query numbers

for i in range(20,test_num):
    output_path = os.path.join(result_root, 'vid-{}'.format(attacked_ids[i]))
    os.mkdir(output_path)
    vid,vid_label = GetPairs(test_data,i)
    features = GetFeatures(vid)
    probs = agent(features[None, :])
    actions = Bernoulli(probs)
    SS = torch.zeros(5, 16)  # 记录采样的动作
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
        AP = pertubation(vid, adv_vid)
        total_query_num += iter_num
        total_P_num += AP
        success_num += 1
        metric_path = os.path.join(output_path, 'metric.txt')  # save metric
        adv_path = os.path.join(output_path, 'adv_vid.npy')
        np.save(adv_path, adv_vid.cpu().numpy())
        f = open(metric_path, 'w')
        f.write(str(iter_num))
        f.write('\n')
        f.write(str(AP.cpu()))
        f.write('\n')
        f.write(str(masklist.cpu().data))
        f.close()
        f1 = open(av_metric, 'a')
        f1.write(str(total_query_num))
        f1.write('\n')
        f1.write(str(total_P_num))
        f1.write('\n')
        f1.close()
    f1 = open(av_metric,'a')
    f1.write('------------')
    f1.write('\n')
    f1.write(str(success_num))
    f1.write('\n')
    f1.close()