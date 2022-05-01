from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datasets.ucf101 import UCF101
from datasets.hmdb51 import HMDB51

# 获取验证数据集
V_P_HMDB = 'datasets/data/HMDB51/hmdb51-jpg'
V_P_UCF = 'datasets/data/UCF101/UCF101-jpg'
V_A_HMDB = 'datasets/data/HMDB51/hmdb51-annotation/hmdb51_1.json'
V_A_UCF = 'datasets/data/UCF101/UCF101-annotation/ucf101_01.json'

# 获取验证数据集
def get_validation_set(dataset,spatial_transform, temporal_transform, target_transform):
    if dataset == 'ucf101':
        video_path=V_P_UCF
        annotation_path=V_A_UCF
        validation_data = UCF101(
            video_path,
            annotation_path,
            'validation',
            1,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=16)
    elif dataset == 'hmdb51':
        video_path=V_P_HMDB
        annotation_path=V_A_HMDB
        validation_data = HMDB51(
            video_path,
            annotation_path,
            'validation',
            1,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=16)
    return validation_data

# 获取训练数据集
def get_training_lrcn_set(dataset,spatial_transform, temporal_transform, target_transform):
    if dataset == 'ucf101':
        video_path = V_P_UCF
        annotation_path = V_A_UCF
        training_data = UCF101(
            video_path,
            annotation_path,
            'training',
            1,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=16)

    elif dataset == 'hmdb51':
        video_path = V_P_HMDB
        annotation_path = V_A_HMDB
        training_data = HMDB51(
            video_path,
            annotation_path,
            'training',
            1,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=16)
    return training_data