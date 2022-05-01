import torch
import pickle
from transforms import *
from hmdb51 import HMDB51
from ucf101 import UCF101
from Yutils import get_mean, get_std
from c3d_opt import hmdb51_parse_opts, ucf101_parse_opts

# ************************************
# 按照C3D中对test的处理方式来进行处理
# ************************************

class DictToAttr(object):
    def __init__(self, args):
        for i in args.keys():
            setattr(self, i, args[i])


def get_test_set(dataset):
    assert dataset in ['ucf101', 'hmdb51']
    if dataset == 'hmdb51':
        with open('/home/yanhuanqian/DFDA/datasets/c3d_dataset/hmdb51_params.pkl', 'rb') as ipt:
            opt = pickle.load(ipt)
        opt = DictToAttr(opt)
    elif dataset == 'ucf101':
        with open('/home/yanhuanqian/DFDA/datasets/c3d_dataset/ucf101_params.pkl', 'rb') as ipt:
            opt = pickle.load(ipt)
        opt = DictToAttr(opt)

    opt.dataset = dataset
    # 开始转换
    opt.mean = get_mean(opt.norm_value, dataset)
    opt.std = get_std(opt.norm_value, dataset)
    torch.manual_seed(opt.manual_seed)
    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)
    spatial_transform = spatial_Compose([
        Scale(int(opt.sample_size / opt.scale_in_test)),
        CornerCrop(opt.sample_size, opt.crop_position_in_test),
        ToTensor(opt.norm_value), norm_method
    ])
    temporal_transform = LoopPadding(opt.sample_duration)
    target_transform = target_Compose([VideoID(), ClassLabel()])
    # 数据转换操作结束
    if opt.dataset == 'ucf101':
        test_data = UCF101(
            opt.video_path,
            opt.annotation_path,
            'validation',
            input_style='rgb',
            n_samples_for_each_video=3,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'hmdb51':
        test_data = HMDB51(
            opt.video_path,
            opt.annotation_path,
            'validation',
            input_style='rgb',
            n_samples_for_each_video=3,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration)
    return test_data


def get_training_set(dataset):
    assert dataset in ['ucf101', 'hmdb51']
    if dataset == 'hmdb51':
        with open('/home/yanhuanqian/DFDA/datasets/c3d_dataset/hmdb51_params.pkl', 'rb') as ipt:
            opt = pickle.load(ipt)
        opt = DictToAttr(opt)
    elif dataset == 'ucf101':
        with open('/home/yanhuanqian/DFDA/datasets/c3d_dataset/ucf101_params.pkl', 'rb') as ipt:
            opt = pickle.load(ipt)
        opt = DictToAttr(opt)
    opt.dataset = dataset
    # transforms begin
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.mean = get_mean(opt.norm_value, dataset=opt.dataset)
    opt.std = get_std(opt.norm_value, dataset=opt.dataset)
    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)
    torch.manual_seed(opt.manual_seed)
    assert opt.train_crop in ['random', 'corner', 'center']
    if opt.train_crop == 'random':
        crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
    elif opt.train_crop == 'corner':
        crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
    elif opt.train_crop == 'center':
        crop_method = MultiScaleCornerCrop(
            opt.scales, opt.sample_size, crop_positions=['c'])
    spatial_transform = spatial_Compose([
        crop_method,
        RandomHorizontalFlip(),
        ToTensor(opt.norm_value), norm_method
    ])
    temporal_transform = TemporalRandomCrop(opt.sample_duration)
    target_transform = target_Compose([VideoID(), ClassLabel()])
    # transforms end
    if opt.dataset == 'ucf101':
        training_data = UCF101(
            opt.video_path,
            opt.annotation_path,
            'training',
            input_style='rgb',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
        )
    elif opt.dataset == 'hmdb51':
        training_data = HMDB51(
            opt.video_path,
            opt.annotation_path,
            'training',
            input_style='rgb',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
        )
    return training_data