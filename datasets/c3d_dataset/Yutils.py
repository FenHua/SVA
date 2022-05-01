# 加载视频帧总数，获取不同数据集的均值和标准差


# 获取多少帧
def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


# 均值
def get_mean(norm_value=255, dataset='hmdb51'):
    assert dataset in ['hmdb51', 'ucf101']
    if dataset == 'hmdb51':
        return [
            95.4070 / norm_value, 93.4680 / norm_value, 82.1443 / norm_value
        ]
    elif dataset == 'ucf101':
        # Kinetics (10 videos for each class)
        return [
            101.2198/ norm_value, 97.5751 / norm_value,
            89.5303 / norm_value
        ]


# 标准方差
def get_std(norm_value=255, dataset='hmdb51'):
    assert dataset in ['hmdb51', 'ucf101']
    if dataset == 'hmdb51':
        return [
            51.674248 / norm_value, 50.311924 / norm_value,
            49.48427 / norm_value
        ]
    elif dataset == 'ucf101':
        return [
            62.08429 / norm_value, 60.398968 / norm_value,
            59.187363 / norm_value
        ]