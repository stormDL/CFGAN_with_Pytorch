import torch
import pandas as pd


def get_train_data(train_file, user_nb, item_nb):
    '''
    :param train_file, 训练数据集的文件路径
    :param user_nb, 用户数量
    :param item_nb，项目数量
    :return: 一个购买矩阵，一个mask矩阵
    '''
    train_df = pd.read_csv(train_file, header=None)
    train_martrix, mask_martrix = torch.zeros(user_nb, item_nb), torch.zeros(user_nb, item_nb)
    for data in train_df.itertuples():
        u_id, i_id = data[1], data[2]
        train_martrix[u_id][i_id] = 1
        mask_martrix[u_id][i_id] = -99
    return train_martrix, mask_martrix


def get_test_data(test_file):
    '''
    :param test_file，测试数据集的文件路径
    :return: 一个key是用户id, value是项目id列表的字典
    '''
    u_ids_dict = {}
    train_df = pd.read_csv(test_file, header=None)
    for data in train_df.itertuples():
        u_id, i_id = data[1], data[2]
        u_ids_dict.setdefault(u_id, []).append(i_id)
    return u_ids_dict


def deal_ml100k(input_file, output_file, split_mark='\t', header=None):
    '''
    处理ml100k数据集，将其处理成 get_test_data get_train_data这两个方法希望的格式
    :param input_file: 输入的数据集文件路径
    :param output_file: 输出的数据集文件路径
    :param split_mark: 一行数据的划分标志
    :param header: 是否包含标题，与pandas.read_csv中的header参数对应
    :return: None
    '''
    df = pd.read_csv(input_file, sep=split_mark, header=header)
    df = df.iloc[:, :-1]
    df[0] -= 1
    df[1] -= 1
    df.to_csv(output_file, index=None, header=header)
    print('ml 100k 文件：', input_file, '处理完毕!')


def split_data(total_file, train_path, test_path, gap=5):
    '''
    gap表示每个多少条数据选择一个测试集，5表示把总的数据集划分为8:2
    '''
    print('正在划分数据...')
    df = pd.read_csv(total_file, sep='::', header=None)
    with open(train_path, 'w') as train, open(test_path, 'w') as test:
        for idx, data in enumerate(df.itertuples()):
            u_id, i_id, score = int(getattr(data, '_1'))-1, int(getattr(data, '_2'))-1, getattr(data, '_3')
            line = str(u_id) + ',' + str(i_id) + ',' + str(score) + '\n'
            if (idx + 1) % gap == 0:
                test.write(line)
            else:
                train.write(line)
    print('数据划分完毕!')


# test case
# deal_ml100k('dataset/ml-100k/u1.base', output_file='dataset/ml-100k/train.csv')
# deal_ml100k('dataset/ml-100k/u1.test', output_file='dataset/ml-100k/test.csv')
# train_martrix, mask_martrix = get_train_data('dataset/ml-100k/train.csv', user_nb=943, item_nb=1682)
# u_ids_dict = get_test_data('dataset/ml-100k/test.csv')

# split_data('dataset/ml-1m/ratings.dat', 'dataset/ml-1m/train.csv', 'dataset/ml-1m/test.csv')