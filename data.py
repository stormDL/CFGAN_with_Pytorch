import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def read_ml100k(train_path, test_path, sep, header):
    train_set_dict, test_set_dict = {}, {}
    df_train = pd.read_csv(train_path, sep=sep, header=header)-1
    df_test = pd.read_csv(test_path, sep=sep, header=header)-1
    # 处理训练集
    for item in df_train.itertuples():
        uid, i_id = item[1], item[2]
        train_set_dict.setdefault(uid, {}).setdefault(i_id, 1)
    for item in df_test.itertuples():
        uid, i_id = item[1], item[2]
        test_set_dict.setdefault(uid, {}).setdefault(i_id, 1)
    return train_set_dict, test_set_dict


def read_ml1m(filepath, sep='::', header='infer'):
    train_set_dict, test_set_dict = {}, {}
    df = pd.read_csv(filepath, sep=sep, header=header).iloc[:, :3]-1
    df = df.values.tolist()
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=1228)
    for uid, iid, score in train_set:
        train_set_dict.setdefault(uid, {}).setdefault(iid, 1)
    for uid, iid, score in test_set:
        test_set_dict.setdefault(uid, {}).setdefault(iid, 1)
    return train_set_dict, test_set_dict


def get_matrix(train_set_dict, test_set_dict, nb_user, nb_item):
    train_set, test_set = np.zeros(shape=(nb_user, nb_item)), np.zeros(shape=(nb_user, nb_item))
    for u in train_set_dict.keys():
        for i in train_set_dict[u].keys():
            # # 模糊过的标签
            # train_set[u][i] = train_set_dict[u][i]
            # 购买过的就是1，不做模糊处理
            train_set[u][i] = 1
    for u in test_set_dict.keys():
        for i in test_set_dict[u]:
            test_set[u][i] = 1
    return train_set, test_set

# # train_set_dict, test_set_dict = read_data(filepath='../dataset/ml-100k/u.data', sep='\t', header=None)
# train_set_dict, test_set_dict = read_data(filepath='../dataset/ml-1m/ratings.dat', sep='::', header=None)
#
# count_train, count_test = 0, 0
# for u in train_set_dict.keys():
#     for i in train_set_dict[u].keys():
#         count_train += 1
#         # print(u, i, train_set_dict[u][i])
#
# for u in test_set_dict.keys():
#     count_test += len(test_set_dict[u])
#     # print(u, len(test_set_dict[u]))
# print(count_train, count_test, count_train/count_test)

# read_ml1m('dataset/ml-1m/ratings.dat')

