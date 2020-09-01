"""
Author: liudong
Date: 2020/9/1
Description: 
"""
import pandas as pd
import numpy as np


def extract_base_info_ml100k(path):
    u_items, i_users = {}, {}
    df = pd.read_csv(path, header=None, sep='\t')
    for data in df.itertuples():
        uid, iid = data[1], data[2]
        u_items.setdefault(uid, []).append(iid)
        i_users.setdefault(iid, []).append(uid)
    return u_items, i_users


def gen_purchase_matrix(data, based='user', user_nb=944, item_nb=1683):
    '''
    if based is "user", then param:data is u_item, else i_users
    '''
    if based == 'user':
        purchase_matrix = np.zeros((user_nb, item_nb))
        for uid in data.keys():
            purchase_matrix[uid, data[uid]] = 1
    else:
        purchase_matrix = np.zeros((item_nb, user_nb))
        for iid in data.keys():
            purchase_matrix[iid, data[iid]] = 1
    return purchase_matrix


# u_items, i_users = extract_base_info_ml100k('dataset/ml-100k/u1.base')
# user_purchase_matrix = gen_purchase_matrix(u_items, based='user')
# item_purchase_matrix = gen_purchase_matrix(i_users, based='item')