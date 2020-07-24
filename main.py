import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from matplotlib import pyplot as plt
from models import G, D
import data_deal


def neg_sample(data_martrix, zr_sample_size, pm_sample_size):
    zr_sample_martrix, pm_sample_martrix = torch.zeros_like(data_martrix), torch.zeros_like(data_martrix)
    for idx in range(len(data_martrix)):
        non_zero_idxs = (data_martrix[idx]-1).cpu().nonzero().numpy().flatten().tolist()
        random.shuffle(non_zero_idxs)
        zr_sample_size = min(len(non_zero_idxs), zr_sample_size)
        sample_idxs = non_zero_idxs[: zr_sample_size]
        zr_sample_martrix[idx][sample_idxs] = 1

        non_zero_idxs = non_zero_idxs[zr_sample_size:]
        pm_sample_size = min(len(non_zero_idxs), pm_sample_size)
        sample_idxs = non_zero_idxs[: pm_sample_size]
        pm_sample_martrix[idx][sample_idxs] = 1
    return zr_sample_martrix, pm_sample_martrix


def cal_common_nb(predict_purchase_vec, target_item_set, top_k):
    '''
    :param predict_purchase_vec: 单个用户的购买向量，是一个List
    :param target_item_set: 目标项目集合，是一个List
    :param top_k: 推荐几个项目给用户
    :return: 共同项目的数量
    '''
    hit = 0
    preference_info = []
    for idx, score in enumerate(predict_purchase_vec):
        preference_info.append((idx, score))
    preference_info.sort(key=lambda x : x[1], reverse=True)
    for data in preference_info[:top_k]:
        if data[0] in target_item_set:
           hit += 1
    return hit


def train(epoches, g_batch_size, d_batch_size, g_lr, d_lr,
          g_steps, d_steps, train_path, test_path, user_nb,
          item_nb, zr_sample_size, pm_sample_size, alpha=0.03, top_k=5):
    precision_list = []
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # 模型相关配置
    g = G(u_info_len=item_nb).to(device)
    d = D(u_info_len=item_nb, condition_len=item_nb).to(device)
    g_opt = optim.Adam(g.parameters(), lr=g_lr)
    d_opt = optim.Adam(d.parameters(), lr=d_lr)
    mse_loss = nn.MSELoss(reduction='sum')
    # mse_loss = nn.MSELoss(reduction='mean')
    bce_loss = nn.BCELoss()
    d_real_label = torch.ones(d_batch_size, 1)
    d_fake_label = torch.zeros(d_batch_size, 1)
    d_labels = torch.cat([d_real_label, d_fake_label], dim=0).to(device)
    # 相关数据加载
    train_martrix, mask_martrix = data_deal.get_train_data(train_file=train_path, user_nb=user_nb, item_nb=item_nb)
    train_martrix, mask_martrix = train_martrix.to(device), mask_martrix.to(device)
    u_ids_dict = data_deal.get_test_data(test_file=test_path)
    all_index_list = list(range(user_nb))
    # 定义训练过程
    for e in range(epoches):
#---------------------------------------------------------------------------------------------------------
#                         开始训练生成器
#---------------------------------------------------------------------------------------------------------
        zr_sample_martrix, pm_sample_martrix = neg_sample(train_martrix, 12, 21)
        zr_sample_martrix, pm_sample_martrix = zr_sample_martrix.to(device), pm_sample_martrix.to(device)
        for g_idx in range(g_steps):
            batch_idx = np.random.choice(all_index_list, g_batch_size, replace=False)
            batch_train_martrix = train_martrix[batch_idx]
            batch_mask_martrix = train_martrix[batch_idx]
            batch_zr = zr_sample_martrix[batch_idx]
            batch_pm = pm_sample_martrix[batch_idx]

            fake_purchase = g(batch_mask_martrix)
            eu_plus_ku = batch_mask_martrix + batch_pm
            d_out = d(u_info=batch_train_martrix, condition=fake_purchase*eu_plus_ku)
            zr_target = torch.zeros_like(fake_purchase) + batch_train_martrix
            loss = torch.mean(torch.log(1-d_out+1e-6)) + alpha*mse_loss(fake_purchase*(batch_zr+batch_train_martrix), zr_target)/len(fake_purchase)
            g_opt.zero_grad()
            loss.backward()
            g_opt.step()

#---------------------------------------------------------------------------------------------------------
#                         开始训练判别器
#---------------------------------------------------------------------------------------------------------
        zr_sample_martrix, pm_sample_martrix = neg_sample(train_martrix, 12, 21)
        zr_sample_martrix, pm_sample_martrix = zr_sample_martrix.to(device), pm_sample_martrix.to(device)
        for d_idx in range(d_steps):
            batch_idx = np.random.choice(all_index_list, d_batch_size, replace=False)
            batch_train_martrix = train_martrix[batch_idx]
            batch_mask_martrix = train_martrix[batch_idx]
            batch_zr = zr_sample_martrix[batch_idx]
            batch_pm = pm_sample_martrix[batch_idx]

            fake_d_out = d(u_info=batch_train_martrix, condition=g(batch_train_martrix)*(batch_mask_martrix+batch_pm))
            real_d_out = d(u_info=batch_train_martrix, condition=batch_train_martrix)
            predict_result = torch.cat([real_d_out, fake_d_out], dim=0)
            loss = bce_loss(predict_result, d_labels)
            d_opt.zero_grad()
            loss.backward()
            d_opt.step()
#---------------------------------------------------------------------------------------------------------
#                         每隔一段时间就测试一波
#---------------------------------------------------------------------------------------------------------
        if e % 10 == 0:
            total_recommend_nb = len(u_ids_dict)*top_k
            total_positive_nb = 0
            total_hit = 0
            with torch.no_grad():
                for uid in u_ids_dict.keys():
                    predict_purchase_vec = (g(train_martrix[uid])+mask_martrix[uid]).cpu().numpy().tolist()
                    total_hit += cal_common_nb(predict_purchase_vec, u_ids_dict[uid], top_k)
                    total_positive_nb += len(u_ids_dict[uid])
            recall, precision = total_hit/total_positive_nb, total_hit/total_recommend_nb
            precision_list.append(precision)
            print(e, 'recall:', recall, 'precision:', precision, 'hits:', total_hit,
                  'total_recommend_nb:', total_recommend_nb, 'total_positive_nb:', total_positive_nb)
    return precision_list


if __name__ == '__main__':
    precision_list = train(epoches=1001,
                          g_batch_size=32,
                          d_batch_size=32,
                          g_lr=0.0001,
                          d_lr=0.0001,
                          g_steps=24,
                          d_steps=12,
                          train_path='dataset/ml-100k/train.csv',
                          test_path='dataset/ml-100k/test.csv',
                          # train_path='dataset/ml-1m/train.csv',
                          # test_path='dataset/ml-1m/test.csv',
                          user_nb=943,
                          item_nb=1682,
                          zr_sample_size=128,
                          pm_sample_size=128,
                          alpha=0.3,
                          top_k=5)
    epoch_list = [i*10 for i in range(len(precision_list))]
    plt.plot(epoch_list, precision_list)
    plt.show()