"""
Author: liudong
Date: 2020/9/1
Description: 
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

from data_util import extract_base_info_ml100k, gen_purchase_matrix


class G(nn.Module):
    def __init__(self, input_len, item_nb):
        super(G, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(input_len, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, item_nb),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.gen(x)


class D(nn.Module):
    def __init__(self, input_len, item_nb):
        super(D, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(input_len, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.dis(x)


class Train_Dataset(Dataset):
    def __init__(self, purchase_matrix, mask_nb, device):
        self.purchase_matrix = torch.tensor(purchase_matrix, dtype=torch.float).to(device)
        self.target_matrix = torch.tensor(purchase_matrix, dtype=torch.float).to(device)
        self.mask_nb = mask_nb

    def __getitem__(self, idx):
        '''
        :return: user_condition, user_info, mask_matrix, target_matrix
        '''
        if idx == 0:
            idx = 1
        user_condition = self.purchase_matrix[idx]
        user_info = self.purchase_matrix[idx]
        mask_matrix = self.purchase_matrix[idx].clone().view(-1)
        target_matrix = self.target_matrix[idx]

        mask_index = random.choices((mask_matrix-1).nonzero().cpu().numpy().reshape(-1), k=self.mask_nb)
        for index in mask_index:
            mask_matrix[index] = 1
        return user_condition, user_info, mask_matrix, target_matrix

    def __len__(self):
        return len(self.purchase_matrix)


def train(dataset, data_loader, u_items, device, batch_size=32, epoches=10, g_lr=0.0001, d_lr=0.0001, g_steps=1, d_steps=1,
          user_nb=944, item_nb=1683):
    #--------------------------------------
    # 设置模型相关的参数
    #--------------------------------------
    gen = G(item_nb, item_nb).to(device)
    dis = D(item_nb*2, item_nb).to(device)
    g_opt = optim.Adam(gen.parameters(), lr=g_lr)
    d_opt = optim.Adam(dis.parameters(), lr=d_lr)
    #--------------------------------------
    # 关于损失相关
    # --------------------------------------
    real_label = torch.ones(batch_size, 1).to(device)
    fake_label = torch.zeros(batch_size, 1).to(device)
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()

    for e in range(epoches):
        #----------------------------------------
        #  开始训练判别器
        #----------------------------------------
        # print('正在训练判别器...')
        for d_step in range(d_steps):
            for user_condition, user_info, mask_matrix, target_matrix in data_loader:
                predict_matrix = gen(user_info)

                fake_user_data = torch.cat([user_condition, predict_matrix*mask_matrix], dim=1)
                real_user_data = torch.cat([user_condition, user_info], dim=1)
                fake_prob = dis(fake_user_data)
                real_prob = dis(real_user_data)
                loss = bce_loss(real_prob, real_label[:len(predict_matrix)]) + bce_loss(fake_prob, fake_label[:len(predict_matrix)])
                d_opt.zero_grad()
                loss.backward()
                d_opt.step()
        # ----------------------------------------
        #  开始训练生成器
        # ----------------------------------------
        # print('正在训练生成器...')
        for g_step in range(g_steps):
            for user_condition, user_info, mask_matrix, target_matrix in data_loader:
                predict_matrix = gen(user_info)
                user_data = torch.cat([user_condition, predict_matrix*mask_matrix], dim=1)
                predict_prob = dis(user_data)

                loss = bce_loss(predict_prob, real_label[:len(predict_matrix)]) + 1*mse_loss(predict_matrix*mask_matrix, target_matrix)
                g_opt.zero_grad()
                loss.backward()
                g_opt.step()
        if e%1 == 0:
            # print('开始测试...')
            print('-'*48, e, '-'*48)
            test(gen, dataset, u_items)


def test(gen_model, dataset, u_items, k=5):
    precisions, recalls = 0, 0
    with torch.no_grad():
        for uid in u_items.keys():
            user_condition, user_info, mask_matrix, target_matrix = dataset[uid]
            # user_data = torch.cat([user_condition, user_info], dim=-1)
            predict_result = gen_model(user_info).cpu().numpy().tolist()
            tmp_list = []
            for idx, prob in enumerate(predict_result):
                tmp_list.append((idx, prob))
            tmp_list.sort(key=lambda x:x[1], reverse=True)
            hit = 0
            for data in tmp_list[:k]:
                if data[0] in u_items[uid]:
                    hit += 1
            precisions += hit/k
            recalls += hit/len(u_items[uid])
    precision = precisions/len(u_items)
    recall = recalls/len(u_items)
    print('precision:', precision, '\trecall:', recall)


if __name__ == '__main__':
    batch_size = 64
    epoches = 100
    u_items, i_users = extract_base_info_ml100k('dataset/ml-100k/u1.base')
    user_purchase_matrix = gen_purchase_matrix(u_items, based='user')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset = Train_Dataset(user_purchase_matrix, mask_nb=128, device=device)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    train(dataset, data_loader, u_items, device, batch_size=batch_size, epoches=epoches,
          g_lr=0.0001, d_lr=0.0001, g_steps=1, d_steps=1,
          user_nb=944, item_nb=1683)