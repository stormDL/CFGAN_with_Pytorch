'''
基于用户购买向量的CFGAN
'''
import random
import torch
import numpy as np
from data import read_ml100k, get_matrix, read_ml1m
from layers import Gen, Dis
from matplotlib import pyplot as plt


def test(gen, test_set_dict, train_set, top_k=5):
    gen.eval()
    users = list(test_set_dict.keys())
    input_data = torch.tensor(train_set[users], dtype=torch.float)
    out = gen(input_data)
    out = (out - 999*input_data).detach().numpy()
    precisions = 0
    recalls = 0
    hits = 0
    total_purchase_nb = 0
    for i, u in enumerate(users):
        hit = 0
        tmp_list = [(idx, value) for idx, value in enumerate(out[i])]
        tmp_list = sorted(tmp_list, key=lambda x:x[1], reverse=True)[:top_k]
        for k, v in tmp_list:
            if k in test_set_dict[u]:
                hit += 1
        recalls += hit/len(test_set_dict[u])
        precisions += hit/top_k
        hits += hit
        total_purchase_nb += len(test_set_dict[u])
    recall = recalls/len(users)
    precision = precisions/len(users)
    print('recall:{}, precision:{}'.format(recall, precision))
    return precision, recall


def select_negative_items(batch_history_data, nb_zr, nb_pm):
    '''
    :param history_data:用户和项目交互的信息
    :param nb_zr:zr采样的个数
    :param nb_pm:pm采样的个数
    :return:
    '''
    data = np.array(batch_history_data)
    idx_zr, idx_pm = np.zeros_like(data), np.zeros_like(data)
    for i in range(data.shape[0]):
        # 得到所有为0的项目下标
        items = np.where(data[i] == 0)[0].tolist()
        # 随机抽取一定数量的下标
        tmp_zr = random.sample(items, nb_zr)
        tmp_pm = random.sample(items, nb_pm)
        # 这些位置的值为1
        idx_zr[i][tmp_zr] = 1
        idx_pm[i][tmp_pm] = 1
    return idx_zr, idx_pm


def train_CFGAN(train_set, nb_item, epoches, batch_size, nb_zr, nb_pm, alpha, test_set_dict, top_k):
    '''
    :param train_set: 矩阵，行是用户，列是项目，值表示感兴趣的程度
    :param epoches:迭代次数
    :param batch_size: 每次训练的样本个数
    :param pro_zr: 生成器辅助损失函数上负样本的个数
    :param pro_pm: 判别器上损失函数上负样本的个数
    :param alpha: 控制辅助函数的权重
    :return:
    '''
    # 收集数据
    epoche_list, precision_list = [], []
    # 创建模型
    gen = Gen(nb_item)
    dis = Dis(nb_item)
    # 创建优化器
    gen_opt = torch.optim.Adam(gen.parameters(), lr=0.0001)
    dis_opt = torch.optim.Adam(dis.parameters(), lr=0.0001)
    # 创建判别准则
    loss_mse = torch.nn.MSELoss(reduction='sum')
    loss_bce = torch.nn.BCELoss()
    # 创建标签
    d_real_label = torch.ones(batch_size, 1, dtype=torch.float)
    d_fake_label = torch.zeros(batch_size, 1, dtype=torch.float)
    # 生成器和判别器的训练步幅
    step_gen, step_dis = 5, 2
    for e in range(epoches):
        #------------------------------------------
        # 判别器的训练
        # 1. 选择样本的下标
        # 2. 负采样
        # 3. 训练数据
        # ------------------------------------------
        dis.train()
        gen.eval()
        for step in range(step_dis):
            # begin_idx = random.randint(0, len(train_set)-1-batch_size)
            # condition_vec = torch.tensor(train_set[begin_idx:begin_idx + batch_size], dtype=torch.float)
            idxs = random.sample(range(len(train_set)), batch_size)
            condition_vec = torch.tensor(train_set[idxs], dtype=torch.float)
            # 负采样
            _, idx_pm = select_negative_items(condition_vec, nb_zr, nb_pm)
            idx_pm = torch.tensor(idx_pm)
            eu = condition_vec
            # 真实的部分
            # predict1 = copy.deepcopy(condition_vec)
            predict1 = condition_vec
            # 生成的部分
            predict2 = gen(condition_vec) * (eu+idx_pm)
            loss1 = loss_bce(dis(condition_vec, predict1), d_real_label)
            loss2 = loss_bce(dis(condition_vec, predict2), d_fake_label)

            # 计算判别器的损失
            loss = loss1 + loss2
            dis_opt.zero_grad()
            loss.backward()
            dis_opt.step()
        # ------------------------------------------
        # 生成器的训练
        # ------------------------------------------
        gen.train()
        dis.eval()
        for step in range(step_gen):
            # begin_idx = random.randint(1, len(train_set) - 1 - batch_size)
            # condition_vec = torch.tensor(train_set[begin_idx:begin_idx + batch_size], dtype=torch.float)
            idxs = random.sample(range(len(train_set)), batch_size)
            condition_vec = torch.tensor(train_set[idxs], dtype=torch.float)
            # 负采样
            idx_zr, idx_pm = select_negative_items(condition_vec, nb_zr, nb_pm)
            idx_zr, idx_pm = torch.tensor(idx_zr), torch.tensor(idx_pm)
            eu = condition_vec
            # 生成假数据
            predict = gen(condition_vec)
            predict_pm = predict*(eu+idx_pm)
            # 只计算负样本
            predict_zr = predict*(idx_zr)
            # 判别器的结果
            out_dis = dis(condition_vec, predict_pm)
            # 计算损失函数
            loss1 = loss_bce(out_dis, d_real_label)
            loss2 = alpha*torch.sum((predict_zr-eu).pow(2).sum(dim=-1)/(idx_zr).sum(-1))
            loss = loss1 + loss2
            gen_opt.zero_grad()
            loss.backward()
            gen_opt.step()
        if (e+1) % 1 == 0:
            print(e+1, '\t', '=='*24)
            precision, _ = test(gen, test_set_dict, train_set, top_k=top_k)
            epoche_list.append(e+1)
            precision_list.append(precision)
    plot_precision(epoche_list, precision_list)


def plot_precision(epoche_list, precision_list):
    plt.title('precision')
    plt.plot(epoche_list, precision_list, marker='o')
    plt.savefig('precision_ml100k.png')
    plt.show()


if __name__ == '__main__':
    # ml100k的数据
    nb_user=943
    nb_item=1682
    top_k = 5
    train_set_dict, test_set_dict = read_ml100k('dataset/ml-100k/u1.base', 'dataset/ml-100k/u1.test', sep='\t', header=None)
    train_set, test_set = get_matrix(train_set_dict, test_set_dict, nb_user=nb_user, nb_item=nb_item)
    train_CFGAN(train_set, nb_item, epoches=300, batch_size=32, nb_zr=128, nb_pm=128, alpha=0.1, test_set_dict=test_set_dict, top_k=top_k)

    # # ml1m的数据
    # nb_user=6040
    # nb_item = 3952
    # top_k = 5
    # train_set_dict, test_set_dict = read_ml1m('dataset/ml-1m/ratings.dat')
    # train_set, test_set = get_matrix(train_set_dict, test_set_dict, nb_user=nb_user, nb_item=nb_item)
    # train_CFGAN(train_set, nb_item, epoches=1000, batch_size=32, nb_zr=128, nb_pm=128, alpha=0.1,
    #             test_set_dict=test_set_dict, top_k=top_k)