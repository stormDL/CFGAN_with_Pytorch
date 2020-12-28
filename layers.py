import torch
import torch.nn as nn


class Dis(nn.Module):
    def __init__(self, nb_item):
        '''
        :param nb_item: 项目的数量
        '''
        super(Dis, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(nb_item*2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, condition, predict):
        data = torch.cat([condition, predict], dim=-1)
        out = self.dis(data)
        return out


class Gen(nn.Module):
    def __init__(self, nb_item):
        super(Gen, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(nb_item, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, nb_item),
            nn.Sigmoid()
        )

    def forward(self, purchase_vec):
        out = self.gen(purchase_vec)
        return out