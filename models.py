import torch
import torch.nn as nn


class G(nn.Module):
    def __init__(self, u_info_len):
        super(G, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(u_info_len, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, u_info_len),
            nn.Sigmoid())

    def forward(self, u_info):
        x = u_info
        out = self.gen(x)
        return out


class D(nn.Module):
    def __init__(self, u_info_len, condition_len):
        super(D, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(u_info_len + condition_len, 512),
            nn.ReLU(True),
            nn.Linear(512, 64),
            nn.ReLU(True),
            nn.Linear(64, 1),
            nn.Sigmoid())

    def forward(self, u_info, condition):
        x = torch.cat([u_info, condition], 1)
        out = self.dis(x)
        return out