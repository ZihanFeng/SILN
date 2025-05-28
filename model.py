import torch.nn as nn
from math import sqrt

from dataLoader import dataProcess
from module import SubGL,TCAN,SSAN,Fusion


class SILN(nn.Module):
    def __init__(self, args):
        super(SILN, self).__init__()
        self.dim = args.dim
        self.user_num = args.user_num

        self.SubGL = SubGL(args)
        self.TCAN = TCAN(args)
        self.SSAN = SSAN(args)

        self.Fusion = Fusion(args)
        self.Predict = nn.Linear(self.dim, self.user_num)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, args, data):
        cascade, cas_mask, label, label_mask, neighbor, relation, dis = dataProcess(args, data)

        # # Step1: Cascade Induced Subgraph Learning Module
        casEmbed = self.SubGL(cascade, neighbor, relation)

        # # Step2: Dual-View Attention Networks (Temporal and Structural)
        ht = self.TCAN(casEmbed, cas_mask)
        hs = self.SSAN(casEmbed, cas_mask, dis)
        h = self.Fusion(ht, hs)

        # # Information Diffusion Prediction (Participant v_{k+1} for C_k)
        pred_user = self.Predict(h) + label_mask
        pred_user = pred_user.view(-1, pred_user.size(-1))

        return pred_user, label
