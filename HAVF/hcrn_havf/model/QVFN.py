import numpy as np
import itertools

import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.nn import functional as F

class QFN(Module):
    def __init__(self, module_dim):
        super(QFN, self).__init__()
        self.query_fc = nn.Linear(2*module_dim, module_dim)
        self.activation = nn.ELU()
    
    def forward(self, ques_feat, visual_feat, audio_feat):
        """
        ques_feat: (bz, module_dim)
        visual_feat: (bz, k, module_dim)
        audio_feat: (bz, module_dim)
        """
        if ques_feat.size(0) != audio_feat.size(0):
            audio_feat = audio_feat.repeat(ques_feat.size(0) // audio_feat.size(0), 1)
        query_feat = torch.cat((ques_feat, audio_feat), dim=-1)
        query_feat = self.query_fc(query_feat)
        query_feat = self.activation(query_feat)
        query_feat = torch.unsqueeze(query_feat, -2) #(bz, 1, module_dim)

        attention_scores = torch.matmul(query_feat, visual_feat.transpose(-1, -2))
        attention_scores = F.softmax(attention_scores, dim=-1) #(bz, 1, k)
        attended_visual_feat = torch.matmul(attention_scores, visual_feat) #(bz, 1, module_dim)
        attended_visual_feat = torch.squeeze(attended_visual_feat, -2) #(bz, module_dim)

        # print('QFN shape:', attended_visual_feat.shape)

        return attended_visual_feat


class VFN(Module):
    def __init__(self, module_dim):
        super(VFN, self).__init__()
        self.value_fc = nn.Linear(2*module_dim, module_dim)
        self.activation = nn.ELU()
    
    def forward(self, ques_feat, visual_feat, audio_feat):
        """
        ques_feat: (bz, module_dim)
        visual_feat: (bz, k, module_dim)
        audio_feat: (bz, module_dim)
        """
        audio_feat = audio_feat.unsqueeze(1)
        if visual_feat.size(0) != audio_feat.size(0):
            audio_feat_repeat = audio_feat.repeat(visual_feat.size(0) // audio_feat.size(0), visual_feat.size(1), 1)
        else:
            audio_feat_repeat = audio_feat.repeat(1, visual_feat.size(1), 1)
        value_feat = torch.cat((visual_feat, audio_feat_repeat), dim=-1) #(bz, k, module_dim*2)
        value_feat = self.value_fc(value_feat) #(bz, k, module_dim)
        value_feat = self.activation(value_feat)

        ques_feat = torch.unsqueeze(ques_feat, -2) #(bz, 1, module_dim)
        attention_scores = torch.matmul(ques_feat, value_feat.transpose(-1, -2))
        attention_scores = F.softmax(attention_scores, dim=-1) #(bz, 1, k)
        attended_value_feat = torch.matmul(attention_scores, value_feat) #(bz, 1, module_dim)
        attended_value_feat = torch.squeeze(attended_value_feat, -2) #(bz, module_dim)

        # print('VFN shape:', attended_value_feat.shape)

        return attended_value_feat
        

class QVFN(Module):
    def __init__(self, module_dim, fuse_type):
        super(QVFN, self).__init__()
        self.fuse_type = fuse_type
        self.qfn = QFN(module_dim)
        self.vfn = VFN(module_dim)
        self.cat = nn.Linear(module_dim*2, module_dim)

    def forward(self, ques_feat, visual_feat, audio_feat):
        if self.fuse_type == 'queryvalue':
            qfn_feat = self.qfn(ques_feat, visual_feat, audio_feat)
            vfn_feat = self.vfn(ques_feat, visual_feat, audio_feat)
            qvfn_feat = torch.cat((qfn_feat, vfn_feat), dim=-1)
            qvfn_feat = self.cat(qvfn_feat)
        elif self.fuse_type == "query":
            qvfn_feat = self.qfn(ques_feat, visual_feat, audio_feat)
        elif self.fuse_type == "value":
            qvfn_feat = self.vfn(ques_feat, visual_feat, audio_feat)

        return qvfn_feat
