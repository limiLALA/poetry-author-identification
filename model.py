#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextRNN(nn.Module):
    """文本分类，RNN模型"""

    def __init__(self, num_embeddings, num_labels):
        super(TextRNN, self).__init__()
        # 三个待输入的数据
        self.embedding = nn.Embedding(num_embeddings, 64)  # 进行词嵌入
        # self.rnn = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, bidirectional=True)
        self.rnn = nn.GRU(input_size=64, hidden_size=128, num_layers=2, bidirectional=True)
        self.f1 = nn.Sequential(nn.Linear(256, 128),
                                nn.Dropout(0.8),
                                nn.ReLU())
        self.f2 = nn.Sequential(nn.Linear(128, num_labels),
                                nn.Softmax())

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = F.dropout(x, p=0.8)
        x = self.f1(x[:, -1, :])
        return self.f2(x)


class TextCNN(nn.Module):
    def __init__(self, num_embeddings, num_labels):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, 64)
        self.conv = nn.Conv1d(64, 256, 5)
        self.f1 = nn.Sequential(nn.Linear(256 * 596, 128),
                                nn.ReLU())
        self.f2 = nn.Sequential(nn.Linear(128, num_labels),
                                nn.Softmax())

    def forward(self, x):
        x = self.embedding(x)
        x = x.detach().cpu().numpy()
        x = np.transpose(x, [0, 2, 1])
        x = torch.Tensor(x).to(device)
        x = Variable(x)
        x = self.conv(x)
        x = x.view(-1, 256 * 596)
        x = self.f1(x)
        return self.f2(x)
