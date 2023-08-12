import argparse

import torch
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix

from tqdm import tqdm

from collections import defaultdict
import random

from collections import OrderedDict

from torch.utils.data import DataLoader,TensorDataset
import torch.nn.functional as F
from torch import optim, nn

import random

class Encoder(nn.Module):
    def __init__(self, hidden, dropout=0.3):
        super(Encoder, self).__init__()
        d1 = OrderedDict()
        for i in range(len(hidden)-1):
            d1['enc_linear' + str(i)] = nn.Linear(hidden[i], hidden[i + 1])
            d1['enc_drop' + str(i)] = nn.Dropout(dropout)
            d1['enc_relu'+str(i)] = nn.ReLU() 
        self.encoder = nn.Sequential(d1)

    def forward(self, x):
        x = self.encoder(x)
        return x

class Decoder_s(nn.Module):
    def __init__(self, hidden, dropout=0.2):
        super(Decoder_s, self).__init__()
        d2 = OrderedDict()
        for i in range(len(hidden) - 1, 0, -1):
            d2['dec_linear' + str(i)] = nn.Linear(hidden[i], hidden[i - 1])
            d2['dec_drop' + str(i)] = nn.Dropout(dropout)
            d2['dec_relu' + str(i)] = nn.ReLU()
        self.decoder = nn.Sequential(d2)

    def forward(self, x):
        x = self.decoder(x)
        return x

class Decoder_t(nn.Module):
    def __init__(self, hidden, dropout=0.2):
        super(Decoder_t, self).__init__()
        d3 = OrderedDict()
        for i in range(len(hidden) - 1, 0, -1):
            d3['dec_linear' + str(i)] = nn.Linear(hidden[i], hidden[i - 1])
            d3['dec_drop' + str(i)] = nn.Dropout(dropout)
            d3['dec_relu' + str(i)] = nn.ReLU()
        self.decoder = nn.Sequential(d3)

    def forward(self, x):
        x = self.decoder(x)
        return x

class Decoder_u(nn.Module):
    def __init__(self, hidden, dropout=0):
        super(Decoder_u, self).__init__()
        d4 = OrderedDict()
        for i in range(len(hidden) - 1, 0, -1):
            d4['dec_linear' + str(i)] = nn.Linear(hidden[i], hidden[i - 1])
            d4['dec_drop' + str(i)] = nn.Dropout(dropout)
            d4['dec_relu' + str(i)] = nn.Sigmoid()
        self.decoder = nn.Sequential(d4)

    def forward(self, x):
        x = self.decoder(x)
        return x