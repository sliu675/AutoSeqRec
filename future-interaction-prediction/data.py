import argparse

import torch
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['ml-100k', 'ml-1m', 'video', 'garden'])
    parser.add_argument('--cuda', default=0, type=int)
    parser.add_argument('--dim', default=256, type=int)
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--lr_enc', default=1e-05, type=float)
    parser.add_argument('--lr_dec_s', default=0.01, type=float)
    parser.add_argument('--lr_dec_u', default=0.001, type=float)
    parser.add_argument('--lr_dec_t', default=1e-05, type=float)
    parser.add_argument('--batch', default=64, type=int)
    parser.add_argument('--weight_basic', default=0.2, type=float)
    parser.add_argument('--weight_item', default=0.2, type=float)
    args = parser.parse_args()
    args.cuda = torch.device('cuda:{}'.format(args.cuda))
    # args.split = [0.8, 0.1, 0.1]
    args.topk = 10
    args.dtype = np.float32
    return args

def get_data(args):
    data_file = '../data/{}.csv'.format(args.dataset)
    df = pd.read_csv(data_file, usecols=[0, 1, 2], names=["user_id", "item_id", "timestamp"])
#     df = pd.read_csv(data_file, index_col=False, usecols=[0, 1, 2])
    user_nums = df['user_id'].nunique()
    item_nums = df['item_id'].nunique()
    interaction_nums = len(df)
    train_end_index = int(interaction_nums * 0.8)
    valid_end_index = int(interaction_nums * 0.9)
    test_end_index = int(interaction_nums)
    train_df = df.iloc[:train_end_index]
    valid_df = df.iloc[train_end_index:valid_end_index]
    test_df = df.iloc[valid_end_index:test_end_index]
    
    return user_nums, item_nums, train_df, valid_df, test_df