import argparse

import torch
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix

import time
from tqdm import tqdm

from collections import defaultdict
import random

from collections import OrderedDict

from torch.utils.data import DataLoader,TensorDataset
import torch.nn.functional as F
from torch import optim, nn

import random

from model import Encoder, Decoder_s, Decoder_u, Decoder_t
def run(user_nums, item_nums, train_df, valid_df, test_df, args):
    Recall_valid, MRR_valid, Recall_test, MRR_test = 0, 0, 0, 0
    weight_user = 1 - args.weight_basic - args.weight_item
    hidden = [item_nums, args.dim]

    encoder = Encoder(hidden).to(args.cuda)
    decoder_s = Decoder_s(hidden).to(args.cuda)
    decoder_u = Decoder_u(hidden).to(args.cuda)
    decoder_t = Decoder_t(hidden).to(args.cuda)

    opt1 = optim.Adam(encoder.parameters(), args.lr_enc)
    opt2 = optim.Adam(decoder_s.parameters(), args.lr_dec_s)
    opt3 = optim.Adam(decoder_u.parameters(), args.lr_dec_u)
    opt4 = optim.Adam(decoder_t.parameters(), args.lr_dec_t)

    for phase in ['train','test']:
        # in train phase, we can only see the training data
        if phase == 'train':
            df = train_df
            z = zip(valid_df['user_id'], valid_df['item_id'], valid_df['timestamp'])
        # in test phase, we can see the training data and validation data
        else:
            df = pd.concat([train_df, valid_df], axis=0)
            z = zip(test_df['user_id'], test_df['item_id'], test_df['timestamp'])
        user_list, item_list, time_list = list(df['user_id']), list(df['item_id']), list(df['timestamp'])
        
        # construct user-item matrix
        data = np.ones([len(time_list)])
        user_graph = coo_matrix((data, [user_list, item_list]), shape=(user_nums, item_nums))
        user_graph_ = torch.from_numpy(user_graph.todense()).float().to(args.cuda)
        
        # construct item-item matrix
        user_sequences = defaultdict(list)
        for x in range(df.shape[0]):
            u, i, _ = df.iloc[x]
            user_sequences[u].append(i)

        item_graph = np.zeros((item_nums, item_nums))
        total_ctr = 0

        for sequence in user_sequences.values():
            sender = sequence[:len(sequence) - 1]
            receive = sequence[1:len(sequence)]
            sender_int = list(map(int, sender))
            receive_int = list(map(int, receive))
            total_ctr += len(sender)
            item_graph[sender_int, receive_int] += 1
        
        item_graph_ = torch.from_numpy(item_graph).float().to(args.cuda)
        item_graph_t = torch.transpose(item_graph_, dim0=1, dim1=0)
        
        dataset_s = TensorDataset(item_graph_)
        dataloader_s = DataLoader(dataset_s,batch_size=args.batch,shuffle=True)
        dataset_t = TensorDataset(item_graph_t)
        dataloader_t = DataLoader(dataset_t,batch_size=args.batch,shuffle=True)
        
        dataset_u = TensorDataset(user_graph_)
        dataloader_u = DataLoader(dataset_u,batch_size=args.batch,shuffle=True)
        
        combine_batch_list, task_label_list = [], []
        for item_batch in dataloader_s:
            combine_batch_list.append(item_batch[0])
            task_label_list.append('s')
        for item_batch_t in dataloader_t:
            combine_batch_list.append(item_batch_t[0])
            task_label_list.append('t')
        
        for user_batch in dataloader_u:
            combine_batch_list.append(user_batch[0])
            task_label_list.append('u')
        
        index = [i for i in range(len(combine_batch_list))] 
        random.shuffle(index)
        new_data, new_label = [], []
        for i in index:
            new_data.append(combine_batch_list[i])
            new_label.append(task_label_list[i])
            
        if phase == 'train':
            for epoch in range(args.epoch):
                loss_sum_s, loss_sum_u, loss_sum_t, length_s, length_u, length_t = 0, 0, 0, 0, 0, 0
                encoder.train()
                decoder_s.train()
                decoder_u.train()
                decoder_t.train()
                for combined_batch in range(len(new_data)):
                    input_value = new_data[combined_batch].to(args.cuda)
                    encoded_emb = encoder(input_value)
    #                 input_value = input_value.to(torch.float64)
    #                 mask = torch.where(input_value == 0, input_value, 1.0)
                    if (new_label[combined_batch] == 's'):
                        opt1.zero_grad()
                        opt2.zero_grad()
                        output = decoder_s(encoded_emb)
    #                     loss_i = F.mse_loss(output * mask, input_value * mask)
                        loss_s = F.mse_loss(output, input_value)
                        loss_s.backward()
                        opt1.step()
                        opt2.step()
                        
                        length_s += output.shape[0]
                        loss_sum_s += loss_s.item()*output.shape[0]
                    elif (new_label[combined_batch] == 'u'):
                        opt1.zero_grad()
                        opt3.zero_grad()
                        output = decoder_u(encoded_emb)
                        
    #                     input_value = input_value.to(torch.float64)
    #                     mask = torch.where(input_value == 0, input_value, 1.0)
    #                     loss_u = F.mse_loss(output * mask, input_value * mask)
                        loss_u = F.mse_loss(output, input_value)
                        loss_u.backward()
                        opt1.step()
                        opt3.step()

                        length_u += output.shape[0]
                        loss_sum_u += loss_u.item()*output.shape[0]
                    else:
                        opt1.zero_grad()
                        opt4.zero_grad()
                        output = decoder_t(encoded_emb)
    #                     loss_it = F.mse_loss(output * mask, input_value * mask)
                        loss_t = F.mse_loss(output, input_value)
                        loss_t.backward()
                        opt1.step()
                        opt4.step()

                        length_t += output.shape[0]
                        loss_sum_t += loss_t.item()*output.shape[0]
                    
                log_str_ = "tr - {} | mse_sgraph - {:.4f} | mse_ugraph - {:.4f} | mse_tgraph - {:.4f} |".format(epoch, loss_sum_s / length_s, loss_sum_u / length_u, loss_sum_t / length_t)
                print()
                print(log_str_)
            
            with torch.no_grad():
                encoder.eval()
                decoder_s.eval()
                decoder_u.eval()
                decoder_t.eval()
                
                item_graph_re = encoder(item_graph_)
                item_graph_con = decoder_s(item_graph_re)
            
                item_graph_t_re = encoder(item_graph_t)
                item_graph_t_con = decoder_t(item_graph_t_re)
                
                user_graph_re = encoder(user_graph_)
                user_graph_con = decoder_u(user_graph_re)

                # initial users_pred
                users_pred = torch.zeros(user_nums,item_nums)
                for uid in user_sequences.keys():
                    item_graph_uid = item_graph_re[int(user_sequences[uid][-1])] # select the last item
                    
                    user_item_uid = torch.unsqueeze(user_graph_re[int(uid)],dim=0)
                    uid_trans_mat = torch.mul(item_graph_uid, user_item_uid)
    #                 users_pred[int(uid)] = torch.mm(uid_trans_mat,torch.transpose(item_graph_t_re, dim0=1, dim1=0)) # [1,d] * [d, item_num]
                    basic_re = torch.mm(uid_trans_mat,torch.transpose(item_graph_t_re, dim0=1, dim1=0)) # [1,d] * [d, item_num]
                    
                    add_item_re = item_graph_con[int(user_sequences[uid][-1])]
                    add_user_re = torch.unsqueeze(user_graph_con[int(uid)],dim=0)
                    users_pred[int(uid)] = args.weight_basic * basic_re + args.weight_item * add_item_re + weight_user * add_user_re

                # update users_pred
                Recall, MRR = 0, 0
                count_first_appears = 0
                for cnt, tup in tqdm(enumerate(z)):
                    uid, iid, t = tup[0], tup[1], tup[2]
                    if uid not in user_sequences.keys():
    #                     pred_index = list(range(0,item_nums))
    #                     random.shuffle(pred_index)
                        _, pred_index = users_pred[int(uid)].sort(descending=True)
                        user_sequences[uid].append(iid)
                        user_graph_[int(uid)][int(iid)] += 1
                        count_first_appears += 1
                    else:             
                        item_graph_re = encoder(item_graph_)
                        item_graph_con = decoder_s(item_graph_re)
                
                        item_graph_t_re = encoder(item_graph_t)
                
                        user_graph_re = encoder(user_graph_)
                        user_graph_con = decoder_u(user_graph_re)
                        
                        item_graph_uid = item_graph_re[int(user_sequences[uid][-1])] # select the last item
                    
                        user_item_uid = torch.unsqueeze(user_graph_re[int(uid)],dim=0)
                        uid_trans_mat = torch.mul(item_graph_uid, user_item_uid) # [1,d] 
    #                     users_pred[int(uid)] = torch.mm(uid_trans_mat,torch.transpose(item_graph_t_re, dim0=1, dim1=0)) # [1,d] * [d, item_num]
                        basic_re = torch.mm(uid_trans_mat,torch.transpose(item_graph_t_re, dim0=1, dim1=0)) # [1,d] * [d, item_num]
                        add_item_re = item_graph_con[int(user_sequences[uid][-1])]
                        add_user_re = torch.unsqueeze(user_graph_con[int(uid)],dim=0)
                        users_pred[int(uid)] = args.weight_basic * basic_re + args.weight_item * add_item_re + weight_user * add_user_re
        
                        users_pred[int(uid)][user_sequences[uid]] = -(1 << 10)
                        _, pred_index = users_pred[int(uid)].sort(descending=True)

                        item_graph_[int(user_sequences[uid][-1])][iid] += 1
                        item_graph_t[iid][int(user_sequences[uid][-1])] += 1
                        user_sequences[uid].append(iid)
                        user_graph_[int(uid)][int(iid)] += 1

                    if iid in pred_index[: args.topk]:
                        Recall += 1
                    MRR += 1 / (list(pred_index).index(int(iid)) + 1)

                Recall_valid = Recall / len(valid_df)
                MRR_valid = MRR / len(valid_df)
            
        else:
            with torch.no_grad():
                encoder.eval()
                decoder_s.eval()
                decoder_u.eval()
                decoder_t.eval()
                
                item_graph_re = encoder(item_graph_)
                item_graph_con = decoder_s(item_graph_re)
            
                item_graph_t_re = encoder(item_graph_t)
                
                user_graph_re = encoder(user_graph_)
                user_graph_con = decoder_u(user_graph_re)

                # initial users_pred
                users_pred = torch.zeros(user_nums,item_nums)
                for uid in user_sequences.keys():
                    item_graph_uid = item_graph_re[int(user_sequences[uid][-1])] # select the last item
                    
                    user_item_uid = torch.unsqueeze(user_graph_re[int(uid)],dim=0)
                    uid_trans_mat = torch.mul(item_graph_uid, user_item_uid)
    #                 users_pred[int(uid)] = torch.mm(uid_trans_mat,torch.transpose(item_graph_t_re, dim0=1, dim1=0)) # [1,d] * [d, item_num]
                    basic_re = torch.mm(uid_trans_mat,torch.transpose(item_graph_t_re, dim0=1, dim1=0)) # [1,d] * [d, item_num]

                    add_item_re = item_graph_con[int(user_sequences[uid][-1])]
                    add_user_re = torch.unsqueeze(user_graph_con[int(uid)],dim=0)
                    users_pred[int(uid)] = args.weight_basic * basic_re + args.weight_item * add_item_re + weight_user * add_user_re
            
                # update users_pred
                Recall, MRR = 0, 0
                count_first_appears = 0
                for cnt, tup in tqdm(enumerate(z)):
                    uid, iid, t = tup[0], tup[1], tup[2]
                    if uid not in user_sequences.keys():
    #                     pred_index = list(range(0,item_nums))
    #                     random.shuffle(pred_index)
                        _, pred_index = users_pred[int(uid)].sort(descending=True)
                        user_sequences[uid].append(iid)
                        user_graph_[int(uid)][int(iid)] += 1
                        count_first_appears += 1
                    else:             
                        item_graph_re = encoder(item_graph_)
                        item_graph_con = decoder_s(item_graph_re)
                
                        item_graph_t_re = encoder(item_graph_t)
                
                        user_graph_re = encoder(user_graph_)
                        user_graph_con = decoder_u(user_graph_re)
                        
                        item_graph_uid = item_graph_re[int(user_sequences[uid][-1])] # select the last item
                    
                        user_item_uid = torch.unsqueeze(user_graph_re[int(uid)],dim=0)
                        uid_trans_mat = torch.mul(item_graph_uid, user_item_uid) # [1,d] 
    #                     users_pred[int(uid)] = torch.mm(uid_trans_mat,torch.transpose(item_graph_t_re, dim0=1, dim1=0)) # [1,d] * [d, item_num]
                        basic_re = torch.mm(uid_trans_mat,torch.transpose(item_graph_t_re, dim0=1, dim1=0)) # [1,d] * [d, item_num]
            
                        add_item_re = item_graph_con[int(user_sequences[uid][-1])]
                        add_user_re = torch.unsqueeze(user_graph_con[int(uid)],dim=0)
                        users_pred[int(uid)] = args.weight_basic * basic_re + args.weight_item * add_item_re + weight_user * add_user_re
            
                        users_pred[int(uid)][user_sequences[uid]] = -(1 << 10)
                        _, pred_index = users_pred[int(uid)].sort(descending=True)

                        item_graph_[int(user_sequences[uid][-1])][iid] += 1
                        item_graph_t[iid][int(user_sequences[uid][-1])] += 1
                        user_sequences[uid].append(iid)
                        user_graph_[int(uid)][int(iid)] += 1

                    if iid in pred_index[: args.topk]:
                        Recall += 1
                    MRR += 1 / (list(pred_index).index(int(iid)) + 1)

                Recall_test = Recall / len(test_df)
                MRR_test = MRR / len(test_df)

    return Recall_valid, MRR_valid, Recall_test, MRR_test

def output(Recall_valid, MRR_valid, Recall_test, MRR_test, args):
    Str = '{:.3f}\t'.format(Recall_valid) + \
          '{:.3f}\t'.format(MRR_valid) + \
          '{:.3f}\t'.format(Recall_test) + \
          '{:.3f}\n'.format(MRR_test)
    print(Str)