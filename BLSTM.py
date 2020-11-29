import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
from numpy import array
from torch.autograd import Variable
import random
import pickle
import random


class BLSTM(nn.Module):
    def __init__(self, args, non_trainable=True):
        super(BLSTM, self).__init__()

        random.seed(1234)
        torch.manual_seed(1234)
        np.random.seed(1234)

        self.args=args
        with open(self.args['input_folder']+'/tokenized_paragraphs.pkl', 'rb') as f:
            [self.weights_matrix, self.segment2tokenized_text] = pickle.load(f)


        self.hidden_dim = 150

        num_embeddings = len(self.weights_matrix)
        embedding_dim = len(self.weights_matrix[0])
        if torch.cuda.is_available():
            self.word_embeddings = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0).cuda()
        else:
            self.word_embeddings = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)

        self.word_embeddings.weight.data.copy_(torch.from_numpy(self.weights_matrix))

        if non_trainable:
            self.word_embeddings.weight.requires_grad = False

       
        if torch.cuda.is_available():
            self.lstm = nn.LSTM(300, self.hidden_dim, bidirectional=True, batch_first=True, dropout=0).cuda()
        else:
            self.lstm = nn.LSTM(300, self.hidden_dim, bidirectional=True, batch_first=True, dropout=0)

    def init_hidden(self, batch_size):
        if torch.cuda.is_available():
            return (torch.zeros(2, batch_size, self.hidden_dim).cuda(),
                    torch.zeros(2, batch_size, self.hidden_dim).cuda())
        return (torch.zeros(2, batch_size, self.hidden_dim),
                torch.zeros(2, batch_size, self.hidden_dim))

    def get_padded(self, X):
        # print X
        X_lengths = [len(sentence) for sentence in X]
        pad_token = 0
        longest_sent = max(X_lengths)
        batch_size = len(X)
        padded_X = np.ones((batch_size, longest_sent)) * pad_token

        for i, x_len in enumerate(X_lengths):
            sequence = X[i]
            padded_X[i, 0:x_len] = sequence[:x_len]
        return padded_X, X_lengths




    def forward(self, X):
       
        self.hidden = self.init_hidden(len(X))
        X = [self.segment2tokenized_text[seg_id] for seg_id in X]
        X, X_lengths=self.get_padded(X)

        if torch.cuda.is_available():
            X = self.word_embeddings(Variable(torch.cuda.LongTensor(np.array(X))))
        else:
            X = self.word_embeddings(Variable(torch.LongTensor(np.array(X))))

        seg_sort_index = sorted(range(len(X_lengths)), key=lambda k: X_lengths[k], reverse=True)
        seg_sort_index_map = {old: new for new, old in enumerate(seg_sort_index)}
        reverse_seg_index = [seg_sort_index_map[i] for i in range(len(seg_sort_index))]
        reverse_seg_index_var = torch.LongTensor(reverse_seg_index)

        if torch.cuda.is_available():
            reverse_seg_index_var = reverse_seg_index_var.cuda()

        seg_lengths_sort = sorted(X_lengths, reverse=True)
        
        X_sort = torch.cat([X[i].unsqueeze(0) for i in seg_sort_index], 0)
        X = torch.nn.utils.rnn.pack_padded_sequence(X_sort, seg_lengths_sort, batch_first=True)

        X, self.hidden = self.lstm(X, self.hidden)

        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        if torch.cuda.is_available():
            seg_embeds = torch.index_select(X, 0, reverse_seg_index_var).sum(1) / torch.cuda.FloatTensor(array(X_lengths)).view((-1, 1))
        else:
            seg_embeds = torch.index_select(X, 0, reverse_seg_index_var).sum(1) / torch.FloatTensor(array(X_lengths)).view((-1, 1))

        return seg_embeds
