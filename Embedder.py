import collections
import glob
import math
from itertools import chain
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from numpy import array
import pickle
import time
import torch.nn as nn
from BLSTM import *
import random


class Embedder(torch.nn.Module):
    def __init__(self, args, cuda_available):
        super(Embedder, self).__init__()
        
        self.args=args
        
        with open(self.args['input_folder']+'/graph_info.pkl', 'rb') as in_file:
            [self.graph, self.id2name, self.name2id, self.doc_start, self.doc_end,\
             self.bigram_start, self.bigram_end, self.trigram_start, self.trigram_end,\
             self.subframe_start, self.subframe_end]=pickle.load(in_file)

        self.doc_count=self.doc_end-self.doc_start+1
        self.bigram_count=self.bigram_end-self.bigram_start+1
        self.trigram_count=self.trigram_end-self.trigram_start+1
        self.subframe_count=self.subframe_end-self.subframe_start+1

        self.cuda_available=cuda_available
        self.embedding_size = 300

        self.embed_rng = np.random.RandomState(1)
        self.scale = 1.0 / math.sqrt(self.embedding_size)

        
        # Initialization of bigram, trigram and subframe embeddings randomly
        if self.cuda_available:
            self.bigram_embeddings = nn.Embedding(self.bigram_count, self.embedding_size).cuda()
            self.trigram_embeddings = nn.Embedding(self.trigram_count, self.embedding_size).cuda()
            self.subframe_embeddings = nn.Embedding(self.subframe_count, self.embedding_size).cuda()
        else:
            self.bigram_embeddings = nn.Embedding(self.bigram_count, self.embedding_size)
            self.trigram_embeddings = nn.Embedding(self.trigram_count, self.embedding_size)
            self.subframe_embeddings = nn.Embedding(self.subframe_count, self.embedding_size)

        bigram_embed_values = self.embed_rng.normal(scale=self.scale, size=(self.bigram_count, self.embedding_size))
        trigram_embed_values = self.embed_rng.normal(scale=self.scale, size=(self.trigram_count, self.embedding_size))
        subframe_embed_values = self.embed_rng.normal(scale=self.scale, size=(self.subframe_count, self.embedding_size))

        self.bigram_embeddings.weight.data.copy_(torch.from_numpy(bigram_embed_values))
        self.trigram_embeddings.weight.data.copy_(torch.from_numpy(trigram_embed_values))
        self.subframe_embeddings.weight.data.copy_(torch.from_numpy(subframe_embed_values))

        # Initialization of the Bi-LSTM text encoder
        if self.cuda_available:
            self.doc_embeddings = BLSTM(self.args).cuda()
        else:
            self.doc_embeddings = BLSTM(self.args)

        self.CrossEntropyLoss = nn.CrossEntropyLoss(reduction='mean')
        self.aspects = ['text2bigram', 'text2trigram', 'bigram2subframe', 'trigram2subframe']



    def forward(self, batch):

        batch_input_index = batch[0]
        batch_gold_index = batch[1]
        batch_negs_index = batch[2]
        text_nodes = batch[3]

        if len(text_nodes) > 0:
            textid2index = {}
            for i in range(0, len(text_nodes)):
                textid2index[text_nodes[i]] = i
            if self.cuda_available:
                output_doc_embeddings = self.doc_embeddings(text_nodes).cuda()
            else:
                output_doc_embeddings = self.doc_embeddings(text_nodes)

        embedding_size = self.embedding_size
        batch_target_index = {}
        batch_input = {}
        batch_target = {}
        input_embed = {}
        target_embed = {}
        sim_score = {}
        loss_all = {}
        loss = 0

        for aspect in self.aspects:
            if (len(batch_gold_index[aspect]) == 0):
                continue
            
            if (aspect == 'text2bigram'):
                #print ('text2bigram')
                target_embeddings = self.bigram_embeddings
            elif (aspect == 'text2trigram'):
                #print ('text2trigram')
                target_embeddings = self.trigram_embeddings
            elif (aspect == 'bigram2subframe'):
                #print ('bigram2sf')
                target_embeddings = self.subframe_embeddings
            elif (aspect == 'trigram2subframe'):
                #print ('trigram2sf')
                target_embeddings = self.subframe_embeddings
            else:
                continue

            
            


            if (aspect == 'text2bigram'):

                if self.cuda_available:
                    index = [textid2index[idx] for idx in batch_input_index[aspect]]
                    batch_input[aspect] = output_doc_embeddings[index]
                    batch_target_index[aspect] = torch.cat((Variable(
                        torch.cuda.LongTensor(batch_gold_index[aspect] - self.bigram_start)), Variable(
                        torch.cuda.LongTensor(batch_negs_index[aspect] - self.bigram_start))), 1)
                else:
                    index = [textid2index[idx] for idx in batch_input_index[aspect]]
                    batch_input[aspect] = output_doc_embeddings[index]
                    batch_target_index[aspect] = torch.cat((Variable(
                        torch.LongTensor(batch_gold_index[aspect] - self.bigram_start)), Variable(
                        torch.LongTensor(batch_negs_index[aspect] - self.bigram_start))), 1)


            elif (aspect == 'text2trigram'):

                if self.cuda_available:
                    index = [textid2index[idx] for idx in batch_input_index[aspect]]
                    batch_input[aspect] = output_doc_embeddings[index]
                    batch_target_index[aspect] = torch.cat((Variable(
                        torch.cuda.LongTensor(batch_gold_index[aspect] - self.trigram_start)), Variable(
                        torch.cuda.LongTensor(batch_negs_index[aspect] - self.trigram_start))), 1)
                else:
                    index = [textid2index[idx] for idx in batch_input_index[aspect]]
                    batch_input[aspect] = output_doc_embeddings[index]
                    batch_target_index[aspect] = torch.cat((Variable(
                        torch.LongTensor(batch_gold_index[aspect] - self.trigram_start)), Variable(
                        torch.LongTensor(batch_negs_index[aspect] - self.trigram_start))), 1)

            
            elif (aspect == 'bigram2subframe'):
                if self.cuda_available:
                    batch_input[aspect] = self.bigram_embeddings(
                        Variable(torch.cuda.LongTensor(batch_input_index[aspect] - self.bigram_start))).view(
                        (-1, self.embedding_size))
                    batch_target_index[aspect] = torch.cat((Variable(
                        torch.cuda.LongTensor(batch_gold_index[aspect] - self.subframe_start)), Variable(
                        torch.cuda.LongTensor(batch_negs_index[aspect] - self.subframe_start))), 1)
                else:
                    batch_input[aspect] = self.bigram_embeddings(
                        Variable(torch.LongTensor(batch_input_index[aspect] - self.bigram_start))).view(
                        (-1, self.embedding_size))
                    batch_target_index[aspect] = torch.cat((Variable(
                        torch.LongTensor(batch_gold_index[aspect] - self.subframe_start)), Variable(
                        torch.LongTensor(batch_negs_index[aspect] - self.subframe_start))), 1)

                   
            elif (aspect == 'trigram2subframe'):
                if self.cuda_available:
                    batch_input[aspect] = self.trigram_embeddings(
                        Variable(torch.cuda.LongTensor(batch_input_index[aspect] - self.trigram_start))).view(
                        (-1, self.embedding_size))
                    batch_target_index[aspect] = torch.cat((Variable(
                        torch.cuda.LongTensor(batch_gold_index[aspect] - self.subframe_start)), Variable(
                        torch.cuda.LongTensor(batch_negs_index[aspect] - self.subframe_start))), 1)
                else:
                    batch_input[aspect] = self.trigram_embeddings(
                        Variable(torch.LongTensor(batch_input_index[aspect] - self.trigram_start))).view(
                        (-1, self.embedding_size))
                    batch_target_index[aspect] = torch.cat((Variable(
                        torch.LongTensor(batch_gold_index[aspect] - self.subframe_start)), Variable(
                        torch.LongTensor(batch_negs_index[aspect] - self.subframe_start))), 1)

            example_size = 5 + 1    # 5 negative samples per 1 positive example

            batch_target[aspect] = target_embeddings(batch_target_index[aspect]).view((-1, example_size, self.embedding_size))

            dropout = nn.Dropout(p=0.7)
            input_layers = [dropout(batch_input[aspect])]
            input_embed[aspect] = input_layers[-1]

            target_layers = [dropout(batch_target[aspect])]
            target_embed[aspect] = target_layers[-1]

            sim_score[aspect] = torch.bmm(
                target_embed[aspect],
                input_embed[aspect].view(-1, embedding_size, 1)).view(-1, example_size)

            if self.cuda_available:
                target = Variable(torch.cuda.LongTensor(batch_input[aspect].size(0)).zero_())
            else:
                target = Variable(torch.LongTensor(batch_input[aspect].size(0)).zero_())


            self.CrossEntropyLoss = nn.CrossEntropyLoss(reduction='mean')
            loss_all[aspect] = self.CrossEntropyLoss(sim_score[aspect], target)

            loss_all[aspect] = torch.sum(loss_all[aspect])


            rate = 1.0
            loss += loss_all[aspect] * rate

        return loss, loss_all


    
    
    
