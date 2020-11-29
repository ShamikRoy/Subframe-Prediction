import numpy
from random import shuffle
from Embedder import *
import torch
import torch.optim as optim
import time
import pickle
import argparse


class NodeInputData:

    def __init__(self, args):
        
        self.args=args

        with open(self.args['input_folder']+'/graph_info.pkl', 'rb') as in_file:
            [self.graph, self.id2name, self.name2id, self.doc_start, self.doc_end,\
             self.bigram_start, self.bigram_end, self.trigram_start, self.trigram_end,\
             self.subframe_start, self.subframe_end]=pickle.load(in_file)

        self.all_docs = set([i for i in range(self.doc_start, self.doc_end + 1)])
        self.all_bigrams = set([i for i in range(self.bigram_start, self.bigram_end + 1)])
        self.all_trigrams = set([i for i in range(self.trigram_start, self.trigram_end + 1)])
        self.all_subframes = set([i for i in range(self.subframe_start, self.subframe_end + 1)])

        # Initialization of positive adjacent list for each element
        self.text2bigram_adj_pos = {}
        self.text2trigram_adj_pos = {}
        self.bigram2subframe_adj_pos = {}
        self.trigram2subframe_adj_pos = {}
        
        # Initialization of negative adjacent list for each element
        self.text2bigram_adj_neg = {}
        self.text2trigram_adj_neg = {}
        self.bigram2subframe_adj_neg = {}
        self.trigram2subframe_adj_neg = {}

        self.batch_size = 30  # batch size of 30 for the embedding learning
        self.neg_sample_rate = 5 # 5 negative examples per positive example

        self.all_nodes_to_train = None
        self.batches = None


    def get_nodes(self, training_graph):
        
        # Creating positive and negative adjacent lists for each paragraph
        for doc in self.all_docs:
            adjacency_list = training_graph[doc]

            if len(adjacency_list & self.all_bigrams) > 0:
                self.text2bigram_adj_pos[doc] = adjacency_list & self.all_bigrams
                self.text2bigram_adj_neg[doc] = self.all_bigrams - self.text2bigram_adj_pos[doc]

            if len(adjacency_list & self.all_trigrams) > 0:
                self.text2trigram_adj_pos[doc] = adjacency_list & self.all_trigrams
                self.text2trigram_adj_neg[doc] = self.all_trigrams - self.text2trigram_adj_pos[doc]

        # Creating positive and negative adjacent lists for each bigram
        for bigram in self.all_bigrams:
            adjacency_list = training_graph[bigram]

            if len(adjacency_list & self.all_subframes) > 0:
                self.bigram2subframe_adj_pos[bigram] = adjacency_list & self.all_subframes
                self.bigram2subframe_adj_neg[bigram] = self.all_subframes - self.bigram2subframe_adj_pos[bigram]

        # Creating positive and negative adjacent lists for each trigram
        for trigram in self.all_trigrams:
            adjacency_list = training_graph[trigram]

            if len(adjacency_list & self.all_subframes) > 0:
                self.trigram2subframe_adj_pos[trigram] = adjacency_list & self.all_subframes
                self.trigram2subframe_adj_neg[trigram] = self.all_subframes - self.trigram2subframe_adj_pos[trigram]

        self.all_nodes_to_train = [i for i in training_graph]

        shuffle(self.all_nodes_to_train)
    
        # Creation of batches for embedding learning
        
        batches = []

        j = 0
        while j<len(self.all_nodes_to_train):
            batch_input = {'text2bigram':[], 'text2trigram':[], 'bigram2subframe':[], 'trigram2subframe':[]}
            batch_gold = {'text2bigram':[], 'text2trigram':[], 'bigram2subframe':[], 'trigram2subframe':[]}
            batch_neg = {'text2bigram':[], 'text2trigram':[], 'bigram2subframe':[], 'trigram2subframe':[]}
            text_nodes=[]
            
            if j+self.batch_size<=len(self.all_nodes_to_train)-1:
                nodes_to_train=self.all_nodes_to_train[j:j+self.batch_size]
            else:
                nodes_to_train = self.all_nodes_to_train[j:]

            if len(nodes_to_train)==0:
                continue

            for node in nodes_to_train:

                if node>=self.bigram_start and node<=self.bigram_end:
                    if node in self.bigram2subframe_adj_pos:
                        batch_input['bigram2subframe']+=[node for i in range(0,len(self.bigram2subframe_adj_pos[node]))]
                        batch_gold['bigram2subframe']+=list(self.bigram2subframe_adj_pos[node])
                        batch_neg['bigram2subframe']+=[random.sample(self.bigram2subframe_adj_neg[node],self.neg_sample_rate) for i in range(0,len(self.bigram2subframe_adj_pos[node]))]

                if node>=self.trigram_start and node<=self.trigram_end:
                    if node in self.trigram2subframe_adj_pos:
                        batch_input['trigram2subframe']+=[node for i in range(0,len(self.trigram2subframe_adj_pos[node]))]
                        batch_gold['trigram2subframe']+=list(self.trigram2subframe_adj_pos[node])
                        batch_neg['trigram2subframe']+=[random.sample(self.trigram2subframe_adj_neg[node],self.neg_sample_rate) for i in range(0,len(self.trigram2subframe_adj_pos[node]))]

                if node>=self.doc_start and node<=self.doc_end:
                    text_nodes.append(node)

                    if node in self.text2bigram_adj_pos:
                        batch_input['text2bigram']+=[node for i in range(0,len(self.text2bigram_adj_pos[node]))]
                        batch_gold['text2bigram']+=list(self.text2bigram_adj_pos[node])
                        batch_neg['text2bigram']+=[random.sample(self.text2bigram_adj_neg[node],self.neg_sample_rate) for i in range(0,len(self.text2bigram_adj_pos[node]))]
                    if node in self.text2trigram_adj_pos:
                        batch_input['text2trigram']+=[node for i in range(0,len(self.text2trigram_adj_pos[node]))]
                        batch_gold['text2trigram']+=list(self.text2trigram_adj_pos[node])
                        batch_neg['text2trigram']+=[random.sample(self.text2trigram_adj_neg[node],self.neg_sample_rate) for i in range(0,len(self.text2trigram_adj_pos[node]))]

            for k in batch_input:
                batch_input[k] = array(batch_input[k])
            for k in batch_gold:
                batch_gold[k] = array([batch_gold[k]]).transpose()
            for k in batch_neg:
                batch_neg[k] = array(batch_neg[k])

            j=j+self.batch_size
            batches.append([batch_input, batch_gold, batch_neg, text_nodes])
        
        self.batches=batches


    def save_embeddings(self, model):
        # Save learned embeddings to output folder
        
        output_dir=self.args['output_folder']+'/'

            
        f=open(output_dir+"bigram.embeddings","w")
        bigram_embd=model.bigram_embeddings.weight.cpu().data.numpy()
        bigram_embd=bigram_embd.tolist()
        for i in range(0,len(bigram_embd)):
            id=i+self.bigram_start
            name=self.id2name[id]
            f.write(str(name))
            embd=bigram_embd[i]
            for j in range(len(embd)):
                if j==0:
                    f.write("\t"+str(embd[j]))
                else:
                    f.write(" "+str(embd[j]))
            f.write("\n")
            
        f=open(output_dir+"trigram.embeddings","w")
        trigram_embd=model.trigram_embeddings.weight.cpu().data.numpy()
        trigram_embd=trigram_embd.tolist()
        for i in range(0,len(trigram_embd)):
            id=i+self.trigram_start
            name=self.id2name[id]
            f.write(str(name))
            embd=trigram_embd[i]
            for j in range(len(embd)):
                if j==0:
                    f.write("\t"+str(embd[j]))
                else:
                    f.write(" "+str(embd[j]))
            f.write("\n")

        
        f = open(output_dir+"subframe.embeddings", "w")
        subframe_embd = model.subframe_embeddings.weight.cpu().data.numpy()
        subframe_embd = subframe_embd.tolist()
        for i in range(0, len(subframe_embd)):
            id = i + self.subframe_start
            name = self.id2name[id]
            f.write(str(name))
            embd = subframe_embd[i]
            for j in range(len(embd)):
                if j==0:
                    f.write("\t"+str(embd[j]))
                else:
                    f.write(" "+str(embd[j]))
            f.write("\n")
        f.close()
        

        f = open(output_dir+"doc.embeddings", "w")
        docs=[k for k in range(self.doc_start, self.doc_end+1)]
        for d in docs:
            embd=model.doc_embeddings([d])
            embd=embd[0]
            embd=embd.cpu().data.numpy()
            embd=embd.tolist()
            name = self.id2name[d]
            f.write(str(name))
            for j in range(len(embd)):
                if j==0:
                    f.write("\t"+str(embd[j]))
                else:
                    f.write(" "+str(embd[j]))
            f.write("\n")
        f.close()


def run():
    global cuda_available
    cuda_available = torch.cuda.is_available()

    NID = NodeInputData(args)
    print ("Creating batches...")
    NID.get_nodes(NID.graph)
    print ("Batches Created!")


    # return
    print ("Initializing the model...")

    model = Embedder(args, cuda_available=cuda_available)
    model1 = Embedder(args, cuda_available=cuda_available)

    print ("Model Initialized!")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    print ("Total Batches: %d" % len(NID.batches))
    
    '''
    for name, param in model1.named_parameters():
        if param.requires_grad:
            print (name)
    '''
    
    min_loss = 10000000
    no_down = 0
    model.train()
    for epoch in range(0, 100):  # initiate number of epoches
        start = time.time()
        average_loss = 0
        for batch in NID.batches:
            optimizer.zero_grad()
            l, loss = model(batch)

            if l == 0:
                continue
                
            l.backward()
            optimizer.step()
            average_loss += l.data

        end = time.time()
        print ("Total Time for epoch: %d: %lf" % (epoch, float(end - start)))
        print ("Loss at epoch %d = %f" % (epoch, average_loss))

        if average_loss < min_loss:
            min_loss = average_loss
            model1.load_state_dict(model.state_dict())

            no_down = 0
        else:
            no_down += 1

        if no_down == 10:
            break
    
    NID.save_embeddings(model1)




if __name__ == '__main__':
    random.seed(1234)
    torch.manual_seed(1234)
    numpy.random.seed(1234)
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i','--input_folder', help='Input folder path', required=True)
    parser.add_argument('-o','--output_folder', help='Output folder path', required=True)
    args = vars(parser.parse_args())
    
    run()
