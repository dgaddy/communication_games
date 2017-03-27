import math
import numpy as np
import random
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

# %%

def sample_gumbel(shape, eps=1e-20):
    # gumbel(0,1)
    u = Variable(torch.rand(*shape).cuda())
    return -torch.log(-torch.log(u+eps)+eps)

def gumbel_binary(inputs, temperature):
    y = inputs + sample_gumbel(inputs.size()) - sample_gumbel(inputs.size())

    return F.sigmoid(y/temperature)

def hard_binary(inputs):
    y = inputs + sample_gumbel(inputs.size()) - sample_gumbel(inputs.size())

    return torch.gt(y, 0).type(torch.cuda.FloatTensor)

class Net(nn.Module):
    def __init__(self, n_words=10, vocab_size=10, hidden_size=32):
        super(Net, self).__init__()

        self.n_words = n_words

        self.emb = nn.Embedding(vocab_size,hidden_size)
        self.fc_out = nn.Linear(hidden_size*2,n_words)
        self.fc_final = nn.Linear(n_words,vocab_size*2)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.temp = 1

    def forward(self,x1,x2,noise_mask=None):

        dropout = 0

        x1 = self.emb(x1)
        x2 = self.emb(x2)
        batch_size = x1.size()[0]

        o = self.fc_out(torch.cat((x1,x2),1))
        if self.training:
            o = gumbel_binary(o, self.temp)
        else:
            o = hard_binary(o)

        if noise_mask is not None:
            o = (1-noise_mask)*o # zero out the bits that are noised

        results = self.fc_final(o)
        result1 = F.log_softmax(results[:,:self.vocab_size])
        result2 = F.log_softmax(results[:,self.vocab_size:])

        return result1, result2, o

    def update_temp(self, temp):
        self.temp = temp

    def fix_solution(self):
        assert self.n_words == 6 and self.vocab_size == 8

        emb = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
        emb = emb*2-1
        full_emb = np.zeros((self.vocab_size, self.hidden_size))
        full_emb[:,:3] = emb
        self.emb.weight = nn.Parameter(torch.cuda.FloatTensor(full_emb))

        fc_out = np.zeros((self.hidden_size*2,self.n_words))
        hs = self.hidden_size
        fc_out[0,0] = 1
        fc_out[1,1] = 1
        fc_out[2,2] = 1
        fc_out[hs,3] = 1
        fc_out[hs+1,4] = 1
        fc_out[hs+2,5] = 1
        self.fc_out.weight = nn.Parameter(torch.cuda.FloatTensor(fc_out.T*100))

        self.fc_out.bias.data.zero_()

        '''
        fc_final = np.zeros((self.vocab_size*2,self.n_words))
        fc_final[:self.vocab_size,:self.n_words/2] = emb
        fc_final[self.vocab_size:,self.n_words/2:] = emb
        self.fc_final.weight = nn.Parameter(torch.cuda.FloatTensor(fc_final*1))
        self.fc_final.bias.data.zero_()
        '''

def loss(output, target):
    values = torch.gather(output, 1, target.view(-1,1))
    return -0.0001*values.mean() - torch.exp(values).mean()

def evaluate(noise):
    net.eval()
    points1 = Variable(values1)
    points2 = Variable(values2)

    out1, out2,_ = net(points1, points2, noise)
    cost = loss(out1, points1) + loss(out2, points2)
    maxes1, out1 = out1.data.max(1)
    maxes2, out2 = out2.data.max(1)

    n_correct = (out1 == points1.data).sum()+(out2 == points2.data).sum()
    n_total = out1.size()[0] * 2
    net.train()
    return n_correct / float(n_total)

def display(vocab_size):
    # %%
    net.eval()

    values1, values2 = basic_input(vocab_size)


    _, _, messages = net(Variable(values1), Variable(values2))
    messages = messages.data.cpu().numpy()
    #messages = np.array([m.data.cpu().numpy() for m in messages]).squeeze().T
    plt.matshow(messages,cmap=plt.cm.gray)
    plt.savefig('basic_comm_sample_message.png')
    plt.close()

    #%%
    messages_sums = np.zeros_like(messages)
    for i in xrange(10):
        input = Variable(torch.LongTensor(range(vocab_size)).cuda())
        _, _, messages = net(Variable(values1), Variable(values2))
        messages = messages.data.cpu().numpy()
        #messages = np.array([m.data.cpu().round_().numpy() for m in messages]).squeeze().T
        messages_sums += messages

    plt.matshow(messages_sums,cmap=plt.cm.gray)
    plt.savefig('basic_comm_averaged_messages.png')
    plt.close()
    net.train()

def basic_input(vocab_size):
    values1 = range(vocab_size) *vocab_size
    values2 = []
    for i in range(vocab_size):
        values2.extend([i]*vocab_size)

    values1 = torch.LongTensor(values1).cuda()
    values2 = torch.LongTensor(values2).cuda()
    return values1, values2

def basic_noise():
    noise_mask = np.zeros(self.n_words)
    if random.random() < .5:
        i = random.randrange(self.n_words-3)
        noise_mask[i:i+3] = 1
    '''for i in xrange(5):
        if random.random() < noise:
            noise_mask[5*i:5*(i+1)] = 1'''
    '''for i in xrange(n_words):
        noise_mask[i] = 1 if random.random() < noise else 0'''
    noise_mask = Variable(torch.cuda.FloatTensor(noise_mask))
    return noise_mask


def make_inputs(vocab_size, length):
    chunk_size = vocab_size*vocab_size
    values1 = range(vocab_size) *vocab_size
    values2 = []
    for i in range(vocab_size):
        values2.extend([i]*vocab_size)

    values1 = torch.LongTensor(values1).cuda()
    values2 = torch.LongTensor(values2).cuda()

    repeats=3
    total_len = vocab_size*vocab_size*repeats
    noise = np.zeros((total_len,length))
    noise[0:chunk_size,0:length/2] = 1
    noise[chunk_size:2*chunk_size,length/2:length] = 1
    '''
    repeats = length-n_noise+1

    for i in xrange(length-n_noise):
        start = i * chunk_size
        end = (i+1) * chunk_size

        noise[start:end,i:i+n_noise] = 1
    '''

    return values1.repeat(repeats), values2.repeat(repeats), torch.cuda.FloatTensor(noise)

# %%

vocab_size = 8
n_words = 6

#values1, values2, noise = make_inputs(vocab_size, n_words)
values1, values2 = basic_input(vocab_size)

net = Net(n_words=n_words, vocab_size=vocab_size, hidden_size=64).cuda()
#net.fix_solution()
#optimizer = optim.Adam(net.fc_final.parameters(),lr=.001)
optimizer = optim.Adam(net.parameters(),lr=.001)
# %%
net.train()
total_cost = 0
n_iters = 50000
temperatures = np.geomspace(2,1.5,n_iters)
for episode in xrange(1,n_iters+1):
    points1 = Variable(values1)
    points2 = Variable(values2)

    net.update_temp(temperatures[episode-1])
    optimizer.zero_grad()

    out1, out2, _ = net(points1, points2)#, Variable(noise))
    cost = loss(out1, points1) + loss(out2, points2)
    cost.backward()
    total_cost += cost.data[0]
    optimizer.step()

    if episode % 1000 == 0:
        print 'episode', episode
        print 'average cost:', total_cost / float(episode)
        print 'temperature', net.temp
        print 'dev', evaluate(0)

        torch.save(net, 'basic_comm_net_2g_nolstm.pkl')
        display(vocab_size)
# %%
#print 'dev accuracy noise', evaluate(.1)
#print 'dev accuracy noise', evaluate(Variable(noise))
print 'dev accuracy no noise', evaluate(0)

#net = torch.load('basic_comm_net_2g_nolstm.pkl')
#display()
