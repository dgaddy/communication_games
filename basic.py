import math
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

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
    def __init__(self, vocab_size=10, hidden_size=32):
        super(Net, self).__init__()
        self.emb = nn.Embedding(vocab_size,hidden_size)
        self.fc_combine = nn.Linear(hidden_size*2, hidden_size)
        self.lstm_out = nn.LSTMCell(1,hidden_size)
        self.fc_out = nn.Linear(hidden_size,1)
        self.lstm_in = nn.LSTMCell(1,hidden_size)
        self.fc_final = nn.Linear(hidden_size,vocab_size*2)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.temp = 1

    def forward(self,x1,x2,noise=0.0):
        dropout = 0.0

        x1 = self.emb(x1)
        x2 = self.emb(x2)
        n_words = 25
        batch_size = x1.size()[0]

        h = Variable(torch.zeros(batch_size, self.hidden_size)).cuda()
        lstm_in = Variable(torch.zeros(batch_size,1)).cuda()
        c = self.fc_combine(F.dropout(torch.cat((x1,x2),1), dropout))
        outs = []
        noise_mask = np.zeros(n_words)
        # two different types of noise
        '''if random.random() < .5:
            i = random.randrange(n_words-5)
            noise_mask[i:i+5] = 1'''
        '''for i in xrange(n_words):
            noise_mask[i] = 1 if random.random() < noise else 0'''
        for wi in xrange(n_words):
            h,c = self.lstm_out(lstm_in,(h,c))

            o = self.fc_out(F.dropout(h, dropout))
            if self.training:
                o = gumbel_binary(o, self.temp)
            else:
                o = hard_binary(o)

            n = float(noise_mask[wi])
            o = (1-n)*o # zero out the bits that are noised

            outs.append(o)
            #lstm_in = o # this would feed back the output bit into the lstm input

        h = Variable(torch.zeros(batch_size, self.hidden_size)).cuda()
        c = Variable(torch.zeros(batch_size, self.hidden_size)).cuda()
        results1 = []
        results2 = []
        for o in outs:
            h,c = self.lstm_in(o,(h,c))

            results = self.fc_final(c)
            results1.append(F.log_softmax(results[:,:10]))
            results2.append(F.log_softmax(results[:,10:]))

        return results1, results2, outs

    def update_temp(self, temp):
        self.temp = temp

def loss(output, target):
    values = torch.gather(output, 1, target.view(-1,1))
    return -values.mean()# - torch.exp(values).mean()

def evaluate(net, values1, values2, noise):
    net.eval()
    points1 = Variable(values1)
    points2 = Variable(values2)

    out1, out2,_ = net(points1, points2, noise)
    _, out1 = out1[-1].data.max(1)
    _, out2 = out2[-1].data.max(1)

    n_correct = (out1 == points1.data).sum()+(out2 == points2.data).sum()
    n_total = out1.size()[0] * 2
    net.train()
    return n_correct / float(n_total)

def make_inputs(vocab_size):
    values1 = range(vocab_size) * vocab_size
    values2 = []
    for i in range(vocab_size):
        values2.extend([i]*vocab_size)
    values1 = torch.LongTensor(values1).cuda()
    values2 = torch.LongTensor(values2).cuda()
    return values1, values2

def train(t1, t2, iterations, all_loss=False, model_file=None):
    vocab_size = 10

    values1, values2 = make_inputs(vocab_size)

    net = Net(vocab_size=vocab_size, hidden_size=64).cuda()
    optimizer = optim.Adam(net.parameters(),lr=.001)

    net.train()
    total_cost = 0
    temperatures = np.geomspace(t1,t2,iterations)
    for episode in xrange(1,iterations+1):
        points1 = Variable(values1)
        points2 = Variable(values2)

        net.update_temp(temperatures[episode-1])
        optimizer.zero_grad()

        out1, out2, _ = net(points1, points2)
        if all_loss:
            cost = Variable(torch.zeros(1).cuda())
            for r in out1:
                cost = cost + loss(r, points1)
            for r in out2:
                cost = cost + loss(r, points2)
        else:
            cost = loss(out1[-1], points1) + loss(out2[-1], points2)
        cost.backward()
        total_cost += cost.data[0]
        optimizer.step()

        if episode % 1000 == 0:
            print 'episode', episode
            print 'average cost:', total_cost / float(episode)
            print 'temperature', net.temp
            print 'eval', evaluate(net, values1, values2, 0)

    if model_file is not None:
        torch.save(net, model_file)

def display(net, vocab_size):
    values1, values2 = make_inputs(vocab_size)
    _, _, messages = net(Variable(values1), Variable(values2))
    messages = np.array([m.data.cpu().numpy() for m in messages]).squeeze().T
    plt.matshow(messages,cmap=plt.cm.gray)
    plt.savefig('basic_comm_sample_message.png')

    messages_sums = np.zeros((vocab_size**2,25))
    for i in xrange(10):
        _, _, messages = net(Variable(values1), Variable(values2))
        messages = np.array([m.data.cpu().round_().numpy() for m in messages]).squeeze().T
        messages_sums += messages

    plt.matshow(messages_sums,cmap=plt.cm.gray)
    plt.savefig('basic_comm_averaged_messages.png')

for repeat in xrange(5):
    t1=5
    t2=1
    i = 25000
    all_loss = True
    print '=============================='
    print 'training with t1, t2=', t1, t2
    print 'iterations=', i
    print 'all_loss=', all_loss
    train(t1, t2, i, all_loss)
