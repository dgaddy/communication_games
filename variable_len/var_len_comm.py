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
    def __init__(self, n_items=10, hidden_size=32):
        super(Net, self).__init__()
        self.emb = nn.Embedding(n_items,hidden_size)
        self.speaker_lstm = nn.LSTMCell(1,hidden_size)
        self.fc_continue = nn.Linear(hidden_size,1)
        self.fc_out = nn.Linear(hidden_size,1)
        self.listener_lstm = nn.LSTMCell(1,hidden_size)
        self.fc_final = nn.Linear(hidden_size,n_items)

        self.hidden_size = hidden_size
        self.temp = 1

    def forward(self,x):
        x = self.emb(x)
        n_words = 10
        batch_size = x.size()[0]

        speaker_h = Variable(torch.zeros(batch_size, self.hidden_size)).cuda()
        lstm_in = Variable(torch.zeros(batch_size,1)).cuda()
        speaker_c = x
        messages = []
        continues = []

        listener_h = Variable(torch.zeros(batch_size, self.hidden_size)).cuda()
        listener_c = Variable(torch.zeros(batch_size, self.hidden_size)).cuda()
        results = []
        for i in xrange(n_words):
            speaker_h,speaker_c = self.speaker_lstm(lstm_in,(speaker_h,speaker_c))

            if i > 0:
                c = self.fc_continue(speaker_h)
                if self.training:
                    c = gumbel_binary(c, self.temp)
                else:
                    c = hard_binary(c)
                continues.append(c)

            o = self.fc_out(speaker_h)
            if self.training:
                o = gumbel_binary(o, self.temp)
            else:
                o = hard_binary(o)
            messages.append(o)

            lstm_in = o

            listener_h,listener_c = self.listener_lstm(o,(listener_h,listener_c))

            result = self.fc_final(listener_c)
            result = F.log_softmax(result)
            results.append(result)

        return results, messages, continues

    def update_temp(self, temp):
        self.temp = temp

def nll(output, target):
    return -torch.gather(output, 1, target.view(-1,1))

def loss(outputs, continues, targets):
    length_penalty = .001
    batch_size = outputs[0].size()[0]

    amt_continuing = Variable(torch.ones(*continues[0].size())).cuda()
    total_loss = Variable(torch.zeros(batch_size, 1)).cuda()

    for i in xrange(len(outputs)):
        out = outputs[i]
        length = i # XXX
        current_loss = nll(out, targets) + length*length * length_penalty

        cont = continues[i] if i < len(continues) else Variable(torch.zeros(*continues[0].size())).cuda()
        amt_stopping = amt_continuing * (1-cont)

        total_loss = total_loss + amt_stopping * current_loss
        amt_continuing = amt_continuing * cont

    return total_loss.mean()

def get_outputs_hard(outputs, messages, continues):
    outputs = [o.max(1)[1] for o in outputs] # convert from one hot
    outputs = [o.data.cpu().squeeze() for o in outputs]
    messages = [m.data.cpu().squeeze() for m in messages]
    continues = [c.data.cpu().squeeze() for c in continues]

    batch_size = outputs[0].size()[0]

    max_len = len(messages)
    all_messages = []
    all_outputs = []
    for batch_ind in xrange(batch_size):
        message = []
        output = None
        for i in xrange(max_len):
            message.append(str(int(messages[i][batch_ind])))
            output = outputs[i][batch_ind]

            if i < len(continues) and continues[i][batch_ind] < .5:
                break

        all_messages.append(''.join(message))
        all_outputs.append(output)

    return all_messages, all_outputs

# %%

n_items = 20
values = range(n_items)
values = torch.LongTensor(values).cuda()

net = Net(n_items=n_items, hidden_size=64).cuda()
optimizer = optim.Adam(net.parameters())

# %%
n_iters = 500
total_cost = 0
temperatures = np.geomspace(5,.25,n_iters)
for episode in xrange(1,n_iters+1):
    points = Variable(values)

    net.update_temp(temperatures[episode-1])
    optimizer.zero_grad()

    out, messages, continues = net(points)
    cost = loss(out, continues, points)
    cost.backward()
    total_cost += cost.data[0]
    optimizer.step()

    if episode % 1000 == 0:
        print 'episode', episode
        print 'average cost:', total_cost / float(episode)
        print 'temperature', net.temp
torch.save(net, 'var_len_net.pkl')

# %%

net.eval()

points = Variable(values)

out, messages, continues = net(points)

messages, output = get_outputs_hard(out, messages, continues)
print messages

n_correct = sum(o == v for v,o in enumerate(output))
n_total = len(output)

print 'dev accuracy', n_correct / float(n_total)

# %%

def display():
    input = Variable(torch.LongTensor(range(n_items)).cuda())
    out, messages, continues = net(input)
    continues = np.array([m.data.cpu().numpy() for m in continues]).squeeze().T
    plt.matshow(continues,cmap=plt.cm.gray)
    messages = np.array([m.data.cpu().numpy() for m in messages]).squeeze().T
    plt.matshow(messages,cmap=plt.cm.gray)

    continues_sums = np.zeros((20,9))

    for i in xrange(10):
        input = Variable(torch.LongTensor(range(n_items)).cuda())
        out, messages, continues = net(input)
        continues = np.array([m.data.cpu().round_().numpy() for m in continues]).squeeze().T
        continues_sums += continues

    plt.matshow(continues_sums,cmap=plt.cm.gray)
