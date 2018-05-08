import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import os
import argparse

from dataset import *


class Discriminator(nn.Module):
    """A CNN for text classification
    architecture: Embedding >> Convolution >> Max-pooling >> Softmax
    """

    def __init__(self, num_classes, vocab_size, emb_dim, filter_sizes, num_filters, dropout):
        super(Discriminator, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, emb_dim)) for (n, f) in zip(num_filters, filter_sizes)
        ])
        self.highway = nn.Linear(sum(num_filters), sum(num_filters))
        self.dropout = nn.Dropout(p=dropout)
        self.lin = nn.Linear(sum(num_filters), 1)
        self.clf_layer = nn.Linear(sum(num_filters), 1)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.init_parameters()

    def forward(self, x):
        """
        Args:
            x: (batch_size * seq_len)
        """
        emb = self.emb(x).unsqueeze(1)  # batch_size * 1 * seq_len * emb_dim
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter * length]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]  # [batch_size * num_filter]
        pred = torch.cat(pools, 1)  # batch_size * num_filters_sum
        highway = self.highway(pred)
        pred = F.sigmoid(highway) * F.relu(highway) + (1. - F.sigmoid(highway)) * pred
        pred1 = self.sigmoid(self.lin(self.dropout(pred)))
        clf = self.sigmoid(self.clf_layer(self.dropout(pred)))
        return pred1, clf

    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)


class Generator(nn.Module):
    """Generator """

    def __init__(self, num_emb, emb_dim, hidden_dim, use_cuda, pretrained_emb=None):
        super(Generator, self).__init__()
        self.num_emb = num_emb
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.use_cuda = use_cuda
        self.emb = nn.Embedding(num_emb, emb_dim)
        self.lstm_layer_number = 2
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=self.lstm_layer_number, batch_first=True)
        self.lin = nn.Linear(hidden_dim, num_emb)
        self.softmax = nn.Softmax()
        self.init_params()
        if pretrained_emb is not None:
            self.emb.from_pretrained(pretrained_emb)

        self.UNK_IDX = 0
        self.PAD_IDX = 1
        self.START_IDX = 2
        self.EOS_IDX = 3
        self.MAX_SENT_LEN = 50

    def forward(self, x, h, c):
        """
        Args:
            x: (batch_size, seq_len), sequence of tokens generated by generator
        """
        emb = self.emb(x)
        # h0, c0 = self.init_hidden(x.size(0))
        output, (h, c) = self.lstm(emb, (h, c))
        pred = self.softmax(self.lin(output.contiguous().view(-1, self.hidden_dim)))
        return pred, h, c

    def step(self, x, h, c):
        """
        Args:
            x: (batch_size,  1), sequence of tokens generated by generator
            h: (1, batch_size, hidden_dim), lstm hidden state
            c: (1, batch_size, hidden_dim), lstm cell state
        """
        emb = self.emb(x)
        output, (h, c) = self.lstm(emb, (h, c))
        pred = F.softmax(self.lin(output.view(-1, self.hidden_dim)))
        return pred, h, c

    def init_hidden(self, batch_size, random=False):
        if random:
            h = torch.rand((self.lstm_layer_number, batch_size, self.hidden_dim))
            c = torch.rand((self.lstm_layer_number, batch_size, self.hidden_dim))
        else:
            h = torch.zeros((self.lstm_layer_number, batch_size, self.hidden_dim))
            c = torch.zeros((self.lstm_layer_number, batch_size, self.hidden_dim))
        emotion = torch.bernoulli(torch.ones(batch_size) * 0.5).expand(self.lstm_layer_number, -1)
        h[:, :, -1] = emotion
        c[:, :, -1] = emotion
        if self.use_cuda:
            h, c = h.cuda(), c.cuda()
        return h, c, emotion[0, :]

    def init_params(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)

    def sample(self, batch_size, emotion):
        self.eval()

        word = torch.LongTensor([self.START_IDX]).expand(batch_size)[:, None].to(device)

        h, c, _ = self.init_hidden(batch_size, random=True)
        h[:, :, -1] = emotion.expand(self.lstm_layer_number, -1)
        c[:, :, -1] = emotion.expand(self.lstm_layer_number, -1)

        sentences = [word]
        for i in range(self.MAX_SENT_LEN):
            word, h, c = self.forward(word, h, c)
            word = torch.multinomial(word, 1).type(torch.long)
            sentences.append(word)
        sentences = torch.cat(sentences, 1)

        self.train()
        return sentences


def prepare_generator_batch(samples, start_letter=2, gpu=False):
    """
    Takes samples (a batch) and returns

    Inputs: samples, start_letter, cuda
        - samples: batch_size x seq_len (Tensor with a sample in each row)

    Returns: inp, target
        - inp: batch_size x seq_len (same as target, but with start_letter prepended)
        - target: batch_size x seq_len (Variable same as samples)
    """

    batch_size, seq_len = samples.size()

    inp = torch.zeros(batch_size, seq_len)
    target = samples
    inp[:, 0] = start_letter
    inp[:, 1:] = target[:, :seq_len-1]

    inp = inp.type(torch.long)
    target = target.type(torch.long)

    if gpu:
        inp = inp.cuda()
        target = target.cuda()

    return inp, target


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', dest='lr', type=float,
                        default=1e-3, help="learning rate.")
    parser.add_argument('--gpu', dest='gpu', type=bool, default=True,
                        help='Use gpu or not')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64,
                        help='Batch size for training')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)

    gpu = args.gpu
    lr = args.lr

    model_dir = 'model'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    device = 'cuda' if gpu and torch.cuda.is_available() else 'cpu'

    dataset = IMDB_Dataset()
    embd = dataset.get_vocab_vectors()
    vocab_size = embd.size(0)
    embd_dim = embd.size(1)

    d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50]
    d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]

    discriminator = Discriminator(2, vocab_size, embd_dim, d_filter_sizes, d_num_filters, dropout=0.5).to(device)

    generator = Generator(embd.size(0), embd.size(1), 64 + 1, gpu, pretrained_emb=embd).to(device)

    print('Pretraining Generator.')
    pretrain_epoch_num = 20000
    optim = torch.optim.Adam(generator.parameters(), lr=lr)
    for epoch in range(pretrain_epoch_num):
        total_loss = 0
        inputs, labels = dataset.next_batch(args.gpu)
        inputs = inputs.transpose_(0, 1)
        batch_size, seq_len = inputs.size()
        h, c, emotion = generator.init_hidden(batch_size)
        for i in range(seq_len-1):
            prediction, h, c = generator(inputs[:, i].unsqueeze(1), h, c)
            loss = F.cross_entropy(prediction, inputs[:, i+1])
            total_loss += loss
        optim.zero_grad()
        total_loss.backward()
        optim.step()

        if (epoch+1) % 500 == 0:
            print('Epoch {}: Loss {}'.format(epoch, total_loss.item()))
            sampled = generator.sample(2, torch.Tensor([0, 1]))
            for i in range(2):
                print('{}: {}'.format(dataset.idx2label(i), dataset.idxs2sentence(sampled[i, :])))


    print('Pretraining Discriminator')
    generator.eval()
    optim = torch.optim.Adagrad(discriminator.parameters(), lr=lr)
    for epoch in range(pretrain_epoch_num):
        total_loss = 0
        inputs, labels = dataset.next_batch(args.gpu)
        inputs = inputs.transpose_(0, 1)
        batch_size, seq_len = inputs.size()
        emotion = torch.zeros(batch_size).to(device)
        emotion[int(batch_size / 2):] += 1
        generated_sentences = generator.sample(batch_size, emotion)

        truth = torch.zeros(batch_size * 2).to(device)
        truth[:batch_size] += 1
        emotion = torch.cat((labels[:, None].type(torch.float), emotion[:, None])).squeeze()

        inputs = torch.cat((inputs, generated_sentences), 0)
        truth_prediction, emotion_prediction = discriminator(inputs)

        truth_loss = F.binary_cross_entropy(truth_prediction, truth)
        emotion_loss = F.binary_cross_entropy(emotion_prediction, emotion)

        loss = truth_loss + 0.1 * emotion_loss

        total_loss += loss.item()

        optim.zero_grad()
        loss.backward()
        optim.step()

        if (epoch+1) % 500 == 0:
            print('Epoch {}: Loss {}'.format(epoch, total_loss))

    adversarial_step_num = 50000
    gen_optim = torch.optim.Adam(generator.parameters(), lr=lr)
    dis_optim = torch.optim.Adagrad(discriminator.parameters(), lr=lr)
    for step in range(adversarial_step_num):
        for i in range(5):
            batch_size = 32
            emotion = torch.zeros(batch_size*2).to(device)
            emotion[batch_size:] += 1
            sampled = generator.sample(batch_size*2, emotion)

            truth_prediction, emotion_prediction = discriminator(sampled)
            emotion_loss = F.binary_cross_entropy(emotion_prediction, emotion)
            rewards = truth_prediction - emotion_loss

            batch_size, seq_len = sampled.size()
            h, c, _ = generator.init_hidden(batch_size, random=True)
            h[:, :, -1] = emotion.expand(generator.lstm_layer_number, -1)
            c[:, :, -1] = emotion.expand(generator.lstm_layer_number, -1)

            # action = torch.zeros(batch_size, seq_len)
            # action[:, 0] = generator.START_IDX
            # action[:, 1:] = sampled[:, seq_len - 1]

            # G = torch.zeros(seq_len, vocab_size)
            # t = batch_size - 1
            # G[t, action[:, t]] = rewards[:, t]

            # action = action.transpose_(0, 1)

            inp, target = prepare_generator_batch(sampled, gpu=True)
            inp = inp.permute(1, 0)  # seq_len x batch_size
            target = target.permute(1, 0)  # seq_len x batch_size

            loss = 0
            for j in range(seq_len):
                out, h, c = generator(inp[j, :].unsqueeze(1), h, c)
                for k in range(batch_size):
                    loss += -out[k][target.data[j, k]] * rewards[k, :]
            loss /= batch_size

            gen_optim.zero_grad()
            loss.backward()
            gen_optim.step()


        for i in range(5):
            inputs, labels = dataset.next_batch(args.gpu)
            inputs = inputs.transpose_(0, 1)
            batch_size, seq_len = inputs.size()
            emotion = torch.zeros(batch_size).to(device)
            emotion[int(batch_size / 2):] += 1
            generated_sentences = generator.sample(batch_size, emotion)

            truth = torch.zeros(batch_size * 2).to(device)
            truth[:batch_size] += 1
            emotion = torch.cat((labels[:, None].type(torch.float), emotion[:, None])).squeeze()

            inputs = torch.cat((inputs, generated_sentences), 0)
            truth_prediction, emotion_prediction = discriminator(inputs)

            truth_loss = F.binary_cross_entropy(truth_prediction, truth)
            emotion_loss = F.binary_cross_entropy(emotion_prediction, emotion)

            loss = truth_loss + emotion_loss

            dis_optim.zero_grad()
            loss.backward()
            dis_optim.step()

        label = [0, 0, 0, 1, 1, 1]
        if (step+1) % 500 == 0:
            print('Epoch {}'.format(step+1))
            sampled = generator.sample(6, torch.Tensor([0, 0, 0, 1, 1, 1]))
            for i in range(6):
                print('{}: {}'.format(dataset.idx2label(label[i]), dataset.idxs2sentence(sampled[i, :])))
            torch.save(generator.state_dict(), os.path.join(model_dir, 'generator-{}'.format(step+1)))
            torch.save(discriminator.state_dict(), os.path.join(model_dir, 'discriminator-{}'.format(step+1)))
