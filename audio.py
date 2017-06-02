from __future__ import division

import helper as helper
import sys
import wave
import struct
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data import DataLoader


import matplotlib.pyplot as plt
import scipy
import scipy.signal as scipy_signal
import glob, os
import argparse
from model import SRResNet, Residual, SubPixelConv



parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='', help='location of the data')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=2, help='input batch size')
parser.add_argument('--sampleSize', type=int, default=16384, help='the number of samples to use in an audio')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--clip',
                    type=float,
                    default=0.4,
                    help='Clipping Gradients. Default=0.4')

opt = parser.parse_args()
print(opt)






dataset = helper.AudioFolder(root=opt.data_dir, sample_length=16000)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))




def save_audio(audio, filename, sample_rate):
    if isinstance(audio, torch.FloatTensor) or isinstance(audio, torch.cuda.FloatTensor):
        audio =  torch.clamp(audio.mul(32768), min=-32767, max=32767)
        npaudio = audio.cpu().numpy()[0,0,:]
        npaudio = [int(x) for x in npaudio.tolist()]
        outfile = wave.open(filename, 'w')
        outfile.setparams((1, 2, sample_rate, 0, 'NONE', 'not compressed'))
        values = struct.pack('<{}h'.format(len(npaudio)), *(npaudio))
        outfile.writeframes(values)
        outfile.close()



cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception('No GPU found, please run without --cuda')

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

model = SRResNet()

criterion = nn.MSELoss(size_average=False)
lr = opt.lr

if cuda:
    model = torch.nn.DataParallel(model).cuda()
    criterion = criterion.cuda()

optimizer = optim.Adam(model.parameters(), lr=lr)
counter = 0

loss_sum = Variable(torch.zeros(1), requires_grad=False)
if cuda:
    loss_sum = loss_sum.cuda()

for epoch in range(opt.niter):

    for (input, target) in dataloader:
        counter = counter + 1
        input = Variable(input)
        target = Variable(target)

        if cuda:
            input = input.cuda()
            target = target.cuda()

        gen = model(input)
        optimizer.zero_grad()
        loss = criterion(gen, target)
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), opt.clip)
        loss_sum.add_(loss)
        optimizer.step()
        if counter % 100 == 0:
            print('Steps: {}'.format(counter))

        if counter % 500 == 0:
            print('sum_of_loss = {}'.format(
                loss_sum.data.select(0, 0)))
            loss_sum = Variable(torch.zeros(1), requires_grad=False)
            if cuda:
                loss_sum = loss_sum.cuda()

#            save_checkpoint(model, epoch)
            save_audio(input.data, 'src.wav', 8000)
            save_audio(target.data, 'tgt.wav', 16000)
            save_audio(gen.data, 'gen.wav', 16000)
