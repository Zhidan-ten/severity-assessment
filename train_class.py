import os
import sys
import argparse
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utiles import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR

def train(epoch):
    start = time.time()
    net.train()
    train_loss = 0
    correct = 0.0
    for (images, labels,sex,age,progress,minus_progress) in tqdm(data_training_loader):
        if epoch <= args.warm:
            warmup_scheduler.step()


        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()
            sex = sex.cuda()
            age = age.cuda()
            progress = progress.cuda()
            minus_progress = minus_progress.cuda()
        optimizer.zero_grad()
        outputs = net(images[:,0:9,:,:],images[:,9:18,:,:],images[:,18:27,:,:],images[:,27:36,:,:],sex,age,progress,minus_progress)
        loss = loss_function(outputs, labels)
        train_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
        loss.backward()
        optimizer.step()
    epoch_loss = train_loss/len(data_training_loader)
    print('Training Epoch: {epoch}  loss:{:0.4f}  Accuracy:{:0.4f} lr:{:0.4f}'.format(
        epoch_loss,
        correct.float() / len(data_training_loader.dataset),
        optimizer.param_groups[0]["lr"],
        epoch=epoch))
    return  train_loss/len(data_training_loader),correct.float() / len(data_training_loader.dataset)

@torch.no_grad()
def eval_training(epoch):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels,sex,age,progress,minus_progress) in tqdm(data_test_loader):
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()
            sex = sex.cuda()
            age = age.cuda()
            progress = progress.cuda()
            minus_progress = minus_progress.cuda()

        outputs = net(images[:,0:9,:,:],images[:,9:18,:,:],images[:,18:27,:,:],images[:,27:36,:,:],sex,age,progress,minus_progress)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    print('Evaluating Network.....')
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        test_loss / len(data_test_loader),
        correct.float() / len(data_test_loader.dataset),
        finish - start
    ))
    print()
    return correct.float() / len(data_test_loader.dataset)

if __name__ == '__main__':
    data_TRAIN_MEAN = 0.011203114#mean value of training images
    data_TRAIN_STD = 0.08672375#std value of training images
    CHECKPOINT_PATH = 'checkpoint'
    EPOCH = 100
    TIME_NOW = datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss')
    SAVE_EPOCH = 3
    SAVE_DIR = './result/'

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='twonet', help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=4, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=5, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    args = parser.parse_args()

    traintxt_path = './traintxt_path/'
    net = get_network(args)
    net = nn.DataParallel(net.cuda())

    #data preprocessing:
    data_training_loader = get_training_dataloader(
        traintxt_path,
        data_TRAIN_MEAN,
        data_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    data_test_loader = get_test_dataloader(
        traintxt_path,
        data_TRAIN_MEAN,
        data_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,weight_decay=1e-5)
    train_scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)
    iter_per_epoch = len(data_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    warmup_scheduler = WarmUpLR( iter_per_epoch * args.warm)
    checkpoint_path = os.path.join(CHECKPOINT_PATH, args.net, TIME_NOW)
    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    result = []
    for epoch in range(EPOCH):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        all_loss,tacc = train(epoch)
        acc = eval_training(epoch)
        result.append((
            all_loss, tacc , acc))

        if  best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = acc
            continue

        if not epoch % SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))
        np.savetxt(SAVE_DIR + 'loss_acc.txt', result, '%.6f')# save the results of Loss, Accuracy metrics
