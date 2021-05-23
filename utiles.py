import sys
import numpy
import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import random
import  os
import numpy as np

def get_network(args):
    """ return given network
    """
    if args.net == 'twonet':
        from StageNet import mymodel
        net = mymodel()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net

def img_loader(path, is_img):
    if is_img:
        img = Image.open(path).convert('L')
    else:
        img = Image.open(path).convert('1')
    return img

def read_txt(txt_name):
    data = []
    for line in open(txt_name, 'r',encoding='gbk'):
        data.append(line)
    random.shuffle(data)
    return data

class MyDataset(Dataset):
    def __init__(self, mode, file, transform=None, loader=img_loader):
        self.file = file
        self.mode = mode
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        fn = self.file[index]
        label = int(fn.split(' ')[6]) - 1
        sex = fn.split(' ')[2]
        age = fn.split(' ')[3]
        progress = fn.split(' ')[4]
        minus_progress = fn.split(' ')[5]
        img = torch.empty(36, 300, 300)
        #data agumentation
        random_number = random.randint(1, 16)
        if random_number == 1 or random_number == 9 and self.mode == 'train':
            #left lung
            for i in range(9):
                slice = self.loader(fn.split(' ')[0] + str(i) + '.png', True)
                slice = slice.transpose(Image.FLIP_LEFT_RIGHT)
                slice_o = self.transform(slice)
                img[i, :, :] = slice_o
            # right lung
            for j in range(9, 18):
                slice = self.loader(fn.split(' ')[1] + str(j) + '.png', True)
                slice = slice.transpose(Image.FLIP_LEFT_RIGHT)
                slice_o = self.transform(slice)
                img[j, :, :] = slice_o
            # left lung of prevous CT scan
            for j in range(18, 27):
                slice = self.loader(fn.split(' ')[1] + str(j - 18) + '.png', True)
                slice = slice.transpose(Image.FLIP_LEFT_RIGHT)
                slice_o = self.transform(slice)
                img[j, :, :] = slice_o
            # right lung of prevous CT scan
            for j in range(27, 36):
                slice = self.loader(fn.split(' ')[1] + str(j - 18) + '.png', True)
                slice = slice.transpose(Image.FLIP_LEFT_RIGHT)
                slice_o = self.transform(slice)
                img[j, :, :] = slice_o
        label = torch.from_numpy(numpy.array(label, numpy.int64, copy=False))
        sex = torch.from_numpy(numpy.array(sex, numpy.float32, copy=False))
        age = torch.from_numpy(numpy.array(age, numpy.float32, copy=False))
        progress = torch.from_numpy(numpy.array(progress, numpy.float32, copy=False))
        minus_progress = torch.from_numpy(numpy.array(minus_progress, numpy.float32, copy=False))
        return img,label,sex,age,progress,minus_progress

    def __len__(self):
        return len(self.file)

def get_training_dataloader(txt_path,mean,std, batch_size=24, num_workers=2, shuffle=True):

    train_data = MyDataset(
        mode='train',
        file=read_txt(txt_path + 'train.txt'),
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean],
                                 std=[std])
        ]))

    cifar100_training_loader = DataLoader(
        train_data, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_dataloader(txt_path,mean,std,batch_size=24, num_workers=2, shuffle=True):

    test_data = MyDataset(
        mode='test',
        file=read_txt(txt_path + 'test.txt'),
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean],
                                 std=[std])
        ]))
    cifar100_test_loader = DataLoader(
        test_data, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return cifar100_test_loader

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]





