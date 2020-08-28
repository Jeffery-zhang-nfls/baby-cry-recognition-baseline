#! python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import os
import sys
import shutil
import time
import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from multiprocessing import cpu_count

import PIL

from utils import Bar, AverageMeter, accuracy, mkdir_p
import models as customized_models
from datasets import TrainDatasetByFeatures
from loss_utils import AMSoftmax


# CONST
train_epochs = 1000  # 100
train_batch = 32  # 256
test_batch = 16  # 100
# OPTIMIZER
weight_decay = 0.0002  # 1e-4

# Models
default_model_names = sorted(name for name in models.__dict__
                             if name.islower() and not name.startswith("__")
                             and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
                                 if name.islower() and not name.startswith("__")
                                 and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch DeepModel Training')

# Datasets
parser.add_argument('-d', '--data', default='path to dataset', type=str)
parser.add_argument('-j', '--workers', default=cpu_count(), type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=1024, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=16, type=int, metavar='N',
                    help='test batchsize (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[30, 60, 90],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.0002, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='se_resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('--classifier_opt', '-cls', metavar='Classifier_Opt', default='am', help="classifier_opt")
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=32, help='ResNet cardinality (group).')
parser.add_argument('--base-width', type=int, default=4, help='ResNet base width.')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--model_weight', dest='model_weight', default=None, type=str,
                    help='custom pretrained model weight')
# Device options
parser.add_argument('--cpu', dest='cpu', action='store_true',
                    help='use cpu mode')
parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
if args.cpu:
    print('Use CPU mode')
    use_cuda = False
    pin_memory = False
else:
    print('Use CUDA mode')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()
    pin_memory = True

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy


################################################################################################################

class totensor(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.FloatTensor(pic.transpose((0, 2, 1)))
            return img


transform = transforms.Compose([totensor()])


################################################################################################################
def main():
    args.checkpoint = "exp/{}".format(args.arch)
    #     args.model_weight = "pretrained/{}.pth.tar".format(args.arch)
    print(args)
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data loading code
    train_dir = SoftmaxDatasetByFeatures(transform=transform)
    # val_dir = ValidationDatasetByFeatures(transform=transform)
    # 因为dataset里面 是随机取样本的，所以这里的data_loader 的shuffle设置为False
    train_loader = DataLoader(train_dir, batch_size=args.train_batch, num_workers=args.workers, shuffle=False)
    # val_loader = DataLoader(val_dir, batch_size=args.test_batch, num_workers=args.workers, shuffle=False)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    elif args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
            baseWidth=args.base_width,
            cardinality=args.cardinality,
        )
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if use_cuda:
        # model = torch.nn.DataParallel(model).cuda() # 会引起各种奇怪的错误，如在使用CPUmode 的时候，map_location也不起作用
        model = model.cuda()
        # cudnn.benchmark = True
    print(model)
    if args.model_weight:
        model_weight = torch.load(args.model_weight)
        model.load_state_dict(model_weight['state_dict'])

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss()
    
    if "am" == args.classifier_opt:
        criterion = AMSoftmax()
        # criterion = AMSoftmax(in_feats=6, n_classes=6)
    elif "a" == args.classifier_opt:
        criterion = AngleLoss()
    else:
        raise Exception("unsupported classifier_opt: {}".format(args.classifier_opt))

    if use_cuda:
        criterion = criterion.cuda()

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # 这里使用 Adagrad
    optimizer = optim.Adagrad(model.parameters(),lr=args.lr,lr_decay=1e-4,weight_decay=args.weight_decay)

    # Resume
    title = 'RegBabyCry-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(val_loader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Top1 Acc:  %.2f ' % (test_loss, test_acc))
        print(' Top1 Err:  %.2f' % (100.0 - test_acc))
        return

    # Train and val
    # test_loss, test_acc = test(val_loader, model, criterion, start_epoch, use_cuda)
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
        # test_loss, test_acc = test(val_loader, model, criterion, epoch, use_cuda)

        # save model
        is_best = train_acc > best_acc
        best_acc = max(train_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': train_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)

    print('Best acc:')
    print(best_acc)


def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    # torch.set_grad_enabled(True)
    torch.autograd.set_detect_anomaly(True)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    bar = Bar('P', max=len(train_loader))
    pbar = tqdm(enumerate(train_loader))
    for batch_idx, (inputs, targets) in pbar:
        # measure data loading time
        data_time.update(time.time() - end)

        # print("inputs: ", inputs.size(), inputs.dtype, inputs.requires_grad)
        # print("targets: ", targets.size(), targets.dtype, targets.requires_grad)
        # print(targets)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            
        # print("inputs: ", inputs.size(), inputs.dtype, inputs.requires_grad)
        # print("targets: ", targets.size(), targets.dtype, targets.requires_grad)

        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # print("inputs: ", inputs.size(), inputs.dtype, inputs.requires_grad)
        # print("targets: ", targets.size(), targets.dtype, targets.requires_grad)

        # compute output
        out_fea, out_cls = model(inputs)
        # print("out_fea: ", out_fea.size(), out_fea.dtype, out_fea.requires_grad)
        # print("out_cls: ", out_cls.size(), out_cls.dtype, out_cls.requires_grad)

        loss = criterion(out_cls, targets, scale=30.0, margin=0.35)
        if "a" == args.classifier_opt:
            prec1, = accuracy(out_cls[0].data, targets.data, topk=(1,))
        elif "am" == args.classifier_opt:
            prec1, = accuracy(out_cls.data, targets.data, topk=(1,))
        # measure accuracy and record loss
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if (batch_idx + 1) % 1 == 0:
            #             print(
            #                 'train ({batch}/{size}) D: {data:.2f}s | B: {bt:.2f}s | T: {total:} | E: {eta:} | L: {loss:.3f} | t1: {top1: .3f} '.format(
            #                     batch=batch_idx + 1,
            #                     size=len(train_loader),
            #                     data=data_time.val,
            #                     bt=batch_time.val,
            #                     total=bar.elapsed_td,
            #                     eta=bar.eta_td,
            #                     loss=losses.avg,
            #                     top1=top1.avg,
            #                 ))
            pbar.set_description(
                'train ({batch}/{size}) D: {data:.2f}s | B: {bt:.2f}s | T: {total:} | E: {eta:} | L: {loss:.3f} | t1: {top1: .3f} '.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                ))
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)


def test(val_loader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()

    bar = Bar('P', max=len(val_loader))
    pbar = tqdm(enumerate(val_loader))
    for batch_idx, (inputs, targets) in pbar:
        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        end = time.time()

        outputs = model(inputs)
        # print("outputs: ", outputs)
        # print("targets: ", targets)
        batch_time.update(time.time() - end)
        loss = criterion(outputs, targets, scale=30.0, margin=0.35)
        prec1, = accuracy(outputs.data, targets.data, topk=(1,))

        # measure accuracy and record loss
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))

        # plot progress
        if (batch_idx + 1) % 1 == 0:
            #             print(
            #                 'test ({batch}/{size}) D: {data:.2f}s | B: {bt:.2f}s | T: {total:} | E: {eta:} | L: {loss:.3f} | t1: {top1: .3f} '.format(
            #                     batch=batch_idx + 1,
            #                     size=len(val_loader),
            #                     data=data_time.avg,
            #                     bt=batch_time.avg,
            #                     total=bar.elapsed_td,
            #                     eta=bar.eta_td,
            #                     loss=losses.avg,
            #                     top1=top1.avg,
            #                 ))
            pbar.set_description(
                'test ({batch}/{size}) D: {data:.2f}s | B: {bt:.2f}s | T: {total:} | E: {eta:} | L: {loss:.3f} | t1: {top1: .3f} '.format(
                    batch=batch_idx + 1,
                    size=len(val_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                ))
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


if __name__ == '__main__':
    main()
