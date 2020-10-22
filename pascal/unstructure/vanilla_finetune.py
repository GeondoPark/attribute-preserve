import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model import *
import logging
from dataset import get_loader
from utils import set_random_seed, compute_mAP, AverageMeter
import argparse
import random
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='PyTorch Pascal VOC Training')
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--seed', default=None, type=int, 
                    help="Random Seed")
parser.add_argument('--arch', default='VGG16_VOC', type=str,
                    help='Model architecture')
parser.add_argument('--data_path', default="./data/VOCdevkit/VOC2012", type=str,
                    help="Data path, here for multi-label classification")
parser.add_argument('--exp_name', default='None', type=str,
                    help='Name of Current Running')

parser.add_argument('--epochs', default=30, type=int, 
                    help='number of total epochs to train')
parser.add_argument('--learning_rate', default=0.001, type=float, 
                    help='initial learning rate')
parser.add_argument('--batch_size', default=64, type=int,
                    help='mini-batch size, this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--momentum', default=0.9, type=float, 
                    help='Optimizer Momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, 
                    help='For Optimizer, weight decay')
parser.add_argument('--print_freq', default=30, type=int, 
                    help='print frequency')
parser.add_argument('--model_path', default='', type=str, 
                    help='Initialize model path & Teacher path')
parser.add_argument('--percent', default=0.2, type=float,
                    help='percentage of weight to prune')

best_mAP = 0

def main():
    global args, best_mAP
    args = parser.parse_args()

    logger = logging.getLogger(__name__)        
    logger.setLevel(logging.INFO)
    streamHandler = logging.StreamHandler()
    file_handler = logging.FileHandler('./loggers/{}.txt'.format(args.exp_name))
    logger.addHandler(file_handler)
    logger.addHandler(streamHandler)

    if args.seed is not None:
        set_random_seed(args.seed)

    args.exp_name = './weights/{}'.format(args.exp_name)

    if not os.path.exists(args.exp_name):
        os.makedirs(args.exp_name)

    # create model
    model = VGG('VGG16_VOC', input_dims=(3,227,227))

    if args.gpu:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    _, mAP, p_rate = load_model(args.model_path, model)

    train_loader = get_loader(data_name='VOC2012', 
                            data_path=args.data_path,
                            split='train', 
                            batch_size= args.batch_size)

    val_loader = get_loader(data_name='VOC2012', 
                            data_path=args.data_path,
                            split='val', 
                            batch_size= args.batch_size)

    criterion = nn.MultiLabelSoftMarginLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    test_mAP = validate(val_loader, model, criterion, 0, logger)

    logger.info('Start Finetuning model with prune_rate {:.3f} with mAP {:.3f}'.format(p_rate.item(), test_mAP))
    # Data loading code

    for epoch in range(0, args.epochs):

        #####################################################################################################
        num_parameters = get_conv_zero_param(model)
        print('Zero parameters: {}'.format(num_parameters))
        num_parameters = sum([param.nelement() for param in model.parameters()])
        print('Parameters: {}'.format(num_parameters))
        #####################################################################################################

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, logger)

        # evaluate on validation set
        mAP = validate(val_loader, model, criterion, epoch, logger)

        # remember best prec@1 and save checkpoint
        is_best = mAP > best_mAP
        best_mAP = max(mAP, best_mAP)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args,
            'p_rate': p_rate,
            'state_dict': model.state_dict(),
            'best_mAP': best_mAP,
            'optimizer' : optimizer.state_dict(),
        }, is_best,checkpoint=args.exp_name)
        logger.info('SAVED_BEST_mAP : {:.3f}'.format(best_mAP))
    return

def get_conv_zero_param(model):
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            total += torch.sum(m.weight.data.eq(0))
    return total

def train(train_loader, model, criterion, optimizer, epoch, logger):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_losses = AverageMeter()
    train_mAP = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu:
            input = input.to('cuda', non_blocking=True)
            target = target.to('cuda', non_blocking=True)

        # compute output
        output, _, _ = model(input)

        loss = criterion(output, target)
        train_losses.update(loss.item(), input.size(0))

        # measure accuracy and record loss
        train_mAP.update(100*compute_mAP(target.data, output.data), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        for k, m in enumerate(model.modules()):
            if isinstance(m, nn.Conv2d):
                weight_copy = m.weight.data.abs().clone()
                mask = weight_copy.gt(0).float().cuda()
                m.weight.grad.data.mul_(mask)

        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'mAP {mAP.val:.3f} ({mAP.avg:.3f})\t'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=train_losses, mAP=train_mAP))

def validate(val_loader, model, criterion, epoch, logger):
    valid_losses = AverageMeter()
    outputs = []
    targets = []

    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if args.gpu:
                images = images.to('cuda', non_blocking=True)
                target = target.to('cuda', non_blocking=True)
            # compute output
            output, _, _ = model(images, mode='eval')
            loss = criterion(output, target)
            outputs.append(output)
            targets.append(target)
            valid_losses.update(loss.item(), images.size(0))

        mAP = 100*compute_mAP(torch.cat(targets, dim=0).data, torch.cat(outputs, dim=0).data)
        logger.info('Epoch[{ep}] : val_loss {loss.avg:.3f} mAP {map_: .3f}'
            .format(ep=epoch, loss=valid_losses, map_=mAP))
    return mAP

def save_checkpoint(state, is_best, checkpoint, filename='finetuned.pth.tar'):
    filepath = os.path.join(checkpoint, 'checkpoint_'+filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best_'+filename))

def load_model(file_path, model, optimizer=None):
    if os.path.isfile(file_path):
        print("=> loading checkpoint '{}'".format(file_path))
        checkpoint = torch.load(file_path)
        start_epoch = checkpoint['epoch']
        p_rate = checkpoint['p_rate']
        best_acc = checkpoint['best_mAP']
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(file_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(file_path))

    return start_epoch, best_acc, p_rate

if __name__ == '__main__':
    main()