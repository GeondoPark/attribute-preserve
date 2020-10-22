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
import numpy as np
from l1_utils import prune_inplace

parser = argparse.ArgumentParser(description='PyTorch Iterative Structure Pruning Pascal Training')
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

##Pruning Arguments
parser.add_argument('--model_path', default='', type=str, 
                    help='Initialize model path & Teacher path')
parser.add_argument('--percent', default=0.3, type=float,
                    help='percentage of weight to prune')
parser.add_argument('--iter', default=1, type=int,
                    help='How many times to repeat')

##Transfer Arguments
parser.add_argument('--transfer_type', default=None, type=str, 
                    help='Distillation Transfer type, choose in [ewa, swa, None]'+
                    'If none, only use logit distillation')
parser.add_argument('--transfer_weight', default=0, type=float, 
                    help='Transfer Weight.')
parser.add_argument('--stochastic', action='store_true', 
                    help='Use stochastic matching or Not')
parser.add_argument('--drop_percent', default=0.1, type=float,
                    help='For stochasticity, drop ratio')


def main():

    global args
    args = parser.parse_args()

    if args.seed is not None:
        set_random_seed(args.seed)

    # create model
    teacher_model = VGG('VGG16_VOC', input_dims=(3,227,227))
    model = VGG('VGG16_VOC', input_dims=(3,227,227))
    if args.gpu:
        model = torch.nn.DataParallel(model).cuda()
        teacher_model = torch.nn.DataParallel(teacher_model).cuda()
    else:
        model = torch.nn.DataParallel(model)
        teacher_model = torch.nn.DataParallel(teacher_model)

    _, mAP_full = load_model(args.model_path, model)
    _, teacher_mAP = load_model(args.model_path, teacher_model)

    train_loader = get_loader(data_name='VOC2012', 
                            data_path=args.data_path,
                            split='train', 
                            batch_size= args.batch_size)
    val_loader = get_loader(data_name='VOC2012', 
                            data_path=args.data_path,
                            split='val', 
                            batch_size= args.batch_size)

    criterion = nn.MultiLabelSoftMarginLoss()

    print('Full Network Starting mAP : {}'.format(mAP_full))
    num_parameters = sum([param.nelement() for param in model.parameters()])
    print('Starting #Parameters: {}'.format(num_parameters))

    for it in range(1, args.iter+1):
        best_mAP = 0
        prune_percent  = 1 - ((1 - args.percent)**it)

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        streamHandler = logging.StreamHandler()
        file_handler = logging.FileHandler('./loggers/{}_{:.3f}.txt'.format(args.exp_name, prune_percent))
        logger.addHandler(file_handler)
        logger.addHandler(streamHandler)

        logger.info('Start Iterative Pruning -- iter : {} (prune_rate : {:.3f})'.format(it, prune_percent))
        save_file = './weights/{}_{:.3f}'.format(args.exp_name, prune_percent)

        if not os.path.exists(save_file):
            os.makedirs(save_file)

        # Prune
        #####################################################################################################
        print('Prune rate: {}'.format(prune_percent))
        num_parameters = sum([param.nelement() for param in model.parameters()])
        print('Parameters BEFORE: {}'.format(num_parameters))
        model, test_mAP, erased_value = prune(prune_percent, model, criterion, val_loader, save_file, logger)
        num_parameters = sum([param.nelement() for param in model.parameters()])
        print('Parameters AFTER: {}'.format(num_parameters))

        #####################################################################################################
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        logger.info('Start Finetuning model with prune_rate with mAP {:.3f}'.format(test_mAP))

        for epoch in range(0, args.epochs):

            # train for one epoch
            train(train_loader, model, teacher_model, criterion, optimizer, epoch, logger)

            # evaluate on validation set
            mAP = validate(val_loader, model, criterion, epoch, logger)

            # remember best prec@1 and save checkpoint
            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args,
                'p_rate': prune_percent,
                'state_dict': model.state_dict(),
                'best_mAP': best_mAP,
                'optimizer' : optimizer.state_dict(),
            }, is_best,checkpoint=save_file)
            logger.info('SAVED_BEST_mAP for prune {:.3f} : {:.3f}'.format(prune_percent, best_mAP))

        with open(os.path.join(save_file, 'result_dual.txt'), 'w') as f:
            f.write('Final Result of Pruned rate {:.3f} : {:.3f}'.format(prune_percent, best_mAP))
    return

def get_conv_zero_param(model):
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            total += torch.sum(m.weight.data.eq(0))
    return total

def train(train_loader, model, teacher_model, criterion, optimizer, epoch, logger):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_losses = AverageMeter()
    train_mAP = AverageMeter()
    train_auxlosses = AverageMeter()

    # switch to train mode
    teacher_model.eval()
    model.train()
    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        aux_loss = 0
        if args.gpu:
            input = input.to('cuda', non_blocking=True)
            target = target.to('cuda', non_blocking=True)

        if args.stochastic:
            sample_number = int(512 * args.drop_percent)
            teacher_erase_channel = random.sample(range(0, 512), sample_number)

        else:
            teacher_erase_channel = None

        # compute output
        output_t, attribute_t, grad_out = teacher_model(input,
                                                    mode=args.transfer_type,
                                                    TS='Teacher',
                                                    erase_channel=teacher_erase_channel)

        output_s, attribute_s, _ = model(input,
                                         mode=args.transfer_type,
                                         TS='Student',
                                         grad_out=grad_out)
        loss = criterion(output_s, target)
        train_losses.update(loss.item(), input.size(0))

        if args.transfer_type == 'swa':
            for l in range(len(attribute_t)):
                aux_loss += cal_l2loss(attribute_t[l], attribute_s[l])
            teacher_model.zero_grad()
            model.zero_grad()

        elif args.transfer_type == 'ewa':
            for l in range(len(attribute_t)):
                aux_loss += cal_l2loss(x=attribute_t[l].pow(2).mean(1).view(output_s.size(0), -1),
                                       y=attribute_s[l].pow(2).mean(1).view(output_s.size(0), -1))

        train_auxlosses.update(args.transfer_weight * aux_loss.item(), input.size(0))
        loss += aux_loss*args.transfer_weight
        # measure accuracy and record loss
        train_mAP.update(100*compute_mAP(target.data, output_s.data), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Aux Loss {aux_loss.val:.5f} ({aux_loss.avg:.5f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'mAP {mAP.val:.3f} ({mAP.avg:.3f})\t'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       aux_loss=train_auxlosses, loss=train_losses, mAP=train_mAP,))

def cal_l2loss(x, y):
    return (F.normalize(x.view(x.size(0), -1)) - F.normalize(y.view(y.size(0), -1))).pow(2).mean()

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
        best_acc = checkpoint['best_mAP']
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(file_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(file_path))
    return start_epoch, best_acc

def prune(prune_rate, model, criterion, val_loader, savefile, logger):

    logger.info('Kill ratio: {0}'.format(prune_rate))

    # if isinstance(prune_rate, float):
    #     prune_rate = [prune_rate] * 100

    layerwise_prune_rate = [prune_rate, 0.0,
                            0.0, 0.0,
                            0.0, 0.0, 0.0,
                            prune_rate, prune_rate, prune_rate,
                            prune_rate, prune_rate, prune_rate]
    logger.info('VGG layerwise kill ratio: {0}'.format(layerwise_prune_rate))

    model, teacher_student = prune_inplace(model, prune_rate=layerwise_prune_rate)
    ##############################################################################################################################

    test_mAP = 0

    test_mAP = validate(val_loader, model, criterion, 0, logger)

    save_checkpoint({
        'epoch': 0,
        'args': args,
        'p_rate': prune_rate,
        'state_dict': model.state_dict(),
        'best_mAP': test_mAP,
        }, False, savefile, filename='beforefinetvalidateune.pth.tar')

    with open(os.path.join(savefile, 'result_{0}.txt'.format(args.transfer_type)), 'w') as f:
        f.write('Layer kill rate: {0}\n'.format(prune_rate))
        f.write('After Pruning: Test mAP:  %.2f\n' % (test_mAP))

    return model, test_mAP, teacher_student

if __name__ == '__main__':
    main()
