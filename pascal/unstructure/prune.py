import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model import *
import logging
from dataset import get_loader
from utils import set_random_seed, compute_mAP, AverageMeter, load_model
import argparse
import random
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--seed', default=None, type=int, 
                    help="Random Seed")
parser.add_argument('--arch', default='VGG16_VOC', type=str,
                    help='Model architecture')
parser.add_argument('--data_path', default="./data/VOCdevkit/VOC2012", type=str,
                    help="Data path, here for multi-label classification")
parser.add_argument('--exp_name', default='None', type=str,
                    help='Name of Current Running')
parser.add_argument('--batch_size', default=64, type=int,
                    help='mini-batch size, this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--model_path', default='', type=str, 
                    help='Initialize model path & Teacher path')
parser.add_argument('--percent', default=0.2, type=float,
                    help='percentage of weight to prune')

def main():
    global args
    args = parser.parse_args()

    if not os.path.exists(args.exp_name):
        os.mkdir(args.exp_name)

    # create model
    model = VGG('VGG16_VOC', input_dims=(3,227,227))

    if args.gpu:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    _, mAP = load_model(args.model_path, model)

    val_loader = get_loader(data_name='VOC2012', 
                            data_path=args.data_path,
                            split='val', 
                            batch_size= args.batch_size)

    criterion = nn.MultiLabelSoftMarginLoss().cuda()

    test_mAP0 = validate(val_loader, model, criterion)
    #############################################################################################################################
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            total += m.weight.data.numel()

    conv_weights = torch.zeros(total).cuda()
    index = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            size = m.weight.data.numel()
            conv_weights[index:(index+size)] = m.weight.data.view(-1).abs().clone()
            index += size

    y, i = torch.sort(conv_weights)
    thre_index = int(total * args.percent)
    thre = y[thre_index]
    pruned = 0
    print('Pruning threshold: {}'.format(thre))
    zero_flag = False
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre).float().cuda()
            pruned = pruned + mask.numel() - torch.sum(mask)
            m.weight.data.mul_(mask)
            if int(torch.sum(mask)) == 0:
                zero_flag = True
            print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.
                format(k, mask.numel(), int(torch.sum(mask))))
    print('Total conv params: {}, Pruned conv params: {}, Pruned ratio: {}'.format(total, pruned, pruned/total))
    ##############################################################################################################################
    test_mAP1 = validate(val_loader, model, criterion)

    save_checkpoint({
        'epoch': 0,
        'arch': args,
        'p_rate':pruned/total,
        'state_dict': model.state_dict(),
        'best_mAP': test_mAP1,
        }, False, args.exp_name)

    with open(os.path.join(args.exp_name, 'pruneinfo.txt'), 'w') as f:
        f.write('Before pruning: Test mAP:  %.2f\n' % (test_mAP0))
        f.write('Total conv params: {}, Pruned conv params: {}, Pruned ratio: {}\n'.format(total, pruned, pruned/total))
        f.write('After Pruning: Test mAP:  %.2f\n' % (test_mAP1))
        if zero_flag:
            f.write("There exists a layer with 0 parameters left.")
    return

def validate(val_loader, model, criterion):
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
    model.train()
    return mAP

def save_checkpoint(state, is_best, checkpoint, filename='pruned.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

if __name__ == '__main__':
    main()