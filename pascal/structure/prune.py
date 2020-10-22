import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torchvision
import torch
import torch.nn as nn
from l1_utils import prune_inplace, save_checkpoint, prune_vgg_cfg
from utils import set_random_seed, compute_mAP, AverageMeter, load_model
import compute_flops
from model import VGG
from model.vgg_cfg import cfg as vgg_config
from dataset import get_loader

parser = argparse.ArgumentParser(description='PyTorch Pascal L1-norm Pruning')

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
parser.add_argument('--percent', default=0.3, type=float,
                    help='percentage of weight to prune')


def main():
    global args, best_prec1
    args = parser.parse_args()

    if not os.path.exists(args.exp_name):
        os.mkdir(args.exp_name)

    # create model
    model = VGG('VGG16_VOC', input_dims=(3,227,227))

    print('Pre-processing Successful!')

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

    # Computation costs before pruning
    print('Pruning threshold: {}'.format(args.percent))
    nparams_original = compute_flops.print_model_param_nums(model)
    flops_original = compute_flops.count_model_param_flops(model, input_res=224, multiply_adds=True)
    print('FLOPS: {0}'.format(flops_original))

    # Computation costs after pruning
    prune_rate = args.percent
    layerwise_prune_rate = [prune_rate, 0.0,
                            0.0, 0.0,
                            0.0, 0.0, 0.0,
                            prune_rate, prune_rate, prune_rate,
                            prune_rate, prune_rate, prune_rate]
    print('VGG layerwise kill ratio: {0}'.format(layerwise_prune_rate))

    prune_inplace(model, prune_rate=layerwise_prune_rate)
    nparams_pruned = compute_flops.print_model_param_nums(model)
    flops_pruned = compute_flops.count_model_param_flops(model, input_res=224, multiply_adds=True)
    print('FLOPS: {0}'.format(flops_pruned))

    test_mAP1 = validate(val_loader, model, criterion)

    save_checkpoint({
        'epoch': 0,
        'arch': args,
        'p_rate':nparams_pruned/nparams_original,
        'state_dict': model.state_dict(),
        'best_mAP': test_mAP1,
        }, False, args.exp_name)

    with open(os.path.join(args.exp_name, 'prune.txt'), 'w') as f:
        f.write('Before pruning: Test mAP:  %.2f\n' % (test_mAP0))
        f.write('Total params: {}, Pruned params: {}, Pruned ratio: {}\n'.format(nparams_original,
                                                                                 nparams_pruned,
                                                                                 nparams_pruned/nparams_original))
        f.write('After Pruning: Test mAP:  %.2f\n' % (test_mAP1))
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

def main_check():

    prune_rate = 0.5

    print('Pre-processing Successful!')

    model = torchvision.models.vgg16_bn()

    layerwise_prune_rate = [prune_rate, 0.0,
                            0.0, 0.0,
                            0.0, 0.0, 0.0,
                            prune_rate, prune_rate, prune_rate,
                            prune_rate, prune_rate, prune_rate]

    #############################################################################################################################

    print('Pruning rate: {}'.format(prune_rate))
    compute_flops.print_model_param_nums(model)
    flops_original = compute_flops.count_model_param_flops(model, input_res=224, multiply_adds=True)
    print('FLOPS: {0}'.format(flops_original))
    print(model)

    prune_inplace(model, prune_rate=layerwise_prune_rate)

    compute_flops.print_model_param_nums(model)
    flops_pruned = compute_flops.count_model_param_flops(model, input_res=224, multiply_adds=True)
    print('FLOPS: {0}'.format(flops_pruned))
    print(model)

    ##############################################################################################################################

    aaa = prune_vgg_cfg(vgg_config['VGG16_ImageNet'], layerwise_prune_rate, 1)
    return

if __name__ == '__main__':
    main()

