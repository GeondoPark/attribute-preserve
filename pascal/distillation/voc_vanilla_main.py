import os 
import sys
import torch
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import argparse
from utils import set_random_seed
#from model import *
from vgg_pretrain import *
from VOC_trainer import VOC2012_vanilla_trainer
from dataset import get_loader

##Imagenet initialize --> Need only 50 Decay : [ 10, 30, 40 ]
parser = argparse.ArgumentParser(description='PyTorch Multi-label calssification PASCAL_VOC Training')

parser.add_argument('--gpu', action='store_true')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--arch', default='vgg16_bn', type=str,
                    help='model architecture')
parser.add_argument('--data_path', default="./data/VOCdevkit/VOC2012", type=str,
                    help="Data path, here for multi-label classification")
parser.add_argument('--exp_name', default='None', type=str, 
                    help='Name of Current Running')
parser.add_argument('--imagenet_pretrained', action='store_true',
                    help='Start with pretrained weights on imagenet or Not')
parser.add_argument('--learning_rate', default=0.1, type=float, 
                    help='initial learning rate')
parser.add_argument('--batch-size', default=64, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--epochs', default=60, type=int, 
                    help='number of total epochs to run'+ 
                    'default is defined for using pretrained imagenet')
parser.add_argument('--momentum', default=0.9, type=float, 
                    help='Optimizer Momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, 
                    help='For Optimizer, weight decay (default: 5e-4)')
parser.add_argument('--print-freq', default=10, type=int, 
                    help='print frequency (default: 10)')
parser.add_argument('--lr_decay_list', type=int, nargs='+', default=[10, 30, 40],
                    help='Decay Learning Rate at this epochs')
parser.add_argument('--lr_decay', type=float, default=0.1,
                    help="Multiplicate current learning rate with this value")

set_random_seed(3)
opt = parser.parse_args()

if opt.imagenet_pretrained:
    if opt.arch == "vgg16_bn":
        model = vgg16_bn(pretrained=True)
    else:
        raise NotImplementedError
else:
    model = VGG(opt.arch, input_dims=(3,227,227))

trainer = VOC2012_vanilla_trainer(model=model,
                                device=torch.device('cuda') if opt.gpu else torch.device('cpu'), 
                                opt=opt)

train_loader = get_loader(data_name='VOC2012', 
                        data_path=opt.data_path,
                        split='train', 
                        batch_size= opt.batch_size)

val_loader = get_loader(data_name='VOC2012', 
                        data_path=opt.data_path,
                        split='val', 
                        batch_size= opt.batch_size)

best_mAP, best_f1 = trainer.train(train_loader, val_loader, opt.epochs)

with open('./results.txt', 'a') as f:
    f.write('{}\t{:.3f}\t{:.3f}'.format(opt.exp_name, best_mAP, best_f1))
print("Train Finished")
