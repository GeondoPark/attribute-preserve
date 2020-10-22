import os, sys
import torch
import argparse
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils import set_random_seed
from model import *
from VOC_trainer import VOC2012_distill_trainer
from dataset import get_loader

parser = argparse.ArgumentParser(description='PyTorch PASCAL_VOC multi-label classification Distillation Training')

parser.add_argument('--gpu', action='store_true')
parser.add_argument('--seed', default=42, type=int, 
                    help="Random Seed")
parser.add_argument('--arch', default='VGG16_VOC_x4', type=str,
                    help='Student Model architecture')
parser.add_argument('--data_path', default="./data/VOCdevkit/VOC2012", type=str,
                    help="Data path, here for multi-label classification")
parser.add_argument('--exp_name', default='None', type=str,
                    help='Name of Current Running')
parser.add_argument('--teacher_path', default='teacher', type=str, required=True,
                    help='Path of teacher model network')

parser.add_argument('--learning_rate', default=0.1, type=float, 
                    help='initial learning rate')
parser.add_argument('--batch_size', default=64, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--epochs', default=350, type=int, 
                    help='number of total epochs to train')
parser.add_argument('--momentum', default=0.9, type=float, 
                    help='Optimizer Momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, 
                    help='For Optimizer, weight decay')
parser.add_argument('--print_freq', default=30, type=int, 
                    help='print frequency')
parser.add_argument('--lr_decay_list', type=int, nargs='+', default=[150, 200, 250],
                    help='Decay Learning Rate at this epochs')
parser.add_argument('--lr_decay', type=float, default=0.1,
                    help="Multiplicate current learning rate with this value")

##Distillation Arguments
parser.add_argument("--alpha", default=0.9, type=float,
                    help="Distillation loss linear weight. Only for distillation.")
parser.add_argument("--temperature", default=4.0, type=float, 
                    help="Distillation temperature. Only for distillation.")
parser.add_argument('--transfer_type', default=None, type=str, 
                    help='Distillation Transfer type, choose in [ewa, swa, None]'+
                    'If none, only use logit distillation')
parser.add_argument('--transfer_weight', default=0, type=float, 
                    help='Transfer Weight.')
parser.add_argument('--stochastic', action='store_true', 
                    help='Use stochastic matching or Not')
parser.add_argument('--drop_percent', default=0.1, type=float,
                    help='For stochasticity, drop ratio')

opt = parser.parse_args()

set_random_seed(opt.seed)
teacher = VGG('VGG16_VOC', input_dims=(3,227,227))
student = VGG(opt.arch, input_dims=(3,227,227))

trainer = VOC2012_distill_trainer(student =student,
                                teacher=teacher,
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
