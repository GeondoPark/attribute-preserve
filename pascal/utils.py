import os
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import average_precision_score

class Hooks():
    def __init__(self, model, target_layer_list, network='VGG'):

        self.gradients = []
        self.activations = []
        if network == 'VGG':
            for layer in target_layer_list:
                features_layer = model.module.features._modules[layer]
                features_layer.register_forward_hook(self.hook_fn)
                features_layer.register_backward_hook(self.hook_back_fn)
        else:
            feature_layer = model.module.layer4[2].bn2.register_forward_hook(self.hook_fn)
            feature_layer = model.module.layer4[2].bn2.register_backward_hook(self.hook_fn)
    def hook_back_fn(self, module, grad_in, grad_out):
        self.gradients.append(grad_out[0])

    def hook_fn(self, module, inp, out):
        self.activations.append(out)

    def reset_(self):
        self.gradients = []
        self.activations = []
    def detach(self):
        self.Hooks.remove()

def set_random_seed(seed):
    if seed is None:
        seed = 4242
    import random
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def save_checkpoint(state, exp_name, is_best, filename='checkpoint.pth.tar'):

    file_name = os.path.join('./weights/', exp_name, exp_name+'_checkpoint.pth.tar')
    torch.save(state, file_name)
    if is_best:
        shutil.copyfile(file_name, os.path.join('./weights/', exp_name, exp_name+'_best.pth.tar'))

def log_epoch_vanilla(logger, epoch, train_loss, train_score, valid_score):
    logger.add_scalar('Loss/Train', train_loss, epoch)
    logger.add_scalar('Train Score', train_score, epoch)
    logger.add_scalar('Vlidation Score', valid_score, epoch)

def log_epoch_transfer(logger, epoch, train_loss, score, aux_loss):
    logger.add_scalar('Loss/Train', train_loss, epoch)
    logger.add_scalar('Train Score', score, epoch)
    logger.add_scalar('aux_loss', aux_loss, epoch)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)

def adjust_learning_rate(optimizer, epoch, opt):
    if epoch in opt.lr_decay_list:
        opt.learning_rate = optimizer.param_groups[0]['lr'] * opt.lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = opt.learning_rate
        print('Decayed learning rate by a factor {} to {}'.format(opt.lr_decay, opt.learning_rate))

def load_model(file_path, model, optimizer=None, strict=True):
    if os.path.isfile(file_path):
        print("=> loading checkpoint '{}'".format(file_path))
        checkpoint = torch.load(file_path)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_mAP']
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'], strict=strict)
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(file_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(file_path))

    return start_epoch, best_acc

def compute_mAP(labels,outputs):
    y_true = labels.cpu().numpy()
    y_pred = outputs.cpu().numpy()
    AP = []
    for i in range(y_true.shape[0]):
        AP.append(average_precision_score(y_true[i],y_pred[i]))
    return np.mean(AP)
