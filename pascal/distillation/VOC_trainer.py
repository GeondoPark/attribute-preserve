import os
import random
import logging
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from utils import save_checkpoint, AverageMeter, adjust_learning_rate, log_epoch_vanilla
from utils import load_model, compute_mAP, log_epoch_transfer
from sklearn.metrics import f1_score, average_precision_score

class VOC2012_vanilla_trainer(object):
    def __init__(self, model, 
                criterion=nn.MultiLabelSoftMarginLoss(),
                device=torch.device('cuda'), 
                opt=None):
        super(VOC2012_vanilla_trainer, self).__init__()
        self.opt = opt
        self.device = device
        self.criterion = criterion
        self.start_epoch = 0
        self.best_mAP = 0
        self.best_f1 = 0

        #### Model, Optimizer, Device Declaration### 
        self.model = torch.nn.DataParallel(model).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                        lr=opt.learning_rate,
                                        momentum=opt.momentum,
                                        weight_decay=opt.weight_decay)

        #### Logger Declair & Make directory
        if not(os.path.isdir(os.path.join('./weights', opt.exp_name))):
            os.makedirs(os.path.join('./weights', opt.exp_name))
        with open('./weights/{}/opt.txt'.format(opt.exp_name), 'w') as f:
            f.write(str(opt))
        self.logger = SummaryWriter(os.path.join("./logs", opt.exp_name))
        self.file_logger = logging.getLogger(__name__)
        self.file_logger.setLevel(logging.INFO)
        streamHandler = logging.StreamHandler()
        file_handler = logging.FileHandler('./loggers/{}.txt'.format(opt.exp_name))
        self.file_logger.addHandler(streamHandler)
        self.file_logger.addHandler(file_handler)

    def train(self, train_loader, val_loader, n_epochs):

        train_losses = AverageMeter()
        train_mAP = AverageMeter()
        train_f1score = AverageMeter()

        self.model.train()
        for epoch in range(self.start_epoch, n_epochs):
            train_losses.reset()
            train_mAP.reset()
            train_f1score.reset()

            adjust_learning_rate(self.optimizer, epoch, self.opt)
            for i, (images, targets) in enumerate(train_loader):

                images, targets = images.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                output, _, _ = self.model(images)
                loss = self.criterion(output, targets)

                train_mAP.update(100*compute_mAP(targets.data, output.data), images.size(0)) 
                train_losses.update(loss.item(), images.size(0))
                train_f1score.update(f1_score(targets.data.cpu().numpy(), output.data.cpu().numpy() > 0.5, average="samples"), images.size(0))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if i % self.opt.print_freq == 0 or i == (len(train_loader) -1):
                    self.file_logger.info("Epoch:[{0}] [{1}/{2}] : train_loss {3:.3f}\tmAP {4:.3f}\tf1 score {5:.3f}"
                                    .format(epoch, i, len(train_loader), train_losses.avg, train_mAP.avg, train_f1score.avg))

            valid_mAP, valid_f1 = self.validate(val_loader, epoch)

            ##Early Stopping with mAP
            is_best = valid_mAP > self.best_mAP
            self.best_mAP = max(valid_mAP, self.best_mAP)
            if is_best:
                self.best_f1 = valid_f1
            save_checkpoint({
                'epoch': epoch + 1,
                'opt_dict': self.opt,
                'state_dict': self.model.state_dict(),
                'best_mAP': self.best_mAP,
                'optimizer' : self.optimizer.state_dict(),
                }, self.opt.exp_name, is_best)

            self.file_logger.info("Saved Best mAP :{:.3f} f1 score: {:.3f}".format(self.best_mAP, self.best_f1))
            log_epoch_vanilla(self.logger, epoch, train_losses.avg, train_mAP.avg, valid_mAP)
        return self.best_mAP, self.best_f1

    def validate(self, val_loader, epoch):
        valid_losses = AverageMeter()
        outputs = []
        targets = []

        self.model.eval()
        with torch.no_grad():
            for i, (images, target) in enumerate(val_loader):
                images = images.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                # compute output
                if self.opt.imagenet_pretrained:
                    output, _ ,_ = self.model(images)
                else:
                    output, _, _ = self.model(images, mode='eval')
                loss = self.criterion(output, target)
                outputs.append(output)
                targets.append(target)
                valid_losses.update(loss.item(), images.size(0))
            mAP = 100*compute_mAP(torch.cat(targets, dim=0).data, torch.cat(outputs, dim=0).data)
            f1 = f1_score(torch.cat(targets, dim=0).data.cpu().numpy(), torch.cat(outputs, dim=0).data.cpu().numpy() > 0.5, average="samples")
            self.file_logger.info('Epoch[{ep}] : val_loss {loss.avg:.3f} mAP {map_: .3f} F1 {f1_: .5f}'
                        .format(ep=epoch, loss=valid_losses, map_=mAP, f1_=f1))
        self.model.train()
        return mAP, f1

class VOC2012_distill_trainer(object):

    def __init__(self, student, teacher,
                 criterion=nn.MultiLabelSoftMarginLoss(),
                 device=torch.device('cuda'),
                 opt=None):

        super(VOC2012_distill_trainer, self).__init__()
        self.opt = opt
        self.criterion = criterion
        self.device = device
        self.student_model = student
        self.teacher_model = teacher
        self.best_mAP = 0
        self.best_f1 = 0
        self.start_epoch = 0

        self.student_model = torch.nn.DataParallel(self.student_model).to(self.device)
        self.teacher_model = torch.nn.DataParallel(self.teacher_model).to(self.device)

        self.optimizer = torch.optim.SGD(self.student_model.parameters(),
                                        lr=opt.learning_rate,
                                        weight_decay=opt.weight_decay)

        _, mAP = load_model(opt.teacher_path, self.teacher_model)
        print("Teacher model's mean Average Precision : %f" % mAP)

        #### Logger Declair & Make directory
        if not (os.path.isdir(os.path.join('./weights', opt.exp_name))):
            os.makedirs(os.path.join('./weights', opt.exp_name))
        with open('./weights/{}/opt.txt'.format(opt.exp_name), 'w') as f:
            f.write(str(opt))

        self.logger = SummaryWriter(os.path.join("./logs", opt.exp_name))
        self.file_logger = logging.getLogger(__name__)
        self.file_logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler('./loggers/{}.txt'.format(opt.exp_name))
        self.file_logger.addHandler(file_handler)

    def train(self, train_loader, val_loader, n_epochs):

        train_losses = AverageMeter()
        train_mAP = AverageMeter()
        train_f1score = AverageMeter()
        train_auxlosses = AverageMeter()

        self.student_model.train()
        self.teacher_model.eval()

        T = self.opt.temperature
        alpha = self.opt.alpha

        for epoch in range(self.start_epoch, n_epochs):

            train_losses.reset()
            train_mAP.reset()
            train_f1score.reset()
            train_auxlosses.reset()
            adjust_learning_rate(self.optimizer, epoch, self.opt)

            for i, (images, targets) in enumerate(train_loader):
                if self.opt.stochastic:
                    ##512 : the number of channles at last feature map in VGG16
                    sample_number = int(512 *self.opt.drop_percent)
                    teacher_erase_channel = random.sample(range(0, 512), sample_number)
                else:
                    teacher_erase_channel = None

                images, targets = images.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)

                output_t, attribute_t, grad_out = self.teacher_model(images,
                                                                    mode=self.opt.transfer_type,
                                                                    TS='Teacher',            
                                                                    erase_channel=teacher_erase_channel)
                output_s, attribute_s, _ = self.student_model(images,
                                                              mode=self.opt.transfer_type,
                                                              TS='Student',
                                                              grad_out=grad_out)

                loss = self.logit_kd(output_s, output_t, targets, T, alpha)
                train_losses.update(loss.item(), images.size(0))
                aux_loss = 0

                if self.opt.transfer_type == 'swa':
                    for l in range(len(attribute_t)):
                        aux_loss += self.cal_l2loss(attribute_t[l], attribute_s[l])
                    self.teacher_model.zero_grad()
                    self.student_model.zero_grad()

                elif self.opt.transfer_type == 'ewa':
                    for l in range(len(attribute_t)):
                        aux_loss += self.cal_l2loss(x=attribute_t[l].pow(2).mean(1).view(output_s.size(0), -1),
                                                    y=attribute_s[l].pow(2).mean(1).view(output_s.size(0), -1))

                if self.opt.transfer_type:
                    train_auxlosses.update(self.opt.transfer_weight * aux_loss.item(), images.size(0))
                else:
                    train_auxlosses.update(self.opt.transfer_weight * aux_loss, images.size(0))

                loss += self.opt.transfer_weight * aux_loss

                train_mAP.update(100*compute_mAP(targets.data, output_s.data), images.size(0)) 
                train_f1score.update(f1_score(targets.data.cpu().numpy(), output_s.data.cpu().numpy() > 0.5, average="samples"), images.size(0))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if i % self.opt.print_freq == 0 or i == (len(train_loader) - 1):
                    self.file_logger.info(
                        "Epoch: [{0}] [{1}/{2}] : train_loss {3:.3f}\taux_loss{4:.3f}\tlr {5:.3f}\tmAP {6:.3f}\tf1_score {7:.3f}"
                        .format(epoch, i, len(train_loader),
                                train_losses.avg,
                                train_auxlosses.avg,
                                self.optimizer.param_groups[0]['lr'],
                                train_mAP.avg,
                                train_f1score.avg))

            valid_mAP, valid_f1 = self.validate(val_loader, epoch)

            ##Early Stopping with mAP
            is_best = valid_mAP > self.best_mAP
            self.best_mAP = max(valid_mAP, self.best_mAP)
            if is_best:
                self.best_f1 = valid_f1
            save_checkpoint({
                'epoch': epoch + 1,
                'opt_dict': self.opt,
                'state_dict': self.student_model.state_dict(),
                'best_mAP': self.best_mAP,
                'optimizer': self.optimizer.state_dict(),
            }, self.opt.exp_name, is_best)
            self.file_logger.info("Saved Best mAP :{:.3f}\t f1 score :{:.3f}".format(self.best_mAP, self.best_f1))
            log_epoch_transfer(self.logger, epoch, train_losses.avg, train_mAP.avg, train_auxlosses.avg)
        return self.best_mAP, self.best_f1

    def validate(self, val_loader, epoch):
        valid_losses = AverageMeter()
        outputs = []
        targets = []
        self.student_model.eval()
        with torch.no_grad():
            for i, (images, target) in enumerate(val_loader):
                images = images.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                # compute output
                output, _, _ = self.student_model(images, mode='eval')
                loss = self.criterion(output, target)
                outputs.append(output)
                targets.append(target)
                valid_losses.update(loss.item(), images.size(0))
            mAP = 100*compute_mAP(torch.cat(targets, dim=0).data, torch.cat(outputs, dim=0).data)
            f1 = f1_score(torch.cat(targets, dim=0).data.cpu().numpy(), torch.cat(outputs, dim=0).data.cpu().numpy() > 0.5, average="samples")
            self.file_logger.info('Epoch[{ep}] : val_loss {loss.avg:.3f} mAP {map_: .3f} F1 {f1_: .5f}'
                        .format(ep=epoch, loss=valid_losses, map_=mAP, f1_=f1))
        self.student_model.train()
        return mAP, f1

    def logit_kd(self, out_s, teacher_scores, labels, T, alpha):
        p = F.sigmoid(out_s / T)
        p = p.unsqueeze(dim=2)
        p = torch.log(torch.cat((p, 1 - p), dim=2) + 1e-7)
        q = F.sigmoid(teacher_scores / T)
        q = q.unsqueeze(dim=2)
        q = torch.cat((q, 1 - q), dim=2)
        p = p.view(-1, 2)
        q = q.view(-1, 2)
        l_kl = F.kl_div(p, q, size_average=False) * (T ** 2) / p.shape[0]
        l_ce = self.criterion(out_s, labels)
        return l_kl * alpha + l_ce * (1. - alpha)

    def cal_l2loss(self, x, y):
        return (F.normalize(x.view(x.size(0), -1)) - F.normalize(y.view(y.size(0), -1))).pow(2).mean()
