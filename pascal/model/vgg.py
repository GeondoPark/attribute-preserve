'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .vgg_cfg import cfg

class VGG(nn.Module):
    def __init__(self, vgg_name, input_dims, specific_cfg = None):
        super(VGG, self).__init__()
        self.vgg_name = vgg_name
        if specific_cfg is None:
            specific_cfg = cfg[vgg_name]
        self.features, self.classifier = self._make_layers(specific_cfg, input_dims)

    def forward(self, x, mode=None, TS=None, grad_out=None, erase_channel=None):

        f0 = self.features[:44](x)
        if mode == "eval":
            pass
        else:
            f0.retain_grad()
        out = self.features[44:](f0)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        if mode == 'swa':
            if not isinstance(grad_out, Variable):
                ind = out.data.max(1)[1]
                grad_out = out.data.clone().fill_(0.0).scatter_(1, ind.unsqueeze(0).t(), 1.0)

            swa = self.cal_grad(out, grad_out, TS, [f0], erase_channel)
            return out, swa, grad_out
        else:
            return out, [f0], None

    def cal_grad(self, out, grad_out, TS, feature, erase_channel):
        attributions = []
        if TS == 'Teacher':
            out.backward(grad_out, retain_graph=True)
            feat = feature[0].clone().detach()
            grad = feature[0].grad.clone().detach()
            if erase_channel is not None:
                for erase_c in erase_channel:
                        feat[:,erase_c,:,:] = 0
            linear = torch.sum(torch.sum(grad, 3, keepdim=True), 2, keepdim=True)        # batch, 512, 1, 1
            channel = linear * feat                                                      # batch, 512, 7, 7
            swa = torch.sum(channel, 1, keepdim=True)                                  # batch, 1, 7, 7
            attributions.append(F.relu(swa))
            return attributions

        elif TS == 'Student':
            out.backward(grad_out, create_graph=True)
            linear = torch.sum(torch.sum(feature[0].grad, 3, keepdim=True), 2, keepdim=True)        # batch, 512, 1, 1
            channel = linear * feature[0]                                                           # batch, 512, 7, 7
            swa = torch.sum(channel, 1, keepdim=True)                                             # batch, 1, 7, 7
            attributions.append(F.relu(swa))
        return attributions

    def _make_layers(self, cfg, input_dims):
        layers = []
        in_channels, in_width, in_height = input_dims
        for x in cfg['features']:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                in_width = int(in_width/2)
                in_height = int(in_height/2)
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        features = nn.Sequential(*layers)

        layers = []

        linear_input_dims = int(in_channels * in_width * in_height)
        for x in cfg['classifier']:
            if x == 'R':
                layers += [nn.ReLU()]
            elif x == 'D':
                layers += [nn.Dropout()]
            else:
                layers += [nn.Linear(linear_input_dims, x)]
                linear_input_dims = x

        classifier = nn.Sequential(*layers)
        return features, classifier

def test():
    net = VGG(vgg_name='VGG13_VOC', input_dims=(3,32,32))
    print(net)
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())
