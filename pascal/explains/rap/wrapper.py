import numpy as np
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from glob import glob
import imageio
import torch.backends.cudnn as cudnn
from explain.rap.modules.vgg import vgg16, vgg16_bn
from explain.rap.modules.resnet import resnet50, resnet101
import matplotlib.cm
from matplotlib.cm import ScalarMappable
from explain.rap.utils import enlarge_image, hm_to_rgb, visualize, compute_pred, one_hot
import argparse


cudnn.benchmark = True
# Args

parser = argparse.ArgumentParser(description='Interpreting the decision of classifier')
parser.add_argument('--method', type=str, default='RAP', metavar='N',
                    help='Method : LRP or RAP')
parser.add_argument('--arc', type=str, default='vgg', metavar='N',
                    help='Model architecture: vgg / resnet')
args = parser.parse_args()
num_workers = 0
batch_size = 1

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
data_name = 'imagenet/'

# define data loader

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder('./data/'+ data_name,
                          transforms.Compose([
                              transforms.Scale([224, 224]),
                              transforms.ToTensor(),
                              normalize,
                          ])),
    batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=True)

# Make RAP-Model
if args.arc == 'vgg':
    model = vgg16(pretrained=True).cuda()
elif args.arc == 'resnet':
    # model = resnet101(pretrained=True).cuda()
    model = resnet50(pretrained=True).cuda()

# method = LRP or RAP
method = args.method
model.eval()

for idx, (input, label) in enumerate(val_loader):
    input = input.cuda()
    input.requires_grad = True
    img_name = val_loader.dataset.imgs[idx][0].split('/')[-1]
    output = model(input)
    T = one_hot(output)
    # T = compute_pred(output)
    if method == 'LRP':
        Res = model.relprop(R = output * T, alpha= 1).sum(dim=1, keepdim=True)
    else:
        RAP = model.RAP_relprop(R=T)
        Res = (RAP).sum(dim=1, keepdim=True)
    # Check relevance value preserved
    print('Pred logit : ' + str((output * T).sum().data.cpu().numpy()))
    print('Relevance Sum : ' + str(Res.sum().data.cpu().numpy()))
    # save results
    heatmap = Res.permute(0, 2, 3, 1).data.cpu().numpy()
    img_name = method + '/' + data_name + img_name
    visualize(heatmap.reshape([batch_size, 224, 224, 1]), img_name)
print('Done')


class RapWrapper(object):
    def __init__(self, model,
                 device=torch.device('cuda'),
                 resume=False,
                 opt=None):

        super(RapWrapper, self).__init__()
        self.opt = opt
        self.device = device
        if args.arc == 'vgg':
            model = vgg16(pretrained=True).cuda()

    def evaluate(self, l_loader, algo='RAP'):
        if method == 'LRP':
            Res = model.relprop(R=output * T, alpha=1).sum(dim=1, keepdim=True)
        else:
            RAP = model.RAP_relprop(R=T)
            Res = (RAP).sum(dim=1, keepdim=True)
        # Check relevance value preserved
        print('Pred logit : ' + str((output * T).sum().data.cpu().numpy()))
        print('Relevance Sum : ' + str(Res.sum().data.cpu().numpy()))
        # save results
        heatmap = Res.permute(0, 2, 3, 1).data.cpu().numpy()
        img_name = method + '/' + data_name + img_name
        visualize(heatmap.reshape([batch_size, 224, 224, 1]), img_name)