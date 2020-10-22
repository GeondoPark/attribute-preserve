import numpy as np
import torch
from torch.autograd import Variable
import imageio
import matplotlib.cm
from matplotlib.cm import ScalarMappable
import torch.nn as nn
from explain.rap.modules.layers import forward_hook, RelProp


# Unstable and Dangerous
def wrap_model(model):
    for m0 in model.modules():
        if issubclass(m0, RelProp):
            m0.register_forward_hook(forward_hook)


# def convert_model(model):
#     for m0 in model.modules():


def get_blueprint(model):
    blueprint = []
    for m0 in model.modules():
        if isinstance(m0, nn.Conv2d):
            args_dict = {'in_channels': m0.in_channels,
                         'out_channels': m0.out_channels,
                         'kernel_size': m0.kernel_size,
                         'stride': m0.stride,
                         'padding': m0.padding,
                         'dilation': m0.dilation,
                         'groups': m0.groups,
                         'bias': hasattr(m0, 'bias'),
                         'padding_mode': m0.padding_mode,
                         }
        elif isinstance(m0, nn.Linear):
            args_dict = {'in_features': m0.in_features,
                         'out_features': m0.out_features,
                         'bias': hasattr(m0, 'bias'),
                         }
        elif isinstance(m0, nn.BatchNorm2d):
            args_dict = {'num_features': m0.num_features,
                         'eps': m0.eps,
                         'momentum': m0.momentum,
                         'affine': m0.affine,
                         'track_running_stats': m0.track_running_stats,
                         }
        elif isinstance(m0, nn.ReLU):
            args_dict = {'inplace': m0.inplace}
        elif isinstance(m0, nn.MaxPool2d):
            args_dict = {'kernel_size': m0.kernel_size,
                         'stride': m0.stride,
                         'padding': m0.padding,
                         'dilation': m0.dilation,
                         'return_indices': m0.return_indices,
                         'ceil_mode': m0.ceil_mode,
                         }
        elif isinstance(m0, nn.AdaptiveAvgPool2d):
            args_dict = {'output_size': m0.output_size,
                         }
        elif isinstance(m0, nn.AvgPool2d):
            args_dict = {'kernel_size': m0.kernel_size,
                         'stride': m0.stride,
                         'padding': m0.padding,
                         'ceil_mode': m0.ceil_mode,
                         'count_include_pad': m0.count_include_pad,
                         }
        elif isinstance(m0, nn.Dropout):
            args_dict = {'p': m0.p,
                         'inplace': m0.inplace,
                         }
        elif isinstance(m0, nn.Sequential):
            args_dict = {'p': m0.p,
                         'inplace': m0.inplace,
                         }
        else:
            args_dict = {}

        blueprint.append((m0.__class__.__name__, args_dict))
    return blueprint





def enlarge_image(img, scaling = 3):
    if scaling < 1 or not isinstance(scaling,int):
        print ('scaling factor needs to be an int >= 1')

    if len(img.shape) == 2:
        H,W = img.shape
        out = np.zeros((scaling*H, scaling*W))
        for h in range(H):
            fh = scaling*h
            for w in range(W):
                fw = scaling*w
                out[fh:fh+scaling, fw:fw+scaling] = img[h,w]
    elif len(img.shape) == 3:
        H,W,D = img.shape
        out = np.zeros((scaling*H, scaling*W,D))
        for h in range(H):
            fh = scaling*h
            for w in range(W):
                fw = scaling*w
                out[fh:fh+scaling, fw:fw+scaling,:] = img[h,w,:]
    return out


def hm_to_rgb(R, scaling = 3, cmap = 'bwr', normalize = True):
    cmap = eval('matplotlib.cm.{}'.format(cmap))
    if normalize:
        R = R / np.max(np.abs(R)) # normalize to [-1,1] wrt to max relevance magnitude
        R = (R + 1.)/2. # shift/normalize to [0,1] for color mapping
    R = R
    R = enlarge_image(R, scaling)
    rgb = cmap(R.flatten())[...,0:3].reshape([R.shape[0],R.shape[1],3])
    return rgb


def visualize(relevances, img_name):
    # visualize the relevance
    n = len(relevances)
    heatmap = np.sum(relevances.reshape([n, 224, 224, 1]), axis=3)
    heatmaps = []
    for h, heat in enumerate(heatmap):
        maps = hm_to_rgb(heat, scaling=3, cmap = 'seismic')
        heatmaps.append(maps)
        imageio.imsave('./results/' + img_name, maps,vmax=1,vmin=-1)


def one_hot(x):
    max_idx = torch.argmax(x, 1)
    one_hot_encoding = torch.cuda.FloatTensor(x.shape).fill_(0)
    one_hot_encoding[list(range(x.shape[0])), max_idx] = 1.0
    return one_hot_encoding

def compute_pred(output):
    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
    print('Pred cls : '+str(pred))
    T = pred.squeeze().cpu().numpy()
    T = np.expand_dims(T, 0)
    T = (T[:, np.newaxis] == np.arange(1000)) * 1.0
    T = torch.from_numpy(T).type(torch.FloatTensor)
    Tt = Variable(T).cuda()
    return Tt