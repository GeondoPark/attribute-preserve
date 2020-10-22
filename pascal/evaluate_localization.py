import sys
import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from dataset import get_loader
from model import *
from model.vgg_cfg import cfg as vgg_config
from utils import AverageMeter, load_model, Hooks
from explains.rap.modules.vgg import vgg16, vgg16_bn, vgg16_bn_voc
from torch.autograd import Variable
import torch.nn as nn
#from explains.ebp.exc_bp import ExcitationBackpropExplainer
import os
from sklearn.metrics import roc_curve, auc

def main():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--data_path', default="./data/VOCdevkit/VOC2012", type=str,
                        help="Data path, here for segmentation ground truth")
    parser.add_argument('--path', default='', type=str,
                        help="Saved Model Path, ref load_model in utils.py")
    parser.add_argument('--arch', default='VGG16_VOC', type=str,
                        help="Predefined Architecture for knowledge distillation" +
                        "For Structure Pruning and Unstructure Pruning, Keep VGG16_VOC")
    parser.add_argument('--iter', default=1, type=int,
                        help="how many iteration for Structure Pruning"+
                            "We use only one-shot pruning in the paper")
    parser.add_argument('--prune_rate', default=0.7, type=float,
                        help="Pruning ratio for each iteration")
    parser.add_argument('--compression', default='', type=str,
                        help='Choose in [kd | unstructured | structured]')
    parser.add_argument('--method', default='', type=str, 
                        help='Choose in [gcam | ebp | lrp | rap]')
    parser.add_argument('--metric', default='auc', type=str, 
                        help='Choose in [miou | iou | auc]')
    parser.add_argument('--save', default='result.txt', type=str,
                        help="Results Recording file")

    args = parser.parse_args()
    compression = args.compression
    method = args.method
    metric = args.metric

    print("Attribution Map Method : {}".format(method))
    print("Compression Method : {}".format(compression))
    print("Evaluation Metric : {}".format(metric))

    if (method == 'gcam') or (method =='ebp'):
        if compression == 'structured':
            layerwise_prune_rate = [args.prune_rate, 0.0,
                            0.0, 0.0,
                            0.0, 0.0, 0.0,
                            args.prune_rate, args.prune_rate, args.prune_rate,
                            args.prune_rate, args.prune_rate, args.prune_rate]
            model_config = prune_vgg_cfg(vgg_config['VGG16_VOC'], layerwise_prune_rate, args.iter)
            model = VGG(None, input_dims=(3,227,227), specific_cfg=model_config)

        elif compression == 'kd':
            model = VGG(args.arch, input_dims=(3, 227, 227))

        elif compression == 'unstructured':
            model = VGG(args.arch, input_dims=(3,227,227))
        else:
            raise NotImplementedError

    elif (method == 'lrp') or (method == 'rap'):
        if compression == 'structured':
            layerwise_prune_rate = [args.prune_rate, 0.0,
                        0.0, 0.0,
                        0.0, 0.0, 0.0,
                        args.prune_rate, args.prune_rate, args.prune_rate,
                        args.prune_rate, args.prune_rate, args.prune_rate]
            model_config = prune_vgg_cfg(vgg_config['VGG16_VOC'], layerwise_prune_rate, args.iter)
            model = vgg16_bn_voc(config=model_config)
        elif compression == 'kd':
            if args.arch == 'VGG16_VOC':
                r = 1
            elif args.arch == 'VGG16_VOC_x2':
                r = 2
            elif args.arch == 'VGG16_VOC_x4':
                r = 4
            else:
                r = 8
            model = vgg16_bn_voc(config=None, reduce=r, pretrained=False)    
        elif compression == 'unstructured':
            r = 1
            model = vgg16_bn_voc(config=None, reduce=r, pretrained=False)
        else:
            raise NotImplementedError

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    
    if args.gpu:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)       

    checkpoint = args.path

    _, best_mAP = load_model(checkpoint, model)

    ## Batch size should be 1
    vocsegment = get_loader(data_name='VOC-seg', 
                data_path= args.data_path,
                split= 'val', 
                batch_size=1)

    if method == 'gcam':
        features = Hooks(model, ['42'])
    elif method == 'ebp':
        layer = model.module.features._modules['42']
        explainer = ExcitationBackpropExplainer(model, layer)          
    else:
        model = model.module

    model.eval()
    auc_values = []
    correct_point = 0
    incorrect_point = 0
    pred_true = 0
    pred_false = 0

    for idx, (images, targets, labels) in enumerate(vocsegment):
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        if method == 'gcam': 
            features.reset_()   
            output, _, _ = model(images)
            activation = features.activations[0]
        
        elif method == 'ebp':
            output, _, _ = model(images)

        elif (method == 'lrp') or (method=='rap'):
            output = model(images)

        ## 0 index : BackGround
        lbl = (labels[:,1:].squeeze(0)).nonzero().cpu().data.numpy()
        predictions = (output.squeeze()>0).nonzero().cpu().data.numpy()
        pointacc_labels = []

        ## Class-wise heatmap
        for i, lb in enumerate(lbl):
            ## Class-wise Segmentation Map
            target = targets.squeeze().data.numpy()
            seg_mask = np.zeros((target.shape[0], target.shape[1]), dtype=np.uint8)
            whe = np.where(target == (lb+1))
            seg_mask[whe] = 1
            for m in range(len(whe[0])):
                pointacc_labels.append([whe[0][m], whe[1][m]])

            grad_out = output.data.clone()
            grad_out.fill_(0.0)
            grad_out[0, lb] = 1

            if method == 'gcam':
                model.zero_grad()
                output.backward(grad_out, retain_graph=True)
                gradient = features.gradients[i]
                linearization = torch.sum(torch.sum(gradient, 3, keepdim=True), 2, keepdim=True)
                channels = linearization * activation
                grad_cam = torch.sum(channels, 1, keepdim=True)
                grad_cam = F.relu(grad_cam)
                heatmap = grad_cam.squeeze().cpu().data.numpy()
                heatmap = cv2.resize(heatmap, (227, 227))

            elif method == 'ebp':
                ebp_heatmap = explainer.explain(images, grad_out)
                heatmap = ebp_heatmap.squeeze().cpu().data.numpy()
                heatmap = cv2.resize(heatmap, (227, 227))                    

            elif method == 'lrp':
                Res = model.relprop(R=output * grad_out, alpha=1).sum(dim=1, keepdim=True)
                heatmap = Res.squeeze().cpu().data.numpy()
                if output[0, lab] < 0:
                    heatmap = -1 * heatmap

            elif method == 'rap':
                RAP = model.RAP_relprop(R=grad_out)
                Res = (RAP).sum(dim=1, keepdim=True)
                heatmap = Res.squeeze().cpu().data.numpy()
                if output[0, lab] < 0:
                    heatmap = -1 * heatmap
                mask = np.int32(heatmap > 0)
                heatmap = heatmap * mask

            if np.isnan(heatmap).any():
                continue
            ##For Point Acc, maxpoint
            max_index = np.where(heatmap == np.max(heatmap))
            max_point = [max_index[0][0], max_index[1][0]]

            ##Get AUC
            h_max = np.nanmax(heatmap)
            h_min = np.nanmin(heatmap)
            heatmap = (heatmap - h_min) / (h_max - h_min + 10e-6)
            auc_value = get_auc(heatmap, target_mask)

            if lb not in predictions:
                pred_false += 1
                continue

            if max_point in pointacc_labels:
                correct_point += 1
            else:
                incorrect_point += 1
            pred_true += 1
            auc_list.append(auc_list)

        if idx %100 == 0:
            print("Processing Image(Current|Total) {}|{}".format(idx, len(vocsegment)))

    meanAUC = sum(auc_list) / len(auc_list)
    print("Correct Prediction Samples: {}".format(pred_true))
    print("Incorrect Prediction Samples: {}".format(pred_false))
    print("Point game correct samples in correct prediction: {}".format(correct_point))
    print("Point game [(Point Correct and Prediction Correct)/(Prediction Correct)] : {}".format(float(correct_point/pred_true)))
    print("Metric AUC : {}".format(meanAUC))

    save_file = os.path.join('./results', args.save)
    with open(save_file, 'a') as f:
        f.write(checkpoint + '\t' + 'Metric AUC : {} \n'.format(float(meanAUC)))
        f.write(checkpoint + '\t' + 'Point Game Acc : {}'.format(float(correct_point/pred_true)))

def get_auc(pred, gt):
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)
    flasePositive, truePositive, _ = roc_curve(gt, pred)
    roc_auc = auc(flasePositive, truePositive)
    return roc_auc

def get_miou(pred, gt, n_classes=2):
    pred_tmp = pred
    gt_tmp = gt
    intersect = [0] * n_classes
    union = [0] * n_classes
    for j in range(n_classes):
        # match = (pred_tmp == j) + (gt_tmp == j)
        match = (pred_tmp == j).astype(np.float) + (gt_tmp == j).astype(np.float)
        it = np.sum(match == 2)
        un = np.sum(match > 0)
        intersect[j] += it
        union[j] += un
    iou = []
    for k in range(n_classes):
        if union[k] == 0:
            continue
        iou.append(intersect[k] / union[k])
    miou = (sum(iou) / len(iou))
    return miou

def get_iou(pred, gt):
    pred_tmp = pred
    gt_tmp = gt
    match = (pred_tmp == 1).astype(np.float) + (gt_tmp == 1).astype(np.float)
    intersect = np.sum(match == 2)
    union = np.sum(match > 0)
    iou = intersect/union
    return iou

def thresh(a, coef):
    return coef * (np.max(a))

def denormalize(image):
    image[0,:,:] = image[0,:,:] * 0.2239 + 0.4589
    image[1,:,:] = image[1,:,:] * 0.2186 + 0.4355
    image[2,:,:] = image[2,:,:] * 0.2206 + 0.4032
    return image

def visualize(map):
    r = map.copy()
    g = map.copy()
    b = map.copy()
    r[np.where(map == 1)] = 255
    g[np.where(map == 1)] = 255
    b[np.where(map == 1)] = 255
    rgb = np.zeros((map.shape[0], map.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return rgb

def compute_pred(output):
    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
    print('Pred cls : '+str(pred))
    T = pred.squeeze().cpu().numpy()
    T = np.expand_dims(T, 0)
    T = (T[:, np.newaxis] == np.arange(20)) * 1.0
    T = torch.from_numpy(T).type(torch.FloatTensor)
    Tt = Variable(T).cuda()
    return Tt

def prune_vgg_cfg(cfg, prune_rate, iter):
    for idx in range(iter):
        # print('iter: {0}'.format(idx))
        conv_id = 0
        new_features = []
        for layer in cfg['features']:
            if isinstance(layer, int):
                new_features.append(int(layer * (1 - prune_rate[conv_id])))
                conv_id += 1
            elif layer == 'M':
                new_features.append('M')
            else:
                assert False
        cfg = {'features': new_features, 'classifier': cfg['classifier']}
    return {'features': new_features, 'classifier': cfg['classifier']}

if __name__ == '__main__':
    main()