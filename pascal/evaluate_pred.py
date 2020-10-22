import torch
import argparse
from dataset import get_loader
from model import *
from model.vgg_cfg import cfg as vgg_config
from utils import load_model, compute_mAP
from sklearn.metrics import f1_score

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

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--data_path', default="./data/VOCdevkit/VOC2012", type=str,
                        help="Data path, here for multi-label classification")
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
    parser.add_argument('--save', default='result.txt', type=str,
                        help="Results Recording file")

    args = parser.parse_args()
    compression = args.compression

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

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    if args.gpu:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    checkpoint = args.path
    _, best_mAP = load_model(checkpoint, model)
    val_loader = get_loader(data_name='VOC2012', 
                        data_path=args.data_path,
                        split='val', 
                        batch_size= 32)
    model.eval()
    outputs = []
    targets = []
    for idx, (images, labels) in enumerate(val_loader):
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        # compute output
        output, _, _ = model(images)

        outputs.append(output.cpu().data)
        targets.append(labels.cpu().data)

        if idx %100 == 0:
            print("Processing Image {} | {}".format(idx, len(val_loader)))

    mAP = 100*compute_mAP(torch.cat(targets, dim=0).data, torch.cat(outputs, dim=0).data)
    f1 = f1_score(torch.cat(targets, dim=0).data.cpu().numpy(), torch.cat(outputs, dim=0).data.cpu().numpy() >= 0.5, average="samples")

    print("Predictive Performance (mAP) : {}".format(mAP))
    print("Predictive Performance (F1) : {}".format(f1))
