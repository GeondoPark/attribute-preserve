import torch
import torch.utils.data
import torch.nn as nn
import os
import numpy as np

def prune_vgg_cfg(cfg, prune_rate, iter):

    for idx in range(iter):
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


def prune_inplace(model, prune_rate=None):
    erased_value = swap_weights(model, prune_rate)
    for m in model.modules():
        sanitize_layer(m)

    return model, erased_value


def swap_weights(model, prune_rate=None, skip=[]):
    # manipulating .data is okay

    cfg_arch, cfg_masks, erased_value =get_kill_indices(model, prune_rate, skip)
    newmodel = model
    layer_id_in_cfg = 0
    first_linear = True

    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        if isinstance(m0, nn.Conv2d):
            mask_in = cfg_masks[layer_id_in_cfg][0]
            mask_out = cfg_masks[layer_id_in_cfg][1]
            tensor_dict = extract_weights(m0, mask_in, mask_out)
            # kill all grad tensor
            if m1.weight.grad is not None:
                m1.weight.grad = None
            if m1.bias.grad is not None:
                m1.bias.grad = None
            m1.weight.data = tensor_dict['weights']
            m1.bias.data = tensor_dict['bias']
            layer_id_in_cfg += 1
        elif isinstance(m0, nn.BatchNorm2d):
            # mask = cfg_masks[layer_id_in_cfg-1]
            mask = cfg_masks[layer_id_in_cfg][0]
            tensor_dict = extract_weights(m0, mask, None)
            # kill all grad tensor
            if m1.weight.grad is not None:
                m1.weight.grad = None
            if m1.bias.grad is not None:
                m1.bias.grad = None
            m1.weight.data = tensor_dict['weights']
            m1.bias.data = tensor_dict['bias']
            m1.running_mean = tensor_dict['running_mean']
            m1.running_var = tensor_dict['running_var']
        elif isinstance(m0, nn.Linear):
            # kill all grad tensor
            if m1.weight.grad is not None:
                m1.weight.grad = None
            if m1.bias.grad is not None:
                m1.bias.grad = None
            if first_linear:
                mask_in = cfg_masks[layer_id_in_cfg][0].repeat(49)
                mask_out = cfg_masks[layer_id_in_cfg][1]
                tensor_dict = extract_weights(m0, mask_in, mask_out)
                m1.weight.data = tensor_dict['weights']
                m1.bias.data = tensor_dict['bias']
                break
            else:
                continue
    return erased_value

def extract_weights(layer, mask_in, mask_out):
    if isinstance(layer, nn.Conv2d):
        idx_out = np.squeeze(np.argwhere(np.asarray(mask_out.cpu().numpy())))
        idx_in = np.squeeze(np.argwhere(np.asarray(mask_in.cpu().numpy())))
        if idx_out.size == 1:
            idx_out = np.resize(idx_out, (1,))
        if idx_in.size == 1:
            idx_in = np.resize(idx_in, (1,))
        w = layer.weight.data[idx_out.tolist(), :, :, :][:, idx_in.tolist(), :, :].clone()
        bias = layer.bias.data[idx_out.tolist()].clone()
        new_weights = {'weights': w, 'bias': bias}

    elif isinstance(layer, nn.BatchNorm2d):
        mask = mask_in
        idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
        if idx.size == 1:
            idx = np.resize(idx, (1,))
        weights = layer.weight.data[idx.tolist()].clone()
        bias = layer.bias.data[idx.tolist()].clone()
        running_mean = layer.running_mean[idx.tolist()].clone()
        running_var = layer.running_var[idx.tolist()].clone()
        new_weights = {'weights': weights, 'bias': bias, 'running_mean':running_mean, 'running_var': running_var}

    elif isinstance(layer, nn.Linear):
        idx_out = np.squeeze(np.argwhere(np.asarray(mask_out.cpu().numpy())))
        idx_in = np.squeeze(np.argwhere(np.asarray(mask_in.cpu().numpy())))
        if idx_out.size == 1:
            idx_out = np.resize(idx_out, (1,))
        if idx_in.size == 1:
            idx_in = np.resize(idx_in, (1,))
        w = layer.weight.data[idx_out.tolist(), :][:, idx_in.tolist()].clone()
        bias = layer.bias.data[idx_out.tolist()].clone()
        new_weights = {'weights': w, 'bias': bias}
    else:
        raise TypeError

    return new_weights

def sanitize_layer(layer):
    if isinstance(layer, nn.Conv2d):

        layer.in_channels = int(layer.weight.data.shape[1])
        layer.out_channels = int(layer.weight.data.shape[0])
        layer.kernel_size = tuple(layer.weight.data.shape[2:])

    elif isinstance(layer, nn.BatchNorm2d):
        layer.num_features = int(layer.weight.data.shape[0])

    elif isinstance(layer, nn.Linear):
        layer.in_features = int(layer.weight.data.shape[1])
        layer.out_features = int(layer.weight.data.shape[0])
    else:
        return

def get_kill_indices(model, prune_rate, skip=[]):
    conv_id = 0
    cfg = []
    cfg_mask = []
    cfg_arch = []
    erased_value = []
    for idx, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d):
            if m.kernel_size == (1,1):
                continue
            out_channels = m.weight.data.shape[0]
            try:
                fan_in = cfg[-1][1]
                fan_in_mask = cfg_mask[-1][1]
            except IndexError:
                fan_in = m.weight.data.shape[1]
                fan_in_mask = torch.ones(fan_in)
            if conv_id in skip:
                cfg_mask.append((fan_in_mask, torch.ones(out_channels)))
                cfg.append((fan_in, out_channels))
                conv_id += 1
                continue

            weight_copy = m.weight.data.abs().clone().cpu().numpy()
            L1_norm = np.sum(weight_copy, axis=(1,2,3))
            num_keep = int(out_channels * (1 - prune_rate[conv_id]))
            arg_max = np.argsort(L1_norm)
            arg_max_rev = arg_max[::-1][:num_keep]
            mask = torch.zeros(out_channels)
            mask[arg_max_rev.tolist()] = 1
            cfg_mask.append((fan_in_mask, mask))
            cfg.append((fan_in, num_keep))
            conv_id += 1
            if idx == 40:
                for ii, number in enumerate(arg_max[::-1][num_keep:]):
                    erased_value.append(number)

        elif isinstance(m, nn.Linear):
            fan_in = cfg[-1][1]
            fan_in_mask = cfg_mask[-1][1]
            cfg_mask.append((fan_in_mask, torch.ones(m.weight.shape[0])))
            cfg.append((fan_in, m.weight.shape[0]))
            # conv_id += 1
            break

    return cfg, cfg_mask, erased_value

def updateBN(model, sparsity):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.weight.grad.data.add_(sparsity * torch.sign(m.weight.data))

def BN_grad_zero(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            mask = (m.weight.data != 0)
            mask = mask.float().cuda()
            m.weight.grad.data.mul_(mask)
            m.bias.grad.data.mul_(mask)

def save_checkpoint(state, is_best, checkpoint, filename='pruned.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

