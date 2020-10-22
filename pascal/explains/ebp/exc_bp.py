import types
import torch
import torch.nn.functional as F 
import torchvision
import torchvision.models as models
from .functions import EBConv2d, EBLinear, EBAvgPool2d
import pdb
import argparse

# From Github ##https://github.com/yulongwang12/visual-attribution/blob/master/explainer/ebp/functions.py
torch.manual_seed(0)
# Input : model, output_layer_keys = ['features','23']

class ExcitationBackpropExplainer(object):
    def __init__(self, model, layer = None):
        self.output_layer = layer
        self.model = model
        self._override_backward()
        self._register_hooks()

    def _override_backward(self):
        def new_linear(self, x):
            return EBLinear.apply(x, self.weight, self.bias)
        def new_conv2d(self, x):
            return EBConv2d.apply(x, self.weight, self.bias, self.stride,
                                  self.padding, self.dilation, self.groups)
        def new_avgpool2d(self, x):
            return EBAvgPool2d.apply(x, self.kernel_size, self.stride,
                                     self.padding, self.ceil_mode, self.count_include_pad)
        def replace(m):
            name = m.__class__.__name__
            if name == 'Linear':
                m.forward = types.MethodType(new_linear, m)
            elif name == 'Conv2d':
                m.forward = types.MethodType(new_conv2d, m)
            elif name == 'AvgPool2d':
                m.forward = types.MethodType(new_avgpool2d, m)

        self.model.apply(replace)

    def _register_hooks(self):
        self.intermediate_vars = []
        def forward_hook(m, i, o):
            self.intermediate_vars.append(o)
        self.output_layer.register_forward_hook(forward_hook)

    def explain(self, inp, grad_out):
        self.intermediate_vars = []
        output, _, _ = self.model(inp)                  #1 x 1000
        output_var = self.intermediate_vars[0]          #1 x 512 x 14 x 14
        self.model.zero_grad()
        attmap_var = torch.autograd.grad(output, output_var, grad_out, retain_graph=True)
        attmap = attmap_var[0].data.clone()
        attmap = torch.clamp(attmap.sum(1).unsqueeze(1), min=0.0)
        return attmap
