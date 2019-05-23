import torch
import torch.nn as nn
import numpy as np

from collections import OrderedDict
from .helper import compute_madd, compute_flops

__all__ = ["scope"]


class ModelSummary(object):
    def __init__(self, model, input_size, batch_size=-1, device='cpu'):
        super(ModelSummary, self).__init__()
        assert device.lower() in ['cuda', 'cpu']
        self.model = model
        self.batch_size = batch_size

        if device == "cuda" and torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor

        # multiple inputs to the network
        if isinstance(input_size, tuple):
            input_size = list(input_size)
        self.input_size = input_size

        # batch_size of 2 for batchnorm
        x = torch.rand([1] + input_size).type(dtype)

        # create properties
        self.summary = OrderedDict()
        self.hooks = list()

        # register hook
        model.apply(self.register_hook)

        # make a forward pass
        model(x)

        # remove hooks
        for h in self.hooks:
            h.remove()

    def register_hook(self, module):
        if len(list(module.children())) == 0:
            self.hooks.append(module.register_forward_hook(self.hook))

    def hook(self, module, input, output):
        class_name = str(module.__class__).split(".")[-1].split("'")[0]
        module_idx = len(self.summary)

        m_key = "%s-%i" % (class_name, module_idx + 1)
        self.summary[m_key] = OrderedDict()
        self.summary[m_key]["input_shape"] = list(input[0].size())
        if isinstance(output, (list, tuple)):
            self.summary[m_key]["output_shape"] = [[-1] + list(o.size())[1:] for o in output]
        else:
            self.summary[m_key]["output_shape"] = list(output.size())

        # -------------------------
        # compute module parameters
        # -------------------------
        params = 0
        if hasattr(module, "weight") and hasattr(module.weight, "size"):
            params += torch.prod(torch.LongTensor(list(module.weight.size())))
            self.summary[m_key]["trainable"] = module.weight.requires_grad
        if hasattr(module, "bias") and hasattr(module.bias, "size"):
            params += torch.prod(torch.LongTensor(list(module.bias.size())))
        self.summary[m_key]["nb_params"] = params

        # -------------------------
        # compute module flops
        # -------------------------
        flops = compute_flops(module, input[0], output)
        self.summary[m_key]["flops"] = flops

        # -------------------------
        # compute module flops
        # -------------------------
        madds = compute_madd(module, input[0], output)
        self.summary[m_key]["madds"] = madds

    def show(self):
        print("------------------------------------------------------------------------------------------------------")
        line = "{:>20}  {:>25} {:>15} {:>15} {:>15}".format("Layer (type)", "Output Shape", "Params", "FLOPs", "Madds")
        print(line)
        print("======================================================================================================")
        total_params, total_output, trainable_params, total_flops, total_madds = 0, 0, 0, 0, 0
        for layer in self.summary:
            line = "{:>20}  {:>25} {:>15} {:>15} {:>15}".format(
                layer,
                str(self.summary[layer]["output_shape"]),
                "{0:,}".format(self.summary[layer]["nb_params"]),
                "{0:,}".format(self.summary[layer]["flops"]),
                "{0:,}".format(self.summary[layer]["madds"]),
            )
            total_params += self.summary[layer]["nb_params"]
            total_output += np.prod(self.summary[layer]["output_shape"])
            total_flops += self.summary[layer]["flops"]
            total_madds += self.summary[layer]["madds"]
            if "trainable" in self.summary[layer]:
                if self.summary[layer]["trainable"] == True:
                    trainable_params += self.summary[layer]["nb_params"]
            print(line)

        total_input_size = abs(np.prod(self.input_size) * self.batch_size / (1024 ** 2.))
        total_output_size = abs(2. * total_output / (1024 ** 2.))  # x2 for gradients
        total_params_size = abs(total_params.numpy() / (1024 ** 2.))
        total_flops_size = abs(total_flops / (1e9))
        total_madds_size = abs(total_madds / (1e9))
        total_size = total_params_size + total_output_size + total_input_size

        print("======================================================================================================")
        print("Total params: {0:,}".format(total_params))
        print("Trainable params: {0:,}".format(trainable_params))
        print("Non-trainable params: {0:,}".format(total_params - trainable_params))
        print("Total FLOPs: {0:,}".format(total_flops))
        print("Total Madds: {0:,}".format(total_madds))
        print("----------------------------------------------------------------")
        print("Input size (MB): %0.2f" % total_input_size)
        print("Forward/backward pass size (MB): %0.2f" % total_output_size)
        print("Params size (MB): %0.2f" % total_params_size)
        print("Estimated Total Size (MB): %0.2f" % total_size)
        print("FLOPs size (GB): %0.2f" % total_flops_size)
        print("Madds size (GB): %0.2f" % total_madds_size)
        print("----------------------------------------------------------------")


def scope(model, input_size, batch_size=-1, device='cpu'):
    summary = ModelSummary(model, input_size, batch_size, device)
    summary.show()
