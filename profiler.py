from typing import Iterator
import torch
import torch.nn as nn
import time
# from torchvision.models.alexnet import AlexNet
from transformers import AutoModelForCausalLM

def join_layers(net):
    layers = [
        *net.features,
        net.avgpool,
        lambda x: torch.flatten(x, 1),
        *net.classifier,
    ]
    return layers

class Profiler:
    def __init__(self, layers: list):
        super(Profiler, self).__init__()
        self.layers = layers
        self.forward_time = {}
        self.backward_time = {}
        self.nn_module_map = {}
        self.param_sizes = []
        self.fw_counter = 0
        self.bw_counter = 0
        for index, layer in enumerate(self.layers):
            params = 0
            if isinstance(layer, nn.Module):
                if isinstance(layer, nn.ReLU):
                    layer.__init__(inplace=False)
                layer.register_forward_pre_hook(self.forward_time_start)
                layer.register_forward_hook(self.forward_time_end)
                # layer.register_full_backward_pre_hook(self.backward_time_start)
                # layer.register_full_backward_hook(self.backward_time_end)
                
                for param_name, param in layer.named_parameters():
                    # print(node[0], ".", param_name, " - ", param.size())
                    if 'weight' in param_name:
                        params += torch.prod(torch.LongTensor(list(param.size())))
                        if param.requires_grad:
                            required_grad = True
                    elif 'bias' in param_name:
                        params += torch.prod(torch.LongTensor(list(param.size())))
                          
                # self.add_module(str(index), layer)
                self.nn_module_map[self.bw_counter] = index
                self.bw_counter += 1

            self.param_sizes.append(int(params))
            self.forward_time[index] = 0
            self.backward_time[index] = 0
            
        self.bw_counter -= 1

    def forward(self, *x):
        for index, layer in enumerate(self.layers):
            if isinstance(layer, nn.Module):
                if type(x) == tuple:
                    x = layer(*x)
                else:
                    x = layer(x)
            else:
                torch.cuda.synchronize()
                start_time = time.time() * 1000
                if type(x) == tuple:
                    x = layer(*x)
                else:
                    x = layer(x)
                torch.cuda.synchronize()
                if index > len(self.forward_time) - 1:
                    self.forward_time[self.fw_counter] = time.time() * 1000 - start_time
                else:
                    self.forward_time[self.fw_counter] = time.time() * 1000 - start_time
                self.fw_counter += 1
        return x
    
    def parameters(self, recurse: bool = True):
        for layer in self.layers:
            if isinstance(layer, nn.Module):
                yield from layer.parameters(recurse)
    
    def forward_time_start(self, layer, input):
        torch.cuda.synchronize()
        start_time = time.time() * 1000
        self.forward_time[self.fw_counter] = start_time

    def forward_time_end(self, layer, input, output):
        torch.cuda.synchronize()
        self.forward_time[self.fw_counter] = time.time() * 1000 - self.forward_time[self.fw_counter]
        self.fw_counter += 1

    def backward_time_start(self, layer, input):
        torch.cuda.synchronize()
        start_time = time.time() * 1000
        self.backward_time[self.nn_module_map[self.bw_counter]] = start_time

    def backward_time_end(self, layer, input, output):
        torch.cuda.synchronize()
        self.backward_time[self.nn_module_map[self.bw_counter]] = time.time() * 1000 - self.backward_time[self.nn_module_map[self.bw_counter]]
        self.bw_counter -= 1
    
    def summary(self):
        print(f"{'Layer ID':>8}{'Layer Name':>20}{'Param Size':>20}{'Forward Time':>20}{'Backward Time':>20}")
        for index, layer in enumerate(self.layers):
            if isinstance(layer, nn.Module):
                # print(index, " ", layer.__class__.__name__, " ", self.forward_time[index], " ", self.param_sizes[index])
                print(f"{index:>8}{layer.__class__.__name__:>20}{self.param_sizes[index]:>20}{self.forward_time[index]:>20.6f}{self.backward_time[index]:>20.6f}")
            else:
                # print(index, " ", layer.__name__, " ", self.forward_time[index], " ", self.param_sizes[index])
                print(f"{index:>8}{layer.__name__:>20}{self.param_sizes[index]:>20}{self.forward_time[index]:>20.6f}{self.backward_time[index]:>20.6f}")

        return self.param_sizes, self.forward_time, self.backward_time
    
if __name__ == '__main__':
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model = Profiler(model.model)
        
