import torch
import torch.nn as nn
import time
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
from typing import Any, Callable, Dict, List, Optional, OrderedDict, Tuple, Union
from transformers.configuration_utils import PretrainedConfig
from core.load_config import Config


class DistributedModel(nn.Module):
    def __init__(self, config: PretrainedConfig, runtime_config: Config, transformer_class: Callable):
        super().__init__()
        self.node_rrefs = []
        for worker in runtime_config.workers:
            worker_name = worker.name
            offset = (int(worker.start), int(worker.end))
            ckpt_path = worker.ckpt_path
            rref = rpc.remote(worker_name, transformer_class, args=(config, offset, ckpt_path))
            self.node_rrefs.append(rref)

        print("All workers initiated")

    def forward(self, **input_data):
        with torch.no_grad():
            
            start_time = time.time()
            out_rrefs = [RRef(input_data)]
            for rref in self.node_rrefs:
                out_new_rref = rref.remote().forward(out_rrefs[-1])
                out_rrefs.append(out_new_rref)

            # print(f"token gen time: {(time.time() - start_time) * 1000} ms ")

            result = out_rrefs[-1].to_here()
            for key, value in result.items():
                if isinstance(value, torch.Tensor):
                    result[key] = value.cuda()
                    
            return result

    
