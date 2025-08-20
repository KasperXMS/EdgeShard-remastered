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
        self.config = config
        self.node_rrefs = []
        self.current_cache_position = 0
        
        for worker in runtime_config.workers:
            worker_name = worker.name
            offset = (int(worker.start), int(worker.end))
            ckpt_path = worker.ckpt_path
            rref = rpc.remote(worker_name, transformer_class, args=(config, offset, ckpt_path))
            self.node_rrefs.append(rref)
        
        print("All workers initiated")

    def forward(self, **input_data):
        with torch.inference_mode():
            # Initialize with input data
            current_data = input_data                
            current_rref = RRef(current_data)
            # Pipeline through stages
            for i, rref in enumerate(self.node_rrefs):
                # Process on current stage
                if i < len(self.node_rrefs):
                    current_rref = rref.remote().forward(current_rref)
                # else:
                #     # Last stage uses async to overlap compute/transfer
                #     future = rref.rpc_async().forward(current_rref)
            
            # return torch.futures.wait_all([future])[0]
            return current_rref.to_here()

    
