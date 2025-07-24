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
        with torch.no_grad():
            # Initialize with input data
            current_data = input_data
            
            # Add cache position tracking
            if 'cache_position' not in current_data:
                seq_length = current_data['input_ids'].shape[1] if 'input_ids' in current_data else current_data['inputs_embeds'].shape[1]
                current_data['cache_position'] = torch.arange(
                    self.current_cache_position,
                    self.current_cache_position + seq_length,
                    device='cuda'
                )
                self.current_cache_position += seq_length
            
            
            current_rref = RRef(current_data)
            # Pipeline through stages
            for i, rref in enumerate(self.node_rrefs):
                # Process on current stage
                if i < len(self.node_rrefs) - 1:
                    current_rref = rref.remote().forward(current_rref)
                else:
                    # Last stage uses async to overlap compute/transfer
                    future = rref.rpc_async().forward(current_rref)
                
                # Memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            return torch.futures.wait_all([future])[0]

    def reset_cache(self):
        """Reset KV cache and position tracking across all stages"""
        self.current_cache_position = 0
        futures = []
        for rref in self.node_rrefs:
            futures.append(rref.rpc_async().reset_kv_cache())
        torch.futures.wait_all(futures)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    
