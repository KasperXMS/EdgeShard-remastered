from transformers.models.llama.modeling_llama import *
import torch
import torch.nn as nn
import time
from pathlib import Path
from torch.distributed.rpc import RRef
from hf.load_config import Config
from hf.distributed import DistributedModel

class CustomLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig, runtime_config: Config):
        # Call the grandparent's (LlamaPreTrainedModel) __init__ directly
        LlamaPreTrainedModel.__init__(self, config)
        GenerationMixin.__init__(self)

        self.runtime_config = runtime_config
        
        # Initialize your custom model instead of the original LlamaModel
        self.model = DistributedModel(config, runtime_config)  # Replace with your custom model
        
        # The following lines are copied from the original LlamaForCausalLM __init__
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights and apply final processing
        self._initialize_weights()

    def _initialize_weights(self):
        self.lm_head.load_state_dict(self.runtime_config.master.lm_head_weight_path)
        

