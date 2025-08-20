from dataclasses import dataclass
from typing import Any, Dict, OrderedDict
from transformers.models.qwen2.modeling_qwen2 import *
import torch
import torch.nn as nn
import time, gc
from pathlib import Path
from torch.distributed.rpc import RRef
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from dataclasses import dataclass
from core.load_config import Config
from core.distributed import DistributedModel
from utils.monitor import MemoryMonitor
from utils.mem_check import aggregate_gpu_tensors, find_shape_grouped_tensor_referrers_in_range