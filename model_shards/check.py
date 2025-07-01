# 重新分片检查脚本
import torch
from collections import defaultdict

def analyze_shard(shard_path):
    state_dict = torch.load(shard_path)
    param_stats = defaultdict(float)
    
    for name, param in state_dict.items():
        param_stats[name.split('.')[0]] += param.numel() * 2 / (1024**3)  # GB
        
    print(f"\n{shard_path} 分析:")
    for module, size in param_stats.items():
        print(f"{module}: {size:.2f}GB")

def detailed_analysis(shard_path):
    state_dict = torch.load(shard_path)
    param_dict = defaultdict(int)
    
    for name, param in state_dict.items():
        module = '.'.join(name.split('.')[:3])  # 取前三级路径
        param_dict[module] += param.numel()
    
    print(f"\n{shard_path} 详细构成:")
    for k, v in sorted(param_dict.items()):
        print(f"{k}: {v/1e6:.2f}M params")

detailed_analysis("shard_0.pth")
detailed_analysis("shard_1.pth")
detailed_analysis("shard_2.pth")  # 重点检查这里是否包含非lm_head参数