import os
import json
import torch
from transformers import AutoModel, AutoConfig
from typing import Dict, List

class ModelSharder:
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        split_layers: List[int],  # 示例 [16, 32] 表示分片点
        embed_patterns: List[str] = ["wte", "wpe", "embeddings"],
        final_patterns: List[str] = ["ln_f", "final_layer", "lm_head"]
    ):
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
        self.config = AutoConfig.from_pretrained(model_name)
        self.output_dir = output_dir
        self.split_layers = split_layers
        self.embed_patterns = embed_patterns
        self.final_patterns = final_patterns
        
    def _is_embed_layer(self, name: str) -> bool:
        return any(pattern in name for pattern in self.embed_patterns)
    
    def _is_final_layer(self, name: str) -> bool:
        return any(pattern in name for pattern in self.final_patterns)
    
    def shard_model(self):
        """执行分片并生成配置文件"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化分片
        shards = [{} for _ in range(len(self.split_layers) + 1)]
        
        for name, param in self.model.state_dict().items():
            # 分配到对应分片
            if self._is_embed_layer(name):
                shards[0][name] = param
            elif self._is_final_layer(name):
                shards[-1][name] = param
            elif "h." in name:
                layer_idx = int(name.split(".")[2])  # 适配不同模型结构
                for i, split in enumerate(self.split_layers):
                    if layer_idx < split:
                        shards[i+1][name] = param
                        break
                else:
                    shards[-1][name] = param
            else:
                shards[0][name] = param  # 其他参数默认在前片
        
        # 保存分片权重
        for i, shard in enumerate(shards):
            torch.save(shard, os.path.join(self.output_dir, f"shard_{i}.pth"))
            
        # 生成配置文件
        self._generate_configs(shards)
    
    def _generate_configs(self, shards: List[Dict]):
        """为每个分片生成部署配置文件"""
        for shard_idx in range(len(shards)):
            config = {
                "shard_id": shard_idx,
                "original_model": self.config.model_type,
                "layer_ranges": self._get_layer_ranges(shard_idx, shards),
                "rpc_config": {
                    "prev_shard": f"localhost:{9000 + shard_idx - 1}" if shard_idx > 0 else None,
                    "next_shard": f"localhost:{9000 + shard_idx + 1}" if shard_idx < len(shards)-1 else None,
                    "port": 9000 + shard_idx
                }
            }
            with open(os.path.join(self.output_dir, f"shard_{shard_idx}_config.json"), "w") as f:
                json.dump(config, f, indent=2)
    
    def _get_layer_ranges(self, shard_idx: int, shards: List[Dict]) -> Dict:
        """提取分片的层范围元数据"""
        layer_nums = []
        for name in shards[shard_idx].keys():
            if "h." in name:
                parts = name.split(".")
                layer_nums.append(int(parts[2]))
        return {
            "min_layer": min(layer_nums) if layer_nums else None,
            "max_layer": max(layer_nums) if layer_nums else None,
            "has_embeddings": any("embed" in name for name in shards[shard_idx].keys()),
            "has_final": any(self._is_final_layer(name) for name in shards[shard_idx].keys())
        }

# 使用示例
sharder = ModelSharder(
    model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    output_dir="./model_shards",
    split_layers=[16, 32]  # 分片点：Shard0(emb+0-15), Shard1(16-31), Shard2(final)
)
sharder.shard_model()