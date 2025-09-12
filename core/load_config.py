import yaml
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class Master:
    ip: str = ""
    port: str = ""
    interface: str = ""
    lm_head_weight_path: str = ""

@dataclass
class Worker:
    name: str = ""
    ip: str = ""
    interface: str = ""
    start: int = 0
    end: int = 0
    ckpt_path: str = ""

class Config:
    def __init__(self, config_dict: Dict[str, Any]):
        # 模型名称（如果未提供则使用默认 Llama-3.1-8B-Instruct）
        self.model_name: str = config_dict.get('model_name', "meta-llama/Llama-3.1-8B-Instruct")
        # Master 节点配置
        self.master: Master = Master(**config_dict.get('master', {}))
        # Worker 列表
        self.workers: List[Worker] = [Worker(**w) for w in config_dict.get('workers', [])]

    def __repr__(self) -> str:
        return f"Config(model_name={self.model_name}, master={self.master}, workers={len(self.workers)})"

def load_config(config_file: str) -> Optional[Config]:
    """从 YAML 文件加载配置.
    返回 Config 实例或 None (解析失败)。"""
    try:
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f) or {}
            return Config(config_dict)
    except FileNotFoundError:
        print(f"配置文件未找到: {config_file}")
    except yaml.YAMLError as e:
        print(f"解析 YAML 出错: {e}")
    return None

if __name__ == "__main__":
    cfg = load_config('config/config_hf.yaml')
    print(cfg)
