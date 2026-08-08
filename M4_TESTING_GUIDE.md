# M4 实机测试指南

本文档指导你如何在真实机器上测试 EdgeShard M4 分布式推理。

## 前置条件

### 硬件
- 至少 2 台带 GPU 的机器（或 1 台多 GPU 机器）
- 每台机器至少 8GB GPU 内存（测试 Qwen2.5-0.5B）
- 机器间网络互通

### 软件
- Python 3.10+
- CUDA 11.8+ 
- PyTorch 2.2+
- transformers, safetensors

### 安装 EdgeShard
```bash
# 在每台机器上
git clone <your-repo>
cd EdgeShard-remastered
pip install -e ".[all]"
```

### 准备模型
```bash
# 在每台机器上（或使用共享存储）
mkdir -p models
# 下载 Qwen2.5-0.5B-Instruct
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/Qwen2.5-0.5B-Instruct
```

## 测试场景 1：单机双 shard（最快验证）

在一台机器上运行两个 shard 进程，验证 pipeline 逻辑。

### 终端 1：启动 Shard 0（前 12 层）
```bash
edgeshard shard start \
    models/Qwen2.5-0.5B-Instruct \
    --shard-id shard-0 \
    --layers 0:12 \
    --first \
    --port 50100
```

### 终端 2：启动 Shard 1（后 12 层 + LM head）
```bash
edgeshard shard start \
    models/Qwen2.5-0.5B-Instruct \
    --shard-id shard-1 \
    --layers 12:24 \
    --last \
    --port 50101
```

### 终端 3：运行推理（TODO：需要完善 infer 命令）
目前 `edgeshard infer` 还是占位符，需要用 Python 脚本测试：

```python
# test_inference.py
import asyncio
import torch
from pathlib import Path
from transformers import AutoTokenizer

from edgeshard.runtime.adapters.qwen2 import Qwen2Adapter
from edgeshard.runtime.shard import ModelShard
from edgeshard.runtime.pipeline import PipelineOrchestrator, ShardEndpoint
from edgeshard.runtime.pipeline_decoder import PipelineDecoder, GenerationConfig

async def main():
    model_path = "models/Qwen2.5-0.5B-Instruct"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Create shard 0 (local)
    adapter0 = Qwen2Adapter()
    adapter0.load(model_path, 0, 12, torch.float16, torch.device("cuda"))
    shard0 = ModelShard("shard-0", adapter0, is_first_shard=True, is_last_shard=False)
    
    # Create shard 1 (local)
    adapter1 = Qwen2Adapter()
    adapter1.load(model_path, 12, 24, torch.float16, torch.device("cuda"))
    shard1 = ModelShard("shard-1", adapter1, is_first_shard=False, is_last_shard=True)
    
    # Create pipeline
    pipeline = PipelineOrchestrator([
        ShardEndpoint(shard0, None, None, True),
        ShardEndpoint(shard1, None, None, True),
    ])
    
    # Create decoder
    decoder = PipelineDecoder(pipeline, tokenizer)
    
    # Generate
    config = GenerationConfig(max_new_tokens=50, eos_token_id=tokenizer.eos_token_id)
    result = await decoder.generate("The capital of France is", config)
    
    print(f"\nGenerated: {result.text}")
    print(f"Tokens: {result.num_tokens}")
    
    await pipeline.close()
    adapter0.unload()
    adapter1.unload()

if __name__ == "__main__":
    asyncio.run(main())
```

运行：
```bash
python test_inference.py
```

## 测试场景 2：双机分布式（真实分布式）

机器 A（192.168.1.10）和机器 B（192.168.1.11）。

### 机器 A：启动 Shard 0
```bash
edgeshard shard start \
    models/Qwen2.5-0.5B-Instruct \
    --shard-id shard-0 \
    --layers 0:12 \
    --first \
    --host 0.0.0.0 \
    --port 50100
```

### 机器 B：启动 Shard 1
```bash
edgeshard shard start \
    models/Qwen2.5-0.5B-Instruct \
    --shard-id shard-1 \
    --layers 12:24 \
    --last \
    --host 0.0.0.0 \
    --port 50100
```

### 任一机器：运行推理客户端

```python
# test_distributed.py
import asyncio
import torch
from transformers import AutoTokenizer

from edgeshard.runtime.adapters.qwen2 import Qwen2Adapter
from edgeshard.runtime.shard import ModelShard
from edgeshard.runtime.pipeline import PipelineOrchestrator, ShardEndpoint
from edgeshard.runtime.pipeline_decoder import PipelineDecoder, GenerationConfig
from edgeshard.transport.grpc_transport import GrpcTensorTransport

async def main():
    model_path = "models/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Shard 0 on machine A (192.168.1.10:50100)
    adapter0 = Qwen2Adapter()
    adapter0.load(model_path, 0, 12, torch.float16, torch.device("cuda"))
    shard0 = ModelShard("shard-0", adapter0, is_first_shard=True, is_last_shard=False)
    
    # Shard 1 on machine B (192.168.1.11:50100)
    adapter1 = Qwen2Adapter()
    adapter1.load(model_path, 12, 24, torch.float16, torch.device("cuda"))
    shard1 = ModelShard("shard-1", adapter1, is_first_shard=False, is_last_shard=True)
    
    # Create transports
    transport0 = GrpcTensorTransport(
        "shard-0",
        {"shard-1": "192.168.1.11:50100"}
    )
    transport1 = GrpcTensorTransport(
        "shard-1",
        {"shard-0": "192.168.1.10:50100"}
    )
    
    # Create pipeline
    pipeline = PipelineOrchestrator([
        ShardEndpoint(shard0, transport0, "192.168.1.10:50100", False),
        ShardEndpoint(shard1, transport1, "192.168.1.11:50100", False),
    ])
    
    # Create decoder
    decoder = PipelineDecoder(pipeline, tokenizer)
    
    # Generate
    config = GenerationConfig(max_new_tokens=50, eos_token_id=tokenizer.eos_token_id)
    result = await decoder.generate("The capital of France is", config)
    
    print(f"\nGenerated: {result.text}")
    print(f"Tokens: {result.num_tokens}")
    
    await pipeline.close()
    adapter0.unload()
    adapter1.unload()

if __name__ == "__main__":
    asyncio.run(main())
```

## 测试场景 3：单元测试

运行 M4 测试套件：

```bash
# 单机测试（需要模型）
pytest tests/unit/test_m4_distributed.py -v

# 指定测试
pytest tests/unit/test_m4_distributed.py::test_tensor_serialization -v
pytest tests/unit/test_m4_distributed.py::test_pipeline_orchestrator_single_process -v
```

## 预期输出

成功的分布式推理应该输出类似：

```
Generated: The capital of France is Paris. It is located in the north-central part of the country and is the most populous city in France.
Tokens: 24
```

## 故障排查

### Shard 启动失败
- 检查模型路径是否正确
- 检查 GPU 内存是否足够
- 检查端口是否被占用

### Tensor 传输失败
- 检查防火墙是否开放端口
- 检查机器间网络是否互通：`ping 192.168.1.11`
- 检查 gRPC 连接：`telnet 192.168.1.11 50100`

### 输出不一致
- 确认所有 shard 使用相同的 dtype（float16）
- 确认层范围正确覆盖（0:12, 12:24）
- 对比单节点参考输出

### 性能问题
- 检查网络带宽（tensor 传输瓶颈）
- 考虑使用 NCCL 替代 gRPC（未来优化）
- 检查 GPU 利用率

## 下一步

M4 验证通过后，继续：
- **M5**: 资源发现增强
- **M6**: 自动 profiling
- **M7**: 调度器（自动切分）
- **M8**: Docker 部署
