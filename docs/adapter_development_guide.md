# ModelAdapter 编写规范

开发ModelAdapter（模型适配器）时的关键规范和常见陷阱。

## 核心原则

### 1. ⭐ Causal Attention Mask（最重要！）

**规则**: 在所有forward调用中都必须传递正确的causal attention mask，包括prefill和decode阶段。

**为什么**: 如果传`attention_mask=None`，transformers的decoder layer可能会创建错误的mask，导致token可以attend到未来位置，产生重复输出。

**实现**:
```python
def forward(self, hidden_states, kv_cache, position_ids):
    seq_len = hidden_states.shape[1]
    
    # 获取已缓存的长度
    cached_len = kv_cache.get_seq_length() if hasattr(kv_cache, 'get_seq_length') else 0
    target_len = seq_len
    source_len = cached_len + seq_len
    
    # 构建causal mask
    # position i可以attend到位置[0, cached_len + i]
    causal_mask = torch.triu(
        torch.full((target_len, source_len), float('-inf'), 
                   device=hidden_states.device, dtype=hidden_states.dtype),
        diagonal=cached_len + 1,
    )
    attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, target, source]
    
    # 计算cache_position
    cache_position = torch.arange(cached_len, cached_len + seq_len, 
                                  device=hidden_states.device)
    
    # 传递给每一层
    for layer in self.layers:
        outputs = layer(
            hidden_states,
            attention_mask=attention_mask,  # 必须！不能是None
            position_ids=position_ids,
            past_key_value=kv_cache,
            use_cache=True,
            position_embeddings=(cos, sin),
            cache_position=cache_position,
        )
        hidden_states = outputs[0]
    
    return hidden_states, kv_cache
```

**验证**: 如果输出是重复的（如"is is is..."），99%是attention mask问题。

---

### 2. Rotary Position Embeddings (RoPE)

**规则**: 正确计算并传递position_embeddings给decoder layer。

**实现**:
```python
def _compute_rotary_embeddings(self, position_ids, seq_len):
    """计算rotary embeddings (cos, sin)"""
    config = self.config
    head_dim = config.hidden_size // config.num_attention_heads
    base = config.rope_theta
    
    # 计算逆频率
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=self.device) / head_dim)
    )
    
    # position_ids: [batch, seq_len]
    batch_size, seq_len_actual = position_ids.shape
    position_ids_flat = position_ids.reshape(-1)
    
    # 计算频率外积: [batch * seq_len, head_dim/2]
    freqs = torch.outer(position_ids_flat.float(), inv_freq)
    
    # 重塑: [batch, seq_len, head_dim/2]
    freqs = freqs.reshape(batch_size, seq_len_actual, -1)
    
    # 拼接并计算cos/sin: [batch, seq_len, head_dim]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    
    # 添加维度用于广播: [batch, 1, seq_len, head_dim]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    
    return cos, sin
```

**关键点**:
- cos/sin的形状必须是`[batch, 1, seq_len, head_dim]`用于正确广播
- 必须在forward开始时计算一次，然后传递给所有layer
- 不要每层都重新计算

---

### 3. KV Cache 管理

**规则**: 使用transformers的DynamicCache，确保layer_idx从0开始连续。

**初始化**:
```python
def init_kv_cache(self, batch_size, max_seq_len, device):
    from transformers.cache_utils import DynamicCache
    return DynamicCache()
```

**Layer创建**:
```python
# ❌ 错误：使用全局layer index
for i in range(layer_start, layer_end):
    layer = Qwen2DecoderLayer(config, layer_idx=i)  # 会导致index out of range

# ✅ 正确：使用局部layer index
for i in range(layer_end - layer_start):
    layer = Qwen2DecoderLayer(config, layer_idx=i)  # 从0开始
```

**使用**:
```python
# DynamicCache会在layer内部自动更新
for layer in self.layers:
    outputs = layer(
        hidden_states,
        past_key_value=kv_cache,  # 传递同一个cache对象
        use_cache=True,
        # ...
    )
    hidden_states = outputs[0]
    # kv_cache已经被layer内部更新，不需要手动处理

return hidden_states, kv_cache  # 返回更新后的cache
```

---

### 4. 权重加载

#### 4.1 Tied Word Embeddings

**规则**: 检查config中的`tie_word_embeddings`标志，正确处理LM head。

```python
def load(self, model_path, layer_start, layer_end, dtype, device):
    # 加载权重到CPU
    state_dict = self._load_weights(model_path, torch.device("cpu"))
    
    # 检查是否tied embeddings
    tie_word_embeddings = self.config.get("tie_word_embeddings", False)
    
    # 过滤权重
    shard_weights = {}
    for key, value in state_dict.items():
        if "model.layers." in key:
            # 处理transformer layers
            layer_idx = int(key.split("model.layers.")[1].split(".")[0])
            if layer_start <= layer_idx < layer_end:
                new_key = key.replace(
                    f"model.layers.{layer_idx}.",
                    f"model.layers.{layer_idx - layer_start}.",
                )
                shard_weights[new_key] = value.to(dtype=dtype, device=device)
        elif key == "model.embed_tokens.weight":
            # 第一个shard需要embedding
            if layer_start == 0:
                shard_weights[key] = value.to(dtype=dtype, device=device)
            # 如果是tied embeddings且是最后一个shard，也需要加载用于LM head
            elif layer_end == num_layers and tie_word_embeddings:
                shard_weights[key] = value.to(dtype=dtype, device=device)
        elif key == "model.norm.weight" and layer_end == num_layers:
            shard_weights[key] = value.to(dtype=dtype, device=device)
        elif key == "lm_head.weight" and layer_end == num_layers:
            shard_weights[key] = value.to(dtype=dtype, device=device)
    
    # 构建模型
    self._build_model(shard_weights, layer_start, layer_end, num_layers)

def _build_model(self, weights, layer_start, layer_end, num_layers):
    # ... 创建layers ...
    
    # 最后一个shard需要LM head
    if layer_end == num_layers:
        if "lm_head.weight" in weights:
            # 独立LM head
            self._lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
            self._lm_head.weight.data = weights["lm_head.weight"]
        elif tie_word_embeddings and "model.embed_tokens.weight" in weights:
            # Tied embeddings: 使用embed_tokens的权重
            self._lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
            self._lm_head.weight.data = weights["model.embed_tokens.weight"]
```

#### 4.2 内存优化

**规则**: 先加载到CPU，过滤后再移到GPU，避免内存爆炸。

```python
# ❌ 错误：直接加载到GPU
state_dict = load_file(model_path, device="cuda")  # 可能OOM

# ✅ 正确：CPU加载，过滤，再移动
state_dict = load_file(model_path, device="cpu")

# 过滤当前shard需要的权重
shard_weights = {}
for key, value in state_dict.items():
    if should_include(key):
        shard_weights[key] = value.to(device=device, dtype=dtype)

# 释放完整state_dict
del state_dict
```

---

### 5. 设备管理

**规则**: 确保所有张量都在正确的设备上。

```python
def __init__(self):
    self._device = torch.device("cpu")
    self._dtype = torch.float32

def load(self, model_path, layer_start, layer_end, dtype, device):
    self._device = device
    self._dtype = dtype
    # ...

def get_device(self) -> torch.device:
    """返回模型所在设备"""
    return self._device

def forward(self, hidden_states, kv_cache, position_ids):
    # 确保新创建的张量在正确的设备上
    cos, sin = self._compute_rotary_embeddings(position_ids, seq_len)
    # cos和sin会自动使用position_ids的设备
    
    causal_mask = torch.full(..., device=hidden_states.device, dtype=hidden_states.dtype)
    # 使用hidden_states的设备
```

---

### 6. Transformers API 适配

**规则**: 根据transformers版本正确适配API。

**transformers 4.44.0 (Qwen2)**:
```python
# Decoder layer需要这些参数
outputs = layer(
    hidden_states,
    attention_mask=attention_mask,      # 必须！
    position_ids=position_ids,
    past_key_value=kv_cache,            # DynamicCache对象
    use_cache=True,
    position_embeddings=(cos, sin),     # 必须！
    cache_position=cache_position,      # 必须！
)
```

**检查方法**:
```python
import transformers
print(transformers.__version__)

# 查看layer的forward签名
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
import inspect
print(inspect.signature(Qwen2DecoderLayer.forward))
```

---

## 验证清单

开发完adapter后，按以下步骤验证：

### Step 1: 权重验证
```python
# 对比每一层的权重
for i in range(num_layers):
    for key in hf_layer.state_dict():
        diff = (hf_sd[key] - our_sd[key]).abs().max()
        assert diff == 0, f"Layer {i} weight mismatch: {key}"
```

### Step 2: 单层验证
```python
# 用相同权重创建单个layer，对比输出
hf_layer = hf_model.model.layers[0]
our_layer = create_layer_with_same_weights()

hf_out = hf_layer(hf_input, ...)
our_out = our_layer(our_input, ...)

assert (hf_out - our_out).abs().max() < 1e-5
```

### Step 3: 多层验证
```python
# 逐步增加layer数量
for num_layers in [1, 2, 4, 8, 12, 24]:
    hf_hidden = run_hf_layers(num_layers)
    our_hidden = run_our_layers(num_layers)
    
    diff = (hf_hidden - our_hidden).abs().max()
    print(f"{num_layers} layers: diff={diff}")
    assert diff < 1e-4
```

### Step 4: Logits验证
```python
# 对比最终logits
hf_logits = hf_model(input_ids).logits
our_logits = adapter.forward(...)

# 检查argmax是否一致
hf_argmax = torch.argmax(hf_logits[0, -1, :])
our_argmax = torch.argmax(our_logits[0, -1, :])
assert hf_argmax == our_argmax, f"HF: {hf_argmax}, Ours: {our_argmax}"
```

### Step 5: 生成验证
```python
# 完整生成测试
output = decoder.generate("The capital of France is", max_new_tokens=10)
# 应该输出"Paris"，不是"is is is..."
```

---

## 常见错误及解决

| 症状 | 原因 | 解决 |
|------|------|------|
| 输出重复（如"is is is..."） | Attention mask错误 | 始终传递正确的causal mask |
| 输出随机/乱码 | 权重加载错误或RoPE错误 | 检查权重diff和rotary embeddings |
| "list index out of range" | layer_idx不连续 | 使用从0开始的局部layer_idx |
| "LM head not found" | 未处理tied embeddings | 检查tie_word_embeddings标志 |
| CUDA OOM | 权重直接加载到GPU | 先加载到CPU，过滤后再移动 |
| 设备不匹配 | 张量在不同设备上 | 使用`get_device()`确保一致性 |
| `Can't call numpy() on Tensor that requires grad` | 序列化时tensor带梯度 | 用`.detach().cpu().numpy()` |
| `coroutine 'Channel.close' was never awaited` | gRPC异步channel未await close | 用`await channel.close()` |

---

## 模板代码

```python
class MyModelAdapter(ModelAdapter):
    def __init__(self):
        self._layers = []
        self._embed_tokens = None
        self._lm_head = None
        self._norm = None
        self._config = {}
        self._device = torch.device("cpu")
        self._dtype = torch.float32
        self._loaded = False
    
    def load(self, model_path, layer_start, layer_end, dtype, device):
        self._device = device
        self._dtype = dtype
        
        # 1. 加载config
        # 2. 加载权重到CPU
        # 3. 过滤并移动到目标设备
        # 4. 构建模型组件
        # 5. 处理tied embeddings
        
        self._loaded = True
    
    def forward(self, hidden_states, kv_cache, position_ids):
        # 1. 计算rotary embeddings (cos, sin)
        # 2. 构建causal attention mask ⭐
        # 3. 计算cache_position
        # 4. 遍历所有layer，传递所有必要参数
        # 5. 应用final norm（如果是最后一个shard）
        
        return hidden_states, kv_cache
    
    def embed(self, input_ids):
        return self._embed_tokens(input_ids)
    
    def compute_logits(self, hidden_states):
        return self._lm_head(hidden_states)
    
    def init_kv_cache(self, batch_size, max_seq_len, device):
        from transformers.cache_utils import DynamicCache
        return DynamicCache()
    
    def get_device(self):
        return self._device
    
    def get_model_info(self):
        return {
            "hidden_size": self._config["hidden_size"],
            "num_attention_heads": self._config["num_attention_heads"],
            # ...
        }
    
    def unload(self):
        self._layers.clear()
        self._embed_tokens = None
        self._lm_head = None
        self._norm = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

---

## 总结

编写ModelAdapter的核心要点：

1. **⭐ Attention mask必须始终正确** - 这是最常见的bug
2. **权重必须完全一致** - 用diff验证
3. **使用DynamicCache** - layer_idx从0开始
4. **处理tied embeddings** - 检查config标志
5. **内存优化** - CPU加载，过滤后移动
6. **逐步验证** - 从单层到多层到完整生成

**记住**: 如果输出是重复的，首先检查attention mask！
