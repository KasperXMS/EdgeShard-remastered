# 分布式LLM推理系统开发Skill

开发基于transformer的分布式LLM推理系统（如EdgeShard）时的关键注意事项和常见陷阱。

## 1. gRPC异步/同步混用

**问题**: 使用`grpc.insecure_channel()`创建同步channel，但在async函数中用`await`调用stub方法。

**解决**: 
```python
# ❌ 错误
channel = grpc.insecure_channel(addr)
stub = shard_pb2_grpc.ShardServiceStub(channel)
await stub.SendTensor(request)  # TypeError!

# ✅ 正确
channel = grpc.aio.insecure_channel(addr)
stub = shard_pb2_grpc.ShardServiceStub(channel)
await stub.SendTensor(request)  # OK
```

## 2. bfloat16张量序列化

**问题**: NumPy不支持bfloat16，直接调用`tensor.numpy()`会崩溃。

**解决**:
```python
# 序列化时：先detach()移除梯度图，再处理bfloat16
if tensor.dtype == torch.bfloat16:
    data = tensor.detach().cpu().to(torch.float32).numpy().tobytes()
else:
    data = tensor.detach().cpu().numpy().tobytes()

# 反序列化时：转回bfloat16
if dtype == torch.bfloat16:
    array = np.frombuffer(data, dtype=np.float32).reshape(shape)
    tensor = torch.from_numpy(array.copy()).to(dtype=dtype, device=device)
```

**注意**: 必须用`.detach()`，因为模型forward可能返回带梯度的tensor，直接调`.numpy()`会报`RuntimeError: Can't call numpy() on Tensor that requires grad`。

## 3. 单Shard Pipeline双重调用

**问题**: 当pipeline只有一个shard时，它同时是first和last shard，导致prefill/decode被调用两次。

**解决**:
```python
async def prefill(self, session_id, input_ids):
    # 单shard快速路径
    if len(self._shards) == 1:
        return await self._shards[0].shard.prefill(session_id, input_ids=input_ids)
    
    # 多shard pipeline逻辑
    # ...
```

## 4. Transformers版本API适配

**问题**: transformers 4.44.0的Qwen2DecoderLayer需要`position_embeddings`参数（cos, sin元组）。

**解决**:
```python
# 计算rotary embeddings
def _compute_rotary_embeddings(self, position_ids, seq_len):
    head_dim = config.hidden_size // config.num_attention_heads
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2) / head_dim))
    
    # 计算cos/sin
    freqs = torch.outer(position_ids.float().reshape(-1), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(1)  # [batch, 1, seq_len, head_dim]
    sin = emb.sin().unsqueeze(1)
    return cos, sin

# 传递给layer
outputs = layer(
    hidden_states,
    attention_mask=attention_mask,
    position_ids=position_ids,
    past_key_value=kv_cache,
    use_cache=True,
    position_embeddings=(cos, sin),  # 必须！
    cache_position=cache_position,
)
```

## 5. 权重加载内存优化

**问题**: 直接加载所有权重到GPU会导致内存爆炸（大模型可能有几十GB）。

**解决**:
```python
# 先加载到CPU，过滤后再移到目标设备
state_dict = self._load_weights(model_path, torch.device("cpu"))

# 过滤当前shard需要的权重
shard_weights = {}
for key, value in state_dict.items():
    if should_include(key, layer_start, layer_end):
        shard_weights[key] = value.to(dtype=dtype, device=device)

# 释放完整state_dict
del state_dict
```

## 6. DynamicCache与layer_idx

**问题**: transformers 4.44.0使用DynamicCache管理KV cache，它依赖layer_idx进行索引。如果layer_idx不连续（从0开始），会导致"list index out of range"。

**解决**:
```python
# 每个shard内部使用局部layer_idx
for i in range(layer_end - layer_start):
    layer = Qwen2DecoderLayer(config, layer_idx=i)  # 从0开始，不是layer_start+i
    # ...
```

## 7. Tied Word Embeddings

**问题**: Qwen2.5等模型使用tied embeddings（LM head权重与embedding权重共享），checkpoint中没有`lm_head.weight`。

**解决**:
```python
# 检查config中的tie_word_embeddings标志
tie_word_embeddings = config.get("tie_word_embeddings", False)

if layer_end == num_layers:
    if "lm_head.weight" in weights:
        # 独立LM head
        self._lm_head.load_state_dict(...)
    elif tie_word_embeddings and "model.embed_tokens.weight" in weights:
        # Tied embeddings: 最后一个shard也需要加载embed_tokens权重作为LM head
        self._lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self._lm_head.weight.data = weights["model.embed_tokens.weight"]
```

## 8. ⭐ Causal Attention Mask（最关键！）

**问题**: 在prefill阶段传`attention_mask=None`给decoder layer，导致layer创建了错误的attention mask（非causal），每个token可以attend到未来的token，产生重复输出。

**解决**:
```python
# 始终构建正确的causal mask，即使是prefill阶段！
cached_len = kv_cache.get_seq_length() if hasattr(kv_cache, 'get_seq_length') else 0
target_len = seq_len
source_len = cached_len + seq_len

# Causal mask: position i可以attend到[0, cached_len + i]
causal_mask = torch.triu(
    torch.full((target_len, source_len), float('-inf'), 
               device=device, dtype=dtype),
    diagonal=cached_len + 1,
)
attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, target, source]

# 传递给layer
outputs = layer(
    hidden_states,
    attention_mask=attention_mask,  # 始终传递，不是None！
    # ...
)
```

**验证方法**:
```python
# 如果输出是重复的（如"is is is..."），99%是attention mask问题
# 对比HF模型的forward()输出，确保hidden states完全一致
```

## 9. asyncio事件循环管理

**问题**: 在CLI命令中调用两次`asyncio.run()`会导致事件循环冲突。

**解决**:
```python
# ❌ 错误
try:
    asyncio.run(run())
except KeyboardInterrupt:
    asyncio.run(cleanup())  # 新的事件循环，可能有问题

# ✅ 正确
async def run_and_cleanup():
    try:
        await start()
        await wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        await cleanup()

asyncio.run(run_and_cleanup())  # 单次事件循环
```

## 10. ABC签名一致性

**问题**: 基类的方法签名与实现类不匹配（如缺少`session_id`参数）。

**解决**: 确保ABC和实现类的方法签名完全一致，包括可选参数。

## 调试技巧

### 逐步验证
1. **权重加载**: 对比每个layer的权重与HF模型，确保diff=0
2. **单层forward**: 用相同权重创建单个layer，对比输出
3. **多层forward**: 逐步增加layer数量（1→2→4→8→12→24），找到divergence点
4. **参数隔离**: 逐个测试`use_cache`、`DynamicCache`、`cache_position`、`position_embeddings`、`attention_mask`

### 关键对比指标
```python
# 对比hidden states
diff = (hf_hidden - our_hidden).abs()
print(f"mean={diff.mean():.6f}, max={diff.max():.6f}")

# 对比logits
hf_argmax = torch.argmax(hf_logits[0, -1, :]).item()
our_argmax = torch.argmax(our_logits[0, -1, :]).item()
# 如果argmax不同，说明forward pass有问题
```

### 常见错误输出模式
- **重复输出**（如"is is is..."）: attention mask问题
- **随机输出**: 权重加载错误或rotary embeddings错误
- **部分正确**: KV cache管理问题

## 总结

开发分布式LLM推理系统时，最关键的是：
1. **始终传递正确的causal attention mask**（即使是prefill）
2. **确保权重加载完全正确**（对比每个参数）
3. **使用正确的transformers API**（版本适配）
4. **验证每一步的输出**（与HF模型对比）

记住：**如果输出是重复的，首先检查attention mask！**
