"""Isolate which parameter causes divergence in forward pass."""
import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.cache_utils import DynamicCache

model_path = "models/Qwen2.5-0.5B-Instruct"
device = "cuda"
dtype = torch.float16

print("Loading HF model...")
hf_model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=dtype, device_map=device,
)
hf_model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
input_ids = tokenizer.encode("The capital of France is", return_tensors="pt").to(device)

with open(Path(model_path) / "config.json") as f:
    config_dict = json.load(f)
config = Qwen2Config(**config_dict)

# Get embeddings
with torch.no_grad():
    hf_hidden = hf_model.model.embed_tokens(input_ids)
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    seq_len = input_ids.shape[1]

# Compute our rotary embeddings
from edgeshard.runtime.adapters.qwen2 import Qwen2Adapter
adapter_tmp = Qwen2Adapter()
adapter_tmp._config = config_dict
adapter_tmp._device = torch.device(device)
adapter_tmp._dtype = dtype
cos, sin = adapter_tmp._compute_rotary_embeddings(position_ids, seq_len)

# Test 1: use_cache=False, past_key_value=None (baseline)
print("\n=== Test 1: use_cache=False, past_key_value=None ===")
hf_h = hf_hidden.clone()
with torch.no_grad():
    for i in range(24):
        out = hf_model.model.layers[i](
            hf_h, attention_mask=None, position_ids=position_ids,
            past_key_value=None, use_cache=False,
        )
        hf_h = out[0]
hf_h = hf_model.model.norm(hf_h)
print(f"After 24 layers: mean={hf_h.float().mean():.6f}, std={hf_h.float().std():.6f}")

# Test 2: use_cache=True, past_key_value=DynamicCache (empty)
print("\n=== Test 2: use_cache=True, past_key_value=DynamicCache (empty) ===")
hf_h2 = hf_hidden.clone()
cache = DynamicCache()
with torch.no_grad():
    for i in range(24):
        out = hf_model.model.layers[i](
            hf_h2, attention_mask=None, position_ids=position_ids,
            past_key_value=cache, use_cache=True,
        )
        hf_h2 = out[0]
hf_h2 = hf_model.model.norm(hf_h2)
print(f"After 24 layers: mean={hf_h2.float().mean():.6f}, std={hf_h2.float().std():.6f}")
diff = (hf_h - hf_h2).abs()
print(f"Diff from Test 1: mean={diff.mean():.6f}, max={diff.max():.6f}")

# Test 3: use_cache=True, DynamicCache, with cache_position
print("\n=== Test 3: use_cache=True, DynamicCache, cache_position ===")
hf_h3 = hf_hidden.clone()
cache3 = DynamicCache()
cache_position = torch.arange(seq_len, device=device)
with torch.no_grad():
    for i in range(24):
        out = hf_model.model.layers[i](
            hf_h3, attention_mask=None, position_ids=position_ids,
            past_key_value=cache3, use_cache=True,
            cache_position=cache_position,
        )
        hf_h3 = out[0]
hf_h3 = hf_model.model.norm(hf_h3)
print(f"After 24 layers: mean={hf_h3.float().mean():.6f}, std={hf_h3.float().std():.6f}")
diff = (hf_h - hf_h3).abs()
print(f"Diff from Test 1: mean={diff.mean():.6f}, max={diff.max():.6f}")

# Test 4: use_cache=True, DynamicCache, cache_position, position_embeddings
print("\n=== Test 4: + position_embeddings ===")
hf_h4 = hf_hidden.clone()
cache4 = DynamicCache()
with torch.no_grad():
    for i in range(24):
        out = hf_model.model.layers[i](
            hf_h4, attention_mask=None, position_ids=position_ids,
            past_key_value=cache4, use_cache=True,
            cache_position=cache_position,
            position_embeddings=(cos, sin),
        )
        hf_h4 = out[0]
hf_h4 = hf_model.model.norm(hf_h4)
print(f"After 24 layers: mean={hf_h4.float().mean():.6f}, std={hf_h4.float().std():.6f}")
diff = (hf_h - hf_h4).abs()
print(f"Diff from Test 1: mean={diff.mean():.6f}, max={diff.max():.6f}")

# Test 5: Same but with our adapter's full forward
print("\n=== Test 5: Our adapter forward (all 24 layers) ===")
adapter = Qwen2Adapter()
adapter.load(model_path, 0, 24, dtype, torch.device(device))
with torch.no_grad():
    our_hidden = adapter.embed(input_ids)
    kv_cache = adapter.init_kv_cache(1, 100, torch.device(device))
    our_h, _ = adapter.forward(our_hidden, kv_cache, position_ids)
print(f"After 24 layers: mean={our_h.float().mean():.6f}, std={our_h.float().std():.6f}")
diff = (hf_h - our_h).abs()
print(f"Diff from Test 1 (HF baseline): mean={diff.mean():.6f}, max={diff.max():.6f}")

print("\n=== Diagnosis ===")
print("If Test 2 diverges from Test 1: use_cache=True or DynamicCache is the problem")
print("If Test 3 diverges from Test 2: cache_position is the problem")
print("If Test 4 diverges from Test 3: position_embeddings is the problem")
print("If Test 5 diverges from Test 4: our adapter forward has additional issues")
