"""Compare weights loaded by our adapter vs HF for all 24 layers."""
import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, Qwen2Config

model_path = "models/Qwen2.5-0.5B-Instruct"
device = "cuda"
dtype = torch.float16

print("Loading HF model...")
hf_model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=dtype, device_map=device,
)
hf_model.eval()

print("Loading our adapter (all 24 layers)...")
from edgeshard.runtime.adapters.qwen2 import Qwen2Adapter
adapter = Qwen2Adapter()
adapter.load(model_path, 0, 24, dtype, torch.device(device))

# Compare each layer's weights
print("\n=== Comparing layer weights ===")
for i in range(24):
    hf_layer = hf_model.model.layers[i]
    our_layer = adapter._layers[i]

    hf_sd = hf_layer.state_dict()
    our_sd = our_layer.state_dict()

    # Check all keys
    all_match = True
    max_diff = 0.0
    for key in hf_sd:
        if key not in our_sd:
            print(f"Layer {i}: MISSING key {key}")
            all_match = False
            continue
        diff = (hf_sd[key].float() - our_sd[key].float()).abs().max().item()
        max_diff = max(max_diff, diff)
        if diff > 0.001:
            print(f"Layer {i}: {key} diff={diff:.6f}")
            all_match = False

    for key in our_sd:
        if key not in hf_sd:
            print(f"Layer {i}: UNEXPECTED key {key}")
            all_match = False

    status = "✅" if all_match else "❌"
    print(f"Layer {i}: {status} max_diff={max_diff:.6f}")

# Compare final norm
print("\n=== Comparing final norm ===")
hf_norm_w = hf_model.model.norm.weight
our_norm_w = adapter._norm.weight
norm_diff = (hf_norm_w.float() - our_norm_w.float()).abs().max().item()
print(f"Norm weight diff: {norm_diff:.6f}")

# Compare LM head
print("\n=== Comparing LM head ===")
hf_lm_w = hf_model.lm_head.weight
our_lm_w = adapter._lm_head.weight
lm_diff = (hf_lm_w.float() - our_lm_w.float()).abs().max().item()
print(f"LM head weight diff: {lm_diff:.6f}")

# Compare embed_tokens
print("\n=== Comparing embed_tokens ===")
hf_embed_w = hf_model.model.embed_tokens.weight
our_embed_w = adapter._embed_tokens.weight
embed_diff = (hf_embed_w.float() - our_embed_w.float()).abs().max().item()
print(f"Embed weight diff: {embed_diff:.6f}")

print("\n=== Summary ===")
print("If all diffs are 0, the problem is in the forward pass logic, not weights.")
