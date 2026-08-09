"""Diagnostic test: compare single-layer output between our adapter and HF."""
import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2Config

model_path = "models/Qwen2.5-0.5B-Instruct"
device = "cuda"
dtype = torch.float16

# Load HF model
print("=== Loading HF model ===")
hf_model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=dtype, device_map=device,
)
hf_model.eval()

# Load config
with open(Path(model_path) / "config.json") as f:
    config_dict = json.load(f)
config = Qwen2Config(**config_dict)

# Create tokenizer and encode
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
input_ids = tokenizer.encode("The capital of France is", return_tensors="pt").to(device)

print(f"Input: {input_ids}")
print(f"Config: head_dim={config.hidden_size // config.num_attention_heads}, "
      f"hidden_size={config.hidden_size}, num_heads={config.num_attention_heads}")
print(f"Config rope_theta={config.rope_theta}")
print(f"Config tie_word_embeddings={config_dict.get('tie_word_embeddings', 'NOT SET')}")

# Get embed_tokens from HF model
hf_embed = hf_model.model.embed_tokens
hf_layers = hf_model.model.layers

# Step 1: Compare embedding output
with torch.no_grad():
    hf_hidden = hf_embed(input_ids)
print(f"\nHF embed output: shape={hf_hidden.shape}, dtype={hf_hidden.dtype}, "
      f"mean={hf_hidden.float().mean():.6f}, std={hf_hidden.float().std():.6f}")
print(f"HF embed first 5 values: {hf_hidden[0, 0, :5]}")

# Step 2: Create our own embedding with same weights
from torch import nn
our_embed = nn.Embedding(config.vocab_size, config.hidden_size)
our_embed.weight.data = hf_embed.weight.data.clone()
our_embed.to(device, dtype)
our_embed.eval()

with torch.no_grad():
    our_hidden = our_embed(input_ids)
print(f"\nOur embed output: shape={our_hidden.shape}, dtype={our_hidden.dtype}, "
      f"mean={our_hidden.float().mean():.6f}, std={our_hidden.float().std():.6f}")
print(f"Our embed first 5 values: {our_hidden[0, 0, :5]}")
print(f"Embed diff: {(hf_hidden - our_hidden).abs().max().item()}")

# Step 3: Run HF layer 0
with torch.no_grad():
    hf_layer0 = hf_layers[0]
    # Need position_ids and position_embeddings
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)

    # Get HF's rotary embeddings
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
    hf_rotary = hf_model.model.rotary_emb
    # In HF 4.44, rotary_emb takes (x, position_ids) and returns (cos, sin)
    # But it might be called internally. Let's get it directly.
    hf_out_layer0 = hf_layer0(
        hf_hidden,
        attention_mask=None,
        position_ids=position_ids,
        past_key_value=None,
        use_cache=False,
    )
    hf_hidden_after_layer0 = hf_out_layer0[0]

print(f"\nHF layer 0 output: shape={hf_hidden_after_layer0.shape}, "
      f"mean={hf_hidden_after_layer0.float().mean():.6f}, std={hf_hidden_after_layer0.float().std():.6f}")
print(f"HF layer 0 first 5 values: {hf_hidden_after_layer0[0, 0, :5]}")

# Step 4: Create our layer 0 with same weights
print("\n=== Creating our layer 0 ===")
our_layer0 = Qwen2DecoderLayer(config, layer_idx=0)
# Load weights from HF layer 0
hf_layer0_sd = hf_layer0.state_dict()
our_layer0.load_state_dict(hf_layer0_sd, strict=True)  # Use strict=True to verify exact match
our_layer0.to(device, dtype)
our_layer0.eval()
print(f"Loaded HF weights into our layer 0 (strict=True)")

# Step 5: Compute our rotary embeddings
from edgeshard.runtime.adapters.qwen2 import Qwen2Adapter
adapter = Qwen2Adapter()
adapter._config = config_dict
adapter._device = torch.device(device)
adapter._dtype = dtype
adapter._loaded = True

with torch.no_grad():
    cos, sin = adapter._compute_rotary_embeddings(position_ids, input_ids.shape[1])
print(f"\nOur cos shape: {cos.shape}, mean={cos.float().mean():.6f}")
print(f"Our sin shape: {sin.shape}, mean={sin.float().mean():.6f}")

# Step 6: Get HF's rotary embeddings for comparison
# In HF 4.44, the rotary_emb is on the model level
with torch.no_grad():
    # The HF rotary_emb.forward takes (x, position_ids)
    x_for_rotary = hf_hidden  # Use the hidden states as input to get correct dtype/device
    hf_cos, hf_sin = hf_rotary(x_for_rotary, position_ids)
print(f"\nHF cos shape: {hf_cos.shape}, mean={hf_cos.float().mean():.6f}")
print(f"HF sin shape: {hf_sin.shape}, mean={hf_sin.float().mean():.6f}")

# Compare
if cos.shape == hf_cos.shape:
    cos_diff = (cos - hf_cos).abs().max().item()
    sin_diff = (sin - hf_sin).abs().max().item()
    print(f"\nCos diff: {cos_diff}")
    print(f"Sin diff: {sin_diff}")
else:
    print(f"\nSHAPE MISMATCH: our cos={cos.shape}, HF cos={hf_cos.shape}")
    print(f"our sin={sin.shape}, HF sin={hf_sin.shape}")

# Step 7: Run our layer 0
with torch.no_grad():
    our_out_layer0 = our_layer0(
        our_hidden,
        attention_mask=None,
        position_ids=position_ids,
        past_key_value=None,
        use_cache=False,
        position_embeddings=(cos, sin),
    )
    our_hidden_after_layer0 = our_out_layer0[0]

print(f"\nOur layer 0 output: shape={our_hidden_after_layer0.shape}, "
      f"mean={our_hidden_after_layer0.float().mean():.6f}, std={our_hidden_after_layer0.float().std():.6f}")
print(f"Our layer 0 first 5 values: {our_hidden_after_layer0[0, 0, :5]}")

# Compare layer outputs
layer_diff = (hf_hidden_after_layer0 - our_hidden_after_layer0).abs()
print(f"\nLayer 0 output diff: mean={layer_diff.mean():.6f}, max={layer_diff.max():.6f}")

if layer_diff.max() < 0.01:
    print("\n✅ Layer 0 outputs MATCH! Problem is in later layers or final norm/lm_head.")
else:
    print("\n❌ Layer 0 outputs DIVERGE! Problem is in rotary embeddings or attention mechanism.")

    # Try with HF's rotary embeddings instead
    print("\n=== Trying with HF rotary embeddings ===")
    with torch.no_grad():
        our_out_with_hf_rotary = our_layer0(
            our_hidden,
            attention_mask=None,
            position_ids=position_ids,
            past_key_value=None,
            use_cache=False,
            position_embeddings=(hf_cos, hf_sin),
        )
        our_hidden_hf_rotary = our_out_with_hf_rotary[0]

    diff_with_hf = (hf_hidden_after_layer0 - our_hidden_hf_rotary).abs()
    print(f"With HF rotary: mean={diff_with_hf.mean():.6f}, max={diff_with_hf.max():.6f}")
    if diff_with_hf.max() < 0.01:
        print("✅ Confirmed: problem is in our rotary embedding computation!")
    else:
        print("❌ Still diverges. Problem is in something else (attention implementation?)")
