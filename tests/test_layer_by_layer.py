"""Find exactly which layer causes divergence."""
import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from safetensors.torch import load_file

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

# Load raw weights
state_dict = load_file(str(Path(model_path) / "model.safetensors"), device="cpu")

# Get HF embeddings
with torch.no_grad():
    hf_hidden = hf_model.model.embed_tokens(input_ids)
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)

# Test layers 12-23 one by one, starting from layer 12 output from HF
print("\n=== Getting HF layer 12 input ===")
hf_h = hf_hidden.clone()
with torch.no_grad():
    for i in range(12):
        out = hf_model.model.layers[i](
            hf_h, attention_mask=None, position_ids=position_ids,
            past_key_value=None, use_cache=False,
        )
        hf_h = out[0]
print(f"HF hidden after 12 layers: mean={hf_h.float().mean():.6f}, std={hf_h.float().std():.6f}")

# Now test each layer 12-23 individually
print("\n=== Testing layers 12-23 one by one ===")
our_h = hf_h.clone()  # Start from same point

for layer_idx in range(12, 24):
    with torch.no_grad():
        # HF layer
        hf_out = hf_model.model.layers[layer_idx](
            hf_h, attention_mask=None, position_ids=position_ids,
            past_key_value=None, use_cache=False,
        )
        hf_h_next = hf_out[0]

        # Our layer - create with same weights
        our_layer = Qwen2DecoderLayer(config, layer_idx=layer_idx)
        # Load weights from HF
        hf_sd = hf_model.model.layers[layer_idx].state_dict()
        missing, unexpected = our_layer.load_state_dict(hf_sd, strict=True)
        our_layer.to(device, dtype)
        our_layer.eval()

        # Compute rotary embeddings
        from edgeshard.runtime.adapters.qwen2 import Qwen2Adapter
        adapter_tmp = Qwen2Adapter()
        adapter_tmp._config = config_dict
        adapter_tmp._device = torch.device(device)
        adapter_tmp._dtype = dtype
        cos, sin = adapter_tmp._compute_rotary_embeddings(position_ids, input_ids.shape[1])

        our_out = our_layer(
            our_h, attention_mask=None, position_ids=position_ids,
            past_key_value=None, use_cache=False,
            position_embeddings=(cos, sin),
        )
        our_h_next = our_out[0]

    diff = (hf_h_next - our_h_next).abs()
    match = "✅" if diff.max() < 0.01 else "❌"
    print(f"Layer {layer_idx}: {match} diff_mean={diff.mean():.6f}, diff_max={diff.max():.6f}, "
          f"hf_mean={hf_h_next.float().mean():.6f}, our_mean={our_h_next.float().mean():.6f}")

    if diff.max() >= 0.01:
        # Found the problematic layer! Let's dig deeper
        print(f"\n=== Deep dive into layer {layer_idx} ===")

        # Check if attention_mask=None is the issue
        # HF layer with proper causal mask
        seq_len = input_ids.shape[1]
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            hf_out_masked = hf_model.model.layers[layer_idx](
                hf_h, attention_mask=causal_mask, position_ids=position_ids,
                past_key_value=None, use_cache=False,
            )
            hf_h_masked = hf_out_masked[0]

            our_out_masked = our_layer(
                our_h, attention_mask=causal_mask, position_ids=position_ids,
                past_key_value=None, use_cache=False,
                position_embeddings=(cos, sin),
            )
            our_h_masked = our_out_masked[0]

        diff_masked = (hf_h_masked - our_h_masked).abs()
        print(f"With causal mask: diff_mean={diff_masked.mean():.6f}, diff_max={diff_masked.max():.6f}")

        # Check intermediate values
        # Let's manually run through the layer step by step
        print(f"\nChecking layer {layer_idx} internals...")

        # Input layernorm
        hf_ln1 = hf_model.model.layers[layer_idx].input_layernorm(hf_h)
        our_ln1 = our_layer.input_layernorm(our_h)
        diff_ln1 = (hf_ln1 - our_ln1).abs()
        print(f"  InputLayerNorm: diff_max={diff_ln1.max():.6f}")
        print(f"  HF weight: {hf_model.model.layers[layer_idx].input_layernorm.weight[:5]}")
        print(f"  Our weight: {our_layer.input_layernorm.weight[:5]}")

        # Self attention
        with torch.no_grad():
            hf_attn_out = hf_model.model.layers[layer_idx].self_attn(
                hf_ln1, position_ids=position_ids, past_key_value=None, use_cache=False,
            )
            our_attn_out = our_layer.self_attn(
                our_ln1, position_ids=position_ids, past_key_value=None, use_cache=False,
                position_embeddings=(cos, sin),
            )
        diff_attn = (hf_attn_out[0] - our_attn_out[0]).abs()
        print(f"  SelfAttention: diff_max={diff_attn.max():.6f}")

        break

    hf_h = hf_h_next
    our_h = our_h_next

print("\n=== Test complete ===")
