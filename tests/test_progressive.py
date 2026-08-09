"""Test full forward pass with increasing number of layers."""
import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

model_path = "models/Qwen2.5-0.5B-Instruct"
device = "cuda"
dtype = torch.float16

# Load HF model
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

# Get HF embeddings
with torch.no_grad():
    hf_hidden = hf_model.model.embed_tokens(input_ids)

# Test with different numbers of layers
for num_layers in [1, 2, 4, 8, 12, 24]:
    print(f"\n=== Testing with {num_layers} layers ===")

    # Run through HF layers
    hf_h = hf_hidden.clone()
    with torch.no_grad():
        position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        for i in range(num_layers):
            out = hf_model.model.layers[i](
                hf_h,
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=None,
                use_cache=False,
            )
            hf_h = out[0]

    # Apply HF final norm if this is all layers
    if num_layers == 24:
        hf_h = hf_model.model.norm(hf_h)

    # Now run through our implementation
    from edgeshard.runtime.adapters.qwen2 import Qwen2Adapter
    adapter = Qwen2Adapter()
    adapter.load(model_path, 0, num_layers, dtype, torch.device(device))

    with torch.no_grad():
        our_hidden_init = adapter.embed(input_ids)
        kv_cache = adapter.init_kv_cache(1, 100, torch.device(device))
        our_h, _ = adapter.forward(our_hidden_init, kv_cache, position_ids)

        # Apply final norm if this is all layers
        if num_layers == 24:
            our_h = adapter._norm(our_h)

    # Compare
    diff = (hf_h - our_h).abs()
    print(f"HF output: mean={hf_h.float().mean():.6f}, std={hf_h.float().std():.6f}")
    print(f"Our output: mean={our_h.float().mean():.6f}, std={our_h.float().std():.6f}")
    print(f"Diff: mean={diff.mean():.6f}, max={diff.max():.6f}")

    if diff.max() < 0.01:
        print(f"✅ {num_layers} layers: MATCH")
    else:
        print(f"❌ {num_layers} layers: DIVERGE")
        break

    adapter.unload()

# If all layers match, test logits
if num_layers == 24:
    print("\n=== Testing logits ===")
    with torch.no_grad():
        # HF logits
        hf_logits = hf_model.lm_head(hf_h)

        # Our logits
        our_logits = adapter.compute_logits(our_h)

    diff = (hf_logits - our_logits).abs()
    print(f"HF logits: mean={hf_logits.float().mean():.6f}, std={hf_logits.float().std():.6f}")
    print(f"Our logits: mean={our_logits.float().mean():.6f}, std={our_logits.float().std():.6f}")
    print(f"Diff: mean={diff.mean():.6f}, max={diff.max():.6f}")

    hf_argmax = torch.argmax(hf_logits[0, -1, :]).item()
    our_argmax = torch.argmax(our_logits[0, -1, :]).item()
    print(f"HF argmax: {hf_argmax} ({tokenizer.decode([hf_argmax])})")
    print(f"Our argmax: {our_argmax} ({tokenizer.decode([our_argmax])})")

    if diff.max() < 0.1:
        print("✅ Logits MATCH!")
    else:
        print("❌ Logits DIVERGE!")
