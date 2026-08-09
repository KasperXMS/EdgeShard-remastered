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

        if num_layers < 24:
            # For partial layers, iterate manually
            for i in range(num_layers):
                out = hf_model.model.layers[i](
                    hf_h,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_value=None,
                    use_cache=False,
                )
                hf_h = out[0]
        else:
            # For full model, use proper HF forward to get correct hidden states
            hf_outputs = hf_model(input_ids, output_hidden_states=True)
            hf_h = hf_outputs.hidden_states[-1]  # Last hidden state (after norm)

    # Now run through our implementation
    from edgeshard.runtime.adapters.qwen2 import Qwen2Adapter
    adapter = Qwen2Adapter()
    adapter.load(model_path, 0, num_layers, dtype, torch.device(device))

    with torch.no_grad():
        our_hidden_init = adapter.embed(input_ids)
        kv_cache = adapter.init_kv_cache(1, 100, torch.device(device))
        our_h, _ = adapter.forward(our_hidden_init, kv_cache, position_ids)
        # Note: adapter.forward() already applies self._norm for last shard
        # No need to apply it again here

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
        # HF logits - use proper HF forward (not manual iteration)
        hf_outputs = hf_model(input_ids)
        hf_logits = hf_outputs.logits

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
