"""Verify the attention mask issue."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "models/Qwen2.5-0.5B-Instruct"
device = "cuda"
dtype = torch.float16

hf_model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=dtype, device_map=device,
)
hf_model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

input_ids = tokenizer.encode("The capital of France is", return_tensors="pt").to(device)
print(f"Input: {input_ids}")

# Method 1: Full HF forward (what generate() uses)
print("\n=== Method 1: Full HF forward ===")
with torch.no_grad():
    out1 = hf_model(input_ids)
logits1 = out1.logits
argmax1 = torch.argmax(logits1[0, -1, :]).item()
print(f"Logits mean: {logits1.float().mean():.6f}, std: {logits1.float().std():.6f}")
print(f"Argmax: {argmax1} ({tokenizer.decode([argmax1])})")

# Method 2: Manual layer iteration with attention_mask=None
print("\n=== Method 2: Manual layers, attention_mask=None ===")
with torch.no_grad():
    hidden = hf_model.model.embed_tokens(input_ids)
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    for i in range(24):
        out = hf_model.model.layers[i](
            hidden, attention_mask=None, position_ids=position_ids,
            past_key_value=None, use_cache=False,
        )
        hidden = out[0]
    hidden = hf_model.model.norm(hidden)
    logits2 = hf_model.lm_head(hidden)
argmax2 = torch.argmax(logits2[0, -1, :]).item()
print(f"Logits mean: {logits2.float().mean():.6f}, std: {logits2.float().std():.6f}")
print(f"Argmax: {argmax2} ({tokenizer.decode([argmax2])})")

# Method 3: Full HF model.forward (explicitly)
print("\n=== Method 3: hf_model.model() then lm_head ===")
with torch.no_grad():
    model_out = hf_model.model(input_ids)
    hidden3 = model_out.last_hidden_state
    logits3 = hf_model.lm_head(hidden3)
argmax3 = torch.argmax(logits3[0, -1, :]).item()
print(f"Logits mean: {logits3.float().mean():.6f}, std: {logits3.float().std():.6f}")
print(f"Argmax: {argmax3} ({tokenizer.decode([argmax3])})")

# Compare hidden states
print("\n=== Hidden state comparison ===")
diff_12 = (hidden.float() - hidden3.float()).abs()
print(f"Method 2 vs Method 3 hidden diff: mean={diff_12.mean():.6f}, max={diff_12.max():.6f}")

# Method 4: Manual layers with PROPER causal mask (like HF generates)
print("\n=== Method 4: Manual layers with proper causal mask ===")
seq_len = input_ids.shape[1]
# HF uses a 4D causal mask: [batch, 1, target_len, source_len]
causal_mask = torch.triu(
    torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype),
    diagonal=1,
).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

with torch.no_grad():
    hidden4 = hf_model.model.embed_tokens(input_ids)
    for i in range(24):
        out = hf_model.model.layers[i](
            hidden4, attention_mask=causal_mask, position_ids=position_ids,
            past_key_value=None, use_cache=False,
        )
        hidden4 = out[0]
    hidden4 = hf_model.model.norm(hidden4)
    logits4 = hf_model.lm_head(hidden4)
argmax4 = torch.argmax(logits4[0, -1, :]).item()
print(f"Logits mean: {logits4.float().mean():.6f}, std: {logits4.float().std():.6f}")
print(f"Argmax: {argmax4} ({tokenizer.decode([argmax4])})")

diff_14 = (hidden.float() - hidden4.float()).abs()
print(f"Method 2 vs Method 4 hidden diff: mean={diff_14.mean():.6f}, max={diff_14.max():.6f}")

diff_34 = (hidden3.float() - hidden4.float()).abs()
print(f"Method 3 vs Method 4 hidden diff: mean={diff_34.mean():.6f}, max={diff_34.max():.6f}")

print("\n=== Conclusion ===")
if argmax1 == argmax3:
    print("✅ Method 1 and 3 match (full HF model gives consistent results)")
if argmax2 != argmax1:
    print("❌ Method 2 (manual, no mask) gives wrong result")
if argmax4 == argmax1:
    print("✅ Method 4 (manual, with causal mask) matches HF! Attention mask is the fix!")
else:
    print(f"❌ Method 4 still wrong. HF argmax={argmax1}, Method 4 argmax={argmax4}")
