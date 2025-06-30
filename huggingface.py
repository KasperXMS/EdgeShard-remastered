import gc
import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

login("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")

# # Replace with the correct model name
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# # Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Load the model
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# # Save the model's state dictionary
output_path = "./llama-3.1-8b/"
torch.save(model.state_dict(), output_path + "llama-3.1-8b.pth")

print(f"Model saved to {output_path}")

# del model
# gc.collect()

# Load the state dictionary back (for verification)
# state_dict = torch.load(output_path)

# Print the keys in the state dictionary
# print("State dictionary keys:", state_dict.keys())

# tokenizer_path = "llama_3.1_8b_tokenizer"
# tokenizer.save_pretrained(tokenizer_path)

# print(f"Tokenizer saved to {tokenizer_path}")