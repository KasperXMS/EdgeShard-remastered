import gc
import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

login(token=os.getenv("HF_TOKEN"))

model_name = "meta-llama/Llama-3.1-8B-Instruct"
output_dir = "./llama-3.1-8b/"
os.makedirs(output_dir, exist_ok=True)

load_config = {
    "torch_dtype": torch.float16,
    "device_map": "cpu",
    "low_cpu_mem_usage": True
}

try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_config)
    
    print("Saving model...")
    torch.save({
        'state_dict': model.state_dict(),
        'config': model.config,
    }, os.path.join(output_dir, "llama-3.1-8b.pth"))
    
    if os.path.exists(os.path.join(output_dir, "llama-3.1-8b.pth")):
        file_size = os.path.getsize(os.path.join(output_dir, "llama-3.1-8b.pth")) / (1024**3)
        print(f"Model successfully saved to {output_dir}")
        print(f"File size: {file_size:.2f} GB")
    else:
        raise RuntimeError("Model save failed")

except Exception as e:
    print(f"Error occurred: {str(e)}")
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
finally:
    del model
    torch.cuda.empty_cache()
    gc.collect()
