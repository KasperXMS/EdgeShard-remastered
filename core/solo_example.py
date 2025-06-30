from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria
import torch
from huggingface_hub import login

login("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")

# 加载模型和分词器
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,  
    device_map="auto"            
)

model.eval()

print("输入 'exit' 结束对话")
while True:
    user_input = input("用户: ")
    if user_input.lower() == "exit":
        break
    
    inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
    prompt_length = inputs.input_ids.shape[1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.4,
        top_p=0.6,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,  
        eos_token_id=tokenizer.eos_token_id,  
    )
    
    response = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    print(f"模型: {response[len(user_input):]}\n")