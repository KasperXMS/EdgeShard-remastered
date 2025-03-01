from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria
import torch
from huggingface_hub import login

login("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")

# 加载模型和分词器
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,  # 使用BF16精度节省内存
    device_map="auto"            # 自动分配设备（GPU/CPU）
)

# 设置模型为评估模式
model.eval()

# 交互循环
print("输入 'exit' 结束对话")
while True:
    user_input = input("用户: ")
    if user_input.lower() == "exit":
        break
    
    # 编码输入并生成回复
    inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.4,
        top_p=0.6,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,  # 显式设置
        eos_token_id=tokenizer.eos_token_id,  # 明确结束符
    )
    
    # 解码并打印回复
    response = tokenizer.decode(outputs[0])
    print(f"模型: {response[len(user_input):]}\n")