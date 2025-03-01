from transformers import AutoConfig, AutoTokenizer
from hf.model import CustomLlamaForCausalLM
from hf.load_config import load_config

def inference():

    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    runtime_config = load_config("config/config_hf.yaml")
    llama_config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer(model_name)
    model = CustomLlamaForCausalLM(llama_config, runtime_config)

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