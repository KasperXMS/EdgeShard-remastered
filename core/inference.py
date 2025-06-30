from transformers import (
    AutoConfig, AutoTokenizer, TextIteratorStreamer
)
import threading, torch, sys
from models.llama_model import CustomLlamaForCausalLM
from core.load_config import load_config
from torch.distributed import rpc

def inference():
    model_name   = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"   
    runtime_cfg  = load_config("config/config_hf.yaml")

    cfg  = AutoConfig.from_pretrained(model_name,
                                      torch_dtype=torch.float16,
                                      use_cache=True)
    tok  = AutoTokenizer.from_pretrained(model_name)

    model  = CustomLlamaForCausalLM(cfg, runtime_cfg).to("cuda:0").eval()
    pad_id = tok.pad_token_id or tok.eos_token_id     

    print("输入 '/exit' 结束对话")
    history = []                                          
    while True:
        user_input = input("用户: ")
        if user_input.lower() == "/exit":
            rpc.shutdown(); sys.exit(0)

        history.append({"role": "user", "content": user_input})
        prompt_ids = tok.apply_chat_template(
            history,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda:0")                                  

        streamer = TextIteratorStreamer(
            tok, 
            skip_special_tokens=False,
            skip_prompt=True,
            clean_up_tokenization_spaces=True
        )

        gen_kwargs = dict(
            input_ids        = prompt_ids,
            max_new_tokens   = 256,
            temperature      = 0.6,
            top_p            = 0.9,
            repetition_penalty = 1.5,
            do_sample        = True,
            pad_token_id     = pad_id,
            eos_token_id     = tok.eos_token_id,
            streamer         = streamer,
        )
        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        print("模型: ", end="", flush=True)
        reply_text = ""
        for new_chunk in streamer:                       
            print(new_chunk, end="", flush=True)
            reply_text += new_chunk

        thread.join(); print("\n")
        history.append({"role": "assistant", "content": reply_text.strip()})
