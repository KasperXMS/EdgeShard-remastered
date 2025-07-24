import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from transformers import AutoConfig, AutoTokenizer, TextIteratorStreamer
import threading, torch, sys, gc
from models.llama_model import CustomLlamaForCausalLM
from core.load_config import load_config
from torch.distributed import rpc
from contextlib import contextmanager
import numpy as np

@contextmanager
def memory_management_ctx():
    """Enhanced memory management context"""
    try:
        torch.cuda.synchronize()
        yield
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

class ConversationManager:
    def __init__(self, tokenizer, max_history=3, max_length=1024):
        self.tokenizer = tokenizer
        self.max_history = max_history
        self.max_length = max_length
        self.history = []
        self.last_user_message = ""
        self.last_bot_message = ""
        
    def add_message(self, role, content):
        content = content.strip()
        
        if not content or (role == "user" and content == self.last_user_message):
            return False
            
        self.history.append({"role": role, "content": content})
        
        if role == "user":
            self.last_user_message = content
        else:
            self.last_bot_message = content
            
        self._trim_history()
        return True

    def get_current_history(self):
        self._trim_history()
        return self.history.copy()
    
    def _trim_history(self):
        while len(self.history) > self.max_history * 2:
            self.history.pop(0)
            
        while True:
            prompt = self.tokenizer.apply_chat_template(
                self.history,
                add_generation_prompt=True,
                tokenize=False
            )
            
            if len(self.history) <= self.max_history * 2:
                break
                
            self.history.pop(0)

def inference():
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"   
    runtime_cfg = load_config("config/config_hf.yaml")

    with memory_management_ctx():
        cfg = AutoConfig.from_pretrained(
            model_name,
            max_position_embeddings=1024,
            use_cache=True,
        )
        tok = AutoTokenizer.from_pretrained(model_name)
        model = CustomLlamaForCausalLM(cfg, runtime_cfg).eval().half()
        pad_id = tok.pad_token_id or tok.eos_token_id
        
        # Warm up model
        with torch.no_grad():
            dummy_input = tok("warmup", return_tensors="pt").to("cuda:0")
            _ = model.generate(**dummy_input, max_new_tokens=1)
            del dummy_input

    conv_manager = ConversationManager(tok, max_history=3)
    print("输入 '/exit' 结束对话")
    
    while True:
        try:
            user_input = input("用户: ").strip()
            if user_input.lower() == "/exit":     
                model.post_process()
                model.memory_monitor.save_to_csv()
                rpc.shutdown()
                sys.exit(0)

            # Add user message and check if it's valid
            if not conv_manager.add_message("user", user_input):
                continue
                
            current_history = conv_manager.get_current_history()
            
            with memory_management_ctx():
                # Clear all caches before new generation
                if hasattr(model, 'reset_cache'):
                    model.reset_cache()
                elif hasattr(model.model, 'reset_kv_cache'):
                    model.model.reset_kv_cache()
                torch.cuda.empty_cache()
                gc.collect()

                # Create prompt with proper attention mask
                prompt_ids = tok.apply_chat_template(
                    current_history,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to("cuda:0")
                
                attention_mask = (prompt_ids != pad_id).to("cuda:0")
                
                streamer = TextIteratorStreamer(
                    tok,
                    skip_special_tokens=True,
                    skip_prompt=True,
                    clean_up_tokenization_spaces=True
                )

                gen_kwargs = {
                    "input_ids": prompt_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": 512,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "repetition_penalty": 1.1,
                    "do_sample": True,
                    "pad_token_id": pad_id,
                    "eos_token_id": tok.eos_token_id,
                    "streamer": streamer,
                }

                # Start generation in a thread
                thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
                thread.start()

                print("模型: ", end="", flush=True)
                reply_text = ""
                for new_chunk in streamer:
                    print(new_chunk, end="", flush=True)
                    reply_text += new_chunk

                # Wait for generation to complete
                thread.join()
                print("\n")
                
                # Only add to history after successful generation
                conv_manager.add_message("assistant", reply_text.strip())

        except KeyboardInterrupt:
            print("\nCleaning up...")
            model.memory_monitor.save_to_csv()
            model.post_process()
            rpc.shutdown()
            sys.exit(0)
        except torch.cuda.OutOfMemoryError:
            print("\n内存不足，正在清理...")
            conv_manager.history = conv_manager.history[-2:]  # Keep only last exchange
            torch.cuda.empty_cache()
            gc.collect()
            continue
        except Exception as e:
            print(f"\n发生错误: {str(e)}")
            torch.cuda.empty_cache()
            continue

if __name__ == "__main__":
    inference()