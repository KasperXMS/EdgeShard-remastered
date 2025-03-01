from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import streamlit as st
import re
from llama import Llama
from torch.distributed import rpc
import os, yaml


# 在侧边栏中创建一个标题和一个链接
with st.sidebar:
    st.markdown("## DeepSeek-R1-Distill-LLaMA-8B LLM")
    "[开源大模型食用指南 self-llm](https://github.com/datawhalechina/self-llm.git)"
    # 创建一个滑块，用于选择最大长度，范围在 0 到 8192 之间，默认值为 8192（DeepSeek-R1-Distill-Qwen-7B 支持 128K 上下文，并能生成最多 8K tokens，我们推荐设为 8192，因为思考需要输出更多的Token数）
    max_length = st.slider("max_length", 0, 512, 512, step=1)

# 创建一个标题和一个副标题
st.title("💬 DeepSeek R1 Distill Chatbot")
st.caption("🚀 A streamlit chatbot powered by Self-LLM")

# 定义模型路径
mode_name_or_path = '/root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'

# 文本分割函数
def split_text(text):
    pattern = re.compile(r'<think>(.*?)</think>(.*)', re.DOTALL) # 定义正则表达式模式
    match = pattern.search(text) # 匹配 <think>思考过程</think>回答
  
    if match: # 如果匹配到思考过程
        think_content = match.group(1).strip() # 获取思考过程
        answer_content = match.group(2).strip() # 获取回答
    else:
        think_content = "" # 如果没有匹配到思考过程，则设置为空字符串
        answer_content = text.strip() # 直接返回回答
  
    return think_content, answer_content

# # 定义一个函数，用于获取模型和 tokenizer
# @st.cache_resource
# def get_model():
#     # 从预训练的模型中获取 tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path, trust_remote_code=True)
#     tokenizer.pad_token = tokenizer.eos_token
#     # 从预训练的模型中获取模型，并设置模型参数
#     model = AutoModelForCausalLM.from_pretrained(mode_name_or_path, torch_dtype=torch.bfloat16,  device_map="auto")

#     return tokenizer, model

# # 加载 Qwen2.5 的 model 和 tokenizer
# tokenizer, model = get_model()

@st.cache_resource
def get_generator(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.3,
    top_p: float = 0.15,
    max_seq_len: int = 2048,
    max_batch_size: int = 1,
    max_gen_len: int = 256,
    ):
    
    config = {}
    with open("config/config.yaml", 'r') as file:
        config = yaml.safe_load(file)

    os.environ['GLOO_SOCKET_IFNAME'] = config['master']['interface']
    os.environ['TP_SOCKET_IFNAME'] = config['master']['interface']
    os.environ['MASTER_ADDR'] = config['master']['ip']
    os.environ['MASTER_PORT'] = config['master']['port']

    rpc.init_rpc("master", rank=0, world_size=len(config['worker'])+1, 
                 rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                rpc_timeout=120))

    generator = Llama.build_distributed(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    return generator

generator = get_generator(
        ckpt_dir="./model_shards",
        tokenizer_path="./llama_3.1_8b_tokenizer/tokenizer.model",
)

# 如果 session_state 中没有 "messages"，则创建一个包含默认消息的列表
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# 遍历 session_state 中的所有消息，并显示在聊天界面上
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 如果用户在聊天输入框中输入了内容，则执行以下操作
if prompt := st.chat_input():

    # 在聊天界面上显示用户的输入
    st.chat_message("user").write(prompt)

    # 将用户输入添加到 session_state 中的 messages 列表中
    st.session_state.messages = [{"role": "user", "content": prompt}]

    print(st.session_state.messages)

    results = generator.chat_completion(
        [st.session_state.messages], max_gen_len=256
    )
    for result in results:
        response = result['generation']['content']
        think_content, answer_content = split_text(response) # 调用split_text函数，分割思考过程和回答
        # 将模型的输出添加到 session_state 中的 messages 列表中
        st.session_state.messages.append({"role": "assistant", "content": response})
    # 在聊天界面上显示模型的输出
    with st.expander("模型思考过程"):
        st.write(think_content) # 展示模型思考过程
    st.chat_message("assistant").write(answer_content) # 输出模型回答
    # print(st.session_state) # 打印 session_state 调试 
