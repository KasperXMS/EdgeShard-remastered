# chat_cli.py
import readline, traceback
from typing import List, Dict, Optional, Generator
from llama.generation import Llama, StreamChatPrediction, RawMessage
from torch.distributed import rpc

class ChatCLI:
    def __init__(self, model):
        self.model = model
        self.chat_history: List[RawMessage] = []
        self.config = {
            'temperature': 0.6,
            'top_p': 0.9,
            'max_len': 512,
            'stream': True
        }

    def print_colored(self, text: str, color: str = 'green', end: str = "\n"):
        """带颜色的打印输出"""
        colors = {
            'green': '\033[92m',
            'blue': '\033[94m',
            'red': '\033[91m',
            'end': '\033[0m'
        }
        print(f"{colors[color]}{text}{colors['end']}", flush=True, end=end)

    def start_interactive(self):
        """启动交互式聊天会话"""
        self.print_colored("\n=== 聊天模式 (输入 '/help' 查看命令) ===", 'blue')
        self.chat_history.append(RawMessage(role="system", content="You are a helpful assistant."))
        
        while True:
            try:
                # 获取用户输入
                user_input = input("\n\033[94m用户: \033[0m").strip()
                
                # 处理命令
                if user_input.startswith('/'):
                    self.handle_command(user_input)
                    continue
                
                # 添加到历史记录
                self.chat_history.append(RawMessage(role="user", content=user_input))
                
                # 流式生成回复
                self.print_colored("\n助手: ", 'green', end='')
                full_response = []
                stream = self.model.chat_completion_stream(
                    messages=self.chat_history,
                    temperature=self.config['temperature'],
                    top_p=self.config['top_p'],
                    max_gen_len=self.config['max_len']
                )
                
                for stream_result in stream:  # 直接遍历生成器
                    if stream_result.is_final:
                        # 处理最终结果
                        final_content = stream_result.generation.content
                        self.chat_history.append(stream_result.generation)
                        print()  # 换行
                        break
                        
                    # 处理中间结果
                    partial_text = stream_result.generation.content
                    print(partial_text, end='', flush=True)
                    full_response.append(partial_text)

            except KeyboardInterrupt:
                self.print_colored("\n\n检测到中断，输入 '/exit' 退出", 'red')
            except Exception as e:
                error_msg = traceback.format_exc()  # 获取完整traceback
                self.print_colored(f"\n错误详情:\n{error_msg}", 'red')

    def handle_command(self, command: str):
        """处理控制命令"""
        cmd = command[1:].split()
        if not cmd:
            return

        action = cmd[0].lower()
        if action == 'exit':
            self.print_colored("退出聊天...", 'blue')
            rpc.shutdown()
            exit()
        elif action == 'reset':
            self.chat_history = []
            self.print_colored("已重置对话历史", 'blue')
        elif action == 'history':
            self.show_history()
        elif action == 'config':
            self.show_config()
        elif action == 'set':
            self.set_config(cmd[1:])
        elif action == 'help':
            self.show_help()
        else:
            self.print_colored("未知命令", 'red')

    def show_help(self):
        """显示帮助信息"""
        help_text = """
        可用命令:
        /exit       退出程序
        /reset      重置对话历史
        /history    显示对话历史
        /config     显示当前配置
        /set [参数] [值]  修改配置参数
        /help       显示此帮助信息
        
        配置参数:
        - temp: 温度 (0.1-2.0)
        - top_p: 核采样 (0.1-1.0)
        - max_len: 最大生成长度 (10-2048)
        """
        self.print_colored(help_text, 'blue')

    def set_config(self, args: List[str]):
        """修改配置参数"""
        if len(args) != 2:
            self.print_colored("用法: /set [参数] [值]", 'red')
            return

        param, value = args
        try:
            if param == 'temp':
                val = float(value)
                if 0.1 <= val <= 2.0:
                    self.config['temperature'] = val
                else:
                    raise ValueError
            elif param == 'top_p':
                val = float(value)
                if 0.1 <= val <= 1.0:
                    self.config['top_p'] = val
                else:
                    raise ValueError
            elif param == 'max_len':
                val = int(value)
                if 10 <= val <= 2048:
                    self.config['max_len'] = val
                else:
                    raise ValueError
            else:
                raise KeyError
            self.print_colored(f"已更新 {param} = {value}", 'blue')
        except:
            self.print_colored("无效参数或值", 'red')

    def show_config(self):
        """显示当前配置"""
        config_str = "\n".join(
            f"{k}: {v}" for k, v in self.config.items()
        )
        self.print_colored(f"\n当前配置:\n{config_str}", 'blue')

    def show_history(self):
        """显示对话历史"""
        if not self.chat_history:
            self.print_colored("暂无历史记录", 'blue')
            return

        hist_str = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" 
            for msg in self.chat_history
        )
        self.print_colored(f"\n对话历史:\n{hist_str}", 'blue')

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.7,
    top_p: float = 0.7,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    # 初始化分布式模型
    print("正在加载模型...")
    model = Llama.build_distributed(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size
    )
    
    # 启动聊天界面
    cli = ChatCLI(model)
    cli.start_interactive()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--max_batch_size", type=int, default=4)
    args = parser.parse_args()
    
    main(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size
    )