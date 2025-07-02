import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

# 设置全局样式
plt.style.use('seaborn')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True

def load_data(log_path):
    """加载并预处理日志数据"""
    df = pd.read_csv(log_path, parse_dates=['timestamp'])
    
    # 计算衍生指标
    df['throughput'] = 1 / df['elapsed_sec'].diff().fillna(0)
    df['nvidia_smi_gb'] = df['nvidia_smi_mb'] / 1024
    df['cumulative_tokens'] = df['token_count'].cumsum()
    
    # 过滤异常值
    df = df[df['throughput'] < 100]  # 移除初始极高吞吐量
    
    return df

def create_combined_plot(df, output_path):
    """创建组合图表并保存为PNG"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    
    # 第一个子图：内存使用情况
    ax1.plot(df['cumulative_tokens'], df['torch_allocated_gb'], 
             label='Torch Allocated', color='#1f77b4', linewidth=2)
    ax1.plot(df['cumulative_tokens'], df['torch_reserved_gb'], 
             label='Torch Reserved', color='#ff7f0e', linestyle='--')
    ax1.plot(df['cumulative_tokens'], df['nvidia_smi_gb'], 
             label='GPU Used', color='#2ca02c', alpha=0.8)
    ax1.set_ylabel('Memory (GB)')
    ax1.legend(loc='upper left')
    ax1.set_title('GPU Memory Usage During Text Generation', pad=20)
    
    # 第二个子图：吞吐量和峰值内存
    ax2.plot(df['cumulative_tokens'], df['throughput'], 
             label='Throughput', color='#d62728', alpha=0.7)
    ax2.fill_between(df['cumulative_tokens'], df['throughput'], 
                    color='#d62728', alpha=0.1)
    ax2.set_xlabel('Cumulative Generated Tokens')
    ax2.set_ylabel('Tokens/sec')
    
    # 添加峰值内存作为第二条y轴
    ax2b = ax2.twinx()
    ax2b.plot(df['cumulative_tokens'], df['torch_peak_gb'], 
              label='Peak Memory', color='#9467bd', linestyle=':')
    ax2b.set_ylabel('Peak Memory (GB)')
    
    # 合并图例
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    
    # 格式调整
    for ax in [ax1, ax2]:
        ax.grid(alpha=0.2)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {output_path}")

def create_individual_plots(df, output_prefix):
    """创建并保存单独的图表"""
    # 1. 内存使用趋势图
    plt.figure(figsize=(12, 6))
    plt.plot(df['cumulative_tokens'], df['torch_allocated_gb'], label='Allocated')
    plt.plot(df['cumulative_tokens'], df['torch_reserved_gb'], label='Reserved')
    plt.plot(df['cumulative_tokens'], df['nvidia_smi_gb'], label='GPU Used')
    plt.xlabel('Cumulative Tokens')
    plt.ylabel('Memory (GB)')
    plt.title('Memory Usage Over Time')
    plt.legend()
    plt.savefig(f"{output_prefix}_memory.png", dpi=300)
    
    # 2. 吞吐量分析图
    plt.figure(figsize=(12, 6))
    plt.plot(df['cumulative_tokens'], df['throughput'], color='red', alpha=0.7)
    plt.fill_between(df['cumulative_tokens'], df['throughput'], color='red', alpha=0.1)
    plt.xlabel('Cumulative Tokens')
    plt.ylabel('Tokens/sec')
    plt.title('Generation Throughput')
    plt.savefig(f"{output_prefix}_throughput.png", dpi=300)
    
    # 3. 峰值内存分析
    plt.figure(figsize=(12, 6))
    plt.plot(df['cumulative_tokens'], df['torch_peak_gb'], color='purple')
    plt.xlabel('Cumulative Tokens')
    plt.ylabel('Peak Memory (GB)')
    plt.title('Peak Memory Usage')
    plt.savefig(f"{output_prefix}_peak_memory.png", dpi=300)

if __name__ == "__main__":
    input_csv = "memory_usage.csv" 
    output_combined = "generation_metrics.png"
    output_prefix = "generation"
    
    df = load_data(input_csv)
    
    create_combined_plot(df, output_combined)
    create_individual_plots(df, output_prefix)