import numpy as np
np.random.seed(0)

num_layers = 34  # 神经网络层数
num_nodes = 5  # 计算节点数

# 节点的计算时间和内存容量
# device_1: Orinnx; device_2: AGX Orin; device_3: rtx3090;
device_1 = [0.476, 5.759, 6.402, 6.133, 6.333, 5.662, 6.071, 6.056, 5.691, 6.029, 5.955, 6.012, 6.09, 6.15, 5.979, 6.095, 5.981, 5.981, 5.84, 6.074, 6.218, 6.052, 5.625, 5.985, 5.912, 6.145, 6.063, 6.027, 6.082, 6.067, 6.019, 6.016, 6.093, 0.594 + 4.888]
device_2 = [0.627, 5.498, 5.48, 5.454, 5.402, 5.376, 5.36, 5.439, 5.357, 5.261, 5.207, 5.25, 5.276, 5.27, 5.334, 5.242, 5.288, 5.226, 5.22, 5.16, 5.211, 5.206, 5.167, 5.168, 5.158, 5.229, 5.23, 5.164, 5.199, 5.272, 5.201, 5.304, 5.255, 0.885 + 3.597]
device_3 = [0.077, 0.766, 0.819, 0.797, 0.804, 0.798, 0.784, 0.801, 0.782, 0.779, 0.832, 0.799, 0.793, 0.788, 0.79, 0.78, 0.783, 0.761, 0.815, 0.8, 0.797, 0.788, 0.784, 0.779, 0.781, 0.817, 0.818, 0.798, 0.783, 0.785, 0.783, 0.783, 0.782, 0.201 + 0.447]

node_compute_times = np.array([device_1, device_1, device_2, device_2, device_3])

node_memories = np.array([14, 14, 29, 29, 22], dtype=np.float32) #要提前查看可用内存，设备可用内存小于额定内存

# 每层的内存需求，GB
layer_memories = np.array([0.488, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 1.526e-05 + 0.488])

# 层之间数据传输大小
data_size = 4*4096*4*8 # bit
output_size = 4*32000*4*8

# 网络配置 Mbps
bandwidth = np.array([[-1, 950, 950, 950, 10], [950, -1, 950, 950, 950], [950, 950, -1, 950, 950], [950, 950, 950, -1, 950], [10, 950, 950, 950, -1]])

# 数据传输时间计算函数 ms
def data_transfer_time(src, dst, data_size):
    bandwidth_src_dst = bandwidth[int(src)][int(dst)]
    if bandwidth_src_dst < 0:
        return 0
    return data_size/1000/bandwidth_src_dst + 2.1# ms  

# 初始化动态规划表
dp = np.full((num_layers, num_nodes), float('inf'))
choices = np.full((num_layers, num_nodes), None, dtype=object)  # 记录选择和内存状态


# # 更新每层的初始状态
# for j in range(num_nodes):
#     if node_memories[j] >= layer_memories[0]:
#         dp[0][j] = node_compute_times[j][0]
#         nodes_memories_tmp = node_memories.copy()
#         nodes_memories_tmp[j] = nodes_memories_tmp[j] - layer_memories[0]
#         choices[0][j] = (j, nodes_memories_tmp)
#         print("choices[{}][{}]: ({},{})".format(0, j, j, nodes_memories_tmp))

#强制为起始节点
dp[0][0] = node_compute_times[0][0]
nodes_memories_tmp = node_memories.copy()
nodes_memories_tmp[0] = nodes_memories_tmp[0] - layer_memories[0]
for j in range(num_nodes):  
    choices[0][j] = (j, nodes_memories_tmp)
    print("choices[{}][{}]: ({},{})".format(0, j, j, nodes_memories_tmp))

# 动态规划填表
for i in range(1, num_layers):
    for j in range(num_nodes):
        if i == 1:
            k = 0 # 强制为起始节点
            if choices[i-1][k] is None or choices[i-1][k][1][j] < layer_memories[i]: # 检查节点内存是否足够当前层
                continue
            transfer_time = data_transfer_time(k, j, data_size) if k != j else 0
            total_time = dp[i-1][k] + node_compute_times[j][i] + transfer_time
            if total_time < dp[i][j]:
                dp[i][j] = total_time
                tmp = choices[i-1][k][1].copy()
                tmp[j] = tmp[j] - layer_memories[i]
                choices[i][j] = (k, tmp) # 更新选择和内存状态
                print("choices[{}][{}]: ({},{})".format(i, j, k, tmp))
            continue
        for k in range(num_nodes):
            if choices[i-1][k] is None or choices[i-1][k][1][j] < layer_memories[i]: # 检查节点内存是否足够当前层
                continue
            if i == (num_layers - 1): # 回传到第一个节点的时间考虑
                transfer_time = data_transfer_time(k, j, data_size) if k != j else 0
                transfer_back_time = data_transfer_time(j, 0, output_size) if j != 0 else 0
                transfer_time = transfer_back_time + transfer_time
            else:
                transfer_time = data_transfer_time(k, j, data_size) if k != j else 0
            
            total_time = dp[i-1][k] + node_compute_times[j][i] + transfer_time
            if total_time < dp[i][j]:
                dp[i][j] = total_time
                tmp = choices[i-1][k][1].copy()
                tmp[j] = tmp[j] - layer_memories[i]
                choices[i][j] = (k, tmp) # 更新选择和内存状态
                print("choices[{}][{}]: ({},{})".format(i, j, k, tmp))
          
                
                


final_times = dp[-1]

min_time = np.min(final_times)
min_node = np.argmin(final_times)

# 回溯以找出分配方案
path = [min_node]
for i in range(num_layers - 1, 0, -1):
    min_node = choices[i][min_node][0]
    path.append(min_node)

# 输出最优路径和最小总执行时间
path.reverse()
print("Optimal path for processing layers and returning output:")
for layer, node in enumerate(path):
    print(f"Layer {layer} -> Node {node}")
print(f"Minimum total execution time including return: {min_time} miliseconds")
