import numpy as np
import time
#########################################################
#              Nodes and Bandwidth Settings            #
#########################################################
num_layers = 34

num_nodes = 6

# Mbps
bandwidth = np.array([[-1, 10, 10, 10, 10, 10], [10, -1, 10, 10, 10, 10], [10, 10, -1, 10, 10, 10], [10, 10, 10, -1, 10, 10], [10, 10, 10, 10, -1, 10], [10, 10, 10, 10, 10, -1]])

# data_size
data_size = 4*4096*4*8 # bit

# device execution time of each layer
# device_1: Orinnx; device_2: AGX Orin; device_3: rtx3090;
# ms
device_1 = [0.476, 5.759, 6.402, 6.133, 6.333, 5.662, 6.071, 6.056, 5.691, 6.029, 5.955, 6.012, 6.09, 6.15, 5.979, 6.095, 5.981, 5.981, 5.84, 6.074, 6.218, 6.052, 5.625, 5.985, 5.912, 6.145, 6.063, 6.027, 6.082, 6.067, 6.019, 6.016, 6.093, 0.594 + 4.888]
device_2 = [0.627, 5.498, 5.48, 5.454, 5.402, 5.376, 5.36, 5.439, 5.357, 5.261, 5.207, 5.25, 5.276, 5.27, 5.334, 5.242, 5.288, 5.226, 5.22, 5.16, 5.211, 5.206, 5.167, 5.168, 5.158, 5.229, 5.23, 5.164, 5.199, 5.272, 5.201, 5.304, 5.255, 0.885 + 3.597]
device_3 = [0.077, 0.766, 0.819, 0.797, 0.804, 0.798, 0.784, 0.801, 0.782, 0.779, 0.832, 0.799, 0.793, 0.788, 0.79, 0.78, 0.783, 0.761, 0.815, 0.8, 0.797, 0.788, 0.784, 0.779, 0.781, 0.817, 0.818, 0.798, 0.783, 0.785, 0.783, 0.783, 0.782, 0.201 + 0.447]
# device_4 = []

#device_1 = [i*0.1 for i in device_1]

node_frequency = ['device_2'] * 4 + ['device_3'] * 1

node_memories = np.array([29, 29, 29, 29, 24], dtype=np.float32)
layer_memories = np.array([0.488, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 0.754, 1.526e-05 + 0.488])

#########################################################
#                      Time function                    #
#########################################################
def layer_device_latency(frequency, layer_i):
	if frequency == 'device_1':
		return device_1[layer_i]
	elif frequency == 'device_2':
		return device_2[layer_i]
	elif frequency == 'device_3':
		return device_3[layer_i]
	else:
		print("Not Found in dataset")

def device_to_latency(node, frequency, num_layers,start_layer, end_layer, use_latency=False, enable_print=False):
	latency = 0
	if node == 0 and start_layer > 0:
		for layer_i in range(start_layer):
			latency += layer_device_latency(frequency, layer_i)
			
	if use_latency == True:
		for layer_i in range(start_layer, end_layer):
			latency += layer_device_latency(frequency, layer_i)
			if enable_print== True:
				print(f"from {start_layer} to {end_layer}, current {layer_i}  frequency is{frequency} ,latency is {latency}")
	
	return latency

def communication_time_between_device(bandwidth = 10000000, data_size = 442368):
	if bandwidth < 0:
		return 0
	return data_size/1000/bandwidth + 2.1# ms




def communication_time_between_id(src, dst, data_size=442368):
	bandwidth_src_dst = bandwidth[int(src)][int(dst)]
	return communication_time_between_device(bandwidth_src_dst, data_size)


#########################################################
#                      DP Solution                     #
#########################################################
time_start = time.time()
mask = 2 ** len(node_frequency)

h = [[[-1 for k in range(num_nodes)] for j in range(mask)] for i in range(num_layers + 1)]

print(bin(mask))
# print(mask)
h[1][1][0] = layer_device_latency('device_1', 0)
parent = {}
parent[(1, 1, 0)] = (0, 0, 0)

node_memories[0] = node_memories[0] - layer_memories[0]

for i in range(1,num_layers):
	print("layer = ", i)
	for S in range(mask):
		if (S & 1) == 1:
			print("S = ", S)
		else:
			continue
		
		for last_used in range(num_nodes):
			current_cost = h[i][S][last_used]
			print(" last h[%d][%d][%d] = %f" %( i, S, last_used, current_cost))
			if current_cost < 0:
				continue
			for j in range(i, num_layers):
				for node in range(len(node_frequency)):
					if node_memories[node] < np.sum(layer_memories[i:j+1]):
						continue
					if (S >> node & 1): ## 1 if used, if node in S
						if S == 1:
							pass
						else:
							continue
					# try use "node" to handle the layers [i, j]
					# frequency, num_layers,start_layer, end_layer, use_latency=False
					next_cost = max(current_cost, device_to_latency(node, node_frequency[node],0, i,j+1,use_latency=True), (communication_time_between_id(last_used, node, data_size) if i > 0 else 0))
					
					
					if S == 1 and node == 0:
						remained = S
					else:
						remained = (S ^ (1 << node)) # flip the node since it was use, add node into S 
						
					if h[j + 1][remained][node] < 0 or next_cost < h[j + 1][remained][node]:
						h[j + 1][remained][node] = next_cost
						# print(f"now node is {node}, last_used is {last_used} \n")
						print("h[%d][%d][%d] = %f" %(j+1, remained, node, h[j + 1][remained][node]))
						if node == 0 and S == 1:
							parent[(j + 1, remained, node)] = (0, node, last_used)
							print("p[%d][%d][%d] = (%d, %d, %d)" %( j+1, remained, node, 0, node, last_used))
						else:
							parent[(j + 1, remained, node)] = (i, node, last_used)
							print("p[%d][%d][%d] = (%d, %d, %d)" %( j+1, remained, node, i, node, last_used))
				print('\n')

#########################################################
#                      Print answer                     #
#########################################################
# print("h is ", h)
# print("parent is", parent)


answer = (-1, -1, -1)
for i in range(mask):
	for j in range(num_nodes):
		if h[num_layers][i][j] >= 0:
			pair = (h[num_layers][i][j], i, j)
			answer = pair if answer[0] < 0 else min(answer, pair)
print("\n\nThe minimum latency is %f ms\n\n" % answer[0])

print(answer)

layer, s, last_used = (num_layers, answer[1], answer[2])
while layer > 0:
	p = parent[(layer, s, last_used)]
	print("layers in [%d , %d], calculated by node %d with frequency %s,\nbandwidth %d between %d and %d -> latency %f" 
		% (p[0], layer - 1, p[1], node_frequency[p[1]], bandwidth[p[1]][p[2]], p[1], p[2], 
		communication_time_between_id(p[1], p[2], data_size)))
	# print("latency for node %d is %f\n"%(p[1], device_to_latency(node_frequency[p[1]] ,layer - p[0] )) )
	print("latency for node %d is %f\n"%(p[1], device_to_latency(p[1], node_frequency[p[1]], 0, p[0], layer, use_latency=True,enable_print=True)) )
	layer = p[0]
	s -= (1 << p[1])
	last_used = p[2]

time_end = time.time()
print("program consumes ", time_end - time_start)