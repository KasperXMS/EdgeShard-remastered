import csv
import subprocess
import time
import torch
from datetime import datetime

class MemoryMonitor:
    def __init__(self):
        self.records = []
        self.start_time = time.time()
        
    def get_nvidia_smi_memory(self):
        try:
            result = subprocess.check_output([
                'nvidia-smi', 
                '--query-gpu=memory.used',
                '--format=csv,nounits,noheader'
            ])
            return int(result.decode().strip())
        except:
            return 0
            
    def record_memory(self, token_count):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'elapsed_sec': round(time.time() - self.start_time, 2),
            'token_count': token_count,
            'torch_allocated_gb': round(torch.cuda.memory_allocated()/1e9, 4),
            'torch_reserved_gb': round(torch.cuda.memory_reserved()/1e9, 4),
            'torch_peak_gb': round(torch.cuda.max_memory_allocated()/1e9, 4),
            'nvidia_smi_mb': self.get_nvidia_smi_memory(),
        }
        self.records.append(entry)
        
    def save_to_csv(self, filename='memory_usage.csv'):
        if not self.records:
            return
            
        keys = self.records[0].keys()
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.records)

class LatencyMonitor:
    """
    For each entry in the latency list, it is also a list contains:
    [0]: the timestamp before calling model.forward()
    [2n-1]: start timestamp of shard n's inference
    [2n]: end timestamp of shard n's inference
    [last]: the timestamp after calling model.forward()
    """
    def __init__(self):
        self.entries = []

    def append_entry(self, entry):
        self.entries.append(entry)

    def report_avg_latency(self):
        overall_latencies = []
        for entry in self.entries:
            total_latency = entry[-1] - entry[0]
            shard_latencies = [(entry[i+1] - entry[i]) for i in range(1, len(entry)-1, 2)]
            shard_latencies.append(total_latency)
            overall_latencies.append(shard_latencies)

        
        # calculate average for each column, which stands for shard 1, shard 2,..., shard n and the final as total
        # output in dictionary
        avg_latencies = [sum(latency) / len(overall_latencies) for latency in zip(*overall_latencies)]
        latency_dict = {f"shard_{i+1}": lat for i, lat in enumerate(avg_latencies)}
        print(f"Average Latencies: {latency_dict}")

    def save_to_csv(self, filename='latency_report.csv'):
        """
        save the latency by each entry
        """
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = [f"shard_{i+1}" for i in range(len(self.entries[0]) // 2 - 1)] + [f"comm_{i+1}" for i in range(len(self.entries[0]) // 2 - 1)] + ["lm_head", "total"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for entry in self.entries:
                total_latency = entry[-1] - entry[0]
                lm_head_latency = entry[-2] - entry[-3] if len(entry) > 2 else 0
                shard_latencies = [(entry[i+1] - entry[i]) for i in range(1, len(entry)-2, 2)]
                comm_latencies = [(entry[i+1] - entry[i]) for i in range(2, len(entry)-1, 2)]
                shard_num = len(shard_latencies)
                comm_num = len(comm_latencies)
                shard_latencies.extend(comm_latencies)
                shard_latencies.append(total_latency)
                row = {f"shard_{i+1}": shard_latencies[i] for i in range(shard_num)}
                row.update({f"comm_{i+1}": comm_latencies[i] for i in range(comm_num)})
                row["lm_head"] = lm_head_latency
                row["total"] = total_latency
                writer.writerow(row)
