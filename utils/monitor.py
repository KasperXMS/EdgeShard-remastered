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