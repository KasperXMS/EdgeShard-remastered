import yaml
from dataclasses import dataclass

@dataclass
class Worker:
    name: str = ""
    ip: str = ""
    interface: str = ""
    start: int = 0
    end: int = 0
    ckpt_path: str = ""

class Config:
    def __init__(self, config_dict):
        self.master = Master(**config_dict['master'])
        # Ensure that workers are processed correctly
        self.workers = []
        for worker_dict in config_dict.get('workers', []):
            self.workers.append(Worker(**worker_dict))

@dataclass
class Master:
    ip: str = ""
    port: int = 0
    interface: str = ""
    lm_head_weight_path: str = ""

# Load the configuration from a YAML file
def load_config(config_file):
    try:
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
            return Config(config_dict)
    except yaml.YAMLError as e:
        print(f"Error loading YAML configuration: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Assuming the YAML content is in a file named 'config.yaml'
    config = load_config('config/config.yaml')
    
    if config is not None:
        # Access master configuration
        print(f"Master IP: {config.master.ip}")
        print(f"Master Port: {config.master.port}")
        print(f"Master Interface: {config.master.interface}")

        # Access worker configurations
        for worker in config.workers:
            print(f"\nWorker name: {worker.name}")
            print(f"Worker IP: {worker.ip}")
            print(f"Interface: {worker.interface}")
            print(f"Start Range: {worker.start}")
            print(f"End Range: {worker.end}")
            print(f"Checkpoint file path: {worker.ckpt_path}")
    else:
        print("Failed to load configuration.")
