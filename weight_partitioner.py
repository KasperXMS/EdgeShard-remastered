import torch
from collections import OrderedDict

def partition_model_weights(state_dict, layer_ranges):
    """
    Partitions the model's state dictionary into multiple shards based on layer ranges.

    Args:
        state_dict (OrderedDict): The model's state dictionary.
        layer_ranges (list): List of layer indices defining the ranges for each shard.
                            For example, [0, 8, 16, 24, 32] means:
                            - Shard 0: Layers 0 to 7 + embedding layer
                            - Shard 1: Layers 8 to 15
                            - Shard 2: Layers 16 to 23
                            - Shard 3: Layers 24 to 31 + norm and output layer

    Returns:
        list: A list of shards, where each shard is an OrderedDict containing a subset of the state dictionary.
    """
    shards = []
    for i in range(len(layer_ranges) - 1):
        start_layer = layer_ranges[i]
        end_layer = layer_ranges[i + 1]
        shard = OrderedDict()

        for key, value in state_dict.items():
            if "model.layers." in key:
                layer_index = int(key.split("model.layers.")[1].split(".")[0])
                if start_layer <= layer_index < end_layer:
                    shard[key] = value

            elif i == 0 and "model.embed_tokens" in key:
                shard[key] = value

            elif i == len(layer_ranges) - 2:
                if "model.norm" in key or "lm_head" in key:
                    shard[key] = value

        shards.append(shard)
    return shards

def save_shards(shards, output_dir):
    """
    Saves each shard to a separate file.

    Args:
        shards (list): List of shards (OrderedDicts).
        output_dir (str): Directory to save the shard files.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    for i, shard in enumerate(shards):
        shard_path = os.path.join(output_dir, f"shard_{i}.pth")
        torch.save(shard, shard_path)
        print(f"Saved shard {i} to {shard_path}")

def load_shards(shard_paths):
    """
    Loads shards from files and combines them into a single state dictionary.

    Args:
        shard_paths (list): List of paths to shard files.

    Returns:
        OrderedDict: The combined state dictionary.
    """
    state_dict = OrderedDict()
    for shard_path in shard_paths:
        shard = torch.load(shard_path)
        state_dict.update(shard)
    return state_dict

# Example usage
if __name__ == "__main__":
    # Load the model's state dictionary
    state_dict = torch.load("llama-3.1-8b/llama-3.1-8b.pth")
    print(state_dict.keys())

    # Define layer ranges for partitioning
    layer_ranges = [0, 16, 32]

    # Partition the state dictionary
    shards = partition_model_weights(state_dict, layer_ranges)

    # Save the shards
    save_shards(shards, "model_shards")

    # Load and reconstruct the model
    shard_paths = [f"model_shards/shard_{i}.pth" for i in range(len(shards))]
    reconstructed_state_dict = load_shards(shard_paths)

    # Verify that the reconstructed state dictionary matches the original
    assert state_dict.keys() == reconstructed_state_dict.keys(), "State dictionaries do not match!"
    print("Model partitioning and reconstruction successful!")