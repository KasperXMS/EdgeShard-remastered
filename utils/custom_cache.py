from transformers.cache_utils import DynamicCache
from typing import List, Tuple
import torch

class CustomDynamicCache(DynamicCache):
    """
    A custom DynamicCache that can be cleared without being destroyed.
    """
    def clear(self):
        """
        Clears the cache content for all layers.
        """
        for i in range(len(self.key_cache)):
            self.key_cache[i] = torch.tensor([], device=self.key_cache[i].device, dtype=self.key_cache[i].dtype)
            self.value_cache[i] = torch.tensor([], device=self.value_cache[i].device, dtype=self.value_cache[i].dtype)
        # Or more aggressively:
        self.key_cache.clear()
        self.value_cache.clear()
        self._seen_tokens = 0

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache 