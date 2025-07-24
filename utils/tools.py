import dataclasses
import torch


# ---------- generic helper ----------
def _move_any(obj, device=None, dtype=None, non_blocking=False):
    """Recursively move every torch.Tensor found in obj to device/dtype."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device=device, dtype=dtype, non_blocking=non_blocking)

    if isinstance(obj, dict):
        return {k: _move_any(v, device, dtype, non_blocking) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_any(v, device, dtype, non_blocking) for v in obj)

    # dataclass (not strictly needed here because we'll handle via dataclasses.replace)
    if dataclasses.is_dataclass(obj):
        # fallback: rebuild generically
        new_vals = {f.name: _move_any(getattr(obj, f.name), device, dtype, non_blocking)
                    for f in dataclasses.fields(obj)}
        return obj.__class__(**new_vals)