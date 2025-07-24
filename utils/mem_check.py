import gc, torch
from collections import defaultdict

def list_gpu_tensors(top_n: int = 50):
    tensors = []
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and obj.is_cuda:
            size_mb = obj.element_size() * obj.numel() / 1024**2
            tensors.append((size_mb, tuple(obj.size()), obj.dtype))
            
    for size_mb, shape, dtype in sorted(tensors, key=lambda x: -x[0])[:top_n]:
        print(f"{size_mb:8.2f} MB  |  shape={shape}  |  dtype={dtype}")


def aggregate_gpu_tensors(top_n: int = None):
    # stats: {(shape, dtype): {'count': int, 'mem': float}}
    stats = defaultdict(lambda: {'count': 0, 'mem': 0.0})
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and obj.is_cuda:
            shape = tuple(obj.size())
            dtype = str(obj.dtype)
            size_mb = obj.element_size() * obj.numel() / 1024**2
            key = (shape, dtype)
            stats[key]['count'] += 1
            stats[key]['mem']   += size_mb

    sorted_stats = sorted(stats.items(), key=lambda x: -x[1]['mem'])
    if top_n is not None:
        sorted_stats = sorted_stats[:top_n]

    print(f"{'Total MB':>10s} | {'Count':>5s} | {'Shape':>20s} | {'Dtype'}")
    print("-" * 60)
    for (shape, dtype), info in sorted_stats:
        print(f"{info['mem']:10.2f} | {info['count']:5d} | {str(shape):20s} | {dtype}")

def find_tensor_referrers(tensor, print_limit=5):
    """
    打印给定 tensor 的所有引用者，帮助定位内存泄漏。
    Args:
        tensor: 需要追踪的 torch.Tensor
        print_limit: 最多打印多少个引用对象
    """
    import gc
    refs = gc.get_referrers(tensor)
    print(f"\n[find_tensor_referrers] Tensor shape: {tuple(tensor.size())}, dtype: {tensor.dtype}, id: {id(tensor)}")
    print(f"找到 {len(refs)} 个引用者：")
    for i, ref in enumerate(refs[:print_limit]):
        print(f"  Ref {i+1}: type={type(ref)}, id={id(ref)}")
        # 尝试打印部分内容
        try:
            if isinstance(ref, dict):
                print(f"    dict keys: {list(ref.keys())[:5]}")
            elif isinstance(ref, list):
                print(f"    list len: {len(ref)}")
            elif isinstance(ref, tuple):
                print(f"    tuple len: {len(ref)}")
            else:
                print(f"    str: {str(ref)[:200]}")
        except Exception as e:
            print(f"    (无法打印内容: {e})")
    if len(refs) > print_limit:
        print(f"  ... 还有 {len(refs) - print_limit} 个未打印 ...")

def find_all_large_tensor_referrers(size_mb_threshold=1, print_limit=3):
    """
    自动遍历所有大于 size_mb_threshold 的 GPU tensor，并打印其引用链。
    """
    import gc, torch
    tensors = []
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and obj.is_cuda:
            size_mb = obj.element_size() * obj.numel() / 1024**2
            if size_mb >= size_mb_threshold:
                tensors.append(obj)
    print(f"共找到 {len(tensors)} 个大于 {size_mb_threshold} MB 的 GPU tensor。")
    for i, t in enumerate(tensors):
        print(f'\n==== Tensor {i+1}/{len(tensors)} ====')
        print(f'shape={tuple(t.size())}, dtype={t.dtype}, size={t.element_size() * t.numel() / 1024**2:.2f} MB')
        find_tensor_referrers(t, print_limit=print_limit)

def find_tensor_referrers_in_range(min_mb=0.5, max_mb=20, print_limit=3):
    """
    遍历所有GPU tensor，只对大小在[min_mb, max_mb]区间的tensor打印引用链。
    Args:
        min_mb: 最小MB阈值
        max_mb: 最大MB阈值
        print_limit: 每个tensor最多打印多少个引用对象
    """
    import gc, torch
    tensors = []
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and obj.is_cuda:
            size_mb = obj.element_size() * obj.numel() / 1024**2
            if min_mb <= size_mb <= max_mb:
                tensors.append((obj, size_mb))
    print(f"共找到 {len(tensors)} 个大小在 [{min_mb}, {max_mb}] MB 的 GPU tensor。")
    for i, (t, size_mb) in enumerate(tensors):
        print(f'\n==== Tensor {i+1}/{len(tensors)} ====')
        print(f'shape={tuple(t.size())}, dtype={t.dtype}, size={size_mb:.2f} MB')
        find_tensor_referrers(t, print_limit=print_limit)

def print_ref_chain(obj, max_depth=3, cur_depth=1, visited=None):
    """
    递归打印对象的引用链，最多 max_depth 层。
    """
    import gc
    if visited is None:
        visited = set()
    if cur_depth > max_depth:
        return
    refs = gc.get_referrers(obj)
    for i, ref in enumerate(refs):
        if id(ref) in visited:
            continue
        visited.add(id(ref))
        print(f"{'  ' * cur_depth}↑ Level {cur_depth}: type={type(ref)}, id={id(ref)}")
        try:
            if isinstance(ref, dict):
                print(f"{'  ' * cur_depth}  dict keys: {list(ref.keys())[:5]}")
            elif isinstance(ref, list):
                print(f"{'  ' * cur_depth}  list len: {len(ref)}")
            elif isinstance(ref, tuple):
                print(f"{'  ' * cur_depth}  tuple len: {len(ref)}")
            else:
                s = str(ref)
                print(f"{'  ' * cur_depth}  str: {s[:200]}")
        except Exception as e:
            print(f"{'  ' * cur_depth}  (无法打印内容: {e})")
        print_ref_chain(ref, max_depth=max_depth, cur_depth=cur_depth+1, visited=visited)

def find_shape_grouped_tensor_referrers_in_range(min_mb=0.5, max_mb=20, print_limit=3):
    """
    按shape分组统计所有GPU tensor，每组总显存在[min_mb, max_mb]区间时，打印该组所有tensor的引用链（递归3级）。
    Args:
        min_mb: 最小MB阈值
        max_mb: 最大MB阈值
        print_limit: 每个tensor最多打印多少个引用对象
    """
    import gc, torch
    from collections import defaultdict
    shape_groups = defaultdict(list)
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and obj.is_cuda:
            shape = tuple(obj.size())
            size_mb = obj.element_size() * obj.numel() / 1024**2
            shape_groups[shape].append((obj, size_mb))
    # 统计每组总显存
    filtered = []
    for shape, items in shape_groups.items():
        total_mb = sum(size_mb for _, size_mb in items)
        if min_mb <= total_mb <= max_mb:
            filtered.append((shape, total_mb, items))
    print(f"共找到 {len(filtered)} 个 shape 分组，总显存在 [{min_mb}, {max_mb}] MB 区间。")
    for i, (shape, total_mb, items) in enumerate(filtered):
        print(f'\n==== Shape Group {i+1}/{len(filtered)} ====')
        print(f'shape={shape}, group_total={total_mb:.2f} MB, tensor_count={len(items)}')
        for j, (t, size_mb) in enumerate(items[:print_limit]):
            print(f'  -- Tensor {j+1}/{len(items)}: size={size_mb:.2f} MB')
            print_ref_chain(t, max_depth=3)