"""Quickly inspect an NPZ file's structure."""
import numpy as np
import sys

path = sys.argv[1]
d = np.load(path, allow_pickle=True)
print(f"File: {path}")
print(f"Keys: {list(d.keys())}")
print()
for k in d.keys():
    v = d[k]
    if hasattr(v, 'shape'):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        if v.ndim == 1 and v.shape[0] <= 20:
            print(f"    values={np.round(v, 4)}")
    else:
        print(f"  {k}: type={type(v)}, value={v}")
