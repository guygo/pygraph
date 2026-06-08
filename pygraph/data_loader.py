import numpy as np
import os


def load_file(path):
    """Return (col_names: list[str], data: float32 ndarray shape (N, M))."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.csv', '.tsv', '.txt', ''):
        return _load_delimited(path)
    elif ext == '.json':
        return _load_json(path)
    elif ext == '.npy':
        return _load_npy(path)
    elif ext == '.npz':
        return _load_npz(path)
    else:
        return _load_delimited(path)


def _load_delimited(path):
    import csv
    with open(path, newline='', encoding='utf-8-sig') as f:
        sample = f.read(8192)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=',\t; |')
        sep = dialect.delimiter
    except Exception:
        sep = ','

    with open(path, newline='', encoding='utf-8-sig') as f:
        rows = [r for r in csv.reader(f, delimiter=sep)]

    if not rows:
        raise ValueError("Empty file")

    # Detect header
    try:
        [float(v.strip()) for v in rows[0] if v.strip()]
        col_names = [f'col{i}' for i in range(len(rows[0]))]
        data_rows = rows
    except ValueError:
        col_names = [v.strip().strip('"') or f'col{i}' for i, v in enumerate(rows[0])]
        data_rows = rows[1:]

    parsed = []
    for row in data_rows:
        try:
            parsed.append([float(v.strip()) if v.strip() else np.nan for v in row])
        except Exception:
            continue
    if not parsed:
        raise ValueError("No numeric data found")

    max_cols = max(len(r) for r in parsed)
    padded   = [r + [np.nan] * (max_cols - len(r)) for r in parsed]
    data     = np.array(padded, dtype=np.float32)

    while len(col_names) < max_cols:
        col_names.append(f'col{len(col_names)}')
    return col_names[:max_cols], data


def _load_json(path):
    import json
    with open(path) as f:
        obj = json.load(f)

    if isinstance(obj, dict):
        keys   = list(obj.keys())
        arrays = [np.array(obj[k], dtype=np.float32).flatten() for k in keys]
        n      = min(len(a) for a in arrays)
        return keys, np.column_stack([a[:n] for a in arrays])

    if isinstance(obj, list):
        if not obj:
            raise ValueError("Empty JSON array")
        if isinstance(obj[0], (int, float)):
            return ['value'], np.array(obj, dtype=np.float32).reshape(-1, 1)
        if isinstance(obj[0], list):
            data = np.array(obj, dtype=np.float32)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            return [f'col{i}' for i in range(data.shape[1])], data
        if isinstance(obj[0], dict):
            keys = list(obj[0].keys())
            rows = [[r.get(k, np.nan) for k in keys] for r in obj]
            return keys, np.array(rows, dtype=np.float32)

    raise ValueError("Unsupported JSON structure")


def _load_npy(path):
    data = np.load(path).astype(np.float32)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.ndim != 2:
        data = data.reshape(-1, 1)
    return [f'col{i}' for i in range(data.shape[1])], data


def _load_npz(path):
    npz  = np.load(path)
    keys = list(npz.keys())
    if not keys:
        raise ValueError("Empty .npz")
    arrays = [npz[k].flatten().astype(np.float32) for k in keys]
    n      = min(len(a) for a in arrays)
    return keys, np.column_stack([a[:n] for a in arrays])
