"""
数据加载器模块 — 纯 Python 实现
支持 CSV/TSV/Excel/ARFF 格式，自动检测分隔符与表头
返回 numpy 数组 (X, y, feature_names, class_names)
"""
import os
import csv
import urllib.request
import numpy as np

# ============================================================
# 配置（从 config 导入，此处做硬兜底）
# ============================================================
try:
    from config import DATASETS, DATA_DIR
except ImportError:
    DATA_DIR = "data"
    DATASETS = {}


def _auto_detect_delimiter(filepath, sample_lines=5):
    """自动检测 CSV 分隔符：尝试 , ; \\t 空格，选列数最多 & 一致的"""
    candidates = [",", ";", "\t", " "]
    best_delim = ","
    best_cols = 0
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        head = [f.readline() for _ in range(sample_lines)]
    for delim in candidates:
        ncols = [len(line.rstrip("\n").split(delim)) for line in head if line.strip()]
        if ncols and len(set(ncols)) == 1 and ncols[0] > 1:
            if ncols[0] > best_cols:
                best_delim = delim
                best_cols = ncols[0]
    return best_delim, best_cols


def _is_numeric(s):
    """判断字符串是否可转浮点数"""
    try:
        float(s)
        return True
    except ValueError:
        return False


def _guess_header(filepath, delim):
    """试探首行是否为表头：如果首行含非数值列名则当作表头"""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        first = f.readline().rstrip("\n").split(delim)
        second = f.readline().rstrip("\n").split(delim) if not "" else []
    # 首行为非数值 & 次行为数值 → 有表头
    numeric_count_1 = sum(_is_numeric(v) for v in first)
    numeric_count_2 = sum(_is_numeric(v) for v in second) if second else 0
    if numeric_count_1 < len(first) * 0.5 and numeric_count_2 > len(second) * 0.5:
        return True
    # 首行列数与次行不同 → 有表头
    if len(first) != len(second):
        return True
    return False


def load_from_csv(filepath,
                  has_header=None,
                  target_col=-1,
                  skip_cols=None,
                  delimiter=None,
                  encoding="utf-8",
                  skip_rows=0):
    """
    通用 CSV 加载器
    - has_header: None=自动检测, True/False 手动指定
    - target_col: 目标列索引（-1=最后一列）。注意：索引基于 skip_cols 移除后的列
    - skip_cols: 需跳过的列索引列表（如 [0] 跳过ID列）。在目标列定位之前移除
    - delimiter: None=自动检测
    返回: X(ndarray), y(ndarray), feature_names(list), class_names(list)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")
    skip_cols = skip_cols or []
    skip_set = set(skip_cols)
    # 支持负数索引（如 -1 最后一列）
    if target_col < 0:
        target_col_abs = -1  # 用绝对值不方便，先保留标记
    else:
        target_col_abs = target_col

    # ----- 自动检测分隔符 -----
    delimiter, ncols = _auto_detect_delimiter(filepath) if delimiter is None else (delimiter, None)

    # ----- 载入原始数据 -----
    raw_rows = []
    with open(filepath, "r", encoding=encoding, errors="replace") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for i, row in enumerate(reader):
            if i < skip_rows:
                continue
            if not row or all(c.strip() == "" for c in row):
                continue
            raw_rows.append([c.strip() for c in row])

    if not raw_rows:
        raise ValueError(f"文件中无有效数据: {filepath}")

    # ----- 判断表头 -----
    if has_header is None:
        has_header = _guess_header(filepath, delimiter)

    # 确定全部列数目
    n_all_cols = len(raw_rows[1]) if len(raw_rows) > 1 and has_header else len(raw_rows[0])

    # 构建列索引映射：input_col → effective_col（移除 skip 后的位置）
    input_to_eff = {}
    for i in range(n_all_cols):
        if i not in skip_set:
            input_to_eff[i] = len(input_to_eff)
    eff_to_input = {v: k for k, v in input_to_eff.items()}

    # 将 target_col（基于 skip_cols 移除后的有效列索引）映射回原始列索引
    if target_col >= 0:
        if target_col in eff_to_input:
            target_input_idx = eff_to_input[target_col]
        else:
            raise ValueError(f"target_col={target_col} 超出范围 (有效列数={len(eff_to_input)})")
    else:
        # target_col < 0: 从末尾倒数（基于移除 skip 后的有效列）
        n_eff = len(input_to_eff)
        target_eff = n_eff + target_col  # e.g. -1 → n_eff-1
        if target_eff in eff_to_input:
            target_input_idx = eff_to_input[target_eff]
        else:
            raise ValueError(f"target_col={target_col} → eff={target_eff} 超出范围")

    # 特征列：移除 skip 和 target 后剩余的有效列
    feature_eff_indices = [i for i in range(len(input_to_eff))
                           if eff_to_input[i] != target_input_idx]
    feature_input_indices = [eff_to_input[i] for i in feature_eff_indices]

    # 特征名
    if has_header:
        all_names = raw_rows[0]
        target_name = all_names[target_input_idx]
        feature_names = [all_names[fi] for fi in feature_input_indices
                         if fi < len(all_names)]
        # 补充缺失的特征名
        while len(feature_names) < len(feature_input_indices):
            feature_names.append(f"f{len(feature_names)}")
    else:
        target_name = f"target"
        feature_names = [f"f{i}" for i in range(len(feature_input_indices))]

    data_start = 1 if has_header else 0

    # ----- 数值转换 -----
    X_list, y_list = [], []
    for row in raw_rows[data_start:]:
        if len(row) < n_all_cols:
            continue
        try:
            features = [float(row[fi]) if _is_numeric(row[fi]) else np.nan
                        for fi in feature_input_indices if fi < len(row)]
            target_raw = row[target_input_idx]
        except (IndexError, ValueError):
            continue
        X_list.append(features)
        y_list.append(target_raw)

    X = np.array(X_list, dtype=np.float64)

    # ----- 编码 y -----
    y_raw = np.array(y_list)
    class_names = sorted(set(y_raw))
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    y = np.array([class_to_idx[v] for v in y_raw], dtype=np.int64)

    return X, y, feature_names, class_names


def load_dataset(dataset_key):
    """按 config 中的 key 加载数据集，自动下载"""
    info = DATASETS.get(dataset_key)
    if info is None:
        available = list(DATASETS.keys())
        raise KeyError(f"未知数据集: {dataset_key}. 可选: {available}")

    local_path = info["local"]
    has_header = info.get("has_header", None)
    target_col = info.get("target_col", -1)   # 优先使用显式指定的目标列
    skip_cols = info.get("skip_cols", [])     # 需跳过的列索引
    columns = info.get("columns", None)

    # 自动下载
    if not os.path.exists(local_path):
        _download_dataset(dataset_key, info, local_path)

    # 加载原始数据
    X, y, fnames, cnames = load_from_csv(
        local_path, has_header=has_header,
        target_col=target_col, skip_cols=skip_cols
    )

    # 用配置中指定的列名覆盖自动检测的列名
    if columns:
        fnames = columns[:-1]
    # cnames 始终使用从数据中提取的实际类别名（不覆盖）

    print(f"[data_loader] {dataset_key}: X{X.shape}, y{y.shape}, "
          f"classes={len(cnames)}, missing={np.isnan(X).sum()}")
    return X, y, fnames, cnames


def _download_dataset(key, info, local_path):
    """下载数据集到本地"""
    url = info.get("url")
    if url is None:
        raise ValueError(f"数据集 {key} 未配置下载 URL")

    os.makedirs(os.path.dirname(local_path) or DATA_DIR, exist_ok=True)
    print(f"[data_loader] 下载 {info['name']} → {local_path} ...")
    try:
        urllib.request.urlretrieve(url, local_path)
        print(f"[data_loader] 下载完成: {key}")
    except Exception as e:
        raise RuntimeError(f"下载失败 {key} ({url}): {e}")


def load_all_datasets(dataset_keys=None):
    """批量加载所有默认数据集"""
    keys = dataset_keys or ["iris", "wine", "breast_cancer", "diabetes", "digits"]
    results = {}
    for key in keys:
        try:
            X, y, fnames, cnames = load_dataset(key)
            results[key] = {
                "X": X, "y": y,
                "feature_names": fnames,
                "class_names": cnames,
                "task": DATASETS.get(key, {}).get("task", "classification"),
            }
        except Exception as e:
            print(f"[data_loader] 加载 {key} 失败: {e}")
    return results


# ============================================================
# 测试入口
# ============================================================
if __name__ == "__main__":
    for ds_key in ["iris", "wine", "breast_cancer", "diabetes", "digits"]:
        try:
            X, y, fn, cn = load_dataset(ds_key)
            print(f"  ✓ {ds_key}: X{X.shape}, y{y.shape}, classes={cn}")
        except Exception as e:
            print(f"  ✗ {ds_key}: {e}")
