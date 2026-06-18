"""
数据预处理与特征选择模块 — 纯 NumPy 实现
- 缺失值处理: mean / median / drop
- 标准化: StandardScaler / MinMaxScaler（手动实现）
- 类别编码: LabelEncoder（手动实现）
- 数据集划分: train_test_split（手动实现）
- 特征选择: 过滤式 / 包裹式 / 嵌入式
"""
import numpy as np
from itertools import combinations

# ============================================================
# 配置兜底
# ============================================================
try:
    from config import PREPROCESSING
    TEST_SIZE = PREPROCESSING.get("test_size", 0.2)
    RANDOM_STATE = PREPROCESSING.get("random_state", 42)
    FS_CONFIG = PREPROCESSING.get("feature_selection", {})
    FS_METHODS = FS_CONFIG.get("methods", ["filter", "wrapper", "embedded"])
    FS_K_BEST = FS_CONFIG.get("k_best", 10)
except ImportError:
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    FS_METHODS = ["filter", "wrapper", "embedded"]
    FS_K_BEST = 10


# ============================================================
# 1. 缺失值处理
# ============================================================
def handle_missing(X, strategy="mean"):
    """
    缺失值处理
    - strategy: "mean" → 均值填充 / "median" → 中位数填充 / "drop" → 删除含缺失的行
    返回: 处理后的 X, 有效行索引 (drop 模式时用于同步 y)
    """
    X = X.astype(np.float64)
    if strategy == "drop":
        mask = ~np.isnan(X).any(axis=1)
        print(f"[preprocess] 缺失值: drop → 保留 {mask.sum()}/{len(X)} 行")
        return X[mask], mask

    nan_mask = np.isnan(X)
    if not nan_mask.any():
        return X, np.ones(len(X), dtype=bool)

    for col in range(X.shape[1]):
        col_nan = nan_mask[:, col]
        if col_nan.any():
            if strategy == "median":
                fill_val = np.nanmedian(X[:, col])
            else:  # mean
                fill_val = np.nanmean(X[:, col])
            X[col_nan, col] = fill_val
    print(f"[preprocess] 缺失值: {strategy} 填充 (共 {nan_mask.sum()} 个)")
    return X, np.ones(len(X), dtype=bool)


# ============================================================
# 2. 标准化 — 手动实现 StandardScaler
# ============================================================
class StandardScaler:
    """纯 NumPy 实现：Z-score 标准化"""
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1e-10  # 避免除以零
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        return X * self.std_ + self.mean_


# ============================================================
# 3. MinMaxScaler — 手动实现
# ============================================================
class MinMaxScaler:
    """纯 NumPy 实现：Min-Max 归一化 [0, 1]"""
    def __init__(self, feature_range=(0, 1)):
        self.min_ = None
        self.max_ = None
        self.range_ = feature_range

    def fit(self, X):
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        return self

    def transform(self, X):
        a, b = self.range_
        denom = self.max_ - self.min_
        denom[denom == 0] = 1e-10
        scale = (b - a) / denom
        return a + (X - self.min_) * scale

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ============================================================
# 4. LabelEncoder — 手动实现
# ============================================================
class LabelEncoder:
    """纯 NumPy 实现：类别 → 整数"""
    def __init__(self):
        self.classes_ = None
        self.class_to_idx_ = None

    def fit(self, y):
        self.classes_ = np.unique(y)
        self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}
        return self

    def transform(self, y):
        return np.array([self.class_to_idx_[v] for v in y], dtype=np.int64)

    def fit_transform(self, y):
        return self.fit(y).transform(y)


# ============================================================
# 5. train_test_split — 手动实现
# ============================================================
def train_test_split(*arrays,
                     test_size=0.2,
                     random_state=RANDOM_STATE,
                     stratify=None):
    """
    手动实现数据划分，支持分层抽样 (stratify)
    """
    n_samples = len(arrays[0])
    indices = np.arange(n_samples)
    rng = np.random.RandomState(random_state)

    if stratify is not None:
        stratify = np.asarray(stratify)
        unique_classes = np.unique(stratify)
        test_idx = np.array([], dtype=np.int64)
        train_idx = np.array([], dtype=np.int64)
        for cls in unique_classes:
            cls_mask = np.where(stratify == cls)[0]
            rng.shuffle(cls_mask)
            n_test = max(1, int(len(cls_mask) * test_size))
            test_idx = np.concatenate([test_idx, cls_mask[:n_test]])
            train_idx = np.concatenate([train_idx, cls_mask[n_test:]])
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)
    else:
        rng.shuffle(indices)
        n_test = max(1, int(n_samples * test_size))
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

    result = []
    for arr in arrays:
        arr = np.asarray(arr)
        result.append(arr[train_idx])
        result.append(arr[test_idx])
    return tuple(result)


# ============================================================
# 6. 特征选择
# ============================================================

# ---- 6.1 过滤式 (Filter) ----
def _pearson_correlation(X, y):
    """计算每个特征与目标变量的 Pearson 相关系数"""
    n_features = X.shape[1]
    scores = np.zeros(n_features)
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std == 0:
        return scores
    for i in range(n_features):
        col = X[:, i]
        col_mean = np.mean(col)
        col_std = np.std(col)
        if col_std == 0:
            scores[i] = 0
        else:
            cov = np.mean((col - col_mean) * (y - y_mean))
            scores[i] = abs(cov / (col_std * y_std))
    return scores


def _variance_filter(X, threshold=0.01):
    """方差过滤：移除低方差特征"""
    variances = np.var(X, axis=0)
    mask = variances > threshold
    return mask, variances


def _mutual_information_discrete(X, y, n_bins=10):
    """
    基于互信息的近似计算（针对离散化后的连续变量）
    对于连续特征先分箱再计算
    """
    n_samples, n_features = X.shape
    scores = np.zeros(n_features)

    # 目标变量的熵
    _, y_counts = np.unique(y, return_counts=True)
    y_prob = y_counts / n_samples
    h_y = -np.sum(y_prob * np.log2(y_prob + 1e-10))

    for fi in range(n_features):
        col = X[:, fi]
        # 分箱
        if np.unique(col).size < n_bins:
            bins = np.sort(np.unique(col))
        else:
            bins = np.percentile(col, np.linspace(0, 100, n_bins + 1))
            bins = np.unique(bins)
        x_disc = np.digitize(col, bins[:-1])
        # 计算互信息
        mi = 0.0
        for xv in np.unique(x_disc):
            mask = x_disc == xv
            px = np.sum(mask) / n_samples
            for yv in np.unique(y):
                pxy = np.sum(mask & (y == yv)) / n_samples
                if pxy > 0:
                    py = np.sum(y == yv) / n_samples
                    mi += pxy * np.log2(pxy / (px * py) + 1e-10)
        scores[fi] = mi / (h_y + 1e-10)  # 归一化
    return scores


def filter_selection(X, y, k=10, method="pearson"):
    """
    过滤式特征选择
    - method: "pearson" | "variance" | "mutual_info"
    - k: 保留特征数
    返回: 选中特征索引, 各特征得分
    """
    if method == "variance":
        mask, scores = _variance_filter(X)
        top_k = np.argsort(scores)[::-1][:k]
        selected = np.array([i for i in range(len(mask)) if mask[i]])[:k]
        return selected, scores
    elif method == "mutual_info":
        scores = _mutual_information_discrete(X, y)
    else:  # pearson
        scores = _pearson_correlation(X, y)

    top_k = np.argsort(scores)[::-1][:k]
    print(f"[preprocess] 过滤式({method}): 从 {X.shape[1]} 选 {k} 特征, "
          f"最高分={scores[top_k[0]]:.4f}")
    return top_k, scores


# ---- 6.2 包裹式 (Wrapper) ----
def _cross_val_score(model_fn, X, y, cv=3):
    """简单交叉验证评估（用于包裹式特征选择）"""
    n = len(X)
    fold_size = n // cv
    scores = []
    indices = np.arange(n)
    np.random.RandomState(42).shuffle(indices)
    for i in range(cv):
        val_idx = indices[i * fold_size: (i + 1) * fold_size]
        train_idx = np.setdiff1d(indices, val_idx)
        model = model_fn()
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        scores.append(np.mean(preds == y[val_idx]))
    return np.mean(scores)


def forward_selection(X, y, model_fn, k=10):
    """
    前向搜索 (Forward Selection)
    - model_fn: 返回 fit+predict 对象的无参函数
    - k: 最多选 k 个特征
    需注意：这是计算密集型操作，对大数据集可能较慢
    """
    n_features = X.shape[1]
    selected = []
    remaining = list(range(n_features))
    best_scores = []

    for _ in range(min(k, n_features)):
        best_score = -1
        best_feat = None
        for feat in remaining:
            candidate = selected + [feat]
            try:
                score = _cross_val_score(model_fn, X[:, candidate], y)
            except Exception:
                score = 0
            if score > best_score:
                best_score = score
                best_feat = feat
        if best_feat is None:
            break
        selected.append(best_feat)
        remaining.remove(best_feat)
        best_scores.append(best_score)
        print(f"[preprocess] 前向搜索: 选入 f{best_feat} (cv_acc={best_score:.4f}), "
              f"已选 {len(selected)} 个")

    print(f"[preprocess] 包裹式(前向搜索): 最终选 {len(selected)} 特征, "
          f"cv_acc={best_scores[-1]:.4f}")
    return np.array(selected), np.array(best_scores)


def backward_elimination(X, y, model_fn, k=10):
    """
    后向消除 (Backward Elimination)
    - 从全特征集开始逐个移除最不重要特征
    """
    n_features = X.shape[1]
    selected = list(range(n_features))
    removed = []

    while len(selected) > max(k, 1):
        worst_score = float("inf")
        worst_feat = None
        for feat in selected[:]:
            candidate = [f for f in selected if f != feat]
            if len(candidate) < 1:
                continue
            try:
                score = _cross_val_score(model_fn, X[:, candidate], y)
            except Exception:
                score = 0
            # 移除后分数下降最少 = 该特征最不重要
            if score >= worst_score or worst_feat is None:
                worst_score = score
                worst_feat = feat
        if worst_feat is None or len(selected) <= k:
            break
        selected.remove(worst_feat)
        removed.append(worst_feat)
        print(f"[preprocess] 后向消除: 移除 f{worst_feat}, "
              f"剩余 {len(selected)} 特征 (cv_acc={worst_score:.4f})")

    return np.array(selected), np.array([worst_score])


def wrapper_selection(X, y, model_fn, k=10, direction="forward"):
    """包裹式特征选择统一入口"""
    if direction == "backward":
        return backward_elimination(X, y, model_fn, k)
    return forward_selection(X, y, model_fn, k)


# ---- 6.3 嵌入式 (Embedded) ----
def _feature_importance_from_tree(X, y, max_depth=5):
    """
    基于决策树的结构计算特征重要性（不依赖 sklearn）
    使用简化版 CART + 信息增益
    返回: 特征重要性数组 (和为 1)
    """
    n_samples, n_features = X.shape

    class Node:
        def __init__(self):
            self.feature = None
            self.threshold = None
            self.left = None
            self.right = None
            self.value = None
            self.samples = 0

    def _entropy(y_sub):
        _, counts = np.unique(y_sub, return_counts=True)
        p = counts / len(y_sub)
        return -np.sum(p * np.log2(p + 1e-10))

    def _best_split(X_sub, y_sub):
        best_gain = 0
        best_feat = None
        best_thresh = None
        n = len(y_sub)
        if n < 2:
            return best_feat, best_thresh, best_gain
        parent_entropy = _entropy(y_sub)
        for fi in range(X_sub.shape[1]):
            values = np.unique(X_sub[:, fi])
            if len(values) < 2:
                continue
            thresholds = (values[:-1] + values[1:]) / 2
            for th in thresholds[:20]:  # 限制候选以加速
                left = y_sub[X_sub[:, fi] <= th]
                right = y_sub[X_sub[:, fi] > th]
                if len(left) == 0 or len(right) == 0:
                    continue
                child_entropy = (len(left) / n) * _entropy(left) + \
                                (len(right) / n) * _entropy(right)
                gain = parent_entropy - child_entropy
                if gain > best_gain:
                    best_gain = gain
                    best_feat = fi
                    best_thresh = th
        return best_feat, best_thresh, best_gain

    def _build_tree(X_sub, y_sub, depth):
        node = Node()
        node.samples = len(y_sub)
        values, counts = np.unique(y_sub, return_counts=True)
        node.value = values[np.argmax(counts)]

        if depth >= max_depth or len(y_sub) < 2 or len(np.unique(y_sub)) == 1:
            return node

        feat, thresh, gain = _best_split(X_sub, y_sub)
        if feat is None or gain < 0.001:
            return node

        node.feature = feat
        node.threshold = thresh
        left_mask = X_sub[:, feat] <= thresh
        node.left = _build_tree(X_sub[left_mask], y_sub[left_mask], depth + 1)
        node.right = _build_tree(X_sub[~left_mask], y_sub[~left_mask], depth + 1)
        return node

    tree = _build_tree(X, y, 0)

    # 遍历树计算重要性
    importances = np.zeros(n_features)

    def _accumulate(node):
        if node is None or node.feature is None:
            return
        importances[node.feature] += node.samples / n_samples
        _accumulate(node.left)
        _accumulate(node.right)

    _accumulate(tree)
    total = importances.sum()
    if total > 0:
        importances /= total
    return importances


def embedded_selection(X, y, k=10):
    """
    嵌入式特征选择：基于决策树特征重要性
    """
    importances = _feature_importance_from_tree(X, y, max_depth=6)
    top_k = np.argsort(importances)[::-1][:k]
    print(f"[preprocess] 嵌入式(决策树重要性): 从 {X.shape[1]} 选 {k} 特征, "
          f"最高重要性={importances[top_k[0]]:.4f}")
    return top_k, importances


# ============================================================
# 7. 统一预处理流水线
# ============================================================
def preprocess_pipeline(X, y,
                        normalize="standard",
                        handle_missing_strategy="mean",
                        feature_select_methods=None,
                        k_best=10,
                        model_fn_for_wrapper=None):
    """
    统一预处理流水线
    - 缺失值处理 → 标准化 → 特征选择
    返回:
        X_processed, y_processed, scaler, selected_indices, selection_report
    """
    feature_select_methods = feature_select_methods or FS_METHODS
    report = {}

    # Step 1: 缺失值处理
    X_clean, valid_mask = handle_missing(X, handle_missing_strategy)
    y_clean = y[valid_mask] if valid_mask.dtype == bool else y

    # Step 2: 标准化
    scaler = None
    if normalize == "standard":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
    elif normalize == "minmax":
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_clean)
    else:
        X_scaled = X_clean

    # Step 3: 特征选择
    n_features = X_scaled.shape[1]
    k = min(k_best, n_features)
    selected_indices = np.arange(n_features)  # 默认全选

    for method in feature_select_methods:
        if method == "filter":
            idx, scores = filter_selection(X_scaled, y_clean, k=k)
            report["filter"] = {"indices": idx.tolist(), "scores": scores.tolist()}
        elif method == "wrapper":
            if model_fn_for_wrapper is not None:
                idx, scores = wrapper_selection(X_scaled, y_clean, model_fn_for_wrapper, k=k)
                report["wrapper"] = {"indices": idx.tolist(), "cv_scores": scores.tolist()}
        elif method == "embedded":
            idx, scores = embedded_selection(X_scaled, y_clean, k=k)
            report["embedded"] = {"indices": idx.tolist(), "importances": scores.tolist()}

    # 取所有方法的并集（交集更激进，并集更保守）
    if feature_select_methods:
        all_indices = set()
        for m in feature_select_methods:
            if m in report:
                all_indices.update(report[m]["indices"])
        selected_indices = np.array(sorted(all_indices)[:k])
        print(f"[preprocess] 特征选择综合: 选 {len(selected_indices)}/{n_features} 特征")

    X_final = X_scaled[:, selected_indices]
    return X_final, y_clean, scaler, selected_indices, report


# ============================================================
# 测试入口
# ============================================================
if __name__ == "__main__":
    from data_loader import load_dataset

    print("=" * 60)
    print("预处理模块测试")
    print("=" * 60)

    # 加载 Iris
    X, y, fnames, cnames = load_dataset("iris")
    print(f"\n原始数据: X{X.shape}, y{y.shape}, classes={cnames}")

    # 测试缺失值
    X2, mask = handle_missing(X, "mean")
    print(f"缺失值处理: {np.isnan(X2).sum()} 个剩余 NaN")

    # 测试标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X2)
    print(f"标准化: mean={X_scaled.mean(axis=0)[:3]}, std={X_scaled.std(axis=0)[:3]}")

    # 测试 train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X_scaled, y, test_size=0.3)
    print(f"划分: train={Xtr.shape}, test={Xte.shape}")

    # 测试特征选择方法
    print("\n--- 过滤式 (Pearson) ---")
    idx, scores = filter_selection(X_scaled, y, k=2)
    print(f"  选中: {idx}, 得分: {scores[idx]}")

    print("\n--- 嵌入式 (决策树重要性) ---")
    idx2, imp = embedded_selection(X_scaled, y, k=2)
    print(f"  选中: {idx2}, 重要性: {imp[idx2]}")
