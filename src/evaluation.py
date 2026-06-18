"""
评估与对比框架模块 — 纯 NumPy 实现
- 基础指标: accuracy / precision / recall / f1
- 混淆矩阵 + ROC / AUC
- K-fold 交叉验证
- 多算法 × 多数据集 统一对比框架
- 多次运行取平均
"""
import time
import numpy as np

# ============================================================
# 配置
# ============================================================
try:
    from config import EVALUATION
    K_FOLD = EVALUATION.get("k_fold", 5)
    METRICS = EVALUATION.get("metrics", ["accuracy", "precision", "recall", "f1"])
    N_RUNS = EVALUATION.get("n_runs", 10)
except ImportError:
    K_FOLD = 5
    METRICS = ["accuracy", "precision", "recall", "f1"]
    N_RUNS = 10


# ============================================================
# 1. 基础评估指标
# ============================================================

def confusion_matrix(y_true, y_pred, n_classes=None):
    """
    混淆矩阵
    返回: ndarray (n_classes, n_classes), 行=真实, 列=预测
    """
    n = n_classes or max(y_true.max(), y_pred.max()) + 1
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def accuracy(y_true, y_pred):
    """准确率"""
    return np.mean(y_true == y_pred)


def precision_recall_f1(y_true, y_pred, average="macro"):
    """
    Precision / Recall / F1
    - average: "macro" (各类平均) / "micro" (全局) / "weighted" (加权)
    返回: (precision, recall, f1)
    """
    n_classes = max(y_true.max(), y_pred.max()) + 1
    cm = confusion_matrix(y_true, y_pred, n_classes)

    tp = np.diag(cm)                                          # (n_classes,)
    fp = cm.sum(axis=0) - tp                                   # 预测为正但真实为负
    fn = cm.sum(axis=1) - tp                                   # 真实为正但预测为负

    # 按类别计算
    precision_per_class = np.divide(tp, tp + fp,
                                    out=np.zeros_like(tp, dtype=np.float64),
                                    where=(tp + fp) > 0)
    recall_per_class = np.divide(tp, tp + fn,
                                  out=np.zeros_like(tp, dtype=np.float64),
                                  where=(tp + fn) > 0)
    f1_per_class = np.divide(2 * precision_per_class * recall_per_class,
                              precision_per_class + recall_per_class,
                              out=np.zeros_like(tp, dtype=np.float64),
                              where=(precision_per_class + recall_per_class) > 0)

    if average == "macro":
        precision = np.mean(precision_per_class)
        recall = np.mean(recall_per_class)
        f1 = np.mean(f1_per_class)
    elif average == "micro":
        tp_sum = tp.sum()
        fp_sum = fp.sum()
        fn_sum = fn.sum()
        precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0
        recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0)
    else:  # weighted
        class_support = cm.sum(axis=1).astype(np.float64)
        total = class_support.sum()
        weights = class_support / total if total > 0 else np.ones_like(class_support) / n_classes
        precision = np.sum(precision_per_class * weights)
        recall = np.sum(recall_per_class * weights)
        f1 = np.sum(f1_per_class * weights)

    return precision, recall, f1


def compute_all_metrics(y_true, y_pred):
    """计算所有基础指标"""
    acc = accuracy(y_true, y_pred)
    prec, rec, f1 = precision_recall_f1(y_true, y_pred, average="macro")
    prec_w, rec_w, f1_w = precision_recall_f1(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
        "precision_weighted": prec_w,
        "recall_weighted": rec_w,
        "f1_weighted": f1_w,
        "confusion_matrix": cm,
    }


# ============================================================
# 2. ROC / AUC
# ============================================================

def roc_curve_binary(y_true, y_score):
    """
    二分类 ROC 曲线
    - y_true: 0/1 标签
    - y_score: 正类预测概率
    返回: (fpr, tpr, thresholds)
    """
    # 按分数降序排列
    order = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return np.array([0, 1]), np.array([0, 1]), np.array([1, 0])

    # 累积 TPR / FPR
    tpr = np.cumsum(y_true_sorted == 1) / n_pos
    fpr = np.cumsum(y_true_sorted == 0) / n_neg

    # 去重：相同分数的保留最后一个（即最高的 TPR/FPR）
    diff_score = np.diff(y_score_sorted) != 0      # length = n-1
    keep_idx = np.where(diff_score)[0]              # 分数变化的位置
    # 加上最后一个点
    keep_idx = np.append(keep_idx, len(y_score_sorted) - 1)

    fpr = fpr[keep_idx]
    tpr = tpr[keep_idx]
    thresholds = y_score_sorted[keep_idx]

    # 确保从 (0,0) 到 (1,1)
    fpr = np.concatenate([[0], fpr])
    tpr = np.concatenate([[0], tpr])
    thresholds = np.concatenate([[np.inf], thresholds])

    return fpr, tpr, thresholds


def auc(fpr, tpr):
    """梯形法计算 AUC"""
    # numpy 2.x 使用 trapezoid, 1.x 使用 trapz
    if hasattr(np, "trapezoid"):
        return np.trapezoid(tpr, fpr)
    # 手动梯形积分兜底
    return float(np.sum((tpr[1:] + tpr[:-1]) * np.diff(fpr) / 2))


def roc_auc_multiclass(y_true, y_score, n_classes):
    """
    多分类 ROC AUC（One-vs-Rest 宏平均）
    - y_score: (n_samples, n_classes) 各类预测概率
    """
    fpr_dict, tpr_dict, auc_dict = {}, {}, {}
    for c in range(n_classes):
        y_bin = (y_true == c).astype(np.int64)
        y_prob = y_score[:, c]
        fpr, tpr, _ = roc_curve_binary(y_bin, y_prob)
        fpr_dict[c] = fpr
        tpr_dict[c] = tpr
        auc_dict[c] = auc(fpr, tpr)
    # 宏平均
    macro_auc = np.mean(list(auc_dict.values()))
    return fpr_dict, tpr_dict, auc_dict, macro_auc


# ============================================================
# 3. K-Fold 交叉验证
# ============================================================

def k_fold_split(n_samples, k=5, shuffle=True, random_state=42):
    """K-Fold 索引生成器"""
    indices = np.arange(n_samples)
    if shuffle:
        np.random.RandomState(random_state).shuffle(indices)
    fold_sizes = np.full(k, n_samples // k, dtype=np.int64)
    fold_sizes[:n_samples % k] += 1
    current = 0
    folds = []
    for size in fold_sizes:
        start, stop = current, current + size
        folds.append(indices[start:stop])
        current = stop
    return folds


def cross_validate(model_class, X, y, k=K_FOLD, n_runs=1):
    """
    K 折交叉验证（支持多次运行）
    - model_class: 返回 fit+predict 对象的类（无参构造）
    返回: metrics 均值与标准差字典
    """
    all_metrics = []
    all_times = []

    for run in range(n_runs):
        folds = k_fold_split(len(X), k=k, shuffle=True, random_state=42 + run * 100)
        for fold_idx in range(k):
            val_idx = folds[fold_idx]
            train_idx = np.concatenate([folds[i] for i in range(k) if i != fold_idx])

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            t0 = time.perf_counter()
            model = model_class()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            elapsed = time.perf_counter() - t0

            metrics = compute_all_metrics(y_val, y_pred)
            all_metrics.append(metrics)
            all_times.append(elapsed)

    # 汇总
    keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro",
            "precision_weighted", "recall_weighted", "f1_weighted"]
    summary = {}
    for kk in keys:
        vals = [m[kk] for m in all_metrics]
        summary[f"{kk}_mean"] = np.mean(vals)
        summary[f"{kk}_std"] = np.std(vals)
    summary["train_time_mean"] = np.mean(all_times)
    summary["train_time_std"] = np.std(all_times)

    return summary


# ============================================================
# 4. 统一对比框架
# ============================================================

def run_benchmark(algorithms, datasets_loader,
                  preprocess_fn=None,
                  metrics=None,
                  n_runs=N_RUNS,
                  verbose=True):
    """
    统一基准测试框架
    - algorithms: {name: model_class}  模型类字典（无参构造函数）
    - datasets_loader: 返回 {key: {"X": X, "y": y, ...}} 的函数
    - preprocess_fn: (X, y) → (X_processed, y_processed) 预处理函数
    - n_runs: 每折运行的次数

    返回:
        results: {dataset_key: {algo_name: {metric: value}}}
        summary: DataFrame-like dict
    """
    metrics = metrics or METRICS
    datasets = datasets_loader()

    all_results = {}
    for ds_key, ds_info in datasets.items():
        X_raw = ds_info["X"]
        y_raw = ds_info["y"]

        # 预处理
        if preprocess_fn:
            X, y = preprocess_fn(X_raw, y_raw)
        else:
            X, y = X_raw, y_raw

        ds_results = {}
        for algo_name, algo_class in algorithms.items():
            if verbose:
                print(f"  [{ds_key}] 运行 {algo_name} ...", end=" ")
            try:
                t0 = time.perf_counter()
                cv_results = cross_validate(algo_class, X, y, k=K_FOLD, n_runs=n_runs)
                total_time = time.perf_counter() - t0
                ds_results[algo_name] = {
                    "accuracy": cv_results.get("accuracy_mean", 0),
                    "accuracy_std": cv_results.get("accuracy_std", 0),
                    "f1_macro": cv_results.get("f1_macro_mean", 0),
                    "f1_macro_std": cv_results.get("f1_macro_std", 0),
                    "precision_macro": cv_results.get("precision_macro_mean", 0),
                    "recall_macro": cv_results.get("recall_macro_mean", 0),
                    "train_time": cv_results.get("train_time_mean", total_time / K_FOLD / n_runs),
                    "raw_cv": cv_results,
                }
                if verbose:
                    print(f"acc={ds_results[algo_name]['accuracy']:.4f}")
            except Exception as e:
                if verbose:
                    print(f"FAIL: {e}")
                ds_results[algo_name] = {"error": str(e)}

        all_results[ds_key] = ds_results

    return all_results


def results_to_table(results, metric="accuracy"):
    """
    将 benchmark 结果转为可打印表格
    返回: (rows, columns) 用于 report
    """
    datasets = list(results.keys())
    algos = set()
    for ds in datasets:
        algos.update(results[ds].keys())
    algos = sorted(algos)

    table = []
    for ds in datasets:
        row = [ds]
        for algo in algos:
            val = results[ds].get(algo, {}).get(metric, "N/A")
            if isinstance(val, float):
                row.append(f"{val:.4f}")
            else:
                row.append(str(val))
        table.append(row)
    return table, ["Dataset"] + algos


def results_to_matrix(results, metric="accuracy"):
    """提取交叉数据集对比矩阵 (n_datasets, n_algos)"""
    datasets = list(results.keys())
    algos = sorted(set().union(*[set(results[ds].keys()) for ds in datasets]))
    matrix = np.zeros((len(datasets), len(algos)))
    for i, ds in enumerate(datasets):
        for j, algo in enumerate(algos):
            matrix[i, j] = results[ds].get(algo, {}).get(metric, 0)
    return matrix, datasets, algos


# ============================================================
# 测试入口
# ============================================================
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_loader import load_dataset
    from src.preprocessing import StandardScaler, train_test_split

    print("=" * 60)
    print("评估模块测试")
    print("=" * 60)

    # 加载 Iris + 预处理
    X, y, fnames, cnames = load_dataset("iris")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.3)

    # 基础指标
    y_fake = np.random.randint(0, 3, len(yte))
    print(f"\n1. 随机预测: acc={accuracy(yte, y_fake):.4f}")
    cm = confusion_matrix(yte, y_fake)
    print(f"   混淆矩阵:\n{cm}")

    # precision/recall/f1
    prec, rec, f1 = precision_recall_f1(yte, y_fake, "macro")
    print(f"2. Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

    # ROC/AUC
    print(f"\n3. ROC/AUC 测试 (二分类模拟)")
    y_bin = (ytr[:100] == ytr[:100][0]).astype(np.int64)
    y_prob = np.random.random(100)
    fpr, tpr, thresh = roc_curve_binary(y_bin, y_prob)
    auc_val = auc(fpr, tpr)
    print(f"   AUC={auc_val:.4f} (random, should be ~0.5)")

    # K-Fold
    print(f"\n4. K-Fold 拆分 (k=5):")
    folds = k_fold_split(len(Xs), k=5)
    for i, f in enumerate(folds):
        print(f"   fold {i}: {len(f)} samples")

    print(f"\n[ evaluation ] 测试通过")
