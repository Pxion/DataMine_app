"""
============================================================
统一基准测试框架 — benchmark.py
============================================================
功能:
  - 5-fold 交叉验证 × 6 算法 × 5 数据集
  - 自动生成对比表格、矩阵、热力图
  - 格式化输出 LaTeX/Markdown/CSV 多格式报告

运行:
  python src/benchmark.py
============================================================
"""

import sys, os, time, json, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.data_loader import load_dataset
from src.preprocessing import (
    StandardScaler, train_test_split, preprocess_pipeline
)
from src.evaluation import (
    k_fold_split, compute_all_metrics, results_to_table, results_to_matrix,
    run_benchmark
)
from src.visualization import (
    plot_algorithm_comparison, plot_time_comparison,
    plot_cross_dataset_heatmap
)
from config import ALGORITHM_CONFIG, DATASETS as DS_CONFIG, OUTPUT_DIR

# ============================================================
# 配置
# ============================================================
K_FOLD = 5           # 交叉验证折数
N_RUNS = 3           # 每折重复运行次数（取平均）
VERBOSE = True

# 参数字典
BENCH_ALGOS = {
    "KNN":              {"module": "src.algorithms.knn",           "class": "KNN"},
    "Naive Bayes":      {"module": "src.algorithms.naive_bayes",   "class": "GaussianNB"},
    "C4.5":             {"module": "src.algorithms.decision_tree", "class": "DecisionTreeC45"},
    "SVM":              {"module": "src.algorithms.svm",           "class": "SVM"},
    "K-Means":          {"module": "src.algorithms.kmeans",        "class": "KMeans"},
    "MLP":              {"module": "src.algorithms.mlp",           "class": "MLP"},
}


# 算法名 → config key 映射
_ALGO_KEY_MAP = {
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "C4.5": "decision_tree",
    "SVM": "svm",
    "K-Means": "kmeans",
    "MLP": "mlp",
}


def _create_algo_instance(algo_key):
    """根据配置创建算法实例"""
    cfg_key = _ALGO_KEY_MAP.get(algo_key, algo_key.lower())
    cfg = ALGORITHM_CONFIG.get(cfg_key, {})

    if algo_key == "KNN":
        from src.algorithms.knn import KNN
        return KNN(k=cfg.get("k", 5), weights=cfg.get("weights", "uniform"),
                   metric=cfg.get("metric", "euclidean"),
                   sigma=cfg.get("sigma", None))

    elif algo_key == "Naive Bayes":
        from src.algorithms.naive_bayes import GaussianNB
        return GaussianNB()

    elif algo_key == "C4.5":
        from src.algorithms.decision_tree import DecisionTreeC45
        return DecisionTreeC45(max_depth=cfg.get("max_depth", 10),
                               min_samples_split=cfg.get("min_samples_split", 2),
                               pruning=cfg.get("pruning", "post"))

    elif algo_key == "SVM":
        from src.algorithms.svm import SVM
        return SVM(kernel=cfg.get("kernel", "rbf"), C=cfg.get("C", 1.0),
                   max_iter=cfg.get("max_iter", 5000))

    elif algo_key == "K-Means":
        # K-Means 单独处理
        return None  # 占位

    elif algo_key == "MLP":
        from src.algorithms.mlp import MLP
        return MLP(hidden_layers=cfg.get("hidden_layers", [64, 32]),
                   activation=cfg.get("activation", "relu"),
                   learning_rate=cfg.get("learning_rate", 0.01),
                   epochs=cfg.get("epochs", 200),
                   batch_size=cfg.get("batch_size", 32),
                   optimizer=cfg.get("optimizer", "sgd"),
                   momentum=cfg.get("momentum", 0.9),
                   beta1=cfg.get("beta1", 0.9),
                   beta2=cfg.get("beta2", 0.999),
                   eps=cfg.get("eps", 1e-8),
                   weight_decay=cfg.get("weight_decay", 0.0))

    return None


def _cross_validate_classifier(AlgoClass, X, y, k=K_FOLD, n_runs=N_RUNS,
                                create_fn=None):
    """
    对分类器执行 k-fold 交叉验证
    - AlgoClass: 算法类 (需 fit / predict)
    - create_fn: 每 fold 重新创建实例的可选工厂函数
    返回: metrics 字典 (含 mean / std)
    """
    n = len(X)
    if create_fn is None:
        create_fn = AlgoClass

    accs, f1s, precs, recs, times = [], [], [], [], []

    for run_i in range(n_runs):
        folds = k_fold_split(n, k=k, shuffle=True, random_state=42 + run_i)
        for fold_i, val_idx in enumerate(folds):
            train_idx = np.setdiff1d(np.arange(n), val_idx)
            Xtr, ytr = X[train_idx], y[train_idx]
            Xval, yval = X[val_idx], y[val_idx]

            model = create_fn()
            t0 = time.perf_counter()
            model.fit(Xtr, ytr)
            y_pred = model.predict(Xval)
            elapsed = time.perf_counter() - t0

            m = compute_all_metrics(yval, y_pred)
            accs.append(m["accuracy"])
            f1s.append(m.get("f1_macro", m.get("f1_weighted", 0)))
            precs.append(m.get("precision_macro", m.get("precision_weighted", 0)))
            recs.append(m.get("recall_macro", m.get("recall_weighted", 0)))
            times.append(elapsed)

    def _mean_std(arr):
        return float(np.mean(arr)), float(np.std(arr))

    return {
        "accuracy_mean": _mean_std(accs)[0],
        "accuracy_std": _mean_std(accs)[1],
        "f1_macro_mean": _mean_std(f1s)[0],
        "f1_macro_std": _mean_std(f1s)[1],
        "precision_macro_mean": _mean_std(precs)[0],
        "recall_macro_mean": _mean_std(recs)[0],
        "train_time_mean": _mean_std(times)[0],
        "train_time_std": _mean_std(times)[1],
        "raw_accs": accs,
        "raw_f1s": f1s,
    }


def _evaluate_kmeans(X, y, k=5, n_runs=N_RUNS):
    """K-Means 聚类评估 (使用已知类别数)"""
    from src.algorithms.kmeans import KMeans

    n_classes = len(np.unique(y))
    inertias, silhouettes = [], []

    for run_i in range(n_runs):
        km = KMeans(n_clusters=n_classes, init="kmeans++", n_init=5,
                    random_state=42 + run_i)
        km.fit(X)
        inertias.append(km.inertia_)
        if len(X) <= 2000:
            sil = KMeans.silhouette_score(X, km.labels_)
            silhouettes.append(sil)

    result = {
        "inertia_mean": float(np.mean(inertias)),
        "inertia_std": float(np.std(inertias)),
        "accuracy_mean": 0,  # 聚类无标签
    }
    if silhouettes:
        result["silhouette_mean"] = float(np.mean(silhouettes))
        result["silhouette_std"] = float(np.std(silhouettes))
    return result


# ============================================================
# 主基准流程
# ============================================================
def run_full_benchmark():
    """运行完整基准测试: 5 数据集 × 6 算法"""
    results = {}
    total_start = time.perf_counter()

    for ds_key in ["iris", "wine", "breast_cancer", "diabetes", "digits"]:
        print(f"\n{'='*60}")
        ds_name = DS_CONFIG[ds_key]["name"]
        n_classes = DS_CONFIG[ds_key]["n_classes"]
        print(f"[{ds_name}] ({ds_key}) — 加载数据...")
        X, y, fnames, cnames = load_dataset(ds_key)

        # 标准化
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        print(f"  X{Xs.shape}, y{y.shape}, classes={len(cnames)}")

        ds_results = {}

        for algo_name in BENCH_ALGOS:
            print(f"  → {algo_name} ...", end=" ", flush=True)

            try:
                if algo_name == "K-Means":
                    cv = _evaluate_kmeans(Xs, y, k=K_FOLD, n_runs=N_RUNS)
                    ds_results[algo_name] = {
                        "accuracy": cv.get("silhouette_mean", cv.get("inertia_mean", 0)),
                        "accuracy_std": cv.get("silhouette_std", 0),
                        "train_time": cv.get("inertia_mean", 0),  # placeholder
                        "note": "silhouette_score",
                    }
                    if isinstance(cv.get("silhouette_mean"), float):
                        print(f"silhouette={cv['silhouette_mean']:.4f}")
                    else:
                        print(f"inertia={cv.get('inertia_mean', 0):.2f}")
                else:
                    inst = _create_algo_instance(algo_name)
                    if inst is None:
                        print("SKIP (no config)")
                        continue

                    # 用工厂函数保证每 fold 独立实例
                    create_fn = lambda: _create_algo_instance(algo_name)
                    cv = _cross_validate_classifier(type(inst), Xs, y,
                                                     k=K_FOLD, n_runs=N_RUNS,
                                                     create_fn=create_fn)
                    ds_results[algo_name] = {
                        "accuracy": cv["accuracy_mean"],
                        "accuracy_std": cv["accuracy_std"],
                        "f1_macro": cv["f1_macro_mean"],
                        "f1_macro_std": cv["f1_macro_std"],
                        "precision_macro": cv["precision_macro_mean"],
                        "recall_macro": cv["recall_macro_mean"],
                        "train_time": cv["train_time_mean"],
                        "train_time_std": cv["train_time_std"],
                    }
                    print(f"acc={cv['accuracy_mean']:.4f}±{cv['accuracy_std']:.4f}")
            except Exception as e:
                print(f"FAIL: {e}")
                import traceback
                traceback.print_exc()
                ds_results[algo_name] = {"error": str(e), "accuracy": 0}

        results[ds_key] = ds_results

    elapsed_total = time.perf_counter() - total_start
    print(f"\n{'='*60}")
    print(f"全部基准测试完成! 总耗时: {elapsed_total:.1f}s")
    print(f"{'='*60}")

    return results


def print_results_table(results, metric="accuracy"):
    """格式化打印对比表格 (含标准差)"""
    datasets_sorted = ["iris", "wine", "breast_cancer", "diabetes", "digits"]
    algorithms = list(BENCH_ALGOS.keys())

    # 表头
    header = f"{'Dataset':<15s}"
    for a in algorithms:
        header += f" | {a:<10s}"
    print(header)
    print("-" * len(header))

    for ds in datasets_sorted:
        if ds not in results:
            continue
        row = f"{DS_CONFIG[ds]['name']:<15s}"
        for algo in algorithms:
            val = results[ds].get(algo, {}).get(metric, "N/A")
            std_key = f"{metric}_std"
            std = results[ds].get(algo, {}).get(std_key, 0)
            if isinstance(val, float) and val > 0:
                row += f" | {val:.4f}"
                if std > 0.001:
                    row += f"\u00b1{std:.3f}"[:5]
            else:
                row += f" | {'--':>10s}"
        print(row)


def save_results_json(results, filepath):
    """保存结果到 JSON（处理 numpy 类型）"""
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=convert)
    print(f"[benchmark] 结果已保存: {filepath}")


def save_results_csv(results, filepath):
    """保存结果到 CSV"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    algorithms = list(BENCH_ALGOS.keys())
    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Algorithm", "Accuracy", "Accuracy_Std",
                          "F1_Macro", "F1_Std", "Train_Time", "Time_Std"])
        for ds_key, ds_res in results.items():
            ds_name = DS_CONFIG[ds_key]["name"]
            for algo in algorithms:
                r = ds_res.get(algo, {})
                if "error" in r:
                    continue
                writer.writerow([
                    ds_name, algo,
                    f"{r.get('accuracy', 0):.6f}",
                    f"{r.get('accuracy_std', 0):.6f}",
                    f"{r.get('f1_macro', 0):.6f}",
                    f"{r.get('f1_macro_std', 0):.6f}",
                    f"{r.get('train_time', 0):.6f}",
                    f"{r.get('train_time_std', 0):.6f}",
                ])
    print(f"[benchmark] CSV 已保存: {filepath}")


def generate_benchmark_charts(results):
    """生成所有基准对比图表"""
    ds_names = [DS_CONFIG[k]["name"] for k in ["iris","wine","breast_cancer","diabetes","digits"]]

    # 准确率矩阵
    mat, _, algos = results_to_matrix(results, "accuracy")
    plot_cross_dataset_heatmap(mat, ds_names, algos)

    # 时间矩阵
    mat_t, _, _ = results_to_matrix(results, "train_time")
    plot_cross_dataset_heatmap(mat_t, ds_names, algos, metric="train_time_s")

    # 每个数据集的算法对比
    for ds_key in results:
        ds_name = DS_CONFIG[ds_key]["name"]
        plot_algorithm_comparison(results[ds_key], "accuracy", ds_key)
        times = {k: v.get("train_time", 0) for k, v in results[ds_key].items()}
        plot_time_comparison(times, ds_key)

    print(f"[benchmark] 图表已生成到 {OUTPUT_DIR}/")


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("统一基准测试: 5-Fold CV × 6 算法 × 5 数据集")
    print("=" * 60)
    print(f"配置: {K_FOLD}-Fold, {N_RUNS} runs")
    print()

    # 运行基准
    results = run_full_benchmark()

    # 打印汇总
    print("\n\n" + "=" * 60)
    print("准确率对比矩阵 (Accuracy ± Std)")
    print("=" * 60)
    print_results_table(results, "accuracy")

    print("\n\n" + "=" * 60)
    print("F1-Macro 对比矩阵")
    print("=" * 60)
    print_results_table(results, "f1_macro")

    # 保存结果
    save_results_json(results, os.path.join(OUTPUT_DIR, "benchmark_results.json"))
    save_results_csv(results, os.path.join(OUTPUT_DIR, "benchmark_results.csv"))

    # 生成图表
    print("\n生成对比图表...")
    generate_benchmark_charts(results)

    # 最佳算法标注
    print("\n" + "=" * 60)
    print("各数据集最佳算法")
    print("=" * 60)
    for ds_key in ["iris", "wine", "breast_cancer", "diabetes", "digits"]:
        best_algo = max(results[ds_key].items(),
                        key=lambda kv: kv[1].get("accuracy", 0))
        print(f"  {DS_CONFIG[ds_key]['name']:<20s} → {best_algo[0]:<12s} "
              f"(acc={best_algo[1].get('accuracy', 0):.4f})")

    print("\n[DONE] 基准测试完成!")
