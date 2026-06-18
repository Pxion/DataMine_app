"""
可视化模块 — matplotlib 纯手写
- 基础图表: 直方图/箱线图/小提琴图/蜂群图/热力图/散点图矩阵
- 降维可视化: PCA 2D/3D（纯 NumPy 手写）
- 高级图表: 平行坐标/雷达图/层次聚类树/ECDF/学习曲线/特征重要性
- 模型评估: 混淆矩阵/ROC曲线/算法对比柱状图
- 报告生成: 自动输出 PNG 到 output/，支持生成 HTML 综合报告
"""
import os
import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互后端，避免 GUI 依赖
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as ticker

# ============================================================
# 配置
# ============================================================
try:
    from config import VISUALIZATION, OUTPUT_DIR
    FIGSIZE = VISUALIZATION.get("figsize", (10, 6))
    DPI = VISUALIZATION.get("dpi", 100)
    CMAP = VISUALIZATION.get("cmap", "viridis")
    STYLE = VISUALIZATION.get("style", "seaborn-v0_8-darkgrid")
    IMG_FORMAT = VISUALIZATION.get("format", "png")
except ImportError:
    FIGSIZE = (10, 6)
    DPI = 100
    CMAP = "viridis"
    STYLE = "seaborn-v0_8-darkgrid"
    IMG_FORMAT = "png"
    OUTPUT_DIR = "output"

# 尝试加载 seaborn 风格
try:
    plt.style.use(STYLE)
except Exception:
    plt.style.use("default")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 中文字体适配
# ============================================================
def _setup_chinese_font():
    """尝试配置中文字体，失败则回退英文"""
    cn_fonts = ["Microsoft YaHei", "SimHei", "Noto Sans SC", "WenQuanYi Micro Hei",
                 "Arial Unicode MS", "DejaVu Sans"]
    for f in cn_fonts:
        try:
            plt.rcParams["font.sans-serif"] = [f, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return f
        except Exception:
            continue
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    return "DejaVu Sans"

_CN_FONT = _setup_chinese_font()

# 配色方案（色调统一，区分度高）
COLORS = ["#0d9488", "#ef4444", "#3b82f6", "#f59e0b", "#8b5cf6",
          "#10b981", "#ec4899", "#6366f1", "#14b8a6", "#f97316"]


def _save_and_close(fig, filename):
    """统一保存+关闭"""
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", format=IMG_FORMAT)
    plt.close(fig)
    print(f"[visualization] 已保存: {path}")
    return path


# ============================================================
# 1. 基础图表
# ============================================================

def plot_histograms(X, feature_names, y=None, class_names=None,
                    dataset_name="dataset", max_features=8):
    """
    特征分布直方图
    - 若提供 y，则按类别分层着色
    """
    n_features = min(X.shape[1], max_features)
    n_rows = (n_features + 3) // 4
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else ([axes] if n_rows == 1 and n_features == 1 else axes)

    for i in range(n_features):
        ax = axes[i]
        if y is not None:
            for c in np.unique(y):
                mask = y == c
                label = class_names[c] if class_names and c < len(class_names) else f"Class {c}"
                ax.hist(X[mask, i], bins=20, alpha=0.6, label=label,
                        color=COLORS[c % len(COLORS)])
            ax.legend(fontsize=7)
        else:
            ax.hist(X[:, i], bins=20, color=COLORS[0], alpha=0.7)
        ax.set_title(f"{feature_names[i]}" if i < len(feature_names) else f"Feature {i}",
                     fontsize=10)
        ax.tick_params(labelsize=8)

    # 隐藏多余子图
    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{dataset_name} — 特征分布直方图", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return _save_and_close(fig, f"{dataset_name}_histograms.{IMG_FORMAT}")


def plot_boxplots(X, feature_names, dataset_name="dataset", max_features=12):
    """特征箱线图"""
    n_features = min(X.shape[1], max_features)
    fig, ax = plt.subplots(figsize=(max(10, n_features * 0.6), 6))
    # 每列作为一个 box
    data_cols = [X[:, i] for i in range(n_features)]
    bp = ax.boxplot(data_cols, patch_artist=True)
    ax.set_xticklabels(feature_names[:n_features], rotation=45, ha="right", fontsize=8)
    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS[0])
        patch.set_alpha(0.5)
    ax.set_title(f"{dataset_name} — 特征箱线图", fontsize=14, fontweight="bold")
    ax.set_ylabel("值")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    return _save_and_close(fig, f"{dataset_name}_boxplots.{IMG_FORMAT}")


def plot_correlation_heatmap(X, feature_names, dataset_name="dataset"):
    """特征相关性热力图"""
    corr = np.corrcoef(X, rowvar=False)
    n = corr.shape[0]
    display_n = min(n, 15)  # 限制显示的特征数
    corr_sub = corr[:display_n, :display_n]
    names_sub = feature_names[:display_n]

    fig, ax = plt.subplots(figsize=(max(8, display_n * 0.65), max(6, display_n * 0.55)))
    im = ax.imshow(corr_sub, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(display_n))
    ax.set_yticks(range(display_n))
    ax.set_xticklabels(names_sub, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(names_sub, fontsize=7)

    # 标注数值
    for i in range(display_n):
        for j in range(display_n):
            ax.text(j, i, f"{corr_sub[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if abs(corr_sub[i, j]) > 0.5 else "black")

    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"{dataset_name} — 特征相关性热力图", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save_and_close(fig, f"{dataset_name}_correlation.{IMG_FORMAT}")


def plot_scatter_matrix(X, feature_names, y=None, class_names=None,
                        dataset_name="dataset", max_features=5):
    """散点图矩阵（取前 max_features 个特征）"""
    n = min(X.shape[1], max_features)
    fig, axes = plt.subplots(n, n, figsize=(n * 2.5, n * 2.5))

    for i in range(n):
        for j in range(n):
            ax = axes[i, j] if n > 1 else axes
            if i == j:
                # 对角 → 直方图
                if y is not None:
                    for c in np.unique(y):
                        ax.hist(X[y == c, i], bins=15, alpha=0.5,
                                color=COLORS[c % len(COLORS)])
                else:
                    ax.hist(X[:, i], bins=15, color=COLORS[0], alpha=0.7)
            else:
                # 非对角 → 散点图
                if y is not None:
                    for c in np.unique(y):
                        mask = y == c
                        label = class_names[c] if class_names and c < len(class_names) else f"Class {c}"
                        ax.scatter(X[mask, j], X[mask, i], s=3, alpha=0.5,
                                   color=COLORS[c % len(COLORS)], label=label)
                else:
                    ax.scatter(X[:, j], X[:, i], s=3, alpha=0.5, color=COLORS[0])

            if i == n - 1 and j < n:
                ax.set_xlabel(feature_names[j] if j < len(feature_names) else f"f{j}", fontsize=7)
            if j == 0 and i > 0:
                ax.set_ylabel(feature_names[i] if i < len(feature_names) else f"f{i}", fontsize=7)
            ax.tick_params(labelsize=6)

    if n == 1:
        axes = np.array([[axes]])
    if y is not None:
        handles, labels = axes[0, 1].get_legend_handles_labels() if n > 1 else ([], [])
        if handles:
            fig.legend(handles, labels, loc="upper right", fontsize=7,
                       bbox_to_anchor=(1.15, 0.98))

    fig.suptitle(f"{dataset_name} — 散点图矩阵", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.92, 0.97])
    return _save_and_close(fig, f"{dataset_name}_scatter_matrix.{IMG_FORMAT}")


# ============================================================
# 2. PCA 降维可视化（纯 NumPy 手写）
# ============================================================

class PCA:
    """主成分分析 — 纯 NumPy 实现"""
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components_ = None    # (n_components, n_features)
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X):
        # 中心化
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # 协方差矩阵
        cov = np.cov(X_centered, rowvar=False)

        # 特征分解
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # 降序排列
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 取前 k 个
        self.components_ = eigenvectors[:, :self.n_components].T
        total_var = np.sum(eigenvalues)
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / total_var
        return self

    def transform(self, X):
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def plot_pca_2d(X, y, class_names=None, dataset_name="dataset"):
    """PCA 2D 降维可视化"""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(9, 7))
    for c in np.unique(y):
        mask = y == c
        label = class_names[c] if class_names and c < len(class_names) else f"Class {c}"
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=20, alpha=0.7,
                   color=COLORS[c % len(COLORS)], label=label, edgecolors="white", linewidth=0.5)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} 方差)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} 方差)", fontsize=11)
    ax.set_title(f"{dataset_name} — PCA 2D 降维可视化", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save_and_close(fig, f"{dataset_name}_pca_2d.{IMG_FORMAT}"), pca.explained_variance_ratio_


def plot_pca_3d(X, y, class_names=None, dataset_name="dataset"):
    """PCA 3D 降维可视化"""
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for c in np.unique(y):
        mask = y == c
        label = class_names[c] if class_names and c < len(class_names) else f"Class {c}"
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                   s=15, alpha=0.7, color=COLORS[c % len(COLORS)],
                   label=label, edgecolors="white", linewidth=0.3)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})")
    ax.set_title(f"{dataset_name} — PCA 3D 降维可视化", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save_and_close(fig, f"{dataset_name}_pca_3d.{IMG_FORMAT}"), pca.explained_variance_ratio_


def plot_pca_explained_variance(X, dataset_name="dataset"):
    """PCA 各主成分方差解释比例"""
    pca_full = PCA(n_components=min(X.shape[0], X.shape[1]))
    pca_full.fit(X)
    ratios = pca_full.explained_variance_ratio_
    cumsum = np.cumsum(ratios)

    n_show = min(20, len(ratios))
    fig, ax1 = plt.subplots(figsize=(10, 5))

    bars = ax1.bar(range(1, n_show + 1), ratios[:n_show], color=COLORS[0], alpha=0.7, label="individual")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio", color=COLORS[0])
    ax1.tick_params(axis="y", labelcolor=COLORS[0])

    ax2 = ax1.twinx()
    ax2.plot(range(1, n_show + 1), cumsum[:n_show], "o-", color=COLORS[1],
             linewidth=2, markersize=6, label="cumulative")
    ax2.set_ylabel("Cumulative Variance Ratio", color=COLORS[1])
    ax2.tick_params(axis="y", labelcolor=COLORS[1])
    ax2.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5, label="90% threshold")

    ax1.set_title(f"{dataset_name} — PCA 方差解释比例", fontsize=14, fontweight="bold")
    ax1.set_xticks(range(1, n_show + 1))
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    fig.tight_layout()
    return _save_and_close(fig, f"{dataset_name}_pca_variance.{IMG_FORMAT}")


# ============================================================
# 3. 模型评估图表
# ============================================================

def plot_confusion_matrix(cm, class_names, dataset_name="dataset", algo_name="algo"):
    """混淆矩阵热力图"""
    n = cm.shape[0]
    fig, ax = plt.subplots(figsize=(max(6, n * 0.7), max(5, n * 0.6)))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")

    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names[:n] if class_names else range(n),
                       rotation=45, ha="right")
    ax.set_yticklabels(class_names[:n] if class_names else range(n))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{algo_name} on {dataset_name} — Confusion Matrix", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    return _save_and_close(fig, f"{dataset_name}_{algo_name}_cm.{IMG_FORMAT}")


def plot_roc_curves(fpr_dict, tpr_dict, auc_dict=None, dataset_name="dataset"):
    """
    ROC 曲线
    - fpr_dict: {algo_name: fpr_array}
    - tpr_dict: {algo_name: tpr_array}
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, (name, fpr) in enumerate(fpr_dict.items()):
        tpr = tpr_dict[name]
        auc_val = auc_dict.get(name, None) if auc_dict else None
        label = f"{name} (AUC={auc_val:.3f})" if auc_val is not None else name
        ax.plot(fpr, tpr, color=COLORS[i % len(COLORS)], linewidth=2, label=label)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"{dataset_name} — ROC Curves", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save_and_close(fig, f"{dataset_name}_roc.{IMG_FORMAT}")


def plot_algorithm_comparison(results_dict, metric="accuracy", dataset_name="dataset"):
    """
    多算法性能对比柱状图
    - results_dict: {algo_name: {metric_name: value, ...}}
    """
    algo_names = list(results_dict.keys())
    values = [results_dict[a].get(metric, 0) for a in algo_names]
    colors = [COLORS[i % len(COLORS)] for i in range(len(algo_names))]

    fig, ax = plt.subplots(figsize=(max(8, len(algo_names) * 1.2), 6))
    bars = ax.bar(algo_names, values, color=colors, alpha=0.85, edgecolor="white", linewidth=1)

    # 数值标注
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(f"{dataset_name} — {metric.upper()} Comparison", fontsize=14, fontweight="bold")
    ax.set_ylim(0, min(1.1, max(values) * 1.2))
    ax.tick_params(axis="x", rotation=30, labelsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save_and_close(fig, f"{dataset_name}_{metric}_comparison.{IMG_FORMAT}")


def plot_time_comparison(times_dict, dataset_name="dataset"):
    """算法训练时间对比"""
    algo_names = list(times_dict.keys())
    values = [times_dict[a] for a in algo_names]

    fig, ax = plt.subplots(figsize=(max(8, len(algo_names) * 1.2), 6))
    bars = ax.barh(algo_names, values, color=[COLORS[i % len(COLORS)] for i in range(len(algo_names))],
                   alpha=0.85, edgecolor="white")

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}s", va="center", fontsize=9)

    ax.set_xlabel("Training Time (seconds)", fontsize=12)
    ax.set_title(f"{dataset_name} — Training Time", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return _save_and_close(fig, f"{dataset_name}_time.{IMG_FORMAT}")


def plot_cross_dataset_heatmap(results_matrix, datasets, algos, metric="accuracy"):
    """
    跨数据集热力图
    - results_matrix: shape (n_datasets, n_algos)
    """
    fig, ax = plt.subplots(figsize=(max(8, len(algos) * 1.0), max(5, len(datasets) * 0.6)))
    im = ax.imshow(results_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    for i in range(len(datasets)):
        for j in range(len(algos)):
            ax.text(j, i, f"{results_matrix[i, j]:.3f}", ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if results_matrix[i, j] < 0.6 else "black")

    ax.set_xticks(range(len(algos)))
    ax.set_yticks(range(len(datasets)))
    ax.set_xticklabels(algos, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(datasets, fontsize=10)
    ax.set_title(f"Cross-Dataset {metric.upper()} Heatmap", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    return _save_and_close(fig, f"cross_dataset_{metric}.{IMG_FORMAT}")


# ============================================================
# 4. 高级图表
# ============================================================

def plot_violin(X, feature_names, y=None, class_names=None,
                dataset_name="dataset", max_features=12):
    """小提琴图 — 展示特征分布密度"""
    n_features = min(X.shape[1], max_features)
    fig, ax = plt.subplots(figsize=(max(10, n_features * 1.2), 6))

    positions = np.arange(n_features)
    if y is not None:
        unique_classes = np.unique(y)
        width = 0.8 / max(len(unique_classes), 1)
        for ci, c in enumerate(unique_classes):
            mask = y == c
            offset = (ci - len(unique_classes) / 2 + 0.5) * width
            data_cols = [X[mask, i] for i in range(n_features)]
            vp = ax.violinplot(data_cols, positions=positions + offset,
                               widths=width * 0.9, showmeans=True,
                               showextrema=True)
            for body in vp['bodies']:
                body.set_facecolor(COLORS[ci % len(COLORS)])
                body.set_alpha(0.5)
            vp['cmeans'].set_color(COLORS[ci % len(COLORS)])
            vp['cmaxes'].set_color(COLORS[ci % len(COLORS)])
            vp['cmins'].set_color(COLORS[ci % len(COLORS)])
            vp['cbars'].set_color(COLORS[ci % len(COLORS)])
        # Legend
        handles = [plt.Line2D([0], [0], color=COLORS[ci % len(COLORS)], lw=4,
                              label=class_names[ci] if class_names and ci < len(class_names) else f"Class {ci}")
                   for ci in range(len(unique_classes))]
        ax.legend(handles=handles, fontsize=8, loc='upper right')
    else:
        data_cols = [X[:, i] for i in range(n_features)]
        vp = ax.violinplot(data_cols, positions=positions, showmeans=True, showextrema=True)
        for body in vp['bodies']:
            body.set_facecolor(COLORS[0])
            body.set_alpha(0.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(feature_names[:n_features], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("值", fontsize=11)
    ax.set_title(f"{dataset_name} — 特征分布小提琴图", fontsize=14, fontweight="bold")
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    return _save_and_close(fig, f"{dataset_name}_violin.{IMG_FORMAT}")


def plot_swarm(X, feature_names, y=None, class_names=None,
               dataset_name="dataset", max_features=8):
    """蜂群图 — 基于抖动散点的分布展示"""
    n_features = min(X.shape[1], max_features)
    n_rows = (n_features + 3) // 4
    fig, axes = plt.subplots(n_rows, min(4, n_features),
                             figsize=(16, 4 * max(n_rows, 1)))
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes

    for i in range(n_features):
        ax = axes[i] if n_features > 1 else axes[0]
        col_data = X[:, i]
        if y is not None:
            unique_classes = np.unique(y)
            for ci, c in enumerate(unique_classes):
                mask = y == c
                vals = col_data[mask]
                # 抖动 x 坐标
                jitter = np.random.RandomState(42).uniform(-0.15, 0.15, len(vals))
                label = class_names[ci] if class_names and ci < len(class_names) else f"Class {ci}"
                ax.scatter(ci + jitter, vals, s=4, alpha=0.5,
                          color=COLORS[ci % len(COLORS)], label=label if i == 0 else "")
            ax.set_xticks(range(len(unique_classes)))
            ax.set_xticklabels([class_names[ci] if class_names and ci < len(class_names) else str(ci)
                               for ci in unique_classes], fontsize=8)
        else:
            jitter = np.random.RandomState(42).uniform(-0.2, 0.2, len(col_data))
            ax.scatter(jitter, col_data, s=4, alpha=0.5, color=COLORS[0])
            ax.set_xticks([])
        ax.set_ylabel(feature_names[i] if i < len(feature_names) else f"f{i}",
                      fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(axis='y', alpha=0.3)

    if y is not None and n_features > 1:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=8, loc='upper right',
                   bbox_to_anchor=(1.02, 0.95))

    # Hide unused subplots
    if n_features > 1:
        for j in range(n_features, len(axes)):
            axes[j].set_visible(False)

    fig.suptitle(f"{dataset_name} — 蜂群图", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])
    return _save_and_close(fig, f"{dataset_name}_swarm.{IMG_FORMAT}")


def plot_parallel_coordinates(X, feature_names, y=None, class_names=None,
                               dataset_name="dataset", max_features=10):
    """平行坐标图 — 多维特征关系可视化"""
    n_features = min(X.shape[1], max_features)
    # 归一化到 [0,1] 便于平行坐标展示
    X_norm = (X[:, :n_features] - X[:, :n_features].min(axis=0)) / \
             (X[:, :n_features].max(axis=0) - X[:, :n_features].min(axis=0) + 1e-10)

    fig, ax = plt.subplots(figsize=(12, 6))

    if y is not None:
        unique_classes = np.unique(y)
        # 采样避免线条过多
        max_samples = 500
        if X.shape[0] > max_samples:
            idx = np.random.RandomState(42).choice(X.shape[0], max_samples, replace=False)
            X_norm, y = X_norm[idx], y[idx]

        for c in unique_classes:
            mask = y == c
            label = class_names[c] if class_names and c < len(class_names) else f"Class {c}"
            for row in X_norm[mask][:100]:  # 每类最多 100 条
                ax.plot(range(n_features), row, color=COLORS[c % len(COLORS)],
                       alpha=0.15, linewidth=0.8)
            # 每类画一条均值线
            mean_line = X_norm[mask].mean(axis=0)
            ax.plot(range(n_features), mean_line, color=COLORS[c % len(COLORS)],
                   linewidth=3, label=label)
    else:
        for row in X_norm[:200]:
            ax.plot(range(n_features), row, color=COLORS[0], alpha=0.1, linewidth=0.5)

    ax.set_xticks(range(n_features))
    ax.set_xticklabels(feature_names[:n_features], fontsize=9)
    ax.set_xlim(0, n_features - 1)
    ax.set_ylabel("归一化值", fontsize=11)
    ax.set_title(f"{dataset_name} — 平行坐标图", fontsize=14, fontweight="bold")
    if y is not None:
        ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save_and_close(fig, f"{dataset_name}_parallel_coords.{IMG_FORMAT}")


def plot_radar(X, y, feature_names, class_names=None,
               dataset_name="dataset", max_features=8):
    """雷达图 — 各类别特征均值对比"""
    n_features = min(X.shape[1], max_features)
    unique_classes = np.unique(y)

    # 计算各类别均值并归一化
    class_means = []
    for c in unique_classes:
        class_means.append(X[y == c, :n_features].mean(axis=0))
    class_means = np.array(class_means)
    # 统一归一化
    mins = class_means.min(axis=0)
    maxs = class_means.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    class_means_norm = (class_means - mins) / ranges

    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for ci in range(len(unique_classes)):
        values = class_means_norm[ci].tolist() + [class_means_norm[ci][0]]
        label = class_names[ci] if class_names and ci < len(class_names) else f"Class {unique_classes[ci]}"
        ax.fill(angles, values, alpha=0.15, color=COLORS[ci % len(COLORS)])
        ax.plot(angles, values, 'o-', linewidth=2, color=COLORS[ci % len(COLORS)],
                label=label, markersize=4)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names[:n_features], fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_title(f"{dataset_name} — 雷达图 (各类别特征均值)", fontsize=14,
                 fontweight="bold", pad=20)
    ax.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save_and_close(fig, f"{dataset_name}_radar.{IMG_FORMAT}")


def plot_dendrogram(X, feature_names=None, dataset_name="dataset",
                    method="ward", max_leafs=30):
    """层次聚类树状图 — 纯 NumPy 实现凝聚聚类"""
    from scipy.cluster.hierarchy import dendrogram, linkage
    # 采样避免过大
    if X.shape[0] > max_leafs:
        idx = np.random.RandomState(42).choice(X.shape[0], max_leafs, replace=False)
        X_sub = X[idx]
        labels_sub = idx.astype(str)
    else:
        X_sub = X
        labels_sub = np.arange(X.shape[0]).astype(str)

    Z = linkage(X_sub, method=method)

    fig, ax = plt.subplots(figsize=(12, max(5, max_leafs * 0.2)))
    dendrogram(Z, labels=labels_sub, leaf_font_size=7,
               color_threshold=0.7 * max(Z[:, 2]),
               above_threshold_color='gray', ax=ax)
    ax.set_title(f"{dataset_name} — 层次聚类树状图 ({method})",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("样本索引", fontsize=11)
    ax.set_ylabel("距离", fontsize=11)
    ax.tick_params(axis='x', labelsize=6, rotation=90)
    fig.tight_layout()
    return _save_and_close(fig, f"{dataset_name}_dendrogram.{IMG_FORMAT}")


def plot_ecdf(X, feature_names, dataset_name="dataset", max_features=8):
    """经验累积分布函数 (ECDF)"""
    n_features = min(X.shape[1], max_features)
    n_cols = min(4, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4, n_rows * 3.5))
    if n_features == 1:
        axes = np.array([[axes]])
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

    for i in range(n_features):
        ax = axes[i]
        sorted_data = np.sort(X[:, i])
        y_vals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

        ax.step(sorted_data, y_vals, where='post', color=COLORS[0], linewidth=2)
        ax.fill_between(sorted_data, y_vals, step='post', alpha=0.15, color=COLORS[0])

        # 标注分位数
        for q in [0.25, 0.5, 0.75]:
            q_val = np.quantile(sorted_data, q)
            ax.axvline(q_val, color=COLORS[3], linestyle='--', alpha=0.5, linewidth=0.8)
            ax.axhline(q, color=COLORS[3], linestyle='--', alpha=0.3, linewidth=0.5)

        ax.set_title(feature_names[i] if i < len(feature_names) else f"f{i}", fontsize=9)
        ax.set_xlabel("值", fontsize=8)
        ax.set_ylabel("CDF", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)

    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{dataset_name} — 经验累积分布 (ECDF)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save_and_close(fig, f"{dataset_name}_ecdf.{IMG_FORMAT}")


def plot_class_pie(y, class_names=None, dataset_name="dataset"):
    """类别分布饼图 + 环形图"""
    unique, counts = np.unique(y, return_counts=True)
    labels = [class_names[c] if class_names and c < len(class_names) else f"Class {c}"
              for c in unique]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 饼图
    wedges1, texts1, autotexts1 = ax1.pie(
        counts, labels=labels, autopct='%1.1f%%',
        colors=[COLORS[i % len(COLORS)] for i in range(len(unique))],
        startangle=90, textprops={'fontsize': 10})
    ax1.set_title("类别分布 (饼图)", fontsize=13, fontweight="bold")

    # 环形图
    wedges2, texts2, autotexts2 = ax2.pie(
        counts, labels=labels, autopct='%1.1f%%',
        colors=[COLORS[i % len(COLORS)] for i in range(len(unique))],
        startangle=90, wedgeprops=dict(width=0.4),
        textprops={'fontsize': 10})
    ax2.set_title("类别分布 (环形图)", fontsize=13, fontweight="bold")

    fig.suptitle(f"{dataset_name} — 类别分布", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save_and_close(fig, f"{dataset_name}_class_pie.{IMG_FORMAT}")


def plot_feature_importance(importances, feature_names, dataset_name="dataset",
                             algo_name="Model", top_n=15):
    """特征重要性条形图"""
    n = min(len(importances), top_n)
    if len(importances) > top_n:
        idx = np.argsort(np.abs(importances))[-top_n:]
    else:
        idx = np.argsort(np.abs(importances))

    sorted_imp = np.array(importances)[idx]
    sorted_names = [feature_names[i] if i < len(feature_names) else f"f{i}"
                    for i in idx]

    colors = [COLORS[0] if v >= 0 else COLORS[1] for v in sorted_imp]

    fig, ax = plt.subplots(figsize=(8, max(5, n * 0.35)))
    bars = ax.barh(range(n), sorted_imp, color=colors, alpha=0.85,
                   edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.set_xlabel("重要性", fontsize=11)
    ax.set_title(f"{dataset_name} — {algo_name} 特征重要性", fontsize=14,
                 fontweight="bold")
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)

    # 标注数值
    for bar, val in zip(bars, sorted_imp):
        x_pos = bar.get_width() + (0.01 if val >= 0 else -0.01)
        ha = 'left' if val >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va='center', ha=ha, fontsize=8)

    fig.tight_layout()
    return _save_and_close(fig, f"{dataset_name}_{algo_name}_importance.{IMG_FORMAT}")


def plot_learning_curve(train_scores, test_scores, epochs, dataset_name="dataset",
                         algo_name="Model"):
    """学习曲线 — 训练/测试准确率随训练轮数变化"""
    fig, ax = plt.subplots(figsize=(8, 5))

    epochs_range = range(1, len(train_scores) + 1) if epochs is None else epochs
    ax.plot(epochs_range, train_scores, 'o-', color=COLORS[0], linewidth=2,
            markersize=4, label='训练集')
    ax.plot(epochs_range, test_scores, 's-', color=COLORS[1], linewidth=2,
            markersize=4, label='测试集')

    # 填充区域
    ax.fill_between(epochs_range, train_scores, test_scores, alpha=0.1,
                     color=COLORS[2])

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(f"{dataset_name} — {algo_name} 学习曲线", fontsize=14,
                 fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return _save_and_close(fig, f"{dataset_name}_{algo_name}_learning_curve.{IMG_FORMAT}")


def plot_pair_density(X, feature_names, y=None, class_names=None,
                       dataset_name="dataset", max_features=5):
    """配对密度图 — 对角 KDE 密度 + 非对角散点"""
    n = min(X.shape[1], max_features)
    fig, axes = plt.subplots(n, n, figsize=(n * 2.8, n * 2.8))

    for i in range(n):
        for j in range(n):
            ax = axes[i][j] if n > 1 else axes
            if i == j:
                # 对角 → KDE 密度曲线
                for ci, c in enumerate(np.unique(y)) if y is not None else [None]:
                    if y is not None:
                        data_col = X[y == c, i]
                        color = COLORS[ci % len(COLORS)]
                        lbl = class_names[ci] if class_names and ci < len(class_names) else f"Class {c}"
                    else:
                        data_col = X[:, i]
                        color = COLORS[0]
                        lbl = None
                    # 简易 KDE
                    bw = np.std(data_col) * 1.06 * len(data_col) ** (-0.2) if len(data_col) > 1 else 0.1
                    bw = max(bw, 1e-6)
                    grid = np.linspace(data_col.min(), data_col.max(), 100)
                    kde_vals = np.mean(
                        np.exp(-0.5 * ((grid[:, None] - data_col[None, :]) / bw) ** 2),
                        axis=1) / (bw * np.sqrt(2 * np.pi))
                    ax.fill_between(grid, 0, kde_vals, alpha=0.3, color=color)
                    ax.plot(grid, kde_vals, color=color, linewidth=1.5, label=lbl)
            else:
                # 非对角 → 散点
                if y is not None:
                    for ci, c in enumerate(np.unique(y)):
                        mask = y == c
                        ax.scatter(X[mask, j], X[mask, i], s=2, alpha=0.4,
                                  color=COLORS[ci % len(COLORS)])
                else:
                    ax.scatter(X[:, j], X[:, i], s=2, alpha=0.4, color=COLORS[0])

            if j == 0 and i > 0 and feature_names and i < len(feature_names):
                ax.set_ylabel(feature_names[i], fontsize=7)
            if i == n - 1 and feature_names and j < len(feature_names):
                ax.set_xlabel(feature_names[j], fontsize=7)
            ax.tick_params(labelsize=5)

    if n == 1:
        axes = np.array([[axes]])
    if y is not None and n > 1:
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, fontsize=7, loc='upper right',
                      bbox_to_anchor=(1.12, 0.98))

    fig.suptitle(f"{dataset_name} — 配对密度图", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.9, 0.96])
    return _save_and_close(fig, f"{dataset_name}_pair_density.{IMG_FORMAT}")


# ============================================================
# 5. HTML 报告生成
# ============================================================

def _fig_to_b64(fig):
    """matplotlib figure → base64 PNG 字符串"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_html_report(dataset_name, X, y, feature_names, class_names,
                          results=None, upload_name=None):
    """
    生成完整 HTML 数据分析报告
    返回 HTML 字符串 + 图表路径列表
    """
    title = upload_name or dataset_name
    sections = []

    # 1. 数据概览
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    missing = int(np.sum(np.isnan(X)))

    sections.append(f"""
    <div class="report-section">
        <h2>1. 数据概览</h2>
        <div class="stats-grid">
            <div class="stat-item"><span class="stat-val">{n_samples}</span><span class="stat-lbl">样本数</span></div>
            <div class="stat-item"><span class="stat-val">{n_features}</span><span class="stat-lbl">特征数</span></div>
            <div class="stat-item"><span class="stat-val">{n_classes}</span><span class="stat-lbl">类别数</span></div>
            <div class="stat-item"><span class="stat-val">{missing}</span><span class="stat-lbl">缺失值</span></div>
        </div>
    </div>""")

    # 2. 统计摘要
    sections.append('<div class="report-section"><h2>2. 统计摘要</h2><table class="rpt-table"><tr><th>特征</th><th>均值</th><th>标准差</th><th>最小值</th><th>25%</th><th>中位数</th><th>75%</th><th>最大值</th></tr>')
    for i in range(min(n_features, 20)):
        col = X[:, i]
        fname = feature_names[i] if i < len(feature_names) and feature_names[i] else f"特征_{i}"
        sections.append(f"<tr><td>{fname}</td><td>{np.mean(col):.3f}</td><td>{np.std(col):.3f}</td>"
                        f"<td>{np.min(col):.3f}</td><td>{np.percentile(col,25):.3f}</td>"
                        f"<td>{np.median(col):.3f}</td><td>{np.percentile(col,75):.3f}</td>"
                        f"<td>{np.max(col):.3f}</td></tr>")
    sections.append('</table></div>')

    # 3. 类别分布
    sections.append('<div class="report-section"><h2>3. 类别分布</h2><table class="rpt-table"><tr><th>类别</th><th>数量</th><th>占比</th></tr>')
    unique, counts = np.unique(y, return_counts=True)
    for c, cnt in zip(unique, counts):
        label = class_names[c] if class_names and c < len(class_names) else f"Class {c}"
        sections.append(f"<tr><td>{label}</td><td>{cnt}</td><td>{cnt/n_samples:.2%}</td></tr>")
    sections.append('</table></div>')

    # 4. 可视化图表 — 全部内嵌为 base64
    sections.append('<div class="report-section"><h2>4. 可视化分析</h2><div class="chart-gallery">')

    chart_funcs = [
        ("特征分布直方图", lambda: plot_histograms(X, feature_names, y, class_names, dataset_name)),
        ("特征箱线图", lambda: plot_boxplots(X, feature_names, dataset_name)),
        ("特征小提琴图", lambda: plot_violin(X, feature_names, y, class_names, dataset_name)),
        ("相关性热力图", lambda: plot_correlation_heatmap(X, feature_names, dataset_name)),
        ("类别分布", lambda: plot_class_pie(y, class_names, dataset_name)),
        ("ECDF分布", lambda: plot_ecdf(X, feature_names, dataset_name)),
    ]

    if n_features >= 2:
        chart_funcs.append(("散点图矩阵", lambda: plot_scatter_matrix(X, feature_names, y, class_names, dataset_name)))
        chart_funcs.append(("配对密度图", lambda: plot_pair_density(X, feature_names, y, class_names, dataset_name)))
        chart_funcs.append(("PCA 2D降维", lambda: plot_pca_2d(X, y, class_names, dataset_name)[0]))
        chart_funcs.append(("PCA 方差解释", lambda: plot_pca_explained_variance(X, dataset_name)))
        chart_funcs.append(("平行坐标图", lambda: plot_parallel_coordinates(X, feature_names, y, class_names, dataset_name)))

    if n_features >= 3:
        chart_funcs.append(("雷达图", lambda: plot_radar(X, y, feature_names, class_names, dataset_name)))
        chart_funcs.append(("PCA 3D降维", lambda: plot_pca_3d(X, y, class_names, dataset_name)[0]))

    if n_features >= 4:
        chart_funcs.append(("蜂群图", lambda: plot_swarm(X, feature_names, y, class_names, dataset_name)))

    # 层次聚类
    try:
        chart_funcs.append(("层次聚类树", lambda: plot_dendrogram(X, feature_names, dataset_name)))
    except Exception:
        pass

    # 生成所有图表
    chart_paths = []
    for chart_name, chart_fn in chart_funcs:
        try:
            path = chart_fn()
            chart_paths.append(path)
            # 生成内嵌 base64
            fig = plt.figure()
            img = plt.imread(path)
            plt.imshow(img)
            plt.axis('off')
            b64 = _fig_to_b64(fig)
            sections.append(
                f'<div class="chart-card"><h4>{chart_name}</h4>'
                f'<img src="data:image/png;base64,{b64}" alt="{chart_name}"></div>')
        except Exception as e:
            print(f"[report] 图表 {chart_name} 生成失败: {e}")

    sections.append('</div></div>')

    # 组合完整 HTML
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — 数据分析报告</title>
<style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", sans-serif;
           background: #f8f9fa; color: #212529; line-height:1.7; padding: 24px; }}
    .report-container {{ max-width: 1100px; margin:0 auto; }}
    h1 {{ font-size: 26px; color: #0d9488; margin-bottom: 4px; }}
    .subtitle {{ color: #6b7280; font-size: 14px; margin-bottom: 28px; }}
    .report-section {{ background: #fff; border-radius: 10px; padding: 24px; margin-bottom: 20px;
                       box-shadow: 0 1px 4px rgba(0,0,0,0.04); border: 1px solid #e5e7eb; }}
    .report-section h2 {{ font-size: 18px; color: #0d9488; margin-bottom: 16px; padding-bottom: 8px;
                         border-bottom: 2px solid #f0fdfa; }}
    .stats-grid {{ display: flex; gap: 16px; flex-wrap: wrap; }}
    .stat-item {{ flex:1; min-width: 100px; text-align:center; padding: 16px;
                 background: #f0fdfa; border-radius: 8px; }}
    .stat-val {{ display:block; font-size: 28px; font-weight: 700; color: #0d9488; }}
    .stat-lbl {{ font-size: 12px; color: #6b7280; text-transform: uppercase; }}
    .rpt-table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    .rpt-table th {{ background: #f0fdfa; color: #0d9488; padding: 8px 10px; text-align: left;
                    font-weight: 600; border-bottom: 2px solid #e5e7eb; }}
    .rpt-table td {{ padding: 6px 10px; border-bottom: 1px solid #f3f4f6; }}
    .chart-gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
                     gap: 16px; }}
    .chart-card {{ background: #fafafa; border-radius: 8px; padding: 12px;
                  border: 1px solid #e5e7eb; }}
    .chart-card h4 {{ font-size: 13px; color: #374151; margin-bottom: 8px; }}
    .chart-card img {{ width: 100%; border-radius: 4px; }}
    @media (max-width: 768px) {{ .chart-gallery {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="report-container">
    <h1>{title}</h1>
    <p class="subtitle">自动生成 · {n_samples} 样本 · {n_features} 特征 · {n_classes} 类别</p>
    {"".join(sections)}
</div>
</body>
</html>"""

    # 保存 HTML 文件
    html_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_report.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"[report] HTML报告已保存: {html_path}")

    return html, chart_paths, html_path


# ============================================================
# 6. 报告生成（兼容旧 API）
# ============================================================

def generate_dataset_report(dataset_name, X, y, feature_names, class_names,
                            results=None):
    """
    为单个数据集生成完整可视化报告
    返回所有图片路径列表
    """
    paths = []
    print(f"\n{'='*50}")
    print(f"[visualization] 生成报告: {dataset_name} ({X.shape[0]} samples, {X.shape[1]} features)")
    print(f"{'='*50}")

    # 基础图表
    paths.append(plot_histograms(X, feature_names, y, class_names, dataset_name))
    paths.append(plot_boxplots(X, feature_names, dataset_name))
    paths.append(plot_correlation_heatmap(X, feature_names, dataset_name))
    if X.shape[1] >= 2:
        paths.append(plot_scatter_matrix(X, feature_names, y, class_names, dataset_name))

    # PCA 降维
    if X.shape[1] >= 2:
        pv, evr = plot_pca_2d(X, y, class_names, dataset_name)
        paths.append(pv)
        paths.append(plot_pca_explained_variance(X, dataset_name))
        if X.shape[1] >= 3:
            p3, _ = plot_pca_3d(X, y, class_names, dataset_name)
            paths.append(p3)

    # 模型评估图（若有结果）
    if results:
        # algorithm comparison
        acc_dict = {algo: {k: v for k, v in res.items()}
                    for algo, res in results.items()}
        paths.append(plot_algorithm_comparison(acc_dict, "accuracy", dataset_name))
        paths.append(plot_algorithm_comparison(acc_dict, "f1", dataset_name))

        # time comparison
        time_dict = {algo: res.get("train_time", 0) for algo, res in results.items()}
        paths.append(plot_time_comparison(time_dict, dataset_name))

        # ROC
        fpr_d, tpr_d, auc_d = {}, {}, {}
        for algo, res in results.items():
            if "fpr" in res and "tpr" in res:
                fpr_d[algo] = np.array(res["fpr"])
                tpr_d[algo] = np.array(res["tpr"])
                auc_d[algo] = res.get("auc", 0)
        if fpr_d:
            paths.append(plot_roc_curves(fpr_d, tpr_d, auc_d, dataset_name))

    print(f"[visualization] {dataset_name}: 生成 {len(paths)} 张图表")
    return paths


# ============================================================
# 测试入口
# ============================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_loader import load_dataset
    from src.preprocessing import StandardScaler

    print("=" * 60)
    print("可视化模块测试")
    print("=" * 60)

    # 加载 Iris
    X, y, fnames, cnames = load_dataset("iris")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # 基础图表
    plot_histograms(Xs, fnames, y, cnames, "iris")
    plot_boxplots(Xs, fnames, "iris")
    plot_correlation_heatmap(Xs, fnames, "iris")
    plot_scatter_matrix(Xs, fnames, y, cnames, "iris")

    # PCA
    pv, evr = plot_pca_2d(Xs, y, cnames, "iris")
    print(f"  PCA 2D EVR: {evr[0]:.1%}, {evr[1]:.1%}")
    plot_pca_explained_variance(Xs, "iris")

    # 混淆矩阵
    cm = np.array([[48, 2, 0], [1, 47, 2], [0, 3, 47]])
    plot_confusion_matrix(cm, cnames, "iris", "KNN")

    print(f"\n[visualization] 所有图表已保存到 {OUTPUT_DIR}/")