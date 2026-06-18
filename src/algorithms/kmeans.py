"""
K-Means 聚类
纯NumPy实现，支持 K-Means++ 初始化和自动 K 选择（肘部法则/轮廓系数）
"""
import numpy as np


class KMeans:
    """K-Means 聚类算法 — 纯NumPy手写"""

    def __init__(self, n_clusters=3, init="kmeans++", max_iter=300,
                 n_init=10, tol=1e-4, random_state=None):
        """
        n_clusters: 聚类数 k
        init: "random" | "kmeans++"
        max_iter: 单次运行最大迭代次数
        n_init: 随机初始化运行次数，取 inertia 最小的一次
        tol: 聚类中心移动阈值（提前停止）
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None  # (k, n_features)
        self.labels_ = None           # (n_samples,)
        self.inertia_ = None          # 簇内平方和

    # ── 训练 ────────────────────────────────────────────
    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]

        if self.n_clusters > n_samples:
            raise ValueError(f"n_clusters({self.n_clusters}) > n_samples({n_samples})")

        best_inertia = np.inf
        best_centers = None
        best_labels = None
        rng = np.random.RandomState(self.random_state)

        for _ in range(self.n_init):
            # 初始化
            if self.init == "kmeans++":
                centers = self._kmeans_plus_plus_init(X, rng)
            else:  # random
                indices = rng.choice(n_samples, self.n_clusters, replace=False)
                centers = X[indices].copy()

            # Lloyd's 迭代
            for iteration in range(self.max_iter):
                # E-step: 分配标签
                labels = self._assign_labels(X, centers)

                # M-step: 更新中心
                new_centers = np.zeros_like(centers)
                empty_clusters = []
                for k in range(self.n_clusters):
                    mask = labels == k
                    if np.sum(mask) > 0:
                        new_centers[k] = np.mean(X[mask], axis=0)
                    else:
                        empty_clusters.append(k)

                # 处理空簇: 重新初始化为最远离中心的点
                for ek in empty_clusters:
                    # 找离最近中心最远的点
                    dists = self._min_dist_to_centers(X, new_centers)
                    farthest = np.argmax(dists)
                    new_centers[ek] = X[farthest].copy()

                # 检查收敛
                shift = np.max(np.sqrt(np.sum((new_centers - centers) ** 2, axis=1)))
                centers = new_centers

                if shift < self.tol:
                    break

            # 最终分配 + 计算 inertia
            labels = self._assign_labels(X, centers)
            inertia = self._compute_inertia(X, centers, labels)

            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.copy()
                best_labels = labels.copy()

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        return self

    # ── 预测 ────────────────────────────────────────────
    def predict(self, X):
        """将新数据分配到最近的聚类中心"""
        X = np.asarray(X, dtype=np.float64)
        if self.cluster_centers_ is None:
            raise RuntimeError("请先调用 fit() 再 predict")
        return self._assign_labels(X, self.cluster_centers_)

    # ── 核心方法 ────────────────────────────────────────
    def _assign_labels(self, X, centers):
        """将每个样本分配到最近的聚类中心"""
        dists = self._pairwise_dist(X, centers)  # (n, k)
        return np.argmin(dists, axis=1)

    def _compute_inertia(self, X, centers, labels):
        """簇内平方和 Σ||x_i - μ_label_i||²"""
        inertia = 0.0
        for k in range(self.n_clusters):
            mask = labels == k
            if np.sum(mask) > 0:
                inertia += np.sum((X[mask] - centers[k]) ** 2)
        return inertia

    @staticmethod
    def _pairwise_dist(X, Y):
        """向量化欧氏距离矩阵: (n_X, n_Y)"""
        X_sq = np.sum(X ** 2, axis=1, keepdims=True)
        Y_sq = np.sum(Y ** 2, axis=1)
        dists = X_sq + Y_sq - 2 * X @ Y.T
        return np.sqrt(np.maximum(dists, 0))

    @staticmethod
    def _min_dist_to_centers(X, centers):
        """每个样本到最近中心的距离"""
        dists = KMeans._pairwise_dist(X, centers)
        return np.min(dists, axis=1)

    # ── K-Means++ 初始化 ────────────────────────────────
    def _kmeans_plus_plus_init(self, X, rng):
        """
        K-Means++ 初始化:
        1. 随机选第一个中心
        2. 后续中心按 D(x)² 加权随机选择 (D=到最近中心的距离)
        """
        n_samples = X.shape[0]
        centers = np.zeros((self.n_clusters, X.shape[1]))

        # 第一个中心: 均匀随机
        centers[0] = X[rng.randint(n_samples)].copy()

        for k in range(1, self.n_clusters):
            # 计算每个点到最近中心的距离
            dists = self._min_dist_to_centers(X, centers[:k])
            dists_sq = dists ** 2

            # 按距离²加权采样
            probs = dists_sq / np.sum(dists_sq)
            cumsum = np.cumsum(probs)
            r = rng.random()
            chosen = np.searchsorted(cumsum, r)
            centers[k] = X[chosen].copy()

        return centers

    # ── 辅助方法 ────────────────────────────────────────
    @staticmethod
    def elbow_method(X, k_range, **kwargs):
        """肘部法则: 返回不同 k 对应的 inertia 列表"""
        X = np.asarray(X, dtype=np.float64)
        inertias = []
        for k in k_range:
            km = KMeans(n_clusters=k, **kwargs)
            km.fit(X)
            inertias.append(km.inertia_)
        return list(k_range), inertias

    @staticmethod
    def silhouette_score(X, labels):
        """
        轮廓系数 (简化版)
        s(i) = (b(i) - a(i)) / max(a(i), b(i))
        a(i): 同簇内平均距离
        b(i): 到最近异簇的平均距离
        """
        X = np.asarray(X, dtype=np.float64)
        n = len(X)
        labels = np.asarray(labels)
        unique_labels = np.unique(labels)

        if len(unique_labels) <= 1:
            return 0.0

        # 预计算全部距离矩阵（内存消耗大，仅适用于小数据集）
        dist_mat = KMeans._pairwise_dist(X, X)

        scores = np.zeros(n)
        for i in range(n):
            label_i = labels[i]

            # 同簇平均距离 a(i)
            same_mask = labels == label_i
            same_mask[i] = False  # 排除自身
            if np.sum(same_mask) == 0:
                a_i = 0.0
            else:
                a_i = np.mean(dist_mat[i, same_mask])

            # 最近异簇平均距离 b(i)
            b_i = np.inf
            for other_label in unique_labels:
                if other_label == label_i:
                    continue
                other_mask = labels == other_label
                b_candidate = np.mean(dist_mat[i, other_mask])
                if b_candidate < b_i:
                    b_i = b_candidate

            if a_i == 0 and b_i == 0:
                scores[i] = 0.0
            else:
                scores[i] = (b_i - a_i) / max(a_i, b_i)

        return float(np.mean(scores))


# ── 测试入口 ────────────────────────────────────────────
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    print("=" * 60)
    print("K-Means 测试")
    print("=" * 60)

    data = load_iris()
    X, y = data.data, data.target

    for init in ["random", "kmeans++"]:
        km = KMeans(n_clusters=3, init=init, n_init=5, random_state=42)
        km.fit(X)
        sil = KMeans.silhouette_score(X, km.labels_)
        print(f"  KMeans(init={init:>8s}): inertia={km.inertia_:.2f}  silhouette={sil:.4f}")

    # 肘部法则
    ks, inertias = KMeans.elbow_method(X, range(1, 8), n_init=3, random_state=42)
    print(f"  Elbow: k={ks}, inertias={[f'{i:.0f}' for i in inertias]}")

    print("\n[K-Means] 测试完成")
