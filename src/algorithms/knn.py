"""
KNN (K-Nearest Neighbors) 分类器
纯NumPy实现，支持 uniform / distance / gaussian 加权
支持欧氏距离和曼哈顿距离
"""
import numpy as np


class KNN:
    """K近邻分类器 — 纯NumPy手写

    距离加权策略:
      - uniform:  等权重 1/k
      - distance: w = 1/(d + eps)
      - gaussian: w = exp(-d^2 / (2*sigma^2))    (高斯核加权)
    """

    def __init__(self, k=5, weights="uniform", metric="euclidean", sigma=None):
        """
        k: 近邻数
        weights: "uniform" | "distance" | "gaussian"
        metric: "euclidean" | "manhattan"
        sigma: 高斯核带宽 (仅 weights="gaussian")，None时自动计算
        """
        self.k = k
        self.weights = weights
        self.metric = metric
        self.sigma = sigma
        self.X_train = None
        self.y_train = None
        self.classes_ = None
        self.sigma_used_ = None  # 训练后记录实际使用的 sigma

    # ── 训练 ────────────────────────────────────────────
    def fit(self, X, y):
        """KNN无显式训练过程，仅存储数据。若 weights=gaussian 且未指定 sigma，自动计算带宽"""
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y)
        self.classes_ = np.unique(self.y_train)

        # 高斯核带宽自动计算: 使用所有成对距离的中位数 / sqrt(2)
        if self.weights == "gaussian" and self.sigma is None:
            # 采样不超过 500 个样本计算中位距离
            n = min(X.shape[0], 500)
            idx = np.random.RandomState(42).choice(X.shape[0], n, replace=False)
            X_sample = self.X_train[idx]
            dists = self._compute_distances(X_sample)
            median_dist = np.median(dists[dists > 0])  # 排除自距离
            self.sigma_used_ = median_dist / np.sqrt(2)
        elif self.sigma is not None:
            self.sigma_used_ = self.sigma
        else:
            self.sigma_used_ = None

        return self

    # ── 预测 ────────────────────────────────────────────
    def predict(self, X):
        """返回类别标签"""
        proba = self.predict_proba(X)
        class_idx = np.argmax(proba, axis=1)
        return self.classes_[class_idx]

    def predict_proba(self, X):
        """返回每个类别的概率分布 (n_samples, n_classes)"""
        X = np.asarray(X, dtype=np.float64)
        if self.X_train is None:
            raise RuntimeError("请先调用 fit() 再 predict")

        # 距离矩阵 (n_query, n_train)
        dists = self._compute_distances(X)

        # 对每个查询样本取 k 近邻
        n_query = X.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_query, n_classes), dtype=np.float64)

        # 按距离升序取前 k 个
        knn_indices = np.argpartition(dists, self.k - 1, axis=1)[:, :self.k]
        knn_dists = np.take_along_axis(dists, knn_indices, axis=1)
        knn_labels = self.y_train[knn_indices]

        # 类别映射: label → column index
        class_to_col = {c: i for i, c in enumerate(self.classes_)}

        if self.weights == "uniform":
            for i in range(n_query):
                for j in range(self.k):
                    col = class_to_col[knn_labels[i, j]]
                    proba[i, col] += 1.0 / self.k
        elif self.weights == "distance":
            eps = 1e-10  # 防止除以零
            for i in range(n_query):
                w = 1.0 / (knn_dists[i] + eps)
                w_sum = np.sum(w)
                for j in range(self.k):
                    col = class_to_col[knn_labels[i, j]]
                    proba[i, col] += w[j] / w_sum
        elif self.weights == "gaussian":
            # 高斯核加权: w = exp(-d^2 / (2*sigma^2))
            sigma = self.sigma_used_ or 1.0
            for i in range(n_query):
                d2 = knn_dists[i] ** 2
                w = np.exp(-d2 / (2.0 * sigma ** 2))
                w_sum = np.sum(w)
                for j in range(self.k):
                    col = class_to_col[knn_labels[i, j]]
                    proba[i, col] += w[j] / max(w_sum, 1e-15)
        else:
            raise ValueError(f"不支持的 weights: {self.weights}")

        return proba

    # ── 距离计算 ────────────────────────────────────────
    def _compute_distances(self, X):
        """
        计算 X (n_query, d) 到 X_train (n_train, d) 的距离矩阵
        使用向量化运算，避免显式双重循环

        欧氏距离: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        曼哈顿距离: sum(|a - b|)
        """
        if self.metric == "euclidean":
            # 向量化欧氏距离
            X_sq = np.sum(X ** 2, axis=1, keepdims=True)          # (n_query, 1)
            Xt_sq = np.sum(self.X_train ** 2, axis=1)             # (n_train,)
            cross = X @ self.X_train.T                             # (n_query, n_train)
            sq_dists = X_sq + Xt_sq - 2 * cross
            # 数值精度修正：防止极小负数开根号
            sq_dists = np.maximum(sq_dists, 0)
            return np.sqrt(sq_dists)

        elif self.metric == "manhattan":
            # 向量化曼哈顿距离
            # |x - y| 对每对样本求和
            # 用广播: X[:, None, :] - X_train[None, :, :]  → (n_query, n_train, d)
            return np.sum(np.abs(X[:, None, :] - self.X_train[None, :, :]), axis=2)

        else:
            raise ValueError(f"不支持的 metric: {self.metric}")


# ── 测试入口 ────────────────────────────────────────────
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    print("=" * 60)
    print("KNN 测试")
    print("=" * 60)

    data = load_iris()
    X, y = data.data, data.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

    for w in ["uniform", "distance"]:
        for m in ["euclidean", "manhattan"]:
            knn = KNN(k=5, weights=w, metric=m)
            knn.fit(X_tr, y_tr)
            y_pred = knn.predict(X_te)
            acc = np.mean(y_pred == y_te)
            print(f"  KNN(k=5, weights={w:>8s}, metric={m:>10s}): acc={acc:.4f}")

    # 加权KNN vs 普通KNN对比
    knn_u = KNN(k=5, weights="uniform")
    knn_u.fit(X_tr, y_tr)
    acc_u = np.mean(knn_u.predict(X_te) == y_te)

    knn_d = KNN(k=5, weights="distance")
    knn_d.fit(X_tr, y_tr)
    acc_d = np.mean(knn_d.predict(X_te) == y_te)

    print(f"\n  普通KNN: {acc_u:.4f}")
    print(f"  加权KNN: {acc_d:.4f}")
    print("\n[KNN] 测试完成")
