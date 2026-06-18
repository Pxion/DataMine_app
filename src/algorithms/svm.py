"""
SVM 支持向量机
纯NumPy实现，SMO (Sequential Minimal Optimization) 求解器
支持线性核 / RBF核 / 多项式核，支持多分类 (OvO)
"""
import numpy as np


class SVM:
    """SVM 分类器 (SMO 求解 + OvO 多分类)"""

    def __init__(self, kernel="rbf", C=1.0, gamma="scale",
                 degree=3, coef0=0.0, tol=0.001, max_iter=1000):
        """
        kernel: "linear" | "rbf" | "poly"
        C: 正则化参数（越大越不允许误分类）
        gamma: RBF/多项式核系数，"scale" 时 = 1/(n_features * X.var())
        degree: 多项式核次数
        coef0: 多项式核常数项
        tol: SMO容忍度
        max_iter: SMO最大迭代次数
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter

        # 训练后设置
        self.classes_ = None
        self.classifiers_ = []  # [(class_a, class_b, alpha, b, sv_X, sv_y, gamma)]

    # ── 训练 ────────────────────────────────────────────
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # 计算 gamma
        if self.gamma == "scale":
            gamma = 1.0 / (X.shape[1] * np.var(X) + 1e-10)
        elif self.gamma == "auto":
            gamma = 1.0 / X.shape[1]
        else:
            gamma = float(self.gamma)

        # OvO: 每对类别训练一个二分类 SVM
        self.classifiers_ = []
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                c_i, c_j = self.classes_[i], self.classes_[j]
                mask = (y == c_i) | (y == c_j)
                X_pair = X[mask]
                y_pair = y[mask]
                # 转为 ±1 标签
                y_bin = np.where(y_pair == c_i, 1, -1)

                alpha, b, sv_X, sv_y = self._smo_train(X_pair, y_bin, gamma)
                self.classifiers_.append((c_i, c_j, alpha, b, sv_X, sv_y, gamma))

        return self

    # ── SMO 训练 ────────────────────────────────────────
    def _smo_train(self, X, y, gamma):
        """
        Platt's SMO 算法
        返回: (alpha, b, support_vectors_X, support_vectors_y)
        """
        n = X.shape[0]
        alpha = np.zeros(n)
        b = 0.0

        # 预计算核矩阵（缓存加速）
        K = self._kernel_matrix(X, X, gamma)

        # 误差缓存
        E = -y.astype(np.float64)  # 初始 alpha=0, f(x)=0, E_i = f(x_i) - y_i = -y_i

        iter_count = 0
        entire_set = True
        num_changed = 0

        while iter_count < self.max_iter and (num_changed > 0 or entire_set):
            num_changed = 0

            if entire_set:
                # 遍历全部样本
                idxs = range(n)
            else:
                # 遍历非边界支持向量 (0 < alpha < C)
                idxs = [i for i in range(n) if 1e-8 < alpha[i] < self.C - 1e-8]

            for i in idxs:
                Ei = E[i]
                yi = y[i]

                # 检查 KKT 条件
                if (yi * Ei < -self.tol and alpha[i] < self.C) or \
                   (yi * Ei > self.tol and alpha[i] > 0):
                    # 选择第二个变量 j（启发式: 使 |Ei - Ej| 最大）
                    j = self._select_j(i, n, E, Ei)
                    if j is None:
                        continue
                    Ej = E[j]
                    yj = y[j]

                    # 保存旧 alpha
                    alpha_i_old = alpha[i]
                    alpha_j_old = alpha[j]

                    # 计算 L, H 边界
                    if yi != yj:
                        L = max(0, alpha[j] - alpha[i])
                        H = min(self.C, self.C + alpha[j] - alpha[i])
                    else:
                        L = max(0, alpha[i] + alpha[j] - self.C)
                        H = min(self.C, alpha[i] + alpha[j])

                    if L >= H - 1e-10:
                        continue

                    # 计算 eta
                    eta = K[i, i] + K[j, j] - 2 * K[i, j]
                    if eta <= 1e-10:
                        continue

                    # 更新 alpha_j (未裁剪)
                    alpha_j_new = alpha_j_old + yj * (Ei - Ej) / eta

                    # 裁剪
                    if alpha_j_new > H:
                        alpha_j_new = H
                    elif alpha_j_new < L:
                        alpha_j_new = L

                    if abs(alpha_j_new - alpha_j_old) < 1e-10:
                        continue

                    # 更新 alpha_i
                    alpha_i_new = alpha_i_old + yi * yj * (alpha_j_old - alpha_j_new)

                    # 更新 b
                    b1 = b - Ei - yi * (alpha_i_new - alpha_i_old) * K[i, i] \
                         - yj * (alpha_j_new - alpha_j_old) * K[i, j]
                    b2 = b - Ej - yi * (alpha_i_new - alpha_i_old) * K[i, j] \
                         - yj * (alpha_j_new - alpha_j_old) * K[j, j]

                    alpha[i] = alpha_i_new
                    alpha[j] = alpha_j_new

                    if 1e-8 < alpha[i] < self.C - 1e-8:
                        b = b1
                    elif 1e-8 < alpha[j] < self.C - 1e-8:
                        b = b2
                    else:
                        b = (b1 + b2) / 2

                    # 更新误差缓存
                    E[i] = self._decision_single(X[i], X, y, alpha, b, K[i], gamma) - yi
                    E[j] = self._decision_single(X[j], X, y, alpha, b, K[j], gamma) - yj

                    num_changed += 1

            iter_count += 1

            if entire_set:
                entire_set = False
            elif num_changed == 0:
                entire_set = True

        # 提取支持向量 (alpha > 1e-8)
        sv_mask = alpha > 1e-8
        return alpha, b, X[sv_mask], y[sv_mask]

    # ── 预测 ────────────────────────────────────────────
    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)

        if len(self.classes_) == 2:
            # 二分类
            c_i, c_j, alpha, b, sv_X, sv_y, gamma = self.classifiers_[0]
            scores = self._decision_function(X, sv_X, sv_y, alpha, b, gamma)
            return np.where(scores >= 0, c_i, c_j)

        # 多分类 OvO 投票
        n_samples = X.shape[0]
        votes = np.zeros((n_samples, len(self.classes_)))

        for c_i, c_j, alpha, b, sv_X, sv_y, gamma in self.classifiers_:
            scores = self._decision_function(X, sv_X, sv_y, alpha, b, gamma)
            winner = np.where(scores >= 0, c_i, c_j)
            for k in range(n_samples):
                idx = list(self.classes_).index(winner[k])
                votes[k, idx] += 1

        return self.classes_[np.argmax(votes, axis=1)]

    # ── 决策函数 ────────────────────────────────────────
    def _decision_function(self, X, sv_X, sv_y, alpha, b, gamma):
        """f(x) = Σ α_i y_i K(x_i, x) + b (只用支持向量)"""
        sv_alpha = alpha[alpha > 1e-8]
        K_test = self._kernel_matrix(X, sv_X, gamma)
        return K_test @ (sv_alpha * sv_y) + b

    def _decision_single(self, x, X, y, alpha, b, K_row, gamma):
        """单样本决策值（SMO内部使用，K_row是预计算的核向量）"""
        return np.sum(alpha * y * K_row) + b

    # ── α_j 选择 ────────────────────────────────────────
    @staticmethod
    def _select_j(i, n, E, Ei):
        """启发式选择 j: 使 |Ei - Ej| 最大"""
        best_j, best_delta = None, -1
        # 随机起始偏移
        start = np.random.randint(n)
        for offset in range(n):
            j = (start + offset) % n
            if j == i:
                continue
            delta = abs(Ei - E[j])
            if delta > best_delta:
                best_j = j
                best_delta = delta
        return best_j

    # ── 核函数 ──────────────────────────────────────────
    def _kernel_matrix(self, X1, X2, gamma):
        """计算核矩阵 K(X1, X2)"""
        if self.kernel == "linear":
            return X1 @ X2.T
        elif self.kernel == "poly":
            return (gamma * (X1 @ X2.T) + self.coef0) ** self.degree
        elif self.kernel == "rbf":
            # K(x,z) = exp(-gamma * ||x-z||²)
            # ||x-z||² = ||x||² + ||z||² - 2*x·z
            X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
            X2_sq = np.sum(X2 ** 2, axis=1)
            sq_dists = X1_sq + X2_sq - 2 * (X1 @ X2.T)
            sq_dists = np.maximum(sq_dists, 0)
            return np.exp(-gamma * sq_dists)
        else:
            raise ValueError(f"不支持的 kernel: {self.kernel}")


# ── 测试入口 ────────────────────────────────────────────
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    print("=" * 60)
    print("SVM (SMO) 测试")
    print("=" * 60)

    data = load_iris()
    X, y = data.data, data.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

    for ker in ["linear", "rbf", "poly"]:
        svm = SVM(kernel=ker, C=1.0, max_iter=5000)
        svm.fit(X_tr, y_tr)
        y_pred = svm.predict(X_te)
        acc = np.mean(y_pred == y_te)
        n_sv = sum(len(clf[3]) for clf in svm.classifiers_)  # count SV
        print(f"  SVM(kernel={ker:>6s}): acc={acc:.4f}  total_SV={n_sv}")

    print("\n[SVM] 测试完成")
