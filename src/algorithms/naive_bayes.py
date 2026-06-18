"""
朴素贝叶斯分类器
纯NumPy实现：GaussianNB（连续特征）+ MultinomialNB（离散特征/文本）
"""
import numpy as np


class GaussianNB:
    """
    高斯朴素贝叶斯 — 假设每类每个特征服从高斯分布

    P(class|x) ∝ P(class) * ∏ P(x_i|class)
    P(x_i|class) = N(μ_class,i, σ²_class,i)
    """

    def __init__(self, var_smoothing=1e-9):
        """
        var_smoothing: 方差平滑项，防止除零和数值下溢
        """
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.priors_ = None       # log P(class)
        self.means_ = None        # (n_classes, n_features)
        self.vars_ = None         # (n_classes, n_features)

    # ── 训练 ────────────────────────────────────────────
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.priors_ = np.zeros(n_classes)
        self.means_ = np.zeros((n_classes, n_features))
        self.vars_ = np.zeros((n_classes, n_features))

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.priors_[idx] = np.log(len(X_c) / len(y))    # log prior
            self.means_[idx] = np.mean(X_c, axis=0)
            self.vars_[idx] = np.var(X_c, axis=0) + self.var_smoothing

        return self

    # ── 预测 ────────────────────────────────────────────
    def predict(self, X):
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]

    def predict_proba(self, X):
        """返回类别概率（softmax over log joint）"""
        jll = self._joint_log_likelihood(X)
        # log-sum-exp 数值稳定 softmax
        jll_max = np.max(jll, axis=1, keepdims=True)
        exp_jll = np.exp(jll - jll_max)
        return exp_jll / np.sum(exp_jll, axis=1, keepdims=True)

    # ── 核心：联合对数似然 ─────────────────────────────
    def _joint_log_likelihood(self, X):
        """
        计算 log P(class) + Σ log P(x_i|class)
        P(x_i|class) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
        log P(x_i|class) = -0.5 * log(2πσ²) - 0.5 * (x-μ)²/σ²
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        # (n_classes, n_samples, n_features)
        log_likelihood = np.zeros((n_classes, n_samples), dtype=np.float64)

        for idx in range(n_classes):
            # 高斯对数似然: -0.5 * Σ [log(2πσ²) + (x-μ)²/σ²]
            diff = X - self.means_[idx]                            # (n_samples, n_features)
            term1 = -0.5 * np.sum(np.log(2 * np.pi * self.vars_[idx]))
            term2 = -0.5 * np.sum(diff ** 2 / self.vars_[idx], axis=1)
            log_likelihood[idx] = term1 + term2 + self.priors_[idx]

        return log_likelihood.T  # (n_samples, n_classes)


class MultinomialNB:
    """
    多项式朴素贝叶斯 — 适用于离散特征（词频/计数等）

    假设特征服从多项分布，使用拉普拉斯平滑
    P(x_i|c) = (count(x_i in c) + alpha) / (sum(count in c) + alpha * n_features)
    """

    def __init__(self, alpha=1.0):
        """
        alpha: 拉普拉斯平滑参数（alpha=1 即经典拉普拉斯平滑）
        """
        self.alpha = alpha
        self.classes_ = None
        self.priors_ = None          # log P(class)
        self.feature_log_probs_ = None  # log P(x_i|class)

    # ── 训练 ────────────────────────────────────────────
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        # 确保非负（多项式要求）
        if np.any(X < 0):
            raise ValueError("多项式朴素贝叶斯要求特征值非负")

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.priors_ = np.zeros(n_classes)
        self.feature_log_probs_ = np.zeros((n_classes, n_features))

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.priors_[idx] = np.log(len(X_c) / len(y))

            # 每类每特征计数 + 拉普拉斯平滑
            counts = np.sum(X_c, axis=0) + self.alpha
            total = np.sum(counts)
            self.feature_log_probs_[idx] = np.log(counts) - np.log(total)

        return self

    # ── 预测 ────────────────────────────────────────────
    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        # log P(X|c) = Σ x_i * log P(x_i|c)
        log_probs = X @ self.feature_log_probs_.T + self.priors_  # (n, n_classes)
        return self.classes_[np.argmax(log_probs, axis=1)]

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        log_probs = X @ self.feature_log_probs_.T + self.priors_
        log_probs_max = np.max(log_probs, axis=1, keepdims=True)
        exp_probs = np.exp(log_probs - log_probs_max)
        return exp_probs / np.sum(exp_probs, axis=1, keepdims=True)


# ── 测试入口 ────────────────────────────────────────────
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    print("=" * 60)
    print("朴素贝叶斯 测试")
    print("=" * 60)

    data = load_iris()
    X, y = data.data, data.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

    # GaussianNB
    gnb = GaussianNB()
    gnb.fit(X_tr, y_tr)
    y_pred = gnb.predict(X_te)
    acc_g = np.mean(y_pred == y_te)
    print(f"  GaussianNB: acc={acc_g:.4f}")

    # 验证概率分布
    proba = gnb.predict_proba(X_te[:5])
    print(f"  proba[:5] sum check: {proba.sum(axis=1).round(4)}")

    # MultinomialNB (用 MinMaxScaler 确保非负)
    from sklearn.preprocessing import MinMaxScaler
    mms = MinMaxScaler()
    X_mm = mms.fit_transform(X) * 10 + 1  # 放大并 +1 避免零值
    X_trm, X_tem, y_tr2, y_te2 = train_test_split(X_mm, y, test_size=0.3, random_state=42)

    mnb = MultinomialNB(alpha=1.0)
    mnb.fit(X_trm, y_tr2)
    y_pred2 = mnb.predict(X_tem)
    acc_m = np.mean(y_pred2 == y_te2)
    print(f"  MultinomialNB: acc={acc_m:.4f}")

    print("\n[朴素贝叶斯] 测试完成")
