"""
MLP 多层感知机（神经网络）
纯NumPy实现，反向传播，支持 ReLU/Sigmoid/Tanh

优化器: SGD / Momentum / Adam (Kingma & Ba, ICLR 2015)
"""
import numpy as np


class MLP:
    """
    多层感知机分类器 — 纯NumPy手写

    架构: Input → Hidden[0] → ... → Hidden[-1] → Softmax Output
    损失: 交叉熵 (Cross-Entropy)
    优化器: SGD / Momentum / Adam
        - Adam: Kingma & Ba (2015), "Adam: A Method for Stochastic Optimization"
                自适应学习率，兼顾 Momentum 与 RMSprop 优势
    """

    def __init__(self, hidden_layers=None, activation="relu",
                 learning_rate=0.01, batch_size=32, epochs=200,
                 optimizer="adam", momentum=0.9,
                 beta1=0.9, beta2=0.999, eps=1e-8,
                 weight_decay=0.0, random_state=None):
        """
        hidden_layers: 隐层神经元数列表，如 [64, 32]
        activation: "relu" | "sigmoid" | "tanh"
        learning_rate: 学习率
        batch_size: 批量大小
        epochs: 训练轮数
        optimizer: "sgd" | "momentum" | "adam"
        momentum: 动量系数 (仅 sgd/momentum 使用)
        beta1: Adam 一阶矩衰减率 (仅 adam)
        beta2: Adam 二阶矩衰减率 (仅 adam)
        eps: Adam 数值稳定常数 (仅 adam)
        weight_decay: L2 正则化系数
        """
        if hidden_layers is None:
            hidden_layers = [64, 32]
        self.hidden_layers = hidden_layers
        self.activation_name = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.random_state = random_state

        # 训练后设置
        self.weights_ = []      # list of W (不包含偏置)
        self.biases_ = []       # list of b
        self.classes_ = None
        self.loss_history_ = []

    # ── 训练 ────────────────────────────────────────────
    def fit(self, X, y, verbose=False):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # 将标签转为 one-hot
        y_onehot = np.zeros((len(y), n_classes))
        for i, label in enumerate(y):
            idx = list(self.classes_).index(label)
            y_onehot[i, idx] = 1.0

        # 初始化参数
        self._init_weights(X.shape[1], n_classes)

        # 优化器状态缓存 (惰性初始化)
        v_W = None   # Momentum: 速度  /  Adam: m_t (一阶矩)
        v_b = None   # Momentum: 速度  /  Adam: m_t (一阶矩)
        s_W = None   # Adam: v_t (二阶矩)
        s_b = None   # Adam: v_t (二阶矩)
        t = 0        # Adam: 时间步计数器

        n_samples = X.shape[0]
        self.loss_history_ = []

        for epoch in range(self.epochs):
            # 打乱数据
            indices = np.random.RandomState(self.random_state + epoch if self.random_state else None).permutation(n_samples)

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                X_batch = X[batch_idx]
                y_batch = y_onehot[batch_idx]

                # 前向传播
                cache = self._forward(X_batch)

                # 计算损失
                loss = self._cross_entropy_loss(cache["output"], y_batch)
                epoch_loss += loss

                # 反向传播
                grads = self._backward(cache, y_batch)

                # L2 正则化梯度
                if self.weight_decay > 0:
                    for g in grads["dW"]:
                        g_idx = grads["dW"].index(g)
                        g += self.weight_decay * self.weights_[g_idx]

                # 参数更新
                t += 1

                if self.optimizer == "adam":
                    # ── Adam 优化器 ──
                    # 参考: Kingma & Ba, ICLR 2015
                    if v_W is None:
                        v_W = [np.zeros_like(w) for w in self.weights_]
                        v_b = [np.zeros_like(b) for b in self.biases_]
                        s_W = [np.zeros_like(w) for w in self.weights_]
                        s_b = [np.zeros_like(b) for b in self.biases_]

                    for i in range(len(self.weights_)):
                        # 一阶矩估计 (动量)
                        v_W[i] = self.beta1 * v_W[i] + (1 - self.beta1) * grads["dW"][i]
                        v_b[i] = self.beta1 * v_b[i] + (1 - self.beta1) * grads["db"][i]
                        # 二阶矩估计 (RMSprop)
                        s_W[i] = self.beta2 * s_W[i] + (1 - self.beta2) * grads["dW"][i] ** 2
                        s_b[i] = self.beta2 * s_b[i] + (1 - self.beta2) * grads["db"][i] ** 2

                        # 偏差校正
                        v_W_hat = v_W[i] / (1 - self.beta1 ** t)
                        v_b_hat = v_b[i] / (1 - self.beta1 ** t)
                        s_W_hat = s_W[i] / (1 - self.beta2 ** t)
                        s_b_hat = s_b[i] / (1 - self.beta2 ** t)

                        # 参数更新: θ = θ - lr * m_hat / (√v_hat + ε)
                        self.weights_[i] -= self.learning_rate * v_W_hat / (np.sqrt(s_W_hat) + self.eps)
                        self.biases_[i] -= self.learning_rate * v_b_hat / (np.sqrt(s_b_hat) + self.eps)

                elif self.optimizer == "momentum":
                    if v_W is None:
                        v_W = [np.zeros_like(w) for w in self.weights_]
                        v_b = [np.zeros_like(b) for b in self.biases_]
                    for i in range(len(self.weights_)):
                        v_W[i] = self.momentum * v_W[i] - self.learning_rate * grads["dW"][i]
                        v_b[i] = self.momentum * v_b[i] - self.learning_rate * grads["db"][i]
                        self.weights_[i] += v_W[i]
                        self.biases_[i] += v_b[i]
                else:  # sgd
                    for i in range(len(self.weights_)):
                        self.weights_[i] -= self.learning_rate * grads["dW"][i]
                        self.biases_[i] -= self.learning_rate * grads["db"][i]

                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.loss_history_.append(avg_loss)

            if verbose and (epoch % 50 == 0 or epoch == self.epochs - 1):
                y_pred = self.predict(X)
                acc = np.mean(y_pred == y)
                print(f"  Epoch {epoch:4d}: loss={avg_loss:.6f}  acc={acc:.4f}")

        return self

    # ── 预测 ────────────────────────────────────────────
    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        cache = self._forward(X)
        return cache["output"]

    # ── 前向传播 ────────────────────────────────────────
    def _forward(self, X):
        """
        前向传播: X → Linear → Activation → ... → Softmax
        返回 cache dict 供反向传播使用
        """
        cache = {"input": X}  # A[0] 的别称
        A = X
        n_layers = len(self.weights_)

        for i in range(n_layers - 1):
            Z = A @ self.weights_[i] + self.biases_[i]   # 线性变换
            A = self._activate(Z, self.activation_name)    # 激活
            cache[f"Z{i}"] = Z
            cache[f"A{i}"] = A

        # 输出层: Linear + Softmax
        Z_out = A @ self.weights_[-1] + self.biases_[-1]
        A_out = self._softmax(Z_out)
        cache[f"Z{n_layers - 1}"] = Z_out
        cache["output"] = A_out

        return cache

    # ── 反向传播 ────────────────────────────────────────
    def _backward(self, cache, y_onehot):
        """
        反向传播计算梯度
        返回 {"dW": [...], "db": [...]}
        """
        n_layers = len(self.weights_)
        n_samples = y_onehot.shape[0]

        grads_W = [None] * n_layers
        grads_b = [None] * n_layers

        # 输出层梯度: softmax + cross-entropy → dZ = A_out - y
        A_out = cache["output"]
        dZ = (A_out - y_onehot) / n_samples                          # (n, n_classes)

        # 逐层反向传播
        for i in range(n_layers - 1, -1, -1):
            if i == n_layers - 1:
                A_prev = cache[f"A{i - 1}"] if i > 0 else cache["input"]
            else:
                A_prev = cache[f"A{i - 1}"] if i > 0 else cache["input"]

            grads_W[i] = A_prev.T @ dZ                                 # (d_prev, d_curr)
            grads_b[i] = np.sum(dZ, axis=0)                            # (d_curr,)

            if i > 0:
                dA = dZ @ self.weights_[i].T                           # 反向传播到激活前
                Z_prev = cache[f"Z{i - 1}"]
                dZ = dA * self._activate_grad(Z_prev, self.activation_name)

        return {"dW": grads_W, "db": grads_b}

    # ── 参数初始化 ──────────────────────────────────────
    def _init_weights(self, n_features, n_classes):
        """He初始化 (适用于ReLU) / Xavier初始化 (适用于Sigmoid/Tanh)"""
        self.weights_ = []
        self.biases_ = []
        rng = np.random.RandomState(self.random_state)

        layer_dims = [n_features] + self.hidden_layers + [n_classes]

        for i in range(len(layer_dims) - 1):
            d_in, d_out = layer_dims[i], layer_dims[i + 1]

            if self.activation_name == "relu":
                # He 初始化
                std = np.sqrt(2.0 / d_in)
            else:
                # Xavier 初始化
                std = np.sqrt(1.0 / d_in)

            W = rng.randn(d_in, d_out) * std
            b = np.zeros(d_out)
            self.weights_.append(W)
            self.biases_.append(b)

    # ── 激活函数 ────────────────────────────────────────
    @staticmethod
    def _activate(Z, name):
        if name == "relu":
            return np.maximum(0, Z)
        elif name == "sigmoid":
            # 数值稳定 sigmoid
            Z_clipped = np.clip(Z, -500, 500)
            return 1.0 / (1.0 + np.exp(-Z_clipped))
        elif name == "tanh":
            return np.tanh(Z)
        else:
            raise ValueError(f"不支持 activation: {name}")

    @staticmethod
    def _activate_grad(Z, name):
        """激活函数导数（输入为激活前的 Z）"""
        if name == "relu":
            return (Z > 0).astype(np.float64)
        elif name == "sigmoid":
            sig = MLP._activate(Z, "sigmoid")
            return sig * (1 - sig)
        elif name == "tanh":
            t = np.tanh(Z)
            return 1 - t ** 2
        else:
            raise ValueError(f"不支持 activation: {name}")

    # ── Softmax ─────────────────────────────────────────
    @staticmethod
    def _softmax(Z):
        """数值稳定 softmax"""
        Z_max = np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z - Z_max)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    # ── 损失函数 ────────────────────────────────────────
    @staticmethod
    def _cross_entropy_loss(probs, y_onehot):
        """交叉熵损失 (平均)"""
        eps = 1e-15
        return -np.mean(np.sum(y_onehot * np.log(probs + eps), axis=1))


# ── 测试入口 ────────────────────────────────────────────
if __name__ == "__main__":
    from sklearn.datasets import load_iris, load_digits
    from sklearn.model_selection import train_test_split

    print("=" * 60)
    print("MLP 神经网络 测试")
    print("=" * 60)

    # Iris
    data = load_iris()
    X, y = data.data, data.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    for act in ["relu", "sigmoid", "tanh"]:
        mlp = MLP(hidden_layers=[16, 8], activation=act,
                  learning_rate=0.05, epochs=300, batch_size=16,
                  optimizer="momentum", random_state=42)
        mlp.fit(X_tr, y_tr)
        acc = np.mean(mlp.predict(X_te) == y_te)
        print(f"  MLP(act={act:>7s}): acc={acc:.4f}  final_loss={mlp.loss_history_[-1]:.6f}")

    print("\n[MLP] 测试完成")
