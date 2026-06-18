"""
决策树 C4.5 分类器
纯NumPy实现，基于信息增益率分裂，支持预剪枝和后剪枝
"""
import numpy as np
from collections import Counter


class DecisionTreeC45:
    """
    C4.5 决策树

    特点: 信息增益率选特征、连续值二分处理、预剪枝(max_depth/min_samples_split)
    和后剪枝(基于验证集错误率剪枝)
    """

    def __init__(self, max_depth=10, min_samples_split=2, min_info_gain=1e-5,
                 pruning="post", val_split=0.2):
        """
        max_depth: 最大深度
        min_samples_split: 节点最少样本数才继续分裂
        min_info_gain: 最小信息增益，低于此值停止分裂
        pruning: "none" | "post"（后剪枝）
        val_split: 后剪枝用的验证集比例（从训练集中划分）
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_info_gain = min_info_gain
        self.pruning = pruning
        self.val_split = val_split
        self.tree_ = None
        self.classes_ = None

    # ── 训练 ────────────────────────────────────────────
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        if self.pruning == "post" and self.val_split > 0:
            # 划分验证集用于后剪枝
            n_val = int(len(X) * self.val_split)
            indices = np.random.RandomState(42).permutation(len(X))
            val_idx, train_idx = indices[:n_val], indices[n_val:]
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        # 构建决策树
        self.tree_ = self._build_tree(X_train, y_train, depth=0)

        # 后剪枝
        if self.pruning == "post" and X_val is not None:
            self.tree_ = self._post_prune(self.tree_, X_val, y_val)

        return self

    # ── 预测 ────────────────────────────────────────────
    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._predict_one(x, self.tree_) for x in X])

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        n_classes = len(self.classes_)
        proba = np.zeros((len(X), n_classes))
        for i, x in enumerate(X):
            counts = self._predict_counts(x, self.tree_)
            total = sum(counts.values()) or 1
            for c, cnt in counts.items():
                idx = list(self.classes_).index(c)
                proba[i, idx] = cnt / total
        return proba

    # ── 树结构 ──────────────────────────────────────────
    # 树节点为 dict:
    #   {"is_leaf": bool, "class": int, "counts": {c: n}, "n_samples": int}
    #   分支节点额外:
    #   {"feature": int, "threshold": float, "gain_ratio": float,
    #    "left": node, "right": node, "n_samples": int}

    def _build_tree(self, X, y, depth):
        """递归构建决策树"""
        n_samples = len(y)
        counts = Counter(y)
        majority_class = counts.most_common(1)[0][0]

        # 停止条件
        if (depth >= self.max_depth or
                n_samples < self.min_samples_split or
                len(counts) == 1):
            return {"is_leaf": True, "class": majority_class,
                    "counts": counts, "n_samples": n_samples}

        # 寻找最佳分裂
        best = self._best_split(X, y)

        if best is None or best["gain_ratio"] < self.min_info_gain:
            return {"is_leaf": True, "class": majority_class,
                    "counts": counts, "n_samples": n_samples}

        # 分裂
        left_mask = X[:, best["feature"]] <= best["threshold"]
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return {"is_leaf": True, "class": majority_class,
                    "counts": counts, "n_samples": n_samples}

        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            "is_leaf": False,
            "feature": best["feature"],
            "threshold": best["threshold"],
            "gain_ratio": best["gain_ratio"],
            "left": left,
            "right": right,
            "n_samples": n_samples,
            "class": majority_class,  # 分支节点也存多数类（剪枝时用）
        }

    def _best_split(self, X, y):
        """
        遍历所有特征和可能的分裂点，返回信息增益率最大的
        连续值: 二分法，取相邻值的均值作为候选分裂点
        """
        n_samples = len(y)
        parent_entropy = self._entropy(y)
        best = None

        for feature in range(X.shape[1]):
            values = X[:, feature]
            unique_sorted = np.sort(np.unique(values))

            if len(unique_sorted) < 2:
                continue

            # 候选分裂点: 相邻唯一值的均值
            thresholds = (unique_sorted[:-1] + unique_sorted[1:]) / 2.0

            # 对每个分裂点计算信息增益率
            for threshold in thresholds:
                left_mask = values <= threshold
                right_mask = ~left_mask

                n_left = np.sum(left_mask)
                n_right = n_samples - n_left

                if n_left == 0 or n_right == 0:
                    continue

                y_left, y_right = y[left_mask], y[right_mask]

                # 信息增益
                entropy_left = self._entropy(y_left)
                entropy_right = self._entropy(y_right)
                cond_entropy = (n_left / n_samples) * entropy_left + \
                               (n_right / n_samples) * entropy_right
                info_gain = parent_entropy - cond_entropy

                # 分裂信息（intrinsic value）
                pl = n_left / n_samples
                pr = n_right / n_samples
                split_info = -pl * self._log2_safe(pl) - pr * self._log2_safe(pr)

                # 信息增益率
                if split_info < 1e-10:
                    gain_ratio = 0.0
                else:
                    gain_ratio = info_gain / split_info

                if best is None or gain_ratio > best["gain_ratio"]:
                    best = {
                        "feature": feature,
                        "threshold": threshold,
                        "gain_ratio": gain_ratio,
                        "info_gain": info_gain,
                    }

        return best

    # ── 后剪枝 ──────────────────────────────────────────
    def _post_prune(self, node, X_val, y_val):
        """基于验证集的后剪枝 (Reduced Error Pruning)"""
        if node["is_leaf"] or len(X_val) == 0:
            return node

        # 剪枝前精度
        y_pred_before = np.array([self._predict_one(x, node) for x in X_val])
        acc_before = np.mean(y_pred_before == y_val)

        # 尝试剪枝（变为叶节点）
        pruned = {"is_leaf": True, "class": node["class"],
                  "counts": Counter(y_val), "n_samples": len(y_val)}
        y_pred_after = np.array([self._predict_one(x, pruned) for x in X_val])
        acc_after = np.mean(y_pred_after == y_val)

        if acc_after >= acc_before:
            return pruned

        # 递归剪枝子节点
        left_mask = X_val[:, node["feature"]] <= node["threshold"]
        right_mask = ~left_mask

        node["left"] = self._post_prune(node["left"], X_val[left_mask], y_val[left_mask])
        node["right"] = self._post_prune(node["right"], X_val[right_mask], y_val[right_mask])

        return node

    # ── 预测辅助 ────────────────────────────────────────
    def _predict_one(self, x, node):
        """单样本预测"""
        if node["is_leaf"]:
            return node["class"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        else:
            return self._predict_one(x, node["right"])

    def _predict_counts(self, x, node):
        """返回叶子节点的类别计数"""
        if node["is_leaf"]:
            return node["counts"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_counts(x, node["left"])
        else:
            return self._predict_counts(x, node["right"])

    # ── 熵计算 ──────────────────────────────────────────
    @staticmethod
    def _entropy(y):
        """计算信息熵 H(Y) = -Σ p_i log2(p_i)"""
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs + 1e-15))

    @staticmethod
    def _log2_safe(x):
        """安全 log2"""
        return np.log2(max(x, 1e-15))


# ── 测试入口 ────────────────────────────────────────────
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    print("=" * 60)
    print("决策树 C4.5 测试")
    print("=" * 60)

    data = load_iris()
    X, y = data.data, data.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

    for prune in ["none", "post"]:
        dt = DecisionTreeC45(max_depth=5, min_info_gain=0.001, pruning=prune, val_split=0.2)
        dt.fit(X_tr, y_tr)
        y_pred = dt.predict(X_te)
        acc = np.mean(y_pred == y_te)
        print(f"  C4.5(pruning={prune:>4s}): acc={acc:.4f}")

    # 概率预测
    proba = dt.predict_proba(X_te[:3])
    print(f"  proba[:3] sum: {proba.sum(axis=1).round(4)}")

    print("\n[决策树 C4.5] 测试完成")
