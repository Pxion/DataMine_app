"""
全局配置文件
所有可调参数硬编码于此，不依赖 .env 文件
"""

# ============================================================
# 路径配置
# ============================================================
DATA_DIR = "data"               # 数据集存放目录
OUTPUT_DIR = "output"           # 输出图表/报告目录
MODEL_DIR = "output/models"     # 模型保存目录

# ============================================================
# 数据集配置
# ============================================================
DATASETS = {
    "iris": {
        "name": "Iris",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        "local": "data/iris.csv",
        "task": "classification",
        "n_classes": 3,
        "n_features": 4,
        "has_header": False,
        "target_col": 4,  # 最后一列
        "columns": ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"],
    },
    "wine": {
        "name": "Wine",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
        "local": "data/wine.csv",
        "task": "classification",
        "n_classes": 3,
        "n_features": 13,
        "has_header": False,
        "target_col": 0,  # 首列为类别
        "columns": [
            "class", "alcohol", "malic_acid", "ash", "alcalinity_of_ash",
            "magnesium", "total_phenols", "flavanoids", "nonflavanoid_phenols",
            "proanthocyanins", "color_intensity", "hue",
            "od280_od315", "proline"
        ],
    },
    "breast_cancer": {
        "name": "Breast Cancer Wisconsin",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
        "local": "data/breast_cancer.csv",
        "task": "classification",
        "n_classes": 2,
        "n_features": 30,
        "has_header": False,
        "target_col": 0,      # skip ID后，Diagnosis为有效列0
        "skip_cols": [0],     # 跳过第1列 ID
        "columns": None,
    },
    "diabetes": {
        "name": "Pima Indians Diabetes",
        "url": "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv",
        "local": "data/diabetes.csv",
        "task": "classification",
        "n_classes": 2,
        "n_features": 8,
        "has_header": True,   # Kaggle 镜像含表头
        "target_col": 8,  # 最后一列为 outcome
        "columns": [
            "pregnancies", "glucose", "blood_pressure", "skin_thickness",
            "insulin", "bmi", "diabetes_pedigree", "age", "outcome"
        ],
    },
    "digits": {
        "name": "Optical Digits",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra",
        "local": "data/digits.csv",
        "task": "classification",
        "n_classes": 10,
        "n_features": 64,
        "has_header": False,
    },
}

# 默认使用的数据集列表（5个）
DEFAULT_DATASETS = ["iris", "wine", "breast_cancer", "diabetes", "digits"]

# ============================================================
# 数据预处理配置
# ============================================================
PREPROCESSING = {
    "test_size": 0.2,              # 测试集比例
    "random_state": 42,            # 随机种子
    "normalize": "standard",       # 标准化方式: standard / minmax / none
    "handle_missing": "mean",      # 缺失值处理: mean / median / drop
    "encoding": "label",           # 类别编码: label / onehot
    "feature_selection": {
        "methods": ["filter", "wrapper", "embedded"],  # 特征选择方法
        "k_best": 10,              # 过滤式保留特征数
    }
}

# ============================================================
# 算法参数配置
# ============================================================
ALGORITHM_CONFIG = {
    "knn": {
        "k": 5,
        "weights": "gaussian",     # uniform / distance / gaussian
        "metric": "euclidean",
        "sigma": None,             # 高斯核带宽 (仅 gaussian, None=auto)
    },
    "naive_bayes": {
        "variant": "gaussian",     # gaussian / multinomial
        "alpha": 1.0,              # 拉普拉斯平滑
    },
    "decision_tree": {
        "max_depth": 10,
        "min_samples_split": 2,
        "criterion": "entropy",    # entropy (C4.5) / gini (CART)
        "pruning": "post",         # 剪枝策略: none / pre / post
    },
    "svm": {
        "kernel": "rbf",           # linear / rbf / poly
        "C": 1.0,
        "gamma": "scale",
        "tol": 0.001,
        "max_iter": 5000,          # 基准测试使用 5000
    },
    "kmeans": {
        "n_clusters": 3,
        "init": "kmeans++",        # random / kmeans++
        "max_iter": 300,
        "n_init": 10,
    },
    "mlp": {
        "hidden_layers": [64, 32],
        "activation": "relu",      # relu / sigmoid / tanh
        "learning_rate": 0.001,    # Adam 建议更小学习率
        "batch_size": 32,
        "epochs": 200,
        "optimizer": "adam",       # sgd / momentum / adam
        "momentum": 0.9,
        "beta1": 0.9,              # Adam 一阶矩衰减
        "beta2": 0.999,            # Adam 二阶矩衰减
        "eps": 1e-8,               # Adam 数值稳定
    }
}

# ============================================================
# 评估配置
# ============================================================
EVALUATION = {
    "k_fold": 5,                   # K折交叉验证
    "metrics": ["accuracy", "precision", "recall", "f1"],
    "n_runs": 10,                  # 多次运行取平均
}

# ============================================================
# Web 服务配置
# ============================================================
WEB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5000,
    "debug": True,
    "upload_folder": "data/uploads",
    "max_content_length": 16 * 1024 * 1024,  # 16MB
}

# ============================================================
# 可视化配置
# ============================================================
VISUALIZATION = {
    "figsize": (10, 6),
    "dpi": 100,
    "cmap": "viridis",
    "style": "seaborn-v0_8-darkgrid",
    "format": "png",              # 输出格式
}
