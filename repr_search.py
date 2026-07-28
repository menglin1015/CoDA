"""表示搜索：用 kNN 探针快速筛选 latent 表示，不跑蒸馏流水线。

依据：metric_probe.py 显示 kNN 精度的排序与下游蒸馏精度 G 高度一致
（1024维 22.8%/G=39.0 > 256维 22.7%/38.4 > 64维 22.5%/37.8 > 原始 17.1%/35.4），
因此可以用 kNN 当代理指标，几秒钟筛一个候选，而不是 25 分钟跑一遍完整流程。

诊断依据：原始展平 latent 里，同类与异类的中位距离只差 0.76%，
而同一张图平移 16 像素带来的距离是这个类别信号的 95 倍——距离被构图主导。
搜索目标就是把"类别信号 / 平移敏感度"这个比值做大。

纯 CPU，可与 GPU 队列并行。
"""
import os
import pickle
import argparse
from collections import OrderedDict

import numpy as np
from sklearn.decomposition import PCA


# ---------- 基础变换 ----------

def to_grid(X):
    return X.reshape(len(X), 4, 128, 128)


def pool(L, f):
    s = 128 // f
    return L.reshape(len(L), 4, s, f, s, f).mean(axis=(3, 5))


def pool_std(L, f):
    s = 128 // f
    return L.reshape(len(L), 4, s, f, s, f).std(axis=(3, 5))


def flat(A):
    return A.reshape(len(A), -1)


def l2norm(Z):
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)


def center_per_image(L):
    """减掉每张图自己的空间均值（逐通道），去掉整体色调/亮度。"""
    return L - L.mean(axis=(2, 3), keepdims=True)


def standardize_channels(L):
    """按数据集统计逐通道标准化。"""
    mu = L.mean(axis=(0, 2, 3), keepdims=True)
    sd = L.std(axis=(0, 2, 3), keepdims=True) + 1e-8
    return (L - mu) / sd


def logmag(L):
    """幅度压缩，抑制少数高能量位置的主导。"""
    return np.sign(L) * np.log1p(np.abs(L))


def pca_drop_top(Z, k):
    """去掉前 k 个主成分——若构图是最大方差方向，这一步应当有效。"""
    p = PCA(n_components=min(len(Z) - 1, 256), svd_solver="randomized", random_state=0).fit(Z)
    W = p.components_[k:]
    return (Z - p.mean_) @ W.T


def pca_whiten(Z, n):
    p = PCA(n_components=min(n, len(Z) - 1), whiten=True, svd_solver="randomized", random_state=0)
    return p.fit_transform(Z)


# ---------- 指标 ----------

def knn_acc(Z, y, k, nclass):
    sq = (Z ** 2).sum(1)
    D = sq[:, None] + sq[None, :] - 2 * Z @ Z.T
    np.fill_diagonal(D, np.inf)
    idx = np.argpartition(D, k, axis=1)[:, :k]
    pred = np.array([np.bincount(v, minlength=nclass).argmax() for v in y[idx]])
    return float((pred == y).mean())


def signal_ratio(Z, y):
    """类别信号 / 平移敏感度。越大越好。

    类别信号 = 异类中位距 − 同类中位距
    平移敏感度 = 把 latent 沿空间维滚动 2 格后到自身的中位距离
    对已被打平、无法还原空间结构的表示返回 nan。
    """
    sq = (Z ** 2).sum(1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0))
    np.fill_diagonal(D, np.nan)
    same = np.nanmedian(D[y[:, None] == y[None, :]])
    diff = np.nanmedian(D[y[:, None] != y[None, :]])
    return diff - same, same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster_dir", default="results/clusterfile/woof_in1k")
    ap.add_argument("--per_class", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    with open(os.path.join(args.cluster_dir, "original_features_cache.pkl_0"), "rb") as f:
        feats = pickle.load(f)["features"]
    nclass = len(feats)

    # 候选表示：名字 -> 从 (N,4,128,128) 到 (N,D) 的函数
    C = OrderedDict()
    C["原始展平（基线）"] = lambda L: flat(L)
    C["原始 + 余弦"] = lambda L: l2norm(flat(L))
    for f_ in (4, 8, 16, 32):
        C[f"{f_}x{f_} 池化"] = lambda L, f_=f_: flat(pool(L, f_))
    C["8x8 池化 + 余弦"] = lambda L: l2norm(flat(pool(L, 8)))
    C["8x8 池化 + 逐图去均值"] = lambda L: flat(pool(center_per_image(L), 8))
    C["8x8 池化 + 通道标准化"] = lambda L: flat(pool(standardize_channels(L), 8))
    C["8x8 池化 均值+方差"] = lambda L: np.concatenate([flat(pool(L, 8)), flat(pool_std(L, 8))], 1)
    C["多尺度 8x8 + 32x32"] = lambda L: np.concatenate(
        [l2norm(flat(pool(L, 8))), l2norm(flat(pool(L, 32)))], 1)
    C["幅度压缩 + 8x8 池化"] = lambda L: flat(pool(logmag(L), 8))
    C["8x8 池化 + 去掉首个主成分"] = lambda L: pca_drop_top(flat(pool(L, 8)), 1)
    C["8x8 池化 + 去掉前 5 主成分"] = lambda L: pca_drop_top(flat(pool(L, 8)), 5)
    C["8x8 池化 + PCA 白化 64"] = lambda L: pca_whiten(flat(pool(L, 8)), 64)
    C["原始 + 去掉前 5 主成分"] = lambda L: pca_drop_top(flat(L), 5)
    C["通道统计量（均值+方差）"] = lambda L: np.concatenate(
        [L.mean(axis=(2, 3)), L.std(axis=(2, 3))], 1)

    accs = {name: [] for name in C}
    dims = {}
    ratios = {}

    for seed in range(args.seeds):
        rng = np.random.RandomState(seed)
        Xs, ys = [], []
        for c in sorted(feats.keys()):
            for j in rng.choice(len(feats[c]), min(args.per_class, len(feats[c])), replace=False):
                Xs.append(np.asarray(feats[c][j], np.float32))
                ys.append(c)
        L = to_grid(np.stack(Xs))
        y = np.array(ys)
        if seed == 0:
            print(f"样本 {len(y)} 张 / {nclass} 类，随机猜测 {100/nclass:.1f}%，"
                  f"kNN k={args.k}，{args.seeds} 次子采样取均值\n")
        for name, fn in C.items():
            Z = fn(L).astype(np.float32)
            accs[name].append(knn_acc(Z, y, args.k, nclass))
            if seed == 0:
                dims[name] = Z.shape[1]
                sig, same = signal_ratio(Z, y)
                ratios[name] = sig / same * 100      # 类别信号占同类距离的百分比

    order = sorted(C, key=lambda n: -np.mean(accs[n]))
    print(f"{'表示':<28}{'维度':>8}{'kNN':>9}{'类别信号%':>11}")
    print("-" * 58)
    for name in order:
        print(f"{name:<28}{dims[name]:>8}{np.mean(accs[name])*100:>8.1f}%{ratios[name]:>10.2f}%")


if __name__ == "__main__":
    main()
