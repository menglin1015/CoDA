"""度量诊断：直接测各特征空间携带多少类别信息，不经过蒸馏流程。

假设：展平的 VAE latent 上算欧氏距离，主要由空间对齐（构图）主导而非语义。
若成立，则空间池化后的特征应当携带更多类别信息。

做法：每类抽 N 张，做留一法 kNN 分类，比较不同池化粒度下的精度。
随机猜测 = 10%。这个指标不依赖任何下游训练，是对度量本身的直接测量。
"""
import os
import pickle
import argparse

import numpy as np


def transform(X, mode):
    if mode == "none":
        return X
    N = X.shape[0]
    L = X.reshape(N, 4, 128, 128)
    if mode == "stats":
        return np.concatenate([L.mean(axis=(2, 3)), L.std(axis=(2, 3))], axis=1)
    f = int(mode[1:])
    s = 128 // f
    return L.reshape(N, 4, s, f, s, f).mean(axis=(3, 5)).reshape(N, -1)


def knn_loo_acc(Z, y, k):
    """留一法 kNN 精度。"""
    sq = (Z ** 2).sum(axis=1)
    D = sq[:, None] + sq[None, :] - 2 * Z @ Z.T
    np.fill_diagonal(D, np.inf)               # 排除自身
    idx = np.argpartition(D, k, axis=1)[:, :k]
    votes = y[idx]
    pred = np.array([np.bincount(v, minlength=10).argmax() for v in votes])
    return float((pred == y).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cluster_dir", default="results/clusterfile/woof_in1k")
    p.add_argument("--per_class", type=int, default=120)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.RandomState(args.seed)
    with open(os.path.join(args.cluster_dir, "original_features_cache.pkl_0"), "rb") as f:
        feats = pickle.load(f)["features"]

    Xs, ys = [], []
    for c in sorted(feats.keys()):
        n = len(feats[c])
        pick = rng.choice(n, size=min(args.per_class, n), replace=False)
        for j in pick:
            Xs.append(np.asarray(feats[c][j], dtype=np.float32))
            ys.append(c)
    X = np.stack(Xs)
    y = np.array(ys)
    print(f"样本 {X.shape[0]} 张，{len(set(ys))} 类，随机猜测 = {100/len(set(ys)):.1f}%\n")

    print(f"{'特征空间':<26}{'维度':>8}{'kNN 精度':>12}")
    print("-" * 46)
    for mode, name in [("none", "原始展平 latent"),
                       ("p4", "4x4 池化"),
                       ("p8", "8x8 池化"),
                       ("p16", "16x16 池化"),
                       ("p32", "32x32 池化"),
                       ("stats", "通道统计量")]:
        Z = transform(X, mode)
        acc = knn_loo_acc(Z, y, args.k)
        print(f"{name:<26}{Z.shape[1]:>8}{acc*100:>11.1f}%")


if __name__ == "__main__":
    main()
