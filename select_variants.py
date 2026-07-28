"""选点策略的变体对照，用于检验两个假设。

Idea 1（度量问题）：展平的 VAE latent 上算欧氏距离，主要由空间对齐主导——
    同一只狗平移几十个格子，L2 距离可能比两个不同犬种还大。
    做法：聚类前先做空间池化，消除平移敏感性。不引入任何外部模型。

Idea 2（覆盖 vs 代表性）：低 IPC 下蒸馏集需要的可能是覆盖度而非密度代表性。
    做法：用 k-center greedy（最远点采样）代替 K-Means。

聚类在变换后的特征上做，但**存下来的引导目标始终是被选中真实样本的完整 latent**，
因为生成端需要 4x128x128 的原始 latent。这样变量严格隔离在"用什么度量选点"上。

路径标记：
    n_1_s_0  kmeans  + 16x16 平均池化 (4x8x8 = 256 维)
    n_2_s_0  kmeans  + 通道统计量 (每通道均值+方差 = 8 维)
    n_3_s_0  kcenter + 原始 65536 维
对照组（已跑）：
    n_0_s_0  kmeans  + 原始 65536 维    R=32.4  G=35.4
    n_0_s_1  随机
"""
import os
import pickle
import argparse

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from tqdm import tqdm


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cluster_dir", type=str, required=True)
    p.add_argument("--out_root", type=str, required=True)
    # 用另一份缓存的特征来选点（如 U-Net 特征），但引导目标仍存原始 VAE latent。
    # 两份缓存的类别顺序与类内样本顺序必须一致（unet_features.py 保证了这一点）。
    p.add_argument("--select_cache", type=str, default=None)
    p.add_argument("--class_file", type=str, default="./misc/class_woof.txt")
    p.add_argument("--method", choices=["kmeans", "kcenter"], default="kmeans")
    p.add_argument("--pool", choices=["none", "p8", "p16", "p32", "stats",
                                      "cos", "p8cos", "ms"], default="none")
    p.add_argument("--path_tag", type=str, required=True)
    p.add_argument("--nclass", type=int, default=10)
    p.add_argument("--IPC", type=int, default=10)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sample_step", type=int, default=25)
    p.add_argument("--denoising_factor", type=float, default=1.0)
    p.add_argument("--guideTPercent", type=float, default=0.9)
    p.add_argument("--CoDA_guidance_scale", type=float, default=0.05)
    return p.parse_args()


def transform(X, mode):
    """X: (N, 65536) -> 变换后的特征。原始 latent 形状是 (4, 128, 128)。"""
    if mode == "none":
        return X
    N = X.shape[0]
    L = X.reshape(N, 4, 128, 128)
    if mode.startswith("p"):
        f = int(mode[1:])            # 池化窗口，128/f 是输出边长
        s = 128 // f
        return L.reshape(N, 4, s, f, s, f).mean(axis=(3, 5)).reshape(N, -1)
    if mode == "stats":
        # 每通道的空间均值和方差 -> (N, 8)，完全平移不变
        return np.concatenate([L.mean(axis=(2, 3)), L.std(axis=(2, 3))], axis=1)

    # 以下三种来自 repr_search.py 的 kNN 探针筛选结果（基线 16.8%）
    def _pool(f):
        s = 128 // f
        return L.reshape(N, 4, s, f, s, f).mean(axis=(3, 5)).reshape(N, -1)

    def _l2(Z):
        return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)

    if mode == "cos":        # 只做 L2 归一化，不池化。kNN 21.8%
        return _l2(X)
    if mode == "p8cos":      # 8x8 池化 + 余弦。kNN 24.3%
        return _l2(_pool(8))
    if mode == "ms":         # 多尺度 8x8 + 32x32，各自归一化后拼接。kNN 25.1%
        return np.concatenate([_l2(_pool(8)), _l2(_pool(32))], axis=1)
    raise ValueError(mode)


def kcenter_greedy(X, k, seed=0):
    """最远点采样：反复加入离当前已选集合最远的点，最大化覆盖。"""
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    sq = (X ** 2).sum(axis=1)
    first = int(rng.randint(n))
    picked = [first]
    d = np.sqrt(np.maximum(sq + sq[first] - 2 * X @ X[first], 0))
    for _ in range(k - 1):
        j = int(np.argmax(d))
        picked.append(j)
        dj = np.sqrt(np.maximum(sq + sq[j] - 2 * X @ X[j], 0))
        d = np.minimum(d, dj)
    return picked


def main():
    args = get_args()
    with open(args.class_file) as fp:
        sel_classes = [l.strip() for l in fp if l.strip()][: args.nclass]

    save_dir = os.path.join(
        args.out_root,
        f"Step-{args.sample_step}/IPC-{args.IPC}/"
        f"DF-{args.denoising_factor}-GTP-{args.guideTPercent}-gamma-{args.CoDA_guidance_scale}/"
        f"{args.path_tag}")
    os.makedirs(save_dir, exist_ok=True)
    cache = os.path.join(args.cluster_dir, "original_features_cache.pkl")

    sel_cache = (os.path.join(args.select_cache, "original_features_cache.pkl")
                 if args.select_cache else None)

    for chunk_id in range(args.nclass // 10):
        with open(f"{cache}_{chunk_id}", "rb") as f:
            data = pickle.load(f)
        feats, paths = data["features"], data["paths"]

        sel = None
        if sel_cache:
            with open(f"{sel_cache}_{chunk_id}", "rb") as f:
                sel = pickle.load(f)["features"]

        clusters_centers = {}
        for c in tqdm(sorted(feats.keys()), desc=f"{args.method}/{args.pool} chunk {chunk_id}"):
            X_full = np.stack(feats[c]).astype(np.float32)      # 原始 latent，用于存引导目标
            if sel is None:
                Z = transform(X_full, args.pool)                 # 变换后的特征，用于选点
            else:
                Z = np.stack(sel[c]).astype(np.float32)          # 外部特征（如 U-Net）直接用
                assert len(Z) == len(X_full), f"两份缓存类 {c} 的样本数不一致"

            if args.method == "kmeans":
                km = KMeans(n_clusters=args.IPC, random_state=args.seed, n_init="auto").fit(Z)
                pick = []
                for k in range(args.IPC):
                    d = np.linalg.norm(Z - km.cluster_centers_[k], axis=1)
                    pick.append(int(np.argmin(d)))
                sizes = np.bincount(km.labels_, minlength=args.IPC).tolist()
                print(f"[Class {c}] {args.method}/{args.pool} dim={Z.shape[1]} "
                      f"sizes={sorted(sizes, reverse=True)}")
            else:
                pick = kcenter_greedy(Z, args.IPC, seed=args.seed)
                print(f"[Class {c}] {args.method}/{args.pool} dim={Z.shape[1]} picked={sorted(pick)}")

            out_dir = os.path.join(save_dir, "real_images", sel_classes[c])
            os.makedirs(out_dir, exist_ok=True)
            centers = []
            for i, j in enumerate(pick):
                centers.append(X_full[j])                        # 存完整 latent
                Image.open(paths[c][j]).convert("RGB").resize(
                    (args.size, args.size), Image.Resampling.LANCZOS).save(
                    os.path.join(out_dir, f"{i}.png"))
            clusters_centers[c] = np.array(centers)

        out_pkl = os.path.join(
            args.cluster_dir, f"{args.IPC}_{args.path_tag}_saved_clusters_{chunk_id}.pkl")
        with open(out_pkl, "wb") as f:
            pickle.dump(clusters_centers, f)
        print(f"Saved centers to: {out_pkl}")
        del data, feats, paths

    print(f"Done. -> {save_dir}")


if __name__ == "__main__":
    main()
