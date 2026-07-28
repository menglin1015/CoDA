"""按锚点离心率选点：直接控制"锚点离类中心多远"，检验它与 G 的因果关系。

观察依据（woof / ImageNet-1K 子集，锚点几何 vs 最终精度）：

    方法           G      到类心/类内均值
    池化 1024     38.5        0.869
    朴素 KMeans   36.5        0.812
    混合          36.5        0.778
    CoDA 原版     33.3        0.974
    U-Net(L2)     31.9        0.977
    随机          28.4        0.984

离心比 >= 0.97 的三个全部落在 28~33，<= 0.87 的三个全部落在 36~38.5，分界很干净，
且峰值似乎在 0.87 附近而非越低越好。但这些点在"选点算法"上也各不相同，
相关性可能被别的因素混淆。

本脚本把离心率作为唯一自变量：先按目标离心率 tau 圈定候选池，
再在池内用 latent KMeans 保证多样性、取组内 latent 质心最近点作锚点。
这样多样性结构保持一致，只有半径在变。

离心率定义： e_i = ||x_i - mu_c|| / mean_j ||x_j - mu_c||
随机采样的期望离心率约为 1.0。
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
    p.add_argument("--cluster_dir", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--class_file", default="./misc/class_woof.txt")
    p.add_argument("--path_tag", required=True)
    p.add_argument("--tau", type=float, required=True, help="目标离心率")
    p.add_argument("--pool_frac", type=float, default=0.4, help="候选池占该类的比例")
    p.add_argument("--nclass", type=int, default=10)
    p.add_argument("--IPC", type=int, default=10)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sample_step", type=int, default=25)
    p.add_argument("--denoising_factor", type=float, default=1.0)
    p.add_argument("--guideTPercent", type=float, default=0.9)
    p.add_argument("--CoDA_guidance_scale", type=float, default=0.05)
    return p.parse_args()


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

    achieved = []
    for chunk in range(args.nclass // 10):
        with open(os.path.join(args.cluster_dir, f"original_features_cache.pkl_{chunk}"), "rb") as f:
            src = pickle.load(f)
        feats, paths = src["features"], src["paths"]
        centers_all = {}

        for c in tqdm(sorted(feats.keys()), desc=f"ecc tau={args.tau} chunk {chunk}"):
            L = np.stack(feats[c]).astype(np.float32)
            mu = L.mean(0)
            d = np.linalg.norm(L - mu, axis=1)
            e = d / d.mean()                                  # 离心率，随机样本期望 ~1.0

            # 圈定离心率最接近 tau 的候选池
            n_pool = max(args.IPC * 3, int(len(L) * args.pool_frac))
            pool = np.argsort(np.abs(e - args.tau))[:n_pool]

            # 池内按 latent 聚类保证多样性，取组内 latent 质心最近点
            labels = KMeans(n_clusters=args.IPC, random_state=args.seed,
                            n_init="auto").fit_predict(L[pool])
            pick = []
            for k in range(args.IPC):
                sub = pool[labels == k]
                if len(sub) == 0:
                    sub = pool
                cmu = L[sub].mean(0)
                pick.append(int(sub[np.argmin(np.linalg.norm(L[sub] - cmu, axis=1))]))

            got = e[pick].mean()
            achieved.append(got)
            print(f"[Class {c}] 目标 {args.tau:.2f} -> 实际 {got:.3f}  池大小 {n_pool}")

            out_dir = os.path.join(save_dir, "real_images", sel_classes[c])
            os.makedirs(out_dir, exist_ok=True)
            for i, j in enumerate(pick):
                Image.open(paths[c][j]).convert("RGB").resize(
                    (args.size, args.size), Image.Resampling.LANCZOS).save(
                    os.path.join(out_dir, f"{i}.png"))
            centers_all[c] = L[pick]

        out_pkl = os.path.join(args.cluster_dir,
                               f"{args.IPC}_{args.path_tag}_saved_clusters_{chunk}.pkl")
        with open(out_pkl, "wb") as f:
            pickle.dump(centers_all, f)
        print(f"Saved centers to: {out_pkl}")

    print(f"\n目标离心率 {args.tau:.2f}，全类实际均值 {np.mean(achieved):.3f}")


if __name__ == "__main__":
    main()
