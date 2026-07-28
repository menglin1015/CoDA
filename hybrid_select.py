"""混合选点：语义空间做分组，latent 空间挑代表。

诊断依据（锚点几何统计，woof / ImageNet-1K 子集）：

    方法            G      锚点两两距离   到类心/类内平均
    朴素 KMeans    36.5      241.8          0.812
    池化 1024      38.5      265.1          0.869
    CoDA 原版      33.3      298.3          0.974
    U-Net(L2)      31.9      298.2          0.977
    随机           28.4      300.5          0.984

U-Net 选出的锚点在 VAE latent 几何上与随机选无法区分（0.977 vs 0.984），
而 G 几乎完全跟着"锚点是否靠近类心"走。原因是引导项 γ(s_j − ẑ0) 把轨迹推向 s_j；
若 s_j 是 latent 空间的非典型点，就是在往低密度区域推。

在非 latent 空间里选点，等于对 latent 几何不设任何约束——语义再好也没用在刀刃上。

本脚本把两件事拆开：
  1. 用外部语义特征（U-Net）做 KMeans 分组，保留其语义划分能力
  2. 组内按 VAE latent 到该组 latent 质心的距离挑代表，保证锚点典型

对照 select_variants.py --select_cache：那里第 2 步是在语义空间里挑离语义质心
最近的点，锚点的 latent 位置完全不受控。
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
    p.add_argument("--cluster_dir", required=True, help="VAE latent 缓存（引导目标来源）")
    p.add_argument("--select_cache", required=True, help="语义特征缓存（仅用于分组）")
    p.add_argument("--out_root", required=True)
    p.add_argument("--class_file", default="./misc/class_woof.txt")
    p.add_argument("--path_tag", required=True)
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

    for chunk in range(args.nclass // 10):
        with open(os.path.join(args.cluster_dir, f"original_features_cache.pkl_{chunk}"), "rb") as f:
            src = pickle.load(f)
        with open(os.path.join(args.select_cache, f"original_features_cache.pkl_{chunk}"), "rb") as f:
            sel = pickle.load(f)["features"]

        feats, paths = src["features"], src["paths"]
        centers_all = {}

        for c in tqdm(sorted(feats.keys()), desc=f"hybrid chunk {chunk}"):
            L = np.stack(feats[c]).astype(np.float32)          # VAE latent，引导目标
            S = np.stack(sel[c]).astype(np.float32)            # 语义特征，仅用于分组
            assert len(L) == len(S), f"类 {c} 两份缓存样本数不一致"
            S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-8)

            labels = KMeans(n_clusters=args.IPC, random_state=args.seed,
                            n_init="auto").fit_predict(S)

            pick = []
            for k in range(args.IPC):
                idx = np.where(labels == k)[0]
                if len(idx) == 0:                              # 空簇：退回全类
                    idx = np.arange(len(L))
                mu = L[idx].mean(0)                            # 该组在 latent 空间的质心
                pick.append(int(idx[np.argmin(np.linalg.norm(L[idx] - mu, axis=1))]))

            sizes = np.bincount(labels, minlength=args.IPC).tolist()
            print(f"[Class {c}] 语义簇大小 {sorted(sizes, reverse=True)}")

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


if __name__ == "__main__":
    main()
