"""随机基线：每类随机抽 IPC 张真实图当引导目标，不做任何聚类。

用途：Idea 3 显示 G 对"有几个不同的引导目标"极其敏感（10 个 -> 35.0，3~5 个 -> 24.4），
但对"用哪些目标"不敏感（HDBSCAN 35.0 vs K-Means 35.4）。本脚本把这个观察推到极限——
如果纯随机的 10 个锚点也能到 35 左右，那整个 Distribution Discovery 阶段
（UMAP + HDBSCAN + 三重后处理）都可以被一行 np.random.choice 替代。

输出格式与 CoDA_main.py 一致，生成端一行都不用改。
"""
import os
import pickle
import argparse

import numpy as np
from PIL import Image
from tqdm import tqdm


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cluster_dir", type=str, required=True, help="放特征缓存和中心 pkl 的目录")
    p.add_argument("--out_root", type=str, required=True, help="results/<spec> 那一层")
    p.add_argument("--class_file", type=str, default="./misc/class_woof.txt")
    p.add_argument("--nclass", type=int, default=10)
    p.add_argument("--IPC", type=int, default=10)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--path_tag", type=str, default="n_0_s_1")
    p.add_argument("--sample_step", type=int, default=25)
    p.add_argument("--denoising_factor", type=float, default=1.0)
    p.add_argument("--guideTPercent", type=float, default=0.9)
    p.add_argument("--CoDA_guidance_scale", type=float, default=0.05)
    return p.parse_args()


def main():
    args = get_args()
    rng = np.random.RandomState(args.seed)

    with open(args.class_file) as fp:
        sel_classes = [l.strip() for l in fp if l.strip()][: args.nclass]

    save_dir = os.path.join(
        args.out_root,
        f"Step-{args.sample_step}/IPC-{args.IPC}/"
        f"DF-{args.denoising_factor}-GTP-{args.guideTPercent}-gamma-{args.CoDA_guidance_scale}/"
        f"{args.path_tag}")
    os.makedirs(save_dir, exist_ok=True)

    cache = os.path.join(args.cluster_dir, "original_features_cache.pkl")
    for chunk_id in range(args.nclass // 10):
        with open(f"{cache}_{chunk_id}", "rb") as f:
            data = pickle.load(f)
        feats, paths = data["features"], data["paths"]

        clusters_centers = {}
        for c in tqdm(sorted(feats.keys()), desc=f"Random chunk {chunk_id}"):
            n = len(feats[c])
            pick = rng.choice(n, size=args.IPC, replace=False)
            print(f"[Class {c}] pool={n} picked={sorted(pick.tolist())}")

            out_dir = os.path.join(save_dir, "real_images", sel_classes[c])
            os.makedirs(out_dir, exist_ok=True)
            centers = []
            for i, j in enumerate(pick):
                centers.append(np.asarray(feats[c][j]))
                img = Image.open(paths[c][j]).convert("RGB")
                img.resize((args.size, args.size), Image.Resampling.LANCZOS).save(
                    os.path.join(out_dir, f"{i}.png"))
            clusters_centers[c] = np.array(centers)

        out_pkl = os.path.join(
            args.cluster_dir,
            f"{args.IPC}_{args.path_tag}_saved_clusters_{chunk_id}.pkl")
        with open(out_pkl, "wb") as f:
            pickle.dump(clusters_centers, f)
        print(f"Saved centers to: {out_pkl}")
        del data, feats, paths

    print(f"Done. -> {save_dir}")


if __name__ == "__main__":
    main()
