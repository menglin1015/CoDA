"""Idea 3：解耦"模式数 M"与"存储预算 IPC"。

CoDA 的后处理必须把 HDBSCAN 找到的 M 个模式凑到 IPC 个（woof 上 M=3~5，要编造出 5~7 个）。
本脚本去掉全部补齐逻辑，只保留 HDBSCAN 真正找到的 M 个模式，改为按簇大小给每个模式
分配生成配额（最大余数法，每个模式至少 1 张），配额之和 = IPC。

UMAP / HDBSCAN 的参数与 s=55 那一轮完全一致（random_state 固定），所以得到的是
同一批簇，唯一的差别就是"补齐" vs "配额"。

输出格式与 CoDA_main.py 完全相同：把配额展开成长度为 IPC 的中心数组，
第 j 项 = 第 j 张图该用的 mode latent，因此 generated.py 一行都不用改。

路径标记用 min_cluster_size=155 表示"55 的配额变体"。
"""
import os
import copy
import pickle
import argparse

import numpy as np
import hdbscan
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from umap import UMAP

import warnings
warnings.filterwarnings("ignore")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--program_path", type=str, default="./")
    p.add_argument("--spec", type=str, default="woof")
    p.add_argument("--nclass", type=int, default=10)
    p.add_argument("--IPC", type=int, default=10)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--n_neighbors", type=int, default=85)
    p.add_argument("--min_cluster_size", type=int, default=55)
    p.add_argument("--min_samples", type=int, default=3)
    p.add_argument("--path_tag", type=int, default=155, help="写进输出路径的 min_cluster_size 标记")
    p.add_argument("--sample_step", type=int, default=25)
    p.add_argument("--denoising_factor", type=float, default=1.0)
    p.add_argument("--guideTPercent", type=float, default=0.9)
    p.add_argument("--CoDA_guidance_scale", type=float, default=0.05)
    return p.parse_args()


SPEC_FILES = {
    "woof": "./misc/class_woof.txt",
    "nette": "./misc/class_nette.txt",
    "imageA": "./misc/imagenet-a.txt",
    "imageB": "./misc/imagenet-b.txt",
    "imageC": "./misc/imagenet-c.txt",
    "imageD": "./misc/imagenet-d.txt",
    "imageE": "./misc/imagenet-e.txt",
}


def allocate_quota(sizes, ipc):
    """按簇大小分配配额，最大余数法，每个模式至少 1 张。"""
    m = len(sizes)
    if m >= ipc:
        # 模式数够多，退化成 CoDA 的做法：取最大的 IPC 个，每个 1 张
        keep = np.argsort(sizes)[::-1][:ipc]
        return {int(k): 1 for k in keep}

    sizes = np.asarray(sizes, dtype=np.float64)
    raw = sizes / sizes.sum() * (ipc - m)      # 先每人保底 1 张，剩下的按比例分
    base = np.floor(raw).astype(int) + 1
    remainder = ipc - base.sum()
    if remainder > 0:
        order = np.argsort(raw - np.floor(raw))[::-1]
        for k in order[:remainder]:
            base[k] += 1
    return {int(k): int(v) for k, v in enumerate(base)}


def main():
    args = get_args()

    with open(SPEC_FILES[args.spec], "r") as fp:
        sel_classes = [line.strip() for line in fp if line.strip()][: args.nclass]

    cluster_dir = os.path.join(args.program_path, "results/clusterfile", args.spec)
    features_cache_path = os.path.join(cluster_dir, "original_features_cache.pkl")

    save_dir = os.path.join(
        args.program_path, "results", args.spec,
        f"Step-{args.sample_step}/IPC-{args.IPC}/"
        f"DF-{args.denoising_factor}-GTP-{args.guideTPercent}-gamma-{args.CoDA_guidance_scale}/"
        f"n_{args.n_neighbors}_s_{args.path_tag}",
    )
    os.makedirs(save_dir, exist_ok=True)

    for chunk_id in range(args.nclass // 10):
        with open(f"{features_cache_path}_{chunk_id}", "rb") as f:
            cache_data = pickle.load(f)
        features_per_class = copy.deepcopy(cache_data["features"])
        paths_per_class = copy.deepcopy(cache_data["paths"])
        del cache_data

        clusters_centers = {}
        for c in tqdm(sorted(features_per_class.keys()), desc=f"Quota chunk {chunk_id}"):
            X_orig = np.stack(features_per_class[c])
            paths = paths_per_class[c]

            # 与 CoDA 完全一致的前处理
            X = StandardScaler().fit_transform(X_orig)
            X_proc = UMAP(n_components=50, n_neighbors=args.n_neighbors,
                          min_dist=0.0, random_state=42).fit_transform(X)

            clusterer = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size,
                                        min_samples=args.min_samples, prediction_data=True)
            labels = clusterer.fit_predict(X_proc)
            M = len(np.unique(labels)) - (1 if -1 in labels else 0)

            # 每个簇取隶属概率最大的真实点，和 CoDA 一样
            mode_latents, mode_paths, sizes = [], [], []
            for cid in range(M):
                mask = labels == cid
                gidx = np.where(mask)[0]
                best = gidx[int(np.argmax(clusterer.probabilities_[mask]))]
                mode_latents.append(X_orig[best])
                mode_paths.append(paths[best])
                sizes.append(int(mask.sum()))

            quota = allocate_quota(sizes, args.IPC)
            print(f"[Class {c}] M={M} sizes={sizes} -> quota={[quota.get(k, 0) for k in range(M)]}")

            # 展开成长度 IPC 的数组：第 j 项 = 第 j 张图用的 mode
            centers, out_paths = [], []
            for cid, n in sorted(quota.items()):
                for _ in range(n):
                    centers.append(mode_latents[cid])
                    out_paths.append(mode_paths[cid])
            assert len(centers) == args.IPC, f"quota sum {len(centers)} != IPC {args.IPC}"

            out_dir = os.path.join(save_dir, "real_images", sel_classes[c])
            os.makedirs(out_dir, exist_ok=True)
            for i, p in enumerate(out_paths):
                img = Image.open(p).convert("RGB")
                img.resize((args.size, args.size), Image.Resampling.LANCZOS).save(
                    os.path.join(out_dir, f"{i}.png"))

            clusters_centers[c] = np.array(centers)

        out_pkl = os.path.join(
            cluster_dir, f"{args.IPC}_n_{args.n_neighbors}_s_{args.path_tag}_saved_clusters_{chunk_id}.pkl")
        with open(out_pkl, "wb") as f:
            pickle.dump(clusters_centers, f)
        print(f"Saved centers to: {out_pkl}")

    print(f"Done. -> {save_dir}")


if __name__ == "__main__":
    main()
