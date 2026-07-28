"""MGD3 式的朴素 K-Means 选择器，用于替换 CoDA 的 Distribution Discovery 模块。

消融目的：保持生成端（gamma 引导 / SDXL / 全部超参）完全不变，只把选 mode 的方式
从 UMAP + HDBSCAN + 三重后处理换成"直接在 VAE latent 上 KMeans(k=IPC)"，
看最终精度是否变化，以判断 CoDA 的增益到底来自 Discovery 还是 Alignment。

与 CoDA 的唯一差异就是聚类算法本身：
  - 特征：复用同一份 VAE latent 缓存（不重算）
  - 中心：KMeans 质心吸附到最近的真实样本（和 CoDA 一样落到真实图，保证可比）
  - 输出：写成与 CoDA_main.py 完全相同的目录结构和 pickle 格式

用 n_neighbors=0 / min_cluster_size=0 作为该消融的路径标记，这样
scripts/CoDA_woof.sh 不用改一行就能接着跑生成和评测。
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
    p.add_argument("--program_path", type=str, default="./")
    p.add_argument("--spec", type=str, default="woof")
    p.add_argument("--nclass", type=int, default=10)
    p.add_argument("--IPC", type=int, default=10)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    # 生成端超参，只用来拼出和 CoDA 一致的输出路径
    p.add_argument("--sample_step", type=int, default=25)
    p.add_argument("--denoising_factor", type=float, default=1.0)
    p.add_argument("--guideTPercent", type=float, default=0.9)
    p.add_argument("--CoDA_guidance_scale", type=float, default=0.05)
    return p.parse_args()


SPEC_FILES = {
    "woof": "./misc/class_woof.txt",
    "nette": "./misc/class_nette.txt",
    "imagenet100": "./misc/class100.txt",
    "imagenet1k": "./misc/class_indices.txt",
    "IDC": "./misc/class_IDC.txt",
    "imageA": "./misc/imagenet-a.txt",
    "imageB": "./misc/imagenet-b.txt",
    "imageC": "./misc/imagenet-c.txt",
    "imageD": "./misc/imagenet-d.txt",
    "imageE": "./misc/imagenet-e.txt",
}


def main():
    args = get_args()
    np.random.seed(args.seed)

    with open(SPEC_FILES[args.spec], "r") as fp:
        sel_classes = [line.strip() for line in fp if line.strip()]
    sel_classes = sel_classes[: args.nclass]

    cluster_dir = os.path.join(args.program_path, "results/clusterfile", args.spec)
    features_cache_path = os.path.join(cluster_dir, "original_features_cache.pkl")

    save_dir = os.path.join(
        args.program_path, "results", args.spec,
        f"Step-{args.sample_step}/IPC-{args.IPC}/"
        f"DF-{args.denoising_factor}-GTP-{args.guideTPercent}-gamma-{args.CoDA_guidance_scale}/"
        f"n_0_s_0",
    )
    os.makedirs(save_dir, exist_ok=True)

    num_chunks = args.nclass // 10
    for chunk_id in range(num_chunks):
        with open(f"{features_cache_path}_{chunk_id}", "rb") as f:
            cache_data = pickle.load(f)
        features_per_class = cache_data["features"]
        paths_per_class = cache_data["paths"]

        clusters_centers = {}
        for c in tqdm(sorted(features_per_class.keys()), desc=f"KMeans chunk {chunk_id}"):
            X = np.stack(features_per_class[c]).astype(np.float32)
            paths = paths_per_class[c]

            # MGD3 的做法：直接在展平的 VAE latent 上 KMeans，k 强行等于 IPC
            km = KMeans(n_clusters=args.IPC, random_state=args.seed, n_init="auto").fit(X)

            # 质心是虚点，吸附到最近的真实样本（与 CoDA 保持一致，保证可比）
            centers, chosen = [], []
            for k in range(args.IPC):
                d = np.linalg.norm(X - km.cluster_centers_[k], axis=1)
                j = int(np.argmin(d))
                centers.append(X[j])
                chosen.append(j)

            sizes = np.bincount(km.labels_, minlength=args.IPC).tolist()
            print(f"[Class {c}] cluster sizes: {sorted(sizes, reverse=True)}")

            out_dir = os.path.join(save_dir, "real_images", sel_classes[c])
            os.makedirs(out_dir, exist_ok=True)
            for i, j in enumerate(chosen):
                img = Image.open(paths[j]).convert("RGB")
                img.resize((args.size, args.size), Image.Resampling.LANCZOS).save(
                    os.path.join(out_dir, f"{i}.png")
                )

            clusters_centers[c] = np.array(centers)

        out_pkl = os.path.join(cluster_dir, f"{args.IPC}_n_0_s_0_saved_clusters_{chunk_id}.pkl")
        with open(out_pkl, "wb") as f:
            pickle.dump(clusters_centers, f)
        print(f"Saved centers to: {out_pkl}")

        del cache_data, features_per_class, paths_per_class, clusters_centers

    print(f"Done. real_images -> {save_dir}/real_images")


if __name__ == "__main__":
    main()
