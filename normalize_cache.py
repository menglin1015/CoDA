"""把特征缓存做 L2 归一化后另存一份，不修改任何已有文件。

用途：unet_probe.py 的 kNN 用的是余弦距离（先 L2 归一化），
而 select_variants.py 的 --select_cache 路径直接用欧氏距离做 KMeans。
两者度量不一致，导致探针测出的 35.1% 和下游选点跑的不是同一件事。
U-Net 激活的模长在样本间差异很大（与图像对比度、纹理密度强相关），
不归一化的 KMeans 很可能主要按激活能量分组。

本脚本产出一份归一化后的缓存，喂给现有的 --select_cache 即可，
已有结果全部保持可复现。
"""
import os
import pickle
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="源缓存目录")
    ap.add_argument("--dst", required=True, help="目标缓存目录")
    ap.add_argument("--chunks", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    for chunk in range(args.chunks):
        name = f"original_features_cache.pkl_{chunk}"
        with open(os.path.join(args.src, name), "rb") as f:
            data = pickle.load(f)

        out = dict(data)                      # 保留 paths 等其它键
        feats = {}
        for c, vecs in data["features"].items():
            Z = np.stack([np.asarray(v, np.float32) for v in vecs])
            n = np.linalg.norm(Z, axis=1, keepdims=True)
            print(f"  类 {c}: {Z.shape}  模长 中位 {np.median(n):.1f} "
                  f"最小 {n.min():.1f} 最大 {n.max():.1f} 极差比 {n.max()/max(n.min(),1e-8):.1f}x")
            feats[c] = list(Z / (n + 1e-8))
        out["features"] = feats

        with open(os.path.join(args.dst, name), "wb") as f:
            pickle.dump(out, f)
        print(f"已写出 {os.path.join(args.dst, name)}")


if __name__ == "__main__":
    main()
