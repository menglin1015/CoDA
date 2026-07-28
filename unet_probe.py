"""探测 SDXL U-Net 中间层特征能否支撑 mode discovery。

动机：诊断显示 VAE latent 在欧氏距离下几乎不携带类别信息
（10 类 kNN 16.8%，随机 10%；同类/异类中位距离只差 0.76%）。
但扩散 U-Net 的中间激活是已知语义丰富的（DIFT 一系工作），
而读它**不需要任何训练、不引入任何外部模型**——仍是同一个 off-the-shelf 模型，
只是换个地方取特征，不违反 training-free 约束。

本脚本只做探针，不跑蒸馏：复用已有的 VAE latent 缓存（省去重新编码），
在若干时间步加噪后过一次 U-Net，钩出各层激活，空间池化后做 kNN。
判定：若某层能明显超过当前最好的表示（多尺度池化 25.1%），
说明整条路线一直在错误的地方读特征。

条件用**空提示词**，与类别无关——否则类别信息会从文本条件泄漏进特征，kNN 就没有意义了。
"""
import os
import gc
import pickle
import argparse

import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline, DDPMScheduler


def knn_acc(Z, y, k, nclass):
    Z = Z.astype(np.float32)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)   # 余弦，已证明对高维特征更好
    sq = (Z ** 2).sum(1)
    D = sq[:, None] + sq[None, :] - 2 * Z @ Z.T
    np.fill_diagonal(D, np.inf)
    idx = np.argpartition(D, k, axis=1)[:, :k]
    pred = np.array([np.bincount(v, minlength=nclass).argmax() for v in y[idx]])
    return float((pred == y).mean())


def spatial_pool(t, out_hw):
    """(B,C,H,W) -> (B, C*out_hw*out_hw)，自适应平均池化。"""
    return torch.nn.functional.adaptive_avg_pool2d(t.float(), out_hw).flatten(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster_dir", default="results/clusterfile/woof_in1k")
    ap.add_argument("--model", default="model/SDXL-Refiner/sdxl-base")
    ap.add_argument("--per_class", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--timesteps", type=int, nargs="+", default=[100, 300, 500])
    ap.add_argument("--pool_hw", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda"
    torch.set_grad_enabled(False)

    # ---- 取样本（复用 VAE latent 缓存，不重新编码图片）----
    rng = np.random.RandomState(args.seed)
    with open(os.path.join(args.cluster_dir, "original_features_cache.pkl_0"), "rb") as f:
        feats = pickle.load(f)["features"]
    nclass = len(feats)
    Xs, ys = [], []
    for c in sorted(feats.keys()):
        for j in rng.choice(len(feats[c]), min(args.per_class, len(feats[c])), replace=False):
            Xs.append(np.asarray(feats[c][j], np.float32))
            ys.append(c)
    X = np.stack(Xs).reshape(-1, 4, 128, 128)
    y = np.array(ys)
    del feats, Xs
    gc.collect()
    print(f"样本 {len(y)} 张 / {nclass} 类，随机猜测 {100/nclass:.1f}%")

    # ---- 载入 base pipeline（不要 refiner）----
    print("加载 SDXL base ...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, use_safetensors=True)
    pipe.to(device)
    unet = pipe.unet.eval()
    sched = DDPMScheduler.from_pretrained(args.model, subfolder="scheduler")
    acp = sched.alphas_cumprod.to(device)

    # 空提示词：与类别无关，避免标签信息从文本条件泄漏
    pe, _, pooled, _ = pipe.encode_prompt(prompt="", device=device,
                                          num_images_per_prompt=1,
                                          do_classifier_free_guidance=False)
    add_time = torch.tensor([[1024., 1024., 0., 0., 1024., 1024.]], device=device,
                            dtype=torch.float16)

    # ---- 钩住若干层 ----
    taps = {
        "down2": unet.down_blocks[2],
        "mid": unet.mid_block,
        "up0": unet.up_blocks[0],
        "up1": unet.up_blocks[1],
    }
    grabbed = {}

    def mk_hook(name):
        def hook(_m, _i, out):
            grabbed[name] = out[0] if isinstance(out, tuple) else out
        return hook

    handles = [m.register_forward_hook(mk_hook(n)) for n, m in taps.items()]

    results = {}
    for t_val in args.timesteps:
        buf = {f"{n}@{hw}": [] for n in taps for hw in args.pool_hw}
        t = torch.tensor([t_val], device=device)
        a = acp[t_val].sqrt().item()
        b = (1 - acp[t_val]).sqrt().item()

        for i in range(0, len(X), args.batch):
            z0 = torch.from_numpy(X[i:i + args.batch]).to(device, torch.float16)
            B = z0.shape[0]
            zt = a * z0 + b * torch.randn(z0.shape, device=device, dtype=torch.float16,
                                          generator=torch.Generator(device).manual_seed(i))
            unet(zt, t.expand(B),
                 encoder_hidden_states=pe.expand(B, -1, -1),
                 added_cond_kwargs={"text_embeds": pooled.expand(B, -1),
                                    "time_ids": add_time.expand(B, -1)},
                 return_dict=False)
            for n in taps:
                for hw in args.pool_hw:
                    buf[f"{n}@{hw}"].append(spatial_pool(grabbed[n], hw).cpu().numpy())
            grabbed.clear()
            if i % (args.batch * 50) == 0:
                print(f"  t={t_val}  {i}/{len(X)}", flush=True)

        for key, chunks in buf.items():
            Z = np.concatenate(chunks, 0)
            results[f"t{t_val} {key}"] = (Z.shape[1], knn_acc(Z, y, args.k, nclass))
            del chunks
        del buf
        gc.collect()

    for h in handles:
        h.remove()

    print(f"\n{'层 @ 池化边长':<22}{'维度':>10}{'kNN':>9}")
    print("-" * 42)
    for key in sorted(results, key=lambda k: -results[k][1]):
        d, acc = results[key]
        print(f"{key:<22}{d:>10}{acc*100:>8.1f}%")
    print("\n参照：VAE latent 原始展平 16.8% / 8x8池化 21.7% / 多尺度+余弦 25.1% / 随机 10.0%")


if __name__ == "__main__":
    main()
