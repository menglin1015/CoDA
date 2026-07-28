"""抽取 SDXL U-Net 编码器中间特征，产出与 VAE 缓存同格式的特征文件。

依据 unet_probe.py 的探测结果：down_blocks[2] 输出做全局平均池化、t=100 时
10 类 kNN 达 35.1%，而五篇相关工作使用的 VAE 展平 latent 只有 16.8%（随机 10%）。

产物格式与 results/clusterfile/*/original_features_cache.pkl_0 一致，
因此 select_variants.py 可直接消费；配合 --center_cache 用 U-Net 特征选点、
但仍保存被选中样本的完整 VAE latent 作为引导目标。

条件用空提示词，与类别无关，避免标签信息从文本分支泄漏。
特征只有 1280 维，全量缓存约 64 MB（VAE 那份是 13 GB）。
"""
import os
import gc
import pickle
import argparse

import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline, DDPMScheduler


class _Stop(Exception):
    """在目标层触发，提前中止前向——后面的 mid/up 块不用算，省一多半时间。"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_cache", default="results/clusterfile/woof_in1k")
    ap.add_argument("--out_dir", default="results/clusterfile/woof_unet")
    ap.add_argument("--model", default="model/SDXL-Refiner/sdxl-base")
    ap.add_argument("--layer", default="down2", choices=["down1", "down2", "mid"])
    ap.add_argument("--timestep", type=int, default=100)
    ap.add_argument("--pool_hw", type=int, default=1)
    ap.add_argument("--batch", type=int, default=6)
    args = ap.parse_args()

    device = "cuda"
    torch.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    print("加载 SDXL base ...", flush=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, use_safetensors=True)
    pipe.to(device)
    unet = pipe.unet.eval()
    sched = DDPMScheduler.from_pretrained(args.model, subfolder="scheduler")
    a = sched.alphas_cumprod[args.timestep].sqrt().item()
    b = (1 - sched.alphas_cumprod[args.timestep]).sqrt().item()

    pe, _, pooled, _ = pipe.encode_prompt(prompt="", device=device,
                                          num_images_per_prompt=1,
                                          do_classifier_free_guidance=False)
    add_time = torch.tensor([[1024., 1024., 0., 0., 1024., 1024.]],
                            device=device, dtype=torch.float16)
    del pipe.vae, pipe.text_encoder, pipe.text_encoder_2
    gc.collect(); torch.cuda.empty_cache()

    tap = {"down1": unet.down_blocks[1], "down2": unet.down_blocks[2],
           "mid": unet.mid_block}[args.layer]
    box = {}

    def hook(_m, _i, out):
        box["f"] = out[0] if isinstance(out, tuple) else out
        raise _Stop
    tap.register_forward_hook(hook)

    t = torch.tensor([args.timestep], device=device)

    for chunk in sorted(f for f in os.listdir(args.src_cache)
                        if f.startswith("original_features_cache.pkl_")):
        with open(os.path.join(args.src_cache, chunk), "rb") as f:
            data = pickle.load(f)
        out_feats, out_paths = {}, {}

        for c in sorted(data["features"].keys()):
            lat = np.stack(data["features"][c]).astype(np.float32).reshape(-1, 4, 128, 128)
            vecs = []
            for i in range(0, len(lat), args.batch):
                z0 = torch.from_numpy(lat[i:i + args.batch]).to(device, torch.float16)
                B = z0.shape[0]
                g = torch.Generator(device).manual_seed(1234 + i)
                zt = a * z0 + b * torch.randn(z0.shape, device=device,
                                              dtype=torch.float16, generator=g)
                try:
                    unet(zt, t.expand(B), encoder_hidden_states=pe.expand(B, -1, -1),
                         added_cond_kwargs={"text_embeds": pooled.expand(B, -1),
                                            "time_ids": add_time.expand(B, -1)},
                         return_dict=False)
                except _Stop:
                    pass
                v = torch.nn.functional.adaptive_avg_pool2d(box.pop("f").float(), args.pool_hw)
                vecs.append(v.flatten(1).cpu().numpy())
            out_feats[c] = list(np.concatenate(vecs, 0))
            out_paths[c] = data["paths"][c]
            print(f"  类 {c}: {len(out_feats[c])} 张 -> {out_feats[c][0].shape[0]} 维", flush=True)
            del lat, vecs

        dst = os.path.join(args.out_dir, chunk)
        with open(dst, "wb") as f:
            pickle.dump({"features": out_feats, "paths": out_paths}, f)
        sz = os.path.getsize(dst) / 1e6
        print(f"写出 {dst}  ({sz:.0f} MB)", flush=True)
        del data, out_feats, out_paths
        gc.collect()

    print("DONE")


if __name__ == "__main__":
    main()
