"""VLA 백본 정밀도 비교 벤치마크: bf16 vs 8bit vs 4bit(NF4).

동일한 관측(3 카메라 + subtask + state)에 대해
  - 로드 VRAM / 추론 peak VRAM
  - predict() 지연시간 (action chunk 1회)
  - bf16 대비 백본 특징(last-token + mean hidden)의 cosine similarity / 상대오차
를 측정한다. (액션 헤드는 학습 전이므로 백본 특징 보존도로 품질을 비교)

    uv run python scripts/bench_vla_quant.py --modes bf16 4bit 8bit
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

import numpy as np
import torch

from vla_arm.sim import PandaPickPlaceEnv, ScriptedPickPlace, subtask_texts
from vla_arm.vla import QwenVLA, VLAConfig


def collect_obs(n: int, seed: int) -> list[tuple[dict, str, np.ndarray]]:
    env = PandaPickPlaceEnv()
    samples = []
    ep = 0
    while len(samples) < n:
        obs = env.reset(seed=seed + ep)
        pol = ScriptedPickPlace(env)
        texts = subtask_texts(env.task)
        for t in range(60):
            if t % 12 == 0:
                samples.append(({k: v.copy() for k, v in obs["images"].items()},
                                texts[pol.subtask_index], obs["state"].copy()))
            obs, _, done, _ = env.step(pol.act())
            if done or pol.done or len(samples) >= n:
                break
        ep += 1
    env.close()
    return samples[:n]


@torch.inference_mode()
def features(model: QwenVLA, img, txt) -> torch.Tensor:
    inputs = model.build_inputs([img], [txt])
    h = model.backbone(**inputs, use_cache=False, return_dict=True).last_hidden_state.float()
    m = inputs["attention_mask"].unsqueeze(-1).float()
    return torch.cat([h[:, -1], (h * m).sum(1) / m.sum(1)], -1)[0].cpu()


def run(mode: str, samples) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cfg = VLAConfig(load_in_4bit=mode == "4bit", load_in_8bit=mode == "8bit")
    t0 = time.time()
    model = QwenVLA(cfg).eval()
    load_s = time.time() - t0
    load_gb = torch.cuda.memory_allocated() / 2**30

    feats = [features(model, img, txt) for img, txt, _ in samples]
    # 지연시간 (warmup 2회 후)
    for img, txt, st in samples[:2]:
        model.predict(img, txt, st)
    torch.cuda.synchronize()
    lat = []
    for img, txt, st in samples:
        t = time.perf_counter()
        model.predict(img, txt, st)
        torch.cuda.synchronize()
        lat.append(time.perf_counter() - t)
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"mode": mode, "load_s": load_s, "load_gb": load_gb, "peak_gb": peak_gb,
            "latency_ms": 1000 * float(np.mean(lat)), "latency_p90_ms": 1000 * float(np.percentile(lat, 90)),
            "feats": torch.stack(feats)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["bf16", "4bit", "8bit"])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=100)
    args = ap.parse_args()

    torch.zeros(1, device="cuda")
    samples = collect_obs(args.n, args.seed)
    print(f"collected {len(samples)} observations")

    results = [run(m, samples) for m in args.modes]
    ref = next((r for r in results if r["mode"] == "bf16"), None)
    rows = []
    for r in results:
        row = {k: v for k, v in r.items() if k != "feats"}
        if ref is not None:
            a, b = r["feats"], ref["feats"]
            row["cos_vs_bf16"] = float(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean())
            row["rel_err_vs_bf16"] = float(((a - b).norm(dim=-1) / b.norm(dim=-1)).mean())
            # 서로 다른 관측을 구별하는 능력이 유지되는지 (관측 간 유사도 행렬의 상관)
            def sim(x):
                x = torch.nn.functional.normalize(x - x.mean(0), dim=-1)
                return (x @ x.T)[~torch.eye(len(x), dtype=bool)]
            row["struct_corr_vs_bf16"] = float(np.corrcoef(sim(a), sim(b))[0, 1])
        rows.append(row)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}))


if __name__ == "__main__":
    main()
