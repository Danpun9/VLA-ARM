"""VLA 학습 스크립트:
  1) 기본(고속) 모드: Qwen3.5-4B 백본 특징(vis_tokens, llm_img_tokens, lang_token)을 1회 추출/캐싱한 뒤
     LanguageSpatialSoftmax(ctx_proj) + state_proj + ActionHead 를 수천 스텝 고속 학습.
  2) --lora 모드: 백본에 LoRA 어댑터를 붙여 end-to-end QLoRA 학습.

사용법:
    uv run python scripts/train_vla.py --data ~/data/panda_pickplace --out checkpoints/vla_spatial --steps 5000
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from vla_arm.vla import QwenVLA, VLAConfig
from vla_arm.vla.dataset import DemoDataset, collate

LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj|gate_proj|up_proj|down_proj)"


def extract_or_load_cache(model: QwenVLA, ds: DemoDataset, cache_path: Path, batch_size: int, workers: int) -> dict:
    if cache_path.exists():
        print(f"[cache] loading cached backbone features from {cache_path}")
        return torch.load(cache_path, map_location="cpu", mmap=True, weights_only=True)

    N = len(ds)
    base = model._base_model()
    D = base.config.text_config.hidden_size
    s0 = ds[0]
    print(f"[cache] pre-allocating buffers and extracting backbone features for {N} samples (one-time pass)...")
    all_vis = torch.empty((N, 192, D), dtype=torch.float16)
    all_llm = torch.empty((N, 192, D), dtype=torch.float16)
    all_lang = torch.empty((N, D), dtype=torch.float16)
    all_st = torch.empty((N, *s0["state"].shape), dtype=torch.float32)
    all_act = torch.empty((N, *s0["actions"].shape), dtype=torch.float32)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=workers)
    t0 = time.time()
    done = 0
    with torch.inference_mode():
        for b in loader:
            inp = model.build_inputs(b["images"], b["texts"])
            B = len(b["texts"])
            vis_list = base.get_image_features(inp["pixel_values"], inp["image_grid_thw"]).pooler_output
            vis = torch.cat(vis_list, dim=0).to(torch.float16).view(B, 192, D)
            out = model.backbone(**inp, use_cache=False, return_dict=True)
            h = out.last_hidden_state.to(torch.float16)
            img_mask = inp["input_ids"] == 248056
            llm = h[img_mask].view(B, 192, D)
            lang = h[:, -1]
            all_vis[done:done + B].copy_(vis, non_blocking=True)
            all_llm[done:done + B].copy_(llm, non_blocking=True)
            all_lang[done:done + B].copy_(lang, non_blocking=True)
            all_st[done:done + B] = torch.from_numpy(b["state"])
            all_act[done:done + B] = torch.from_numpy(b["actions"])
            done += B
            if done % (batch_size * 10) == 0 or done == N:
                print(f"  extracted {done}/{N} ({time.time() - t0:.1f}s)", flush=True)

    cache = {
        "vis": all_vis,
        "llm": all_llm,
        "lang": all_lang,
        "state": all_st,
        "actions": all_act,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    print(f"[cache] saved to {cache_path} ({time.time() - t0:.1f}s)")
    return cache


def save_ckpt(model: QwenVLA, path: Path, use_lora: bool):
    path.mkdir(parents=True, exist_ok=True)
    if not use_lora:
        for f in path.glob("adapter_*"):
            f.unlink()
    else:
        model.backbone.save_pretrained(path)
    model.save_head(path)
    print(f"[save] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("checkpoints/vla_qlora"))
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr-head", type=float, default=5e-4)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--lora", action="store_true", help="백본 LoRA end-to-end 학습 활성화 (미지정 시 고속 캐시 학습)")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--resume-head", type=Path, default=None)
    args = ap.parse_args()

    ycfg = yaml.safe_load(args.config.read_text())
    cfg = VLAConfig(**ycfg["vla"]["model"])
    torch.manual_seed(0)

    eps = list(range(args.max_episodes)) if args.max_episodes else None
    ds = DemoDataset(args.data, chunk_size=cfg.chunk_size, episodes=eps, image_aug=args.lora)
    tag = f"ep{len(ds.files)}_n{len(ds)}"
    cache_path = Path(args.data).expanduser() / f"backbone_cache_{tag}.pt"
    need_backbone = args.lora or not cache_path.exists()
    model = QwenVLA(cfg, load_backbone=need_backbone)
    if args.resume_head:
        model.load_head(args.resume_head)

    if not args.lora:
        import gc

        model.eval()
        cache = extract_or_load_cache(model, ds, cache_path, batch_size=16, workers=args.workers)
        model.backbone = None
        model.processor = None
        gc.collect()
        torch.cuda.empty_cache()

        N = len(ds)
        n_val = max(1, int(N * args.val_frac))
        # 에피소드 단위 홀드아웃(마지막 5% 샘플 = 미학습 에피소드)으로 일반화 검증 손실 측정
        tr_idx = torch.arange(0, N - n_val, device="cuda")
        va_idx = torch.arange(N - n_val, N, device="cuda")
        print(f"[train] {len(ds.files)} episodes, {len(tr_idx)} train / {len(va_idx)} unseen-episode val samples")

        st_all = cache["state"].cuda()
        act_all = model.normalize_actions(cache["actions"].cuda())
        vis_all = cache["vis"].cuda()
        llm_all = cache["llm"].cuda()
        lang_all = cache["lang"].cuda()

        model.train()
        params = [p for n in ("ctx_proj", "state_proj", "head") for p in getattr(model, n).parameters()]
        opt = torch.optim.AdamW(params, lr=args.lr_head, weight_decay=2e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-5)
        writer = SummaryWriter(str(args.out / "tb"))
        t0 = time.time()
        best_val = float("inf")

        for step in range(1, args.steps + 1):
            sel = tr_idx[torch.randint(0, len(tr_idx), (args.batch,), device="cuda")]
            v = vis_all[sel].float()
            l = llm_all[sel].float()
            g = lang_all[sel].float()
            ctx = model.ctx_proj(v, l, g)
            st = model.state_proj(model.normalize_state(st_all[sel]))
            loss = model.head.loss(torch.cat([ctx, st], dim=-1), act_all[sel])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()

            if step % args.log_every == 0:
                print(f"step {step}/{args.steps} train_loss {loss.item():.5f} lr {opt.param_groups[0]['lr']:.2e} "
                      f"({time.time() - t0:.1f}s)", flush=True)
                writer.add_scalar("train/loss", loss.item(), step)
            if step % args.save_every == 0 or step == args.steps:
                model.eval()
                with torch.no_grad():
                    v = vis_all[va_idx].float()
                    l = llm_all[va_idx].float()
                    g = lang_all[va_idx].float()
                    ctx = model.ctx_proj(v, l, g)
                    st = model.state_proj(model.normalize_state(st_all[va_idx]))
                    vl = model.head.loss(torch.cat([ctx, st], dim=-1), act_all[va_idx]).item()
                model.train()
                print(f"[val] step {step} unseen_ep_loss {vl:.5f}", flush=True)
                writer.add_scalar("val/loss", vl, step)
                if vl < best_val:
                    best_val = vl
                    save_ckpt(model, args.out / "best", use_lora=False)
                save_ckpt(model, args.out / f"step_{step:06d}", use_lora=False)
                save_ckpt(model, args.out / "last", use_lora=False)
        writer.close()
        return

    # --lora end-to-end mode
    from peft import LoraConfig, get_peft_model

    bb = model.backbone
    for p in bb.parameters():
        p.requires_grad_(False)
    bb.enable_input_require_grads()
    bb.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                      target_modules=LORA_TARGETS, bias="none")
    model.backbone = get_peft_model(bb, lcfg)
    model.backbone.print_trainable_parameters()
    model.train()

    n_val = max(1, int(len(ds) * args.val_frac))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(0))
    bsize = min(args.batch, 8)
    train_loader = DataLoader(train_ds, batch_size=bsize, shuffle=True, drop_last=True,
                              collate_fn=collate, num_workers=args.workers)
    head_params = [p for n in ("ctx_proj", "state_proj", "head") for p in getattr(model, n).parameters()]
    lora_params = [p for p in model.backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": head_params, "lr": args.lr_head},
                             {"params": lora_params, "lr": args.lr_lora}], weight_decay=0.01)
    it = iter(train_loader)
    for step in range(1, args.steps + 1):
        try:
            b = next(it)
        except StopIteration:
            it = iter(train_loader)
            b = next(it)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(b["images"], b["texts"], torch.from_numpy(b["state"]), torch.from_numpy(b["actions"]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head_params + lora_params, 1.0)
        opt.step()
        if step % args.log_every == 0:
            print(f"step {step}/{args.steps} loss {loss.item():.4f}")
        if step % args.save_every == 0 or step == args.steps:
            save_ckpt(model, args.out / "last", use_lora=True)


if __name__ == "__main__":
    main()
