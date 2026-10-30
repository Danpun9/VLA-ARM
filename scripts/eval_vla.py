"""학습된 VLA 체크포인트 평가 스크립트 (선택적으로 Planner 연동 가능).

    # 템플릿 subtask 로 VLA 단독 평가
    uv run python scripts/eval_vla.py --ckpt checkpoints/vla_qlora/last --episodes 10
    # Planner(Qwen3.5-9B) 와 함께 전체 파이프라인 평가
    uv run python scripts/eval_vla.py --ckpt checkpoints/vla_qlora/last --episodes 10 --with-planner
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import imageio.v2 as imageio
import numpy as np
import torch
import yaml

from vla_arm.sim import CAMERAS, PandaPickPlaceEnv, sorting_env_config, subtask_texts
from vla_arm.vla import QwenVLA, VLAConfig, VLAPolicy


def tile(images: dict) -> np.ndarray:
    return np.concatenate([images[c] for c in CAMERAS], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--scene", choices=["default", "sorting"], default="default")
    ap.add_argument("--task-mode", choices=["single", "sort_all"], default="sort_all")
    ap.add_argument("--with-planner", action="store_true")
    ap.add_argument("--save-videos", type=int, default=3, help="저장할 최대 에피소드 영상 수")
    ap.add_argument("--out", type=Path, default=Path("outputs/eval"))
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=True)

    planner = None
    if args.with_planner:
        from vla_arm.planner import PlannerConfig, QwenPlanner

        planner = QwenPlanner(PlannerConfig(**cfg["planner"]))

    vla_cfg = VLAConfig(**cfg["vla"]["model"])
    model = QwenVLA(vla_cfg).eval()
    if (args.ckpt / "adapter_config.json").exists():
        from peft import PeftModel

        model.backbone = PeftModel.from_pretrained(model.backbone, args.ckpt).eval()
    model.load_head(args.ckpt)
    policy = VLAPolicy(model, exec_horizon=cfg["vla"]["exec_horizon"])

    env_cfg = sorting_env_config(task_mode=args.task_mode) if args.scene == "sorting" else None
    env = PandaPickPlaceEnv(env_cfg)
    max_steps = cfg["pipeline"]["max_steps_per_subtask"]
    successes = 0
    for ep in range(args.episodes):
        obs = env.reset(seed=args.seed + ep)
        instruction = env.task["instruction"]
        subtasks = planner.plan(instruction, obs["images"]) if planner else subtask_texts(env.task)
        frames = [tile(obs["images"])] if ep < args.save_videos else None
        done = False
        info = {"success": False}
        t0 = time.time()
        for k, sub in enumerate(subtasks):
            is_last = k == len(subtasks) - 1
            policy.reset()
            sub_done = False
            budget = 55 if args.scene == "sorting" else max_steps
            for _ in range(budget):
                action = policy.act(obs, sub)
                obs, _, done, info = env.step(action)
                if frames is not None:
                    frames.append(tile(obs["images"]))
                if done or env.is_subtask_done(sub, is_last=True):
                    sub_done = True
                    break
            if args.scene == "sorting" and not sub_done and "pick" in sub.lower():
                if env.is_object_grasped(sub):
                    obs = env.finish_pick_lift(frames, tile)
                else:
                    obs = env.return_to_home(frames, tile)
                    policy.reset()
                    for _ in range(55):
                        action = policy.act(obs, sub)
                        obs, _, done, info = env.step(action)
                        if frames is not None:
                            frames.append(tile(obs["images"]))
                        if done or env.is_subtask_done(sub, is_last=True):
                            break
                    if env.is_object_grasped(sub) and not env.is_subtask_done(sub, is_last=True):
                        obs = env.finish_pick_lift(frames, tile)
            if "place" in sub.lower() and not is_last:
                obs = env.lift_after_place(frames, tile)
            if done:
                break
        pairs = env.task.get("pairs", [])
        placed_str = ""
        if len(pairs) > 1:
            placed_cnt = sum(env.is_pair_placed(o, t) for o, t in pairs)
            placed_str = f" placed={placed_cnt}/{len(pairs)}"
        ep_ok = env.is_success()
        successes += int(ep_ok)
        print(f"[ep {ep + 1}/{args.episodes}] success={ep_ok}{placed_str} steps={env.step_count} "
              f"({time.time() - t0:.1f}s) task='{instruction}'")
        if frames is not None:
            imageio.mimsave(args.out / f"eval_ep{ep:03d}_s{int(info['success'])}.mp4",
                            frames, fps=int(env.cfg.control_hz))

    print(f"[summary] success rate: {successes}/{args.episodes} ({100 * successes / max(1, args.episodes):.1f}%)")
    env.close()


if __name__ == "__main__":
    main()
