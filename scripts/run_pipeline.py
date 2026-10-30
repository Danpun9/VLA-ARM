"""LLM Planner(Qwen3.5-9B) + VLA(Qwen3.5-4B) 로 MuJoCo Panda 를 구동하는 전체 파이프라인.

예)
  # 전체 (planner + VLA, VLA 는 학습 전이면 랜덤에 가까운 작은 움직임)
  uv run python scripts/run_pipeline.py
  # planner 없이 템플릿 subtask 로 VLA 만
  uv run python scripts/run_pipeline.py --no-planner
  # planner + 스크립트 정책 (planner 출력/연동 검증용)
  uv run python scripts/run_pipeline.py --policy scripted
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")  # WSL CUDA "device not ready" 회피

import imageio.v2 as imageio
import numpy as np
import torch
import yaml

from vla_arm.sim import CAMERAS, PandaPickPlaceEnv, ScriptedPickPlace, sorting_env_config, subtask_texts


def tile(images: dict) -> np.ndarray:
    return np.concatenate([images[c] for c in CAMERAS], axis=1)


def vram() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return f"{torch.cuda.memory_allocated() / 2**30:.1f}GB alloc / {torch.cuda.max_memory_allocated() / 2**30:.1f}GB peak"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--instruction", type=str, default=None, help="미지정 시 환경이 샘플링한 지시문 사용")
    ap.add_argument("--policy", choices=["vla", "scripted"], default="vla")
    ap.add_argument("--scene", choices=["default", "sorting"], default="default")
    ap.add_argument("--task-mode", choices=["single", "sort_all"], default="sort_all")
    ap.add_argument("--no-planner", action="store_true")
    ap.add_argument("--vla-ckpt", type=Path, default=None, help="학습된 action head / LoRA 디렉토리")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("outputs/pipeline"))
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=True)

    # WSL 주의: VRAM 초과 시 OOM 대신 "CUDA driver error: device not ready" 가 발생함.
    # Planner(9B NF4 ≈7.4GB) + VLA(4B bf16 ≈8.5GB) 는 16GB 초과 ->
    #   keep_planner_loaded=false : planner 로 계획 -> 언로드 -> VLA 로드 (기본)
    #   keep_planner_loaded=true  : VLA 를 8bit 로 강제 로드 (닫힌 루프 check_done 사용 가능)
    keep_planner = cfg["pipeline"]["keep_planner_loaded"] and not args.no_planner
    planner = None
    if not args.no_planner:
        from vla_arm.planner import PlannerConfig, QwenPlanner

        t0 = time.time()
        planner = QwenPlanner(PlannerConfig(**cfg["planner"]))
        print(f"[planner] loaded in {time.time() - t0:.0f}s, VRAM {vram()}")

    env_cfg = sorting_env_config(task_mode=args.task_mode) if args.scene == "sorting" else None
    env = PandaPickPlaceEnv(env_cfg)
    obs = env.reset(seed=args.seed)
    instruction = args.instruction or env.task["instruction"]
    print(f"[task] {instruction}")

    # ---------------- Planner ----------------
    if planner is None:
        subtasks = subtask_texts(env.task)
    else:
        t0 = time.time()
        subtasks = planner.plan(instruction, obs["images"])
        print(f"[planner] {time.time() - t0:.1f}s -> {subtasks}")
        if not keep_planner:
            planner.unload()
            planner = None
            print(f"[planner] unloaded, VRAM {vram()}")

    # ---------------- Low-level policy ----------------
    if args.policy == "vla":
        from vla_arm.vla import QwenVLA, VLAConfig, VLAPolicy

        t0 = time.time()
        vla_cfg = VLAConfig(**cfg["vla"]["model"])
        if keep_planner and not (vla_cfg.load_in_8bit or vla_cfg.load_in_4bit):
            print("[vla] planner 상주 -> VRAM 절약을 위해 VLA 를 4bit 로 로드")
            vla_cfg.load_in_4bit = True
        model = QwenVLA(vla_cfg).eval()
        if args.vla_ckpt:
            if (args.vla_ckpt / "adapter_config.json").exists():
                from peft import PeftModel

                model.backbone = PeftModel.from_pretrained(model.backbone, args.vla_ckpt).eval()
            model.load_head(args.vla_ckpt)
        else:
            print("[vla] WARNING: 체크포인트 없음 -> 학습 전 action head (작은 랜덤 액션) 로 동작")
        print(f"[vla] loaded in {time.time() - t0:.0f}s, VRAM {vram()}")
        policy = VLAPolicy(model, exec_horizon=cfg["vla"]["exec_horizon"])
    else:
        policy = ScriptedPickPlace(env)

    # ---------------- Control loop ----------------
    frames = [tile(obs["images"])]
    max_steps = cfg["pipeline"]["max_steps_per_subtask"]
    check_every = cfg["pipeline"]["check_every"]
    done = False
    info = {"success": False}
    for k, subtask in enumerate(subtasks):
        is_last = k == len(subtasks) - 1
        print(f"[subtask {k}] {subtask}")
        if args.policy == "vla":
            policy.reset()
        t_sub = time.time()
        sub_done = False
        step_budget = 55 if args.scene == "sorting" else max_steps
        for step in range(step_budget):
            if args.policy == "scripted":
                if policy.done or policy.subtask_index > k:
                    break
                action = policy.act()
            else:
                action = policy.act(obs, subtask)
            obs, _, done, info = env.step(action)
            frames.append(tile(obs["images"]))
            if done:
                sub_done = True
                break
            if args.policy == "vla" and env.is_subtask_done(subtask, is_last=True):
                print(f"  subtask '{subtask}' completed at step {step + 1} (tcp_z={obs['state'][2]:.3f})")
                sub_done = True
                break
            # 닫힌 루프: planner 가 subtask 완료 여부를 주기적으로 판단
            if planner is not None and args.policy == "vla" and check_every > 0 and (step + 1) % check_every == 0:
                if env.is_subtask_done(subtask, is_last=True) and planner.check_done(subtask, obs["images"]):
                    print(f"  planner: '{subtask}' done at step {step + 1}")
                    sub_done = True
                    break
        # 다단계 sorting 에서 1차 파지 미완료 시: 이미 목표 물체를 파지했으면 리프트만 완료, 아니면 홈 시야(HOME_QPOS) 복귀 후 재시도
        if args.policy == "vla" and args.scene == "sorting" and not sub_done and "pick" in subtask.lower():
            if env.is_object_grasped(subtask):
                obs = env.finish_pick_lift(frames, tile)
            else:
                print("  returning to home view for pick retry...")
                obs = env.return_to_home(frames, tile)
                policy.reset()
                for step2 in range(55):
                    action = policy.act(obs, subtask)
                    obs, _, done, info = env.step(action)
                    frames.append(tile(obs["images"]))
                    if done or env.is_subtask_done(subtask, is_last=True):
                        print(f"  subtask '{subtask}' completed on retry at step {step2 + 1}")
                        break
                if env.is_object_grasped(subtask) and not env.is_subtask_done(subtask, is_last=True):
                    obs = env.finish_pick_lift(frames, tile)
        if args.policy == "vla" and "place" in subtask.lower() and not is_last:
            obs = env.lift_after_place(frames, tile)
        dt = time.time() - t_sub
        print(f"  steps={step + 1} ({(step + 1) / max(dt, 1e-6):.1f} Hz) tcp={obs['state'][:3].round(3)}")
        if done:
            break

    pairs = env.task.get("pairs", [])
    if len(pairs) > 1:
        placed = sum(env.is_pair_placed(o, t) for o, t in pairs)
        print(f"[sorting] placed={placed}/{len(pairs)} " + ", ".join(
            f"{o}->{t}:{'OK' if env.is_pair_placed(o, t) else 'MISS'}" for o, t in pairs
        ))
    print(f"[result] success={env.is_success()}  total_steps={env.step_count}  VRAM {vram()}")
    video = args.out / f"{args.policy}_seed{args.seed}.mp4"
    imageio.mimsave(video, frames, fps=int(env.cfg.control_hz))
    print(f"[video] {video}")
    env.close()


if __name__ == "__main__":
    main()
