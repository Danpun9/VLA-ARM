"""1단계 검증: 스크립트 정책으로 로봇팔을 움직이고 3개 카메라 영상을 저장.

    MUJOCO_GL=egl uv run python scripts/run_scripted.py --episodes 2
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import numpy as np

from vla_arm.sim import CAMERAS, PandaPickPlaceEnv, ScriptedPickPlace, sorting_env_config, subtask_texts


def tile(images: dict) -> np.ndarray:
    return np.concatenate([images[c] for c in CAMERAS], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scene", choices=["default", "sorting"], default="default")
    ap.add_argument("--task-mode", choices=["single", "sort_all"], default="sort_all")
    ap.add_argument("--out", type=Path, default=Path("outputs/scripted"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    env_cfg = sorting_env_config(task_mode=args.task_mode) if args.scene == "sorting" else None
    env = PandaPickPlaceEnv(env_cfg)
    policy = ScriptedPickPlace(env)
    n_success = 0
    for ep in range(args.episodes):
        obs = env.reset(seed=args.seed + ep)
        policy.reset()
        print(f"[ep {ep}] {env.task['instruction']}  subtasks={subtask_texts(env.task)}")
        frames = [tile(obs["images"])]
        imageio.imwrite(args.out / f"ep{ep:03d}_first.png", frames[0])
        info = {"success": False}
        done = False
        while not done and not policy.done:
            obs, _, done, info = env.step(policy.act())
            frames.append(tile(obs["images"]))
        n_success += int(info["success"])
        print(f"  steps={env.step_count} success={info['success']} final_phase={policy.phase}")
        imageio.mimsave(args.out / f"ep{ep:03d}.mp4", frames, fps=int(env.cfg.control_hz))
    print(f"success {n_success}/{args.episodes} -> {args.out}")
    env.close()


if __name__ == "__main__":
    main()
