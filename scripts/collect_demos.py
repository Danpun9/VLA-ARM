"""스크립트 전문가로 VLA 학습용 데모 수집 (HDF5, 에피소드당 1파일, 멀티프로세스 지원).

    uv run python scripts/collect_demos.py --episodes 200 --workers 8 --out ~/data/panda_pickplace

파일 구조 (episode_XXXXX.h5)
    images/top, images/side, images/wrist : (T, H, W, 3) uint8   (lzf 압축)
    state   : (T, 8)  float32   tcp_pos(3) + tcp_quat(4) + gripper_opening(1)
    qpos    : (T, 7)  float32
    action  : (T, 7)  float32   env 에 적용되는 단위로 클리핑된 "정답" 액션 (노이즈 없음)
    subtask : (T,)    int8      subtask_texts 인덱스
    phase   : (T,)    int8      전문가 상태머신 phase (planner 완료판정 라벨용)
    attrs   : instruction, subtasks(json), object, target, success, seed

DART 스타일 노이즈 주입: 실행 액션에는 노이즈를 섞어 분포 이탈 상태를 만들고,
라벨은 해당 상태에서의 전문가 정답 액션을 저장 -> 공변량 이동(compounding error) 완화.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import h5py
import numpy as np

from vla_arm.sim import CAMERAS, PandaPickPlaceEnv, ScriptedPickPlace, sorting_env_config, subtask_texts

_WORKER_ENV = None
_WORKER_POLICY = None


def _init_worker(scene_name: str, task_mode: str):
    global _WORKER_ENV, _WORKER_POLICY
    os.environ.setdefault("MUJOCO_GL", "egl")
    env_cfg = sorting_env_config(task_mode=task_mode) if scene_name == "sorting" else None
    _WORKER_ENV = PandaPickPlaceEnv(env_cfg)
    _WORKER_POLICY = ScriptedPickPlace(_WORKER_ENV)


def _collect_one(task_args: tuple[int, Path, int, float, float]) -> tuple[int, bool, int]:
    ep_idx, out_dir, base_seed, noise, noise_prob = task_args
    path = out_dir / f"episode_{ep_idx:05d}.h5"
    if path.exists():
        return ep_idx, True, 0

    env = _WORKER_ENV
    policy = _WORKER_POLICY
    lim = np.array([env.cfg.max_dpos] * 3 + [env.cfg.max_drot] * 3 + [1.0])

    attempt = 0
    while True:
        seed = base_seed + ep_idx * 100 + attempt
        attempt += 1
        rng = np.random.default_rng(seed)
        obs = env.reset(seed=seed)
        policy.reset()
        noisy = rng.random() < noise_prob
        buf = {k: [] for k in ("state", "qpos", "action", "subtask", "phase", *[f"img_{c}" for c in CAMERAS])}
        info = {"success": False}
        # 후퇴 단계(phase 7, 안전 높이 복귀)까지 데모에 포함되도록 policy.done 또는 max_steps 까지 실행
        while not policy.done and env.step_count < env.cfg.max_steps:
            a = np.clip(policy.act(), -lim, lim)
            a[6] = float(a[6] > 0.5)
            buf["state"].append(obs["state"])
            buf["qpos"].append(obs["qpos"])
            buf["action"].append(a.astype(np.float32))
            buf["subtask"].append(policy.subtask_index)
            buf["phase"].append(policy.phase)
            for c in CAMERAS:
                buf[f"img_{c}"].append(obs["images"][c])
            exec_a = a.copy()
            if noisy:
                exec_a[:3] += rng.normal(0, noise * env.cfg.max_dpos, 3)
            obs, _, _, info = env.step(exec_a)
        if env.is_success():
            break

    tmp_path = out_dir / f".tmp_episode_{ep_idx:05d}.h5"
    with h5py.File(tmp_path, "w") as f:
        g = f.create_group("images")
        for c in CAMERAS:
            g.create_dataset(c, data=np.stack(buf[f"img_{c}"]), compression="lzf",
                             chunks=(1, *buf[f"img_{c}"][0].shape))
        f["state"] = np.stack(buf["state"])
        f["qpos"] = np.stack(buf["qpos"])
        f["action"] = np.stack(buf["action"])
        f["subtask"] = np.array(buf["subtask"], dtype=np.int8)
        f["phase"] = np.array(buf["phase"], dtype=np.int8)
        f.attrs.update(instruction=env.task["instruction"], subtasks=json.dumps(subtask_texts(env.task)),
                       object=env.task["object"], target=env.task["target"], success=True, seed=seed,
                       noisy=noisy, control_hz=env.cfg.control_hz)
    tmp_path.rename(path)
    return ep_idx, True, len(buf["action"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200, help="저장할 성공 에피소드 수")
    ap.add_argument("--out", type=Path, default=Path("data/panda_pickplace"))
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--scene", choices=["default", "sorting"], default="default")
    ap.add_argument("--task-mode", choices=["single", "sort_all"], default="single")
    ap.add_argument("--noise", type=float, default=0.3, help="실행 액션 노이즈 (max_dpos 대비 비율)")
    ap.add_argument("--noise-prob", type=float, default=0.5, help="에피소드별 노이즈 적용 확률")
    ap.add_argument("--workers", type=int, default=6, help="병렬 수집 프로세스 수")
    args = ap.parse_args()
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)

    tasks = [(i, out, args.seed, args.noise, args.noise_prob) for i in range(args.episodes)]
    t0 = time.time()
    done_cnt = 0

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers, initializer=_init_worker, initargs=(args.scene, args.task_mode)) as pool:
        for ep_idx, _, length in pool.imap_unordered(_collect_one, tasks):
            done_cnt += 1
            if done_cnt % 10 == 0 or done_cnt == args.episodes:
                print(f"[{done_cnt}/{args.episodes}] last_ep={ep_idx} T={length} elapsed={time.time() - t0:.0f}s",
                      flush=True)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
