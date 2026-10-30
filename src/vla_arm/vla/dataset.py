"""HDF5 데모 -> (3 카메라 이미지, subtask 문장, state, action chunk) 샘플."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import Dataset

from .model import CAMERAS


class DemoDataset(Dataset):
    def __init__(self, root: str | Path, chunk_size: int = 8, episodes: list[int] | None = None,
                 image_aug: bool = True):
        self.files = sorted(Path(root).expanduser().glob("episode_*.h5"))
        if episodes is not None:
            self.files = [self.files[i] for i in episodes if i < len(self.files)]
        if not self.files:
            raise FileNotFoundError(f"no episode_*.h5 in {root}")
        self.chunk = chunk_size
        self.image_aug = image_aug
        self.index: list[tuple[int, int]] = []
        self.meta: list[dict] = []
        for i, p in enumerate(self.files):
            with h5py.File(p, "r") as f:
                T = f["action"].shape[0]
                self.meta.append({"subtasks": json.loads(f.attrs["subtasks"]), "T": T})
            self.index += [(i, t) for t in range(T)]
        self._handles: dict[int, h5py.File] = {}

    def __len__(self) -> int:
        return len(self.index)

    def _file(self, i: int) -> h5py.File:
        # DataLoader worker 마다 별도 핸들 (fork 후 lazy open)
        if i not in self._handles:
            self._handles[i] = h5py.File(self.files[i], "r")
        return self._handles[i]

    def _augment(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        # 고정 카메라-월드 좌표계 정합을 유지하기 위해 공간 이동(shift)은 하지 않고 밝기/대비 jitter 만 적용
        out = img.astype(np.float32)
        out = (out - 128.0) * rng.uniform(0.9, 1.1) + 128.0 + rng.uniform(-10, 10)
        return np.clip(out, 0, 255).astype(np.uint8)

    def __getitem__(self, k: int) -> dict:
        i, t = self.index[k]
        f = self._file(i)
        T = self.meta[i]["T"]
        rng = np.random.default_rng()
        images = {}
        for c in CAMERAS:
            img = f["images"][c][t]
            images[c] = self._augment(img, rng) if self.image_aug else img
        sub = int(f["subtask"][t])
        end = min(t + self.chunk, T)
        sub_slice = f["subtask"][t:end]
        same_mask = sub_slice == sub
        if not same_mask.all():
            # 다음 subtask(예: 다음 물체 집기)의 이동 방향이 현재 subtask 액션 청크에 섞이지 않도록 경계에서 절단
            first_diff = int(np.argmax(~same_mask))
            end = t + max(1, first_diff)
        acts = f["action"][t:end]
        if len(acts) < self.chunk:  # 에피소드 또는 subtask 끝: 이동 0 + 마지막 그리퍼 상태 유지로 패딩
            pad = np.zeros((self.chunk - len(acts), acts.shape[1]), np.float32)
            pad[:, 6] = acts[-1, 6]
            acts = np.concatenate([acts, pad], 0)
        return {
            "images": images,
            "text": self.meta[i]["subtasks"][sub],
            "state": f["state"][t].astype(np.float32),
            "actions": acts.astype(np.float32),
        }


def collate(batch: list[dict]) -> dict:
    return {
        "images": [b["images"] for b in batch],
        "texts": [b["text"] for b in batch],
        "state": np.stack([b["state"] for b in batch]),
        "actions": np.stack([b["actions"] for b in batch]),
    }
