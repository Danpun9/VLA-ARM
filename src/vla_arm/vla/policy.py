"""VLA 추론 래퍼: action chunk 를 receding-horizon 방식으로 실행."""

from __future__ import annotations

from collections import deque

import numpy as np

from .model import QwenVLA


class VLAPolicy:
    def __init__(self, model: QwenVLA, exec_horizon: int = 4, gripper_threshold: float | None = 0.5):
        self.model = model
        self.exec_horizon = min(exec_horizon, model.cfg.chunk_size)
        self.gripper_threshold = gripper_threshold
        self.queue: deque[np.ndarray] = deque()
        self.hover_cnt: int = 0

    def reset(self):
        self.queue.clear()
        self.hover_cnt = 0

    def act(self, obs: dict, subtask: str) -> np.ndarray:
        if not self.queue:
            chunk = self.model.predict(obs["images"], subtask, obs["state"])
            self.queue.extend(chunk[: self.exec_horizon])
        a = self.queue.popleft().astype(np.float32)
        if self.gripper_threshold is not None:
            a[6] = float(a[6] > self.gripper_threshold)
        # phase 0 -> phase 1 상공(z≈0.14m) 수평 정렬 완료 시 하강 유도
        tcp_z = float(obs["state"][2])
        grip_open = float(obs["state"][7])
        if "pick" in subtask.lower() and grip_open > 0.6 and 0.06 < tcp_z < 0.165:
            if float(np.linalg.norm(a[:2])) < 0.005:
                self.hover_cnt += 1
                if self.hover_cnt >= 2:
                    a[2] = min(float(a[2]), -0.025)
            else:
                self.hover_cnt = 0
        elif "place" in subtask.lower():
            # 접시 상공(x>0.54, z<0.12)에 도달하기 전에는 조기 그리퍼 개방 방지
            if float(obs["state"][0]) < 0.54 or tcp_z > 0.12:
                a[6] = 1.0
            if float(obs["state"][0]) > 0.54 and 0.065 < tcp_z < 0.165:
                if float(np.linalg.norm(a[:2])) < 0.005:
                    self.hover_cnt += 1
                    if self.hover_cnt >= 2:
                        a[2] = min(float(a[2]), -0.025)
                else:
                    self.hover_cnt = 0
            elif tcp_z <= 0.062:
                a[6] = 0.0
        return a
