"""스크립트 전문가 정책 (pick-and-place 상태머신).

1단계: 시뮬레이터/IK/카메라가 제대로 동작하는지 검증
2단계: VLA 학습용 데모 데이터 생성 (각 스텝에 언어 subtask 라벨을 같이 출력)
"""

from __future__ import annotations

import numpy as np

from .env import OBJECT_NAMES_EN, PandaPickPlaceEnv
from .scene import TABLE_TOP_Z


def subtask_texts(task: dict) -> list[str]:
    """Planner 가 출력해야 하는 subtask 문장 템플릿 (VLA 학습 라벨과 동일하게 유지)."""
    pairs = task.get("pairs", [(task["object"], task["target"])])
    out: list[str] = []
    for obj_key, tgt_key in pairs:
        obj, tgt = OBJECT_NAMES_EN[obj_key], OBJECT_NAMES_EN[tgt_key]
        out.extend([f"pick up the {obj}", f"place the {obj} on the {tgt}"])
    return out


def _wrap(a: float, period: float) -> float:
    return (a + period / 2) % period - period / 2


class ScriptedPickPlace:
    HOVER = 0.12
    LIFT = 0.15

    def __init__(self, env: PandaPickPlaceEnv, gain: float = 1.0):
        self.env = env
        self.gain = gain
        self.reset()

    def reset(self):
        self.pair_idx = 0
        self.phase = 0
        self.counter = 0
        self.done = False

    @property
    def subtask_index(self) -> int:
        return 2 * self.pair_idx + (0 if self.phase <= 3 else 1)

    def _yaw_error(self, obj_name: str) -> float:
        if "cylinder" in obj_name:
            return 0.0  # 원기둥은 축대칭이므로 불필요한 손목 회전 생략
        d = self.env.data
        R_tcp = d.site_xmat[self.env.ik.site_id].reshape(3, 3)
        finger_axis = R_tcp[:, 1]
        R_obj = d.body(obj_name).xmat.reshape(3, 3)
        obj_axis = R_obj[:, 0]
        a_f = np.arctan2(finger_axis[1], finger_axis[0])
        a_o = np.arctan2(obj_axis[1], obj_axis[0])
        return _wrap(a_o - a_f, np.pi / 2)  # 큐브는 90도 대칭

    def act(self) -> np.ndarray:
        env = self.env
        task = env.task
        pairs = task.get("pairs", [(task["object"], task["target"])])
        cur_obj, cur_tgt = pairs[min(self.pair_idx, len(pairs) - 1)]
        obj = env.body_pos(cur_obj)
        tgt = env.data.mocap_pos[env.model.body(cur_tgt).mocapid[0]].copy()
        tcp, _ = env.tcp_pose()

        grip = 0.0
        yaw = 0.0
        if self.phase == 0:  # 물체 위로 접근 + yaw 정렬
            goal = obj + [0, 0, self.HOVER]
            yaw = self._yaw_error(cur_obj)
            if np.linalg.norm(goal - tcp) < 0.01 and abs(yaw) < 0.03:
                self.phase = 1
        elif self.phase == 1:  # 하강
            goal = obj + [0, 0, 0.0]
            yaw = self._yaw_error(cur_obj)
            if np.linalg.norm(goal - tcp) < 0.008:
                self.phase, self.counter = 2, 0
        elif self.phase == 2:  # 파지
            goal, grip = tcp, 1.0
            self.counter += 1
            if self.counter >= 8:
                self.phase = 3
        elif self.phase == 3:  # 들어올리기
            goal, grip = np.array([tcp[0], tcp[1], TABLE_TOP_Z + self.LIFT]), 1.0
            if abs(goal[2] - tcp[2]) < 0.01:
                self.phase = 4
        elif self.phase == 4:  # 목표 위로 이동
            goal, grip = tgt + [0, 0, self.LIFT], 1.0
            if np.linalg.norm(goal - tcp) < 0.01:
                self.phase = 5
        elif self.phase == 5:  # 내려놓기 높이까지 하강
            goal, grip = tgt + [0, 0, 0.05], 1.0
            if np.linalg.norm(goal - tcp) < 0.01:
                self.phase, self.counter = 6, 0
        elif self.phase == 6:  # 놓기
            goal, grip = tcp, 0.0
            self.counter += 1
            if self.counter >= 6:
                self.phase = 7
        else:  # 후퇴 (다음 물체로 이동하기 전 안전 높이 확보)
            goal = np.array([tcp[0], tcp[1], TABLE_TOP_Z + 0.16])
            if abs(goal[2] - tcp[2]) < 0.015:
                if self.pair_idx + 1 < len(pairs):
                    self.pair_idx += 1
                    self.phase = 0
                    self.counter = 0
                else:
                    self.done = True

        dpos = self.gain * (goal - tcp)
        return np.array([*dpos, 0.0, 0.0, yaw, grip], dtype=np.float32)
