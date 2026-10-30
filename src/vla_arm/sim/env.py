"""Panda 테이블탑 pick-and-place 환경.

Action (7,) : [dx, dy, dz, drx, dry, drz, gripper]
    - dx..dz  : 월드 좌표계 TCP 위치 변화량 (m), |.| <= max_dpos 로 클리핑
    - drx..drz: 월드 좌표계 TCP 회전 변화량 (rotvec, rad), |.| <= max_drot
    - gripper : 0 = 열림, 1 = 닫힘 (연속값 허용)
Observation dict:
    - images : {"top", "side", "wrist"} -> (H, W, 3) uint8
    - state  : (8,) [tcp_pos(3), tcp_quat(4, wxyz), gripper_opening(1, 0~1)]
    - qpos   : (7,) 관절각
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .ik import DiffIK, quat_from_rotvec, quat_mul
from .scene import HOME_QPOS, TABLE_TOP_Z, SceneConfig, compile_scene, sorting_scene_config

CAMERAS = ("top", "side", "wrist")
OBJECT_NAMES_EN = {
    "red_cube": "red cube",
    "green_cube": "green cube",
    "blue_cube": "blue cube",
    "yellow_cube": "yellow cube",
    "purple_cylinder": "purple cylinder",
    "red_plate": "red plate",
    "yellow_plate": "yellow plate",
    "blue_plate": "blue plate",
    "purple_plate": "purple plate",
    "green_plate": "green plate",
}
COLOR_MATCH_TARGET = {
    "red_cube": "red_plate",
    "yellow_cube": "yellow_plate",
    "blue_cube": "blue_plate",
    "purple_cylinder": "purple_plate",
    "green_cube": "yellow_plate",
}


@dataclass
class EnvConfig:
    image_size: tuple[int, int] = (256, 256)  # (H, W)
    control_hz: float = 10.0
    max_dpos: float = 0.03
    max_drot: float = 0.15
    max_steps: int = 300
    obj_x_range: tuple[float, float] = (0.35, 0.65)
    obj_y_range: tuple[float, float] = (-0.25, 0.25)
    min_obj_dist: float = 0.10
    task_mode: str = "single"  # "single" (단일 물체 배치) | "sort_all" (다중 물체 색상별 연속 분류)
    scene: SceneConfig = field(default_factory=SceneConfig)


def sorting_env_config(task_mode: str = "sort_all", max_steps: int = 900) -> EnvConfig:
    """다중 오브젝트(큐브 3종 + 보라 원기둥) 및 4색 접시 분류(Sorting) 환경 설정."""
    return EnvConfig(
        max_steps=max_steps,
        obj_x_range=(0.37, 0.49),
        obj_y_range=(-0.21, 0.21),
        min_obj_dist=0.085,
        task_mode=task_mode,
        scene=sorting_scene_config(),
    )


class PandaPickPlaceEnv:
    def __init__(self, cfg: EnvConfig | None = None, render: bool = True):
        self.cfg = cfg or EnvConfig()
        self.model, self.spec = compile_scene(self.cfg.scene)
        self.data = mujoco.MjData(self.model)
        self.ik = DiffIK(self.model, site="tcp")
        self.n_substeps = max(1, int(round(1.0 / (self.cfg.control_hz * self.model.opt.timestep))))
        self.arm_qadr = self.ik.qadr
        self.finger_qadr = np.array([self.model.joint(j).qposadr[0] for j in ("finger_joint1", "finger_joint2")])
        self.object_names = [o.name for o in self.cfg.scene.objects]
        self.target_names = [t.name for t in self.cfg.scene.targets]
        self.renderer = mujoco.Renderer(self.model, *self.cfg.image_size) if render else None
        self.rng = np.random.default_rng()
        self.step_count = 0
        self.task: dict = {}
        self._target_pos = np.zeros(3)
        self._target_quat = np.array([1.0, 0, 0, 0])
        self._gripper_cmd = 0.0

    # ------------------------------------------------------------------ utils
    def body_pos(self, name: str) -> np.ndarray:
        return self.data.body(name).xpos.copy()

    def tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.ik.site_pose(self.data)

    def gripper_opening(self) -> float:
        return float(np.clip(self.data.qpos[self.finger_qadr].sum() / 0.08, 0.0, 1.0))

    def _sample_positions(self, n: int) -> np.ndarray:
        pts: list[np.ndarray] = []
        while len(pts) < n:
            p = np.array([self.rng.uniform(*self.cfg.obj_x_range), self.rng.uniform(*self.cfg.obj_y_range)])
            if all(np.linalg.norm(p - q) >= self.cfg.min_obj_dist for q in pts):
                pts.append(p)
        return np.stack(pts)

    # ------------------------------------------------------------------ API
    def reset(self, seed: int | None = None, task: dict | None = None) -> dict:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.arm_qadr] = HOME_QPOS
        self.data.qpos[self.finger_qadr] = 0.04
        self.data.ctrl[:7] = HOME_QPOS
        self.data.ctrl[7] = 255.0

        if len(self.target_names) == 1:
            xy = self._sample_positions(len(self.object_names) + len(self.target_names))
            obj_xy = xy[:len(self.object_names)]
            tgt_xy = xy[len(self.object_names):]
        else:
            # 다중 접시 모드: 접시들을 전방 열(x ≈ 0.60, y 등간격)에 겹치지 않게 배치하고 물체는 중앙 영역에 샘플링
            obj_xy = self._sample_positions(len(self.object_names))
            ys = np.linspace(-0.21, 0.21, len(self.target_names))
            tgt_xy = np.stack([
                np.array([0.60 + self.rng.uniform(-0.008, 0.008), y + self.rng.uniform(-0.008, 0.008)])
                for y in ys
            ])

        for i, name in enumerate(self.object_names):
            jadr = self.model.joint(f"{name}_joint").qposadr[0]
            half = self.cfg.scene.objects[i].half_size
            yaw = self.rng.uniform(-np.pi, np.pi)
            self.data.qpos[jadr:jadr + 3] = [obj_xy[i, 0], obj_xy[i, 1], TABLE_TOP_Z + half + 0.001]
            self.data.qpos[jadr + 3:jadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        for j, name in enumerate(self.target_names):
            mocap_id = self.model.body(name).mocapid[0]
            self.data.mocap_pos[mocap_id] = [*tgt_xy[j], TABLE_TOP_Z]
            self.data.mocap_quat[mocap_id] = [1.0, 0.0, 0.0, 0.0]

        mujoco.mj_forward(self.model, self.data)
        # 물체 안착
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

        self._target_pos, self._target_quat = self.tcp_pose()
        self._gripper_cmd = 0.0
        self.step_count = 0
        self.task = task or self.sample_task()
        return self.get_obs()

    def sample_task(self) -> dict:
        if self.cfg.task_mode == "sort_all" and len(self.target_names) > 1:
            pairs = [
                (obj, COLOR_MATCH_TARGET[obj])
                for obj in self.object_names
                if COLOR_MATCH_TARGET.get(obj) in self.target_names
            ]
            if self.rng.random() < 0.5:
                idx = self.rng.permutation(len(pairs))
                pairs = [pairs[i] for i in idx]
            names_en = ", ".join(OBJECT_NAMES_EN[o] for o, _ in pairs)
            return {
                "object": pairs[0][0],
                "target": pairs[0][1],
                "pairs": pairs,
                "instruction": f"Sort all objects ({names_en}) onto their matching colored plates.",
            }
        obj = self.object_names[self.rng.integers(len(self.object_names))]
        if len(self.target_names) > 1 and self.rng.random() < 0.6 and COLOR_MATCH_TARGET.get(obj) in self.target_names:
            tgt = COLOR_MATCH_TARGET[obj]
        else:
            tgt = self.target_names[self.rng.integers(len(self.target_names))]
        return {
            "object": obj,
            "target": tgt,
            "pairs": [(obj, tgt)],
            "instruction": f"Put the {OBJECT_NAMES_EN[obj]} on the {OBJECT_NAMES_EN[tgt]}.",
        }

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        assert a.shape == (7,), f"action shape must be (7,), got {a.shape}"
        dpos = np.clip(a[:3], -self.cfg.max_dpos, self.cfg.max_dpos)
        drot = np.clip(a[3:6], -self.cfg.max_drot, self.cfg.max_drot)
        self._gripper_cmd = float(np.clip(a[6], 0.0, 1.0))

        cur_pos, _ = self.tcp_pose()
        self._target_pos = self._target_pos + dpos
        # 목표가 실제 위치에서 너무 멀어지지 않도록 (접촉 시 windup 방지)
        off = self._target_pos - cur_pos
        if np.linalg.norm(off) > 0.08:
            self._target_pos = cur_pos + off / np.linalg.norm(off) * 0.08
        self._target_pos[2] = max(self._target_pos[2], TABLE_TOP_Z + 0.005)
        self._target_quat = quat_mul(quat_from_rotvec(drot), self._target_quat)
        self._target_quat /= np.linalg.norm(self._target_quat)

        q_des = self.ik.solve(self.data.qpos, self._target_pos, self._target_quat, q_nominal=HOME_QPOS)
        self.data.ctrl[:7] = q_des
        self.data.ctrl[7] = 255.0 * (1.0 - self._gripper_cmd)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        success = self.is_success()
        done = success or self.step_count >= self.cfg.max_steps
        info = {"success": success, "step": self.step_count}
        return self.get_obs(), float(success), done, info

    def is_pair_placed(self, obj_name: str, tgt_name: str) -> bool:
        obj = self.body_pos(obj_name)
        tgt = self.data.mocap_pos[self.model.body(tgt_name).mocapid[0]]
        radius = next(t.radius for t in self.cfg.scene.targets if t.name == tgt_name)
        return bool(np.linalg.norm(obj[:2] - tgt[:2]) < radius * 0.95 and obj[2] < TABLE_TOP_Z + 0.05)

    def is_object_grasped(self, subtask: str) -> bool:
        """목표 물체가 그리퍼에 파지되어 테이블에서 들려올려지기 시작했는지 확인."""
        s = subtask.lower()
        obj_key = next((k for k, v in OBJECT_NAMES_EN.items() if v in s and k in self.object_names),
                       self.task.get("object"))
        if not obj_key:
            return False
        obj_z = self.body_pos(obj_key)[2]
        return bool(obj_z > TABLE_TOP_Z + 0.032 and self.gripper_opening() < 0.75)

    def finish_pick_lift(self, frames: list | None = None, tile_fn=None) -> dict:
        """물체 파지 직후 리프트 높이(tcp_z>=0.15m)까지 그리퍼를 닫은 채 수직 상승."""
        for _ in range(8):
            if self.tcp_pose()[0][2] >= 0.15:
                break
            obs, _, _, _ = self.step(np.array([0.0, 0.0, 0.028, 0.0, 0.0, 0.0, 1.0], dtype=np.float32))
            if frames is not None and tile_fn is not None:
                frames.append(tile_fn(obs["images"]))
        return self.get_obs()

    def is_subtask_done(self, subtask: str, is_last: bool = False) -> bool:
        """서브태스크 문자열('pick up the <obj>' / 'place the <obj> on the <tgt>')로부터
        해당 단계의 물리적 완료 여부를 판정."""
        s = subtask.lower()
        obj_key = next((k for k, v in OBJECT_NAMES_EN.items() if v in s and k in self.object_names),
                       self.task.get("object"))
        if not obj_key:
            return False
        if "pick" in s:
            obj_z = self.body_pos(obj_key)[2]
            tcp_z = self.tcp_pose()[0][2]
            return bool(obj_z > 0.06 and tcp_z > 0.13 and self.gripper_opening() < 0.75)
        if "place" in s:
            tgt_key = next((k for k, v in OBJECT_NAMES_EN.items() if v in s and k in self.target_names),
                           self.task.get("target"))
            if not tgt_key:
                return False
            placed = self.is_pair_placed(obj_key, tgt_key) and self.gripper_opening() > 0.6
            if not placed:
                return False
            return True if is_last else bool(self.tcp_pose()[0][2] > 0.11)
        return False

    def lift_after_place(self, frames: list | None = None, tile_fn=None) -> dict:
        """전문가 데모의 phase 7 종료 상태와 일치하도록 접시 위에서 안전 높이(z>=0.155m)까지 수직 상승."""
        for _ in range(6):
            if self.tcp_pose()[0][2] >= 0.155:
                break
            obs, _, _, _ = self.step(np.array([0.0, 0.0, 0.025, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
            if frames is not None and tile_fn is not None:
                frames.append(tile_fn(obs["images"]))
        return self.get_obs()

    def return_to_home(self, frames: list | None = None, tile_fn=None) -> dict:
        """손목 카메라 시야 재확보를 위해 수직 상승 후 HOME_QPOS 로 부드럽게 복귀."""
        for _ in range(5):
            obs, _, _, _ = self.step(np.array([0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
            if frames is not None and tile_fn is not None:
                frames.append(tile_fn(obs["images"]))
        for _ in range(10):
            cur_pos, _ = self.tcp_pose()
            home_pos = np.array([0.50, 0.0, 0.22])
            dpos = np.clip(home_pos - cur_pos, -self.cfg.max_dpos, self.cfg.max_dpos)
            obs, _, _, _ = self.step(np.array([*dpos, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
            if frames is not None and tile_fn is not None:
                frames.append(tile_fn(obs["images"]))
            if np.linalg.norm(home_pos - cur_pos) < 0.015:
                break
        self.data.ctrl[:7] = HOME_QPOS
        self.data.ctrl[7] = 255.0
        for _ in range(30):
            mujoco.mj_step(self.model, self.data)
        self._target_pos, self._target_quat = self.tcp_pose()
        self._gripper_cmd = 0.0
        obs = self.get_obs()
        if frames is not None and tile_fn is not None:
            frames.append(tile_fn(obs["images"]))
        return obs

    def is_success(self) -> bool:
        if not self.task:
            return False
        pairs = self.task.get("pairs", [(self.task["object"], self.task["target"])])
        all_placed = all(self.is_pair_placed(o, t) for o, t in pairs)
        return bool(all_placed and self.gripper_opening() > 0.6)

    def render(self, camera: str) -> np.ndarray:
        assert self.renderer is not None, "render=False 로 생성된 환경입니다."
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def get_obs(self) -> dict:
        pos, quat = self.tcp_pose()
        state = np.concatenate([pos, quat, [self.gripper_opening()]]).astype(np.float32)
        obs = {
            "state": state,
            "qpos": self.data.qpos[self.arm_qadr].astype(np.float32).copy(),
            "instruction": self.task.get("instruction", ""),
        }
        if self.renderer is not None:
            obs["images"] = {c: self.render(c) for c in CAMERAS}
        return obs

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
