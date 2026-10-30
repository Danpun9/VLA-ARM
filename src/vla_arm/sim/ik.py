"""Damped Least Squares 기반 미분 역기구학 (TCP 6DoF 목표 -> 7 관절 목표)."""

from __future__ import annotations

import mujoco
import numpy as np


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.empty(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def quat_from_rotvec(rv: np.ndarray) -> np.ndarray:
    q = np.empty(4)
    angle = float(np.linalg.norm(rv))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mju_axisAngle2Quat(q, rv / angle, angle)
    return q


def quat_error(q_target: np.ndarray, q_current: np.ndarray) -> np.ndarray:
    """월드 프레임 회전 오차 (rotvec). q_err = q_target * q_current^-1."""
    q_inv = np.empty(4)
    mujoco.mju_negQuat(q_inv, q_current)
    q_err = quat_mul(q_target, q_inv)
    rv = np.empty(3)
    mujoco.mju_quat2Vel(rv, q_err, 1.0)
    return rv


class DiffIK:
    def __init__(self, model: mujoco.MjModel, site: str = "tcp", joint_names: list[str] | None = None,
                 damping: float = 1e-2, max_iters: int = 20, tol: float = 1e-4):
        self.model = model
        self.data = mujoco.MjData(model)  # IK 전용 scratch data
        self.site_id = model.site(site).id
        joint_names = joint_names or [f"joint{i}" for i in range(1, 8)]
        self.qadr = np.array([model.joint(j).qposadr[0] for j in joint_names])
        self.dadr = np.array([model.joint(j).dofadr[0] for j in joint_names])
        self.lo = model.jnt_range[[model.joint(j).id for j in joint_names], 0]
        self.hi = model.jnt_range[[model.joint(j).id for j in joint_names], 1]
        self.damping = damping
        self.max_iters = max_iters
        self.tol = tol
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def site_pose(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        pos = data.site_xpos[self.site_id].copy()
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, data.site_xmat[self.site_id])
        return pos, quat

    def solve(self, qpos_full: np.ndarray, target_pos: np.ndarray, target_quat: np.ndarray,
              q_nominal: np.ndarray | None = None, nullspace_gain: float = 0.05) -> np.ndarray:
        """현재 qpos 에서 출발해 목표 TCP 포즈를 만족하는 7관절 각도를 반환."""
        d = self.data
        d.qpos[:] = qpos_full
        q = d.qpos[self.qadr].copy()
        for _ in range(self.max_iters):
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)
            pos, quat = self.site_pose(d)
            err = np.concatenate([target_pos - pos, quat_error(target_quat, quat)])
            if np.linalg.norm(err) < self.tol:
                break
            mujoco.mj_jacSite(self.model, d, self._jacp, self._jacr, self.site_id)
            J = np.vstack([self._jacp, self._jacr])[:, self.dadr]
            JJt = J @ J.T + (self.damping ** 2) * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)
            if q_nominal is not None:  # 영공간에서 home 자세 유지 (팔꿈치 꼬임 방지)
                N = np.eye(7) - np.linalg.pinv(J) @ J
                dq += N @ (nullspace_gain * (q_nominal - q))
            q = np.clip(q + dq, self.lo, self.hi)
        return q
