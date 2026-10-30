"""MuJoCo 씬 빌더: Menagerie Franka Panda + 작업 테이블 + 물체 + 카메라 3대(상단/측면 위/손목).

MjSpec(mujoco>=3.2) 으로 panda.xml 을 수정 없이 불러와 카메라/사이트/물체를 프로그램적으로 추가한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PANDA_XML = PROJECT_ROOT / "assets" / "mujoco_menagerie" / "franka_emika_panda" / "panda.xml"

# Panda "home" 자세 (menagerie keyframe 과 동일)
HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853], dtype=np.float64)

TABLE_TOP_Z = 0.0  # 로봇 베이스와 같은 높이를 작업면으로 사용
WORKSPACE_CENTER = np.array([0.5, 0.0, TABLE_TOP_Z])


@dataclass
class ObjectSpec:
    name: str
    rgba: tuple[float, float, float, float]
    half_size: float = 0.02  # 큐브 반변 길이 또는 원기둥 반높이 (m)
    mass: float = 0.05
    shape: str = "box"  # "box" | "cylinder"


@dataclass
class TargetSpec:
    name: str
    rgba: tuple[float, float, float, float]
    radius: float = 0.06


@dataclass
class SceneConfig:
    panda_xml: Path = DEFAULT_PANDA_XML
    objects: list[ObjectSpec] = field(
        default_factory=lambda: [
            ObjectSpec("red_cube", (0.85, 0.15, 0.15, 1.0)),
            ObjectSpec("green_cube", (0.15, 0.7, 0.2, 1.0)),
            ObjectSpec("blue_cube", (0.15, 0.3, 0.85, 1.0)),
        ]
    )
    targets: list[TargetSpec] = field(
        default_factory=lambda: [
            TargetSpec("yellow_plate", (0.95, 0.85, 0.2, 1.0)),
        ]
    )
    # 카메라 설정 (이름 -> (pos, lookat, fovy))
    top_cam: tuple = ((0.55, 0.0, 1.05), (0.55, 0.0, 0.0), 60.0)
    side_cam: tuple = ((0.75, -0.85, 0.7), (0.5, 0.0, 0.05), 55.0)


def sorting_scene_config() -> SceneConfig:
    """다중 오브젝트(3색 큐브 + 보라색 원기둥) 및 4색 접시 분류(Sorting) 씬 설정."""
    return SceneConfig(
        objects=[
            ObjectSpec("red_cube", (0.88, 0.15, 0.15, 1.0), half_size=0.02, shape="box"),
            ObjectSpec("yellow_cube", (0.95, 0.82, 0.12, 1.0), half_size=0.02, shape="box"),
            ObjectSpec("blue_cube", (0.15, 0.35, 0.90, 1.0), half_size=0.02, shape="box"),
            ObjectSpec("purple_cylinder", (0.65, 0.20, 0.88, 1.0), half_size=0.02, shape="cylinder"),
        ],
        targets=[
            TargetSpec("red_plate", (0.90, 0.25, 0.25, 1.0), radius=0.052),
            TargetSpec("yellow_plate", (0.95, 0.85, 0.20, 1.0), radius=0.052),
            TargetSpec("blue_plate", (0.20, 0.42, 0.92, 1.0), radius=0.052),
            TargetSpec("purple_plate", (0.70, 0.28, 0.90, 1.0), radius=0.052),
        ],
    )


def _lookat_xyaxes(pos, target, up=(0.0, 0.0, 1.0)) -> list[float]:
    """카메라가 target 을 바라보도록 하는 MuJoCo xyaxes 계산 (카메라는 -z 방향을 봄)."""
    pos, target, up = map(lambda v: np.asarray(v, dtype=np.float64), (pos, target, up))
    fwd = target - pos
    fwd /= np.linalg.norm(fwd)
    if abs(np.dot(fwd, up)) > 0.999:  # 수직으로 내려다보는 경우: 이미지 위쪽 = +x(로봇 전방)
        up = np.array([1.0, 0.0, 0.0])
    x = np.cross(fwd, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, fwd)
    return [*x.tolist(), *y.tolist()]


def build_scene(cfg: SceneConfig | None = None) -> mujoco.MjSpec:
    cfg = cfg or SceneConfig()
    if not Path(cfg.panda_xml).exists():
        raise FileNotFoundError(
            f"{cfg.panda_xml} 가 없습니다. 먼저 `bash scripts/setup_wsl.sh` 로 menagerie 에셋을 받으세요."
        )
    spec = mujoco.MjSpec.from_file(str(cfg.panda_xml))
    spec.modelname = "panda_tabletop"

    # 물체 freejoint 추가 시 qpos 크기가 달라지므로 기존 keyframe 제거 (home 자세는 코드에서 설정)
    for key in list(spec.keys):
        spec.delete(key)

    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    spec.visual.quality.shadowsize = 4096

    # ---- 텍스처/재질 ----
    spec.add_texture(
        name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.2, 0.3], width=512, height=512,
    )
    spec.add_material(name="grid", textures=["", "grid"], texrepeat=[5, 5], reflectance=0.1)
    spec.add_material(name="table", rgba=[0.55, 0.42, 0.3, 1.0])

    world = spec.worldbody
    world.add_light(pos=[0.5, 0.0, 2.0], dir=[0, 0, -1], diffuse=[0.7, 0.7, 0.7], castshadow=True)
    world.add_light(pos=[0.0, -1.0, 1.5], dir=[0.3, 0.7, -0.6], diffuse=[0.3, 0.3, 0.3], castshadow=False)

    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                   pos=[0, 0, -0.001], material="grid")
    # 작업 매트 (시각 전용)
    world.add_geom(name="table_mat", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.3, 0.42, 0.0005],
                   pos=[0.5, 0.0, TABLE_TOP_Z - 0.0005], material="table", contype=0, conaffinity=0)

    # ---- 고정 카메라 ----
    for cam_name, (pos, look, fovy) in (("top", cfg.top_cam), ("side", cfg.side_cam)):
        world.add_camera(name=cam_name, pos=list(pos), xyaxes=_lookat_xyaxes(pos, look), fovy=fovy)

    # ---- 손목 카메라 + TCP 사이트 (hand 바디에 부착) ----
    hand = spec.body("hand")
    hand.add_site(name="tcp", pos=[0, 0, 0.1034], size=[0.01, 0.01, 0.01], rgba=[1, 0, 0, 0.0],
                  group=4)
    # hand 프레임: +z = 손가락 방향, 손가락은 ±y 로 벌어짐.
    # 카메라를 hand 의 -x 쪽 뒤에 두고, 손가락 끝을 향해 약간 기울임.
    tilt = np.deg2rad(25.0)
    cam_z = -np.array([np.sin(tilt), 0.0, np.cos(tilt)])  # 카메라 -z(시선) = hand +z 쪽 + 약간 +x
    cam_x = np.array([0.0, -1.0, 0.0])  # 손가락/핸드가 이미지 하단에 오도록
    cam_y = np.cross(cam_z, cam_x)
    hand.add_camera(name="wrist", pos=[-0.065, 0.0, 0.03], xyaxes=[*cam_x, *cam_y], fovy=75.0)

    # ---- 물체 ----
    for i, obj in enumerate(cfg.objects):
        b = world.add_body(name=obj.name, pos=[0.4 + 0.1 * i, 0.0, TABLE_TOP_Z + obj.half_size])
        b.add_freejoint(name=f"{obj.name}_joint")
        if obj.shape == "cylinder":
            geom_type = mujoco.mjtGeom.mjGEOM_CYLINDER
            geom_size = [obj.half_size * 0.95, obj.half_size, 0.0]
        else:
            geom_type = mujoco.mjtGeom.mjGEOM_BOX
            geom_size = [obj.half_size] * 3
        b.add_geom(name=f"{obj.name}_geom", type=geom_type, size=geom_size,
                   rgba=list(obj.rgba), mass=obj.mass, friction=[1.5, 0.01, 0.001],
                   condim=4, solref=[0.01, 1], priority=1)

    for t in cfg.targets:
        b = world.add_body(name=t.name, pos=[0.5, 0.2, TABLE_TOP_Z], mocap=True)
        b.add_geom(name=f"{t.name}_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[t.radius, 0.004, 0],
                   pos=[0, 0, 0.004], rgba=list(t.rgba))

    return spec


def compile_scene(cfg: SceneConfig | None = None) -> tuple[mujoco.MjModel, mujoco.MjSpec]:
    spec = build_scene(cfg)
    model = spec.compile()
    return model, spec
