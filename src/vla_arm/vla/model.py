"""Qwen3.5-4B 백본 + Language-Conditioned Spatial Softmax (2D Keypoint Extractor) + 연속 액션 헤드 VLA.

핵심 구조:
  1) Qwen3.5 비전 인코더(get_image_features)의 순수 패치 특징 (3 카메라 x 64 패치 = 192 패치)과
     언어 모델(language_model)을 통과한 문맥 패치 특징을 결합.
  2) 패치 간 공통 상수 오프셋(outlier)을 제거하기 위해 패치 차원(N=64) 기준 평균 차감(Centering) 수행.
  3) 텍스트 지시문 특징으로 FiLM 변조된 Spatial Softmax 가 각 카메라의 8x8 그리드에서
     K개 핵심 물체/특징점의 연속 2D 좌표 (u, v) in [-1, 1]^2 와 시각 특징을 직접 추출.
  4) 추출된 2D 공간 좌표 + 시각 특징 + 고유수용성 상태(state) -> ActionHead -> (H, 7) 액션 예측.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

CAMERAS = ("top", "side", "wrist")
IMAGE_TOKEN_ID = 248056


@dataclass
class VLAConfig:
    model_id: str = "Qwen/Qwen3.5-4B"
    action_dim: int = 7
    state_dim: int = 8
    chunk_size: int = 8  # 한 번에 예측하는 미래 액션 수 (H)
    head_type: str = "mlp"  # "mlp" | "flow"
    head_hidden: int = 1024
    head_layers: int = 3
    flow_steps: int = 10
    image_size: int = 256
    num_keypoints: int = 32  # 카메라당 Spatial Softmax 가 추출할 2D 특징점 수
    load_in_4bit: bool = False  # NF4 (QLoRA 학습/Planner 동시 상주용)
    load_in_8bit: bool = False
    dtype: str = "bfloat16"
    action_offset: list[float] = field(default_factory=lambda: [0, 0, 0, 0, 0, 0, 0.5])
    action_scale: list[float] = field(default_factory=lambda: [0.03, 0.03, 0.03, 0.15, 0.15, 0.15, 0.5])
    action_loss_weights: list[float] = field(default_factory=lambda: [5.0, 5.0, 3.0, 0.5, 0.5, 2.0, 2.0])
    state_offset: list[float] = field(default_factory=lambda: [0.5, 0.0, 0.2, 0, 0, 0, 0, 0.5])
    state_scale: list[float] = field(default_factory=lambda: [0.3, 0.3, 0.2, 1, 1, 1, 1, 0.5])


def _to_pil(img, size: int) -> Image.Image:
    im = img if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img))
    return im.resize((size, size), Image.BICUBIC) if im.size != (size, size) else im


class SinusoidalTime(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # t: (B,)
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        ang = t.float()[:, None] * 1000.0 * freqs[None]
        return torch.cat([ang.sin(), ang.cos()], dim=-1)


def _mlp(i: int, h: int, o: int, n: int) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(i, h), nn.LayerNorm(h), nn.GELU()]
    for _ in range(n - 2):
        layers += [nn.Linear(h, h), nn.LayerNorm(h), nn.GELU()]
    layers.append(nn.Linear(h, o))
    return nn.Sequential(*layers)


class LanguageSpatialSoftmax(nn.Module):
    """카메라별 8x8 패치 그리드에서 지시문(텍스트) 조건부 Spatial Softmax 로
    목표 물체/특징점들의 연속 2D 좌표 (E[u], E[v]) 와 주목된 시각 특징을 추출."""

    def __init__(self, hidden: int, out_dim: int, num_cams: int = 3, grid_h: int = 8, grid_w: int = 8,
                 num_kp: int = 32, feat_dim: int = 256):
        super().__init__()
        self.num_cams = num_cams
        self.num_patches = grid_h * grid_w
        self.num_kp = num_kp
        self.feat_dim = feat_dim

        # 8x8 정규화 그리드 좌표 [-1, 1]^2 등록
        ys = torch.linspace(-1.0, 1.0, grid_h)
        xs = torch.linspace(-1.0, 1.0, grid_w)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # (64, 2)
        self.register_buffer("grid", grid, persistent=False)

        # 비전 인코더 순수 특징 + 언어 모델 문맥 특징 투영
        self.vis_proj = nn.Linear(hidden, feat_dim)
        self.llm_proj = nn.Linear(hidden, feat_dim)
        self.patch_mlp = nn.Sequential(nn.LayerNorm(feat_dim), nn.GELU(), nn.Linear(feat_dim, feat_dim))

        # 텍스트 지시문(마지막 토큰 + 텍스트 토큰 평균) -> FiLM (gamma, beta) 및 keypoint 쿼리 생성
        self.lang_proj = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, feat_dim), nn.GELU())
        self.film = nn.Linear(feat_dim, num_cams * 2 * feat_dim)
        self.kp_Scoring = nn.Linear(feat_dim, num_kp)
        # 패치별 미세 위치 보존용 저차원 병목 (Hybrid: Spatial Softmax + Patch Flatten)
        self.patch_tiny = nn.Linear(feat_dim, 16)

        # 2D 연속 좌표(Spatial Softmax) 분기와 패치 밀집 특징 분기(드롭아웃 정규화)를 50:50 비율로 균형 결합
        coord_in = num_cams * num_kp * 2 + num_cams * feat_dim + feat_dim
        patch_in = num_cams * self.num_patches * 16
        self.coord_branch = nn.Sequential(
            nn.Linear(coord_in, out_dim // 2),
            nn.LayerNorm(out_dim // 2),
            nn.GELU(),
        )
        self.patch_branch = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(patch_in, out_dim // 2),
            nn.LayerNorm(out_dim // 2),
            nn.GELU(),
        )
        self.out_proj = nn.Sequential(nn.Linear(out_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU())

    def forward(self, vis_tokens: torch.Tensor, llm_img_tokens: torch.Tensor, lang_token: torch.Tensor) -> torch.Tensor:
        """
        vis_tokens    : (B, 192, D) 비전 인코더 직접 출력 (공간/색상 원본)
        llm_img_tokens: (B, 192, D) 언어 모델 통과 후 이미지 토큰
        lang_token    : (B, D)      텍스트 지시문 요약 토큰
        """
        B, _, D = llm_img_tokens.shape
        C, P = self.num_cams, self.num_patches
        v = vis_tokens.view(B, C, P, D)
        l = llm_img_tokens.view(B, C, P, D)

        # 핵심: 카메라 내부 64개 패치 간 공통 상수 배경/outlier 제거 (Spatial Centering)
        v = v - v.mean(dim=2, keepdim=True)
        l = l - l.mean(dim=2, keepdim=True)
        v = F.normalize(v, dim=-1)
        l = F.normalize(l, dim=-1)
        if self.training:
            v = F.dropout(v, p=0.1, training=True)
            l = F.dropout(l, p=0.1, training=True)

        x = self.patch_mlp(self.vis_proj(v) + self.llm_proj(l))  # (B, C, 64, feat_dim)

        # 지시문 조건부 FiLM 변조
        lang = self.lang_proj(lang_token - lang_token.mean(dim=-1, keepdim=True))  # (B, feat_dim)
        if self.training:
            lang = F.dropout(lang, p=0.2, training=True)
        gamma, beta = self.film(lang).view(B, C, 1, 2 * self.feat_dim).chunk(2, dim=-1)
        x = x * (1.0 + gamma) + beta  # (B, C, 64, feat_dim)

        # Spatial Softmax: (B, C, 64, num_kp) -> 패치 차원(dim=2)에 대해 softmax
        logits = self.kp_Scoring(F.gelu(x)).transpose(2, 3)  # (B, C, num_kp, 64)
        attn = F.softmax(logits * 2.0, dim=-1)  # (B, C, num_kp, 64)

        # 기대 2D 좌표 (B, C, num_kp, 2)
        coords = torch.matmul(attn, self.grid.to(attn.dtype))  # (B, C, num_kp, 2)
        # 주목된 패치들의 시각 특징 평균 (B, C, feat_dim)
        pooled_feat = torch.matmul(attn.mean(dim=2, keepdim=True), x).squeeze(2)  # (B, C, feat_dim)
        # 패치 그리드 직접 Flatten (손목 카메라 근접 정렬용)
        patch_flat = self.patch_tiny(F.gelu(x)).reshape(B, -1)

        c_out = self.coord_branch(torch.cat([coords.reshape(B, -1), pooled_feat.reshape(B, -1), lang], dim=-1))
        p_out = self.patch_branch(patch_flat)
        return self.out_proj(torch.cat([c_out, p_out], dim=-1))


class ActionHead(nn.Module):
    def __init__(self, cfg: VLAConfig, cond_dim: int):
        super().__init__()
        self.cfg = cfg
        out = cfg.chunk_size * cfg.action_dim
        if cfg.head_type == "mlp":
            self.net = _mlp(cond_dim, cfg.head_hidden, out, cfg.head_layers)
        elif cfg.head_type == "flow":
            self.time_emb = SinusoidalTime(128)
            self.net = _mlp(cond_dim + out + 128, cfg.head_hidden, out, cfg.head_layers)
        else:
            raise ValueError(cfg.head_type)
        nn.init.zeros_(self.net[-1].bias)
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        self.register_buffer(
            "loss_weights",
            torch.tensor(cfg.action_loss_weights, dtype=torch.float32).view(1, 1, cfg.action_dim),
            persistent=False,
        )

    def loss(self, cond: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        B = cond.shape[0]
        if self.cfg.head_type == "mlp":
            pred = self.net(cond).view(B, self.cfg.chunk_size, self.cfg.action_dim)
            w = self.loss_weights.to(pred.device)
            # 희소 액션(대부분의 스텝에서 dx=dy=drz=0)이 L1 중앙값(0)에 갇히는 현상(Zero-Trapping)을 막기 위해
            # MSE loss + 실제 이동이 일어나는 비영(non-zero) 액션 구간에 4배 가중치 부여 + 첫 액션 정밀도 강조
            active = 1.0 + 4.0 * (actions.abs() > 0.15).float()
            t_decay = torch.linspace(1.0, 0.3, self.cfg.chunk_size, device=pred.device).view(1, -1, 1)
            return ((pred - actions).pow(2) * w * active * t_decay).mean()
        a = actions.reshape(B, -1)
        noise = torch.randn_like(a)
        t = torch.distributions.Beta(1.5, 1.0).sample((B,)).to(a.device)
        x_t = t[:, None] * a + (1 - t[:, None]) * noise
        v = self.net(torch.cat([cond, x_t, self.time_emb(t)], dim=-1))
        return F.mse_loss(v, a - noise)

    @torch.no_grad()
    def sample(self, cond: torch.Tensor) -> torch.Tensor:
        B = cond.shape[0]
        out = self.cfg.chunk_size * self.cfg.action_dim
        if self.cfg.head_type == "mlp":
            a = self.net(cond)
        else:
            a = torch.randn(B, out, device=cond.device, dtype=cond.dtype)
            dt = 1.0 / self.cfg.flow_steps
            for k in range(self.cfg.flow_steps):
                t = torch.full((B,), k * dt, device=cond.device, dtype=cond.dtype)
                a = a + dt * self.net(torch.cat([cond, a, self.time_emb(t)], dim=-1))
        return a.reshape(B, self.cfg.chunk_size, self.cfg.action_dim)


class QwenVLA(nn.Module):
    def __init__(self, cfg: VLAConfig | None = None, device: str = "cuda:0", load_backbone: bool = True):
        super().__init__()
        self.cfg = cfg = cfg or VLAConfig()
        dtype = getattr(torch, cfg.dtype)
        self.backbone = None
        self.processor = None
        hidden = 2560
        if load_backbone:
            from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

            quant = None
            skip = ["visual", "lm_head", "mtp"]  # 비전 인코더는 양자화하지 않음 (공간 정밀도 보존)
            if cfg.load_in_4bit:
                quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
                                           llm_int8_skip_modules=skip)
            elif cfg.load_in_8bit:
                quant = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip)
            self.processor = AutoProcessor.from_pretrained(cfg.model_id, padding_side="left")
            vlm = AutoModelForImageTextToText.from_pretrained(
                cfg.model_id, dtype=dtype, quantization_config=quant, device_map=device,
            )
            self.backbone = vlm.model
            hidden = vlm.config.text_config.hidden_size
            del vlm

        d = cfg.head_hidden
        self.ctx_proj = LanguageSpatialSoftmax(hidden, d, num_kp=cfg.num_keypoints)
        self.state_proj = nn.Sequential(
            nn.Linear(cfg.state_dim, d), nn.LayerNorm(d), nn.GELU(),
            nn.Linear(d, d), nn.LayerNorm(d),
        )
        self.head = ActionHead(cfg, cond_dim=2 * d)
        for m in (self.ctx_proj, self.state_proj, self.head):
            m.to(device=device, dtype=torch.float32)

        self.device = torch.device(device)
        self.dtype = dtype
        self._register_norm_buffers()

    # ---------------------------------------------------------------- norm
    def _register_norm_buffers(self):
        c = self.cfg
        for name in ("action_offset", "action_scale", "state_offset", "state_scale"):
            self.register_buffer(name + "_t", torch.tensor(getattr(c, name), dtype=torch.float32,
                                                           device=self.device), persistent=False)

    def normalize_actions(self, a: torch.Tensor) -> torch.Tensor:
        return (a - self.action_offset_t) / self.action_scale_t

    def denormalize_actions(self, a: torch.Tensor) -> torch.Tensor:
        return a.float() * self.action_scale_t + self.action_offset_t

    def normalize_state(self, s: torch.Tensor) -> torch.Tensor:
        return (s.float() - self.state_offset_t) / self.state_scale_t

    # ---------------------------------------------------------------- inputs
    def build_inputs(self, images: list[dict], texts: list[str]) -> dict:
        convs = []
        for imgs, txt in zip(images, texts):
            content = [{"type": "text", "text": f"Task: {txt}\n"}]
            for cam in CAMERAS:
                content.append({"type": "text", "text": f"{cam}:"})
                content.append({"type": "image", "image": _to_pil(imgs[cam], self.cfg.image_size)})
            content.append({"type": "text", "text": "Action:"})
            convs.append([{"role": "user", "content": content}])
        inputs = self.processor.apply_chat_template(
            convs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
            processor_kwargs={"padding": True}, enable_thinking=False,
        )
        return inputs.to(self.device)

    def _base_model(self):
        return self.backbone.get_base_model() if hasattr(self.backbone, "get_base_model") else self.backbone

    def encode(self, inputs: dict, state: torch.Tensor) -> torch.Tensor:
        B = state.shape[0]
        base = self._base_model()
        with torch.no_grad():
            vis_list = base.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"]).pooler_output
            vis_tokens = torch.cat(vis_list, dim=0).float().view(B, -1, base.config.text_config.hidden_size)

        out = self.backbone(**inputs, use_cache=False, return_dict=True)
        h = out.last_hidden_state.float()  # (B, T, D)
        img_mask = inputs["input_ids"] == IMAGE_TOKEN_ID
        llm_img_tokens = h[img_mask].view(B, -1, h.shape[-1])
        lang_token = h[:, -1]

        ctx = self.ctx_proj(vis_tokens, llm_img_tokens, lang_token)
        st = self.state_proj(self.normalize_state(state))
        return torch.cat([ctx, st], dim=-1)

    # ---------------------------------------------------------------- train / infer
    def forward(self, images: list[dict], texts: list[str], state: torch.Tensor,
                actions: torch.Tensor) -> torch.Tensor:
        cond = self.encode(self.build_inputs(images, texts), state.to(self.device))
        a = self.normalize_actions(actions.to(self.device).float())
        return self.head.loss(cond, a)

    @torch.inference_mode()
    def predict(self, images: dict, text: str, state: np.ndarray) -> np.ndarray:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)[None]
        cond = self.encode(self.build_inputs([images], [text]), s)
        a = self.denormalize_actions(self.head.sample(cond))[0]
        return a.cpu().numpy()

    # ---------------------------------------------------------------- io
    def trainable_head_state_dict(self) -> dict:
        return {f"{n}.{k}": v for n in ("ctx_proj", "state_proj", "head")
                for k, v in getattr(self, n).state_dict().items()}

    def save_head(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.trainable_head_state_dict(), path / "action_head.pt")
        (path / "vla_config.json").write_text(json.dumps(asdict(self.cfg), indent=2))

    def load_head(self, path: str | Path):
        sd = torch.load(Path(path) / "action_head.pt", map_location=self.device)
        for n in ("ctx_proj", "state_proj", "head"):
            getattr(self, n).load_state_dict({k[len(n) + 1:]: v for k, v in sd.items() if k.startswith(n + ".")})
